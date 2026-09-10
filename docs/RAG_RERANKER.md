# 🧠 RAG Retrieval, Cross-Encoder Reranking & Semantic Caching Subsystem

## 1. Overview & Architecture Philosophy

FlashStore's **RAG & Reranker Subsystem** extends the core Log-Structured Merge-Tree (LSM-Tree) key-value engine with native Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) capabilities. Instead of treating vector search and text search as disconnected external databases, FlashStore integrates:

1. **Storage Persistence**: Raw passages, chunked documents, vector embeddings, and metadata dictionaries are committed atomically into FlashStore's Write-Ahead Log (WAL) and SSTables via multi-operation `WriteBatch` transactions.
2. **Hybrid Retrieval**: Parallel sparse lexical search (BM25 with tokenization and inverted index) and dense semantic vector search (cosine similarity, dot product, or Euclidean distance).
3. **Multi-Strategy Fusion**: Dynamic rank aggregation combining lexical and dense candidate lists using Reciprocal Rank Fusion (RRF), Weighted Linear Combination, or Borda Count.
4. **Second-Stage Cross-Encoder Reranking**: High-precision lexical-semantic cross-encoder evaluation with token coverage, phrase/bigram alignment, term proximity, and Maximal Marginal Relevance (MMR) diversity reranking.
5. **Semantic Vector Query Cache**: Multi-tier cache providing sub-millisecond query response for identical or semantically equivalent queries using cosine similarity thresholds, TTL expiration, and LRU eviction.
6. **Prompt Context Assembly & Citations**: Token-budget-aware context packing with configurable truncation (`TruncateLast`, `DropOversized`), structured layouts (Markdown, XML, Numbered, Compact), and 1-based citation references (`[1]`, `[2]`).
7. **IR Telemetry & Evaluation**: Built-in evaluation harness measuring Mean Reciprocal Rank (MRR), Recall@K, Precision@K, NDCG@K, and nanosecond-granularity stage latency tracking.

---

## 2. End-to-End Pipeline Architecture

```
                                 [ User Query ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │     Semantic Vector Cache         │
                     │  (Exact Match / Cosine Threshold) ├───────────┐
                     └─────────────────┬─────────────────┘           │
                                       │ Cache Miss                  │ Cache Hit
                                       ▼                             │ (Fast Return)
                     ┌───────────────────────────────────┐           │
                     │    Query Vector Embedding         │           │
                     │  (EmbeddingProvider Interface)    │           │
                     └─────────────────┬─────────────────┘           │
                                       │                             │
                     ┌─────────────────┴─────────────────┐           │
                     ▼                                   ▼           │
       ┌───────────────────────────┐       ┌─────────────────────────┴─┐
       │    Sparse Lexical Search  │       │    Dense Vector Search    │
       │  (BM25 Inverted Index)    │       │  (Cosine / Dot / L2)      │
       └─────────────┬─────────────┘       └─────────────┬─────────────┘
                     │ Top-K Sparse                      │ Top-K Dense
                     └─────────────────┬─────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │      Hybrid Score Fusion          │
                     │   (RRF / Weighted Linear / Borda) │
                     └─────────────────┬─────────────────┘
                                       │ Fused Candidates
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Cross-Encoder Reranker         │
                     │ (Lexical-Semantic / MMR Diversity)│
                     └─────────────────┬─────────────────┘
                                       │ Reranked Candidates
                                       ▼
                     ┌───────────────────────────────────┐
                     │     Context Assembler & Prompter  │
                     │ (Token Budget / Citations / XML)  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     [ LLM Prompt / Pipeline Result ] ◄──────────────┘
```

---

## 3. FlashStore LSM-Tree Storage Persistence

All RAG assets are persisted deterministically within FlashStore's unified LSM-Tree key-space using standardized binary key prefixes. This guarantees zero schema drift, crash consistency via WAL, and efficient prefix-range scans for rehydration.

### 3.1 Key Prefix Namespace

| Prefix | Constant | Type | Payload Description |
| :--- | :--- | :--- | :--- |
| `rag:doc:` | `PREFIX_DOC` | `[u8; 8]` | Raw UTF-8 document content string. |
| `rag:vec:` | `PREFIX_VEC` | `[u8; 8]` | Dense vector embedding serialized as little-endian IEEE-754 `[f32]` array. |
| `rag:meta:` | `PREFIX_META` | `[u8; 9]` | Bincode or JSON-serialized `DocumentMetadata` map with arbitrary typed values. |
| `rag:chunk:`| `PREFIX_CHUNK`| `[u8; 10]` | Serialized `DocumentChunk` records containing chunk offset, text, and vector. |
| `rag:sys:` | `PREFIX_SYS` | `[u8; 8]` | System metadata, schema versions, and telemetry counters. |

### 3.2 Atomic Ingestion via `WriteBatch`

When a document is ingested via `RagStoreAdapter::persist_document`:
```rust
let mut batch = WriteBatch::new();
// 1. Stage raw text
batch.put(RagStoreAdapter::doc_key(doc.id.as_str()), doc.text.as_bytes());
// 2. Stage embedding vector (if present)
if let Some(emb) = &doc.embedding {
    batch.put(RagStoreAdapter::vec_key(doc.id.as_str()), emb.to_bytes());
}
// 3. Stage serialized metadata
let meta_bytes = serde_json::to_vec(&doc.metadata)?;
batch.put(RagStoreAdapter::meta_key(doc.id.as_str()), meta_bytes);
// 4. Stage individual chunks
for chunk in &doc.chunks {
    let chunk_key = RagStoreAdapter::chunk_key(doc.id.as_str(), chunk.index);
    batch.put(chunk_key, serde_json::to_vec(chunk)?);
}
// 5. Commit atomically to WAL and MemTable
store.write_batch(batch)?;
```
If a power failure occurs midway through ingestion, FlashStore's WAL guarantees all records for the document either commit together or none do.

### 3.3 Recovery & Rehydration

Upon starting `RagEngine::recover(config, store)`:
1. The engine iterates over `PREFIX_DOC` keys to discover all valid document IDs.
2. For each document, it fetches the raw text, vector embedding (`PREFIX_VEC`), metadata (`PREFIX_META`), and chunks (`PREFIX_CHUNK`).
3. In-memory indices (`DenseIndex`, `SparseIndex`) are reconstructed in parallel without requiring re-computation of vector embeddings or re-running tokenizer models.

---

## 4. Document Ingestion & Chunking Pipeline

Large documents must be divided into smaller semantic units for effective dense and sparse retrieval. FlashStore provides multiple configurable chunking strategies:

### 4.1 Chunking Strategies (`ChunkingStrategy`)

1. **`Paragraphs` / `MarkdownAware`**: Splits text on double newlines (`\n\n`), markdown headings (`#`, `##`, `###`), or horizontal rules. Preserves header context for nested sections.
2. **`SentenceAware`**: Splits on sentence boundaries (`. `, `! `, `? `) while honoring max chunk size and overlap boundaries. Avoids cutting phrases mid-thought.
3. **`FixedSize` (Character or Token)**: Splits text into uniform chunks of `chunk_size` with a sliding overlap of `chunk_overlap`.
4. **`SlidingWindow`**: Shifts a window of size $W$ by step size $S = W - O$.

### 4.2 Configuration Parameters (`ChunkingConfig`)

```rust
let config = ChunkingConfig {
    strategy: ChunkingStrategy::SentenceAware,
    chunk_size: 512,      // target characters or tokens per chunk
    chunk_overlap: 64,    // overlapping characters between consecutive chunks
    preserve_headers: true,
};
```

Each chunk retains its parent `DocumentId`, 0-based chunk index, character offset span `[start, end]`, and optional independent embedding vector.

---

## 5. Dense Semantic Vector Search

Dense retrieval encodes queries and passages as continuous multi-dimensional vectors, capturing semantic intent and synonymous meaning.

### 5.1 Similarity Metrics (`SimilarityMetric`)

- **Cosine Similarity** (Default):
  $$\text{Cosine}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
  Normalizes vectors to unit sphere upon insertion. Point lookups compute dot product in $O(D)$ time with SIMD-friendly loops.
- **Dot Product**:
  $$\text{Dot}(\vec{u}, \vec{v}) = \sum_{i=1}^D u_i \cdot v_i$$
  Suitable for pre-normalized embeddings (e.g. OpenAI text-embedding-3, Cohere Embed v3).
- **Euclidean Distance**:
  $$\text{Euclidean}(\vec{u}, \vec{v}) = \frac{1}{1 + \sqrt{\sum_{i=1}^D (u_i - v_i)^2}}$$
  Normalized to $[0.0, 1.0]$ where identical vectors yield $1.0$.

### 5.2 Dense Index Implementation (`DenseIndex`)

`DenseIndex` maintains:
- Contiguous flattened vector storage for CPU cache locality.
- Precomputed L2 norms for instantaneous cosine similarity calculation.
- Fast candidate ranking with bounded min-heaps for top-K extraction.

---

## 6. Sparse Lexical Search (BM25)

Sparse search identifies exact keyword occurrences, technical identifiers, model codes, and rare terminology that dense embeddings often dilute.

### 6.1 Tokenization & Normalization

- Converts text to lowercase.
- Strips punctuation and non-alphanumeric separators.
- Drops short stopwords (configurable).
- Emits clean alphanumeric tokens.

### 6.2 Okapi BM25 Formulation

The relevance score of document $D$ for query $Q = \{q_1, q_2, \dots, q_n\}$ is calculated as:

$$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$ is the term frequency of token $q_i$ in document $D$.
- $|D|$ is the length of document $D$ in tokens.
- $\text{avgdl}$ is the average document length across the entire index.
- $k_1$ controls term frequency saturation (default: `1.2`).
- $b$ controls document length normalization penalty (default: `0.75`).
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$ where $N$ is total documents and $n(q_i)$ is document frequency.

---

## 7. Hybrid Search & Multi-Strategy Fusion

FlashStore runs sparse and dense retrieval in parallel and merges the candidate rankings using one of three fusion strategies:

### 7.1 Reciprocal Rank Fusion (RRF)

RRF combines ranks without requiring score calibration or normalization across heterogeneous scoring distributions:

$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$

Where $k$ is the smoothing constant (default: `60`). Documents appearing near the top of both lists receive strong reinforcement.

### 7.2 Weighted Linear Fusion

Normalizes dense and sparse scores into $[0.0, 1.0]$ via min-max or z-score normalization, then calculates:

$$\text{Score}_{\text{linear}}(d) = w_{\text{dense}} \cdot S_{\text{dense}}(d) + (1.0 - w_{\text{dense}}) \cdot S_{\text{sparse}}(d)$$

### 7.3 Borda Count

Elects consensus winners by awarding points inversely proportional to rank position across both systems.

---

## 8. Second-Stage Cross-Encoder Reranking

Initial retrieval extracts top-$N$ candidates (e.g. $N = 25$). The Cross-Encoder reranker performs intensive pairwise evaluation on $(Query, Candidate)$ pairs to determine the final top-$K$ ordering (e.g. $K = 5$).

### 8.1 `LexicalSemanticCrossEncoder`

Computes a composite relevance score based on three complementary factors:

1. **Token Coverage** (weight 0.5): Ratio of unique query terms present in the document.
2. **Phrase & Bigram Alignment** (weight 0.3): Ratio of contiguous query word pairs matching exact document sequences.
3. **Term Proximity & Density** (weight 0.2): Compactness of the window spanning all matched query terms.

$$\text{Score}_{\text{cross}} = w_{\text{tok}} \cdot C_{\text{tok}} + w_{\text{phr}} \cdot C_{\text{phr}} + w_{\text{prox}} \cdot C_{\text{prox}}$$

### 8.2 Score Explanations (`ScoreExplanation`)

When queried with `--explain`, the reranker emits a diagnostic breakdown:
```json
{
  "token_coverage": 1.0,
  "phrase_match": 0.8,
  "proximity": 0.92,
  "combined_score": 0.924
}
```

### 8.3 Maximal Marginal Relevance (MMR) Diversity Reranking

Reduces redundancy in the returned context passages by penalizing candidates that are overly similar to already selected results:

$$\text{MMR} = \operatorname*{arg\,max}_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right]$$

Where:
- $S$ is the set of already selected documents.
- $R \setminus S$ is the set of remaining unselected candidates.
- $\lambda \in [0.0, 1.0]$ balances relevance vs diversity (default: `0.7`).

---

## 9. Semantic Vector Query Cache

Redundant LLM queries are expensive. FlashStore provides an in-memory semantic cache that intercepts incoming queries before retrieval.

### 9.1 Multi-Tier Hit Detection

1. **Tier 1 — Exact Match ($O(1)$)**: Direct string match against cached query text.
2. **Tier 2 — Cosine Semantic Match ($O(M)$)**: Computes cosine similarity between current query embedding and all cached query embeddings. If $\text{sim}(\vec{q}_{\text{curr}}, \vec{q}_{\text{cached}}) \ge \text{threshold}$ (default: `0.92`), the cached result is returned immediately.

### 9.2 Cache Invalidation & Eviction

- **TTL Expiration**: Entries older than `semantic_cache_ttl_secs` (default: 3600s) are treated as misses and lazily purged.
- **LRU Eviction**: When cache exceeds `semantic_cache_capacity` (default: 10,000), least-recently-used entries are evicted.
- **Atomic Telemetry**: Tracks cumulative hits, misses, hit ratio, and average hit latency.

---

## 10. Prompt Context Assembly & Citations

The `ContextAssembler` converts retrieved `ScoredDocument` candidates into structured prompts tailored for LLM consumption.

### 10.1 Layout Formats (`ContextFormat`)

- **`Markdown`** (Default):
  ```markdown
  Relevant Context Passages:

  ### [1] doc_lsm_overview (Score: 0.942)
  FlashStore uses an append-only Write-Ahead Log...

  ### [2] doc_compaction (Score: 0.887)
  Leveled compaction merges overlapping SSTables...
  ```
- **`Xml`**:
  ```xml
  <documents>
    <document index="1" id="doc_lsm_overview" score="0.942">
      FlashStore uses an append-only Write-Ahead Log...
    </document>
  </documents>
  ```
- **`Numbered`**:
  ```text
  [1] [doc_lsm_overview]: FlashStore uses an append-only Write-Ahead Log...
  [2] [doc_compaction]: Leveled compaction merges overlapping SSTables...
  ```
- **`Compact`**: Stripped delimiters for maximum token density.

### 10.2 Token Budget & Truncation (`TruncationStrategy`)

- **`max_chars`**: Hard upper bound on assembled context character count.
- **`TruncateLast`**: Fills budget with as many whole documents as possible, truncating the final document to fit cleanly.
- **`DropOversized`**: Strictly drops any document that would cause total size to exceed budget.

### 10.3 Structured Prompt Templates (`RagPromptTemplate`)

Generates a unified LLM prompt with system instructions, user query, citations, and assembled context passages.

---

## 11. IR Evaluation & Telemetry

FlashStore includes a complete Information Retrieval evaluation suite for testing search quality against ground-truth benchmarks.

### 11.1 Standard Metrics Supported

- **Recall@K**: Ratio of relevant documents retrieved within the top $K$ positions.
- **Precision@K**: Ratio of retrieved documents within top $K$ that are relevant.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures ranking quality with logarithmic position discounting against ideal ranking (IDCG).

### 11.2 Telemetry Latency Breakdown (`StageLatencyTracker`)

Microsecond-resolution timing across all pipeline stages:
- Vector embedding generation
- Sparse inverted index traversal
- Dense vector dot-product computation
- Fusion rank aggregation
- Cross-encoder reranking
- Context prompt formatting

---

## 12. Rust API Reference

### 12.1 End-to-End Pipeline Setup

```rust
use flash_store::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

fn main() -> Result<()> {
    let dir = tempdir()?;
    let store_opts = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(store_opts)?);

    // 1. Configure RAG Engine
    let rag_config = RagConfigBuilder::new()
        .embedding_dim(64)
        .similarity_metric(SimilarityMetric::Cosine)
        .hybrid_dense_weight(0.5)
        .rrf_k(60)
        .semantic_cache(1000, 0.92, 3600)
        .build();

    // 2. Initialize Engine backed by FlashStore
    let engine = Arc::new(parking_lot::RwLock::new(
        RagEngine::with_store(rag_config, store.clone())
    ));

    // 3. Initialize Embedding Provider and Pipeline
    let embedder = Arc::new(MockEmbeddingProvider::new(64));
    let pipeline = RagPipeline::new(engine, embedder);

    // 4. Ingest Documents
    let doc = DocumentBuilder::new("arch_guide")
        .text("FlashStore utilizes a Write-Ahead Log for durability and MemTables for fast writes.")
        .title("Architecture Overview")
        .author("Zein")
        .tag("storage")
        .tag("lsm")
        .metadata_field("version", 1)
        .build();

    let report = pipeline.ingest_document(doc, ChunkingConfig::default())?;
    println!("Ingested doc with {} chunks in {} ms", report.chunks_created, report.duration_ms);

    // 5. Query with Hybrid Retrieval and Cross-Encoder Reranking
    let query = RagQueryBuilder::new("How does durability work?")
        .top_k(3)
        .dense_weight(0.6)
        .sparse_weight(0.4)
        .fusion(FusionStrategy::ReciprocalRankFusion)
        .explain(true)
        .build();

    let result = pipeline.query_with_context(query, ContextConfig::default())?;
    println!("Retrieved {} candidates in {} ms", result.candidates.len(), result.duration_ms);
    println!("Assembled prompt:\n{}", result.assembled_context.context_text);

    Ok(())
}
```

---

## 13. CLI Command Reference (`flash-cli rag`)

The `flash-cli` tool provides a comprehensive CLI for managing and querying the RAG subsystem.

### 13.1 Ingestion (`ingest`)

```bash
# Ingest raw text
flash-cli rag ingest --id doc1 --text "LSM trees optimize write performance using append-only logs." --title "LSM Intro" --tags lsm,database

# Ingest a text or markdown file
flash-cli rag ingest --file docs/ARCHITECTURE.md --title "Architecture Guide" --chunk-strategy paragraphs --chunk-size 512

# Batch ingest JSON file
flash-cli rag ingest --batch-file data/knowledge_base.json
```

### 13.2 Hybrid Query (`query`)

```bash
# Basic query
flash-cli rag query "How are writes handled?" --top-k 5

# Hybrid query with Reciprocal Rank Fusion and MMR diversity
flash-cli rag query "SSTable compression" --fusion rrf --mmr --mmr-lambda 0.7

# Query with score explanation in JSON
flash-cli rag query "WAL crash recovery" --explain --format json

# Emit ready LLM prompt with citations
flash-cli rag query "Explain block cache" --prompt --format prompt
```

### 13.3 Inspection & Management (`get`, `list`, `delete`, `stats`, `clearcache`)

```bash
# Fetch indexed document
flash-cli rag get doc1

# List indexed documents
flash-cli rag list --limit 10

# Delete document and associated chunks/embeddings
flash-cli rag delete doc1

# View RAG telemetry, cache hit ratio, and index statistics
flash-cli rag stats --format text

# Clear semantic vector cache
flash-cli rag clearcache
```

### 13.4 Information Retrieval Benchmark Evaluation (`eval`)

```bash
flash-cli rag eval --samples-file benchmarks/eval_queries.json --k 5 --format text
```

---

## 14. Configuration & Tuning Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `embedding_dim` | `usize` | `384` | Dimensionality of dense vector embeddings. |
| `similarity_metric` | `SimilarityMetric` | `Cosine` | Vector distance function (`Cosine`, `DotProduct`, `Euclidean`). |
| `bm25_k1` | `f32` | `1.2` | BM25 term frequency saturation parameter. Higher values increase sensitivity to repeated terms. |
| `bm25_b` | `f32` | `0.75` | BM25 document length normalization parameter ($0.0 = \text{no penalty}, 1.0 = \text{full penalty}$). |
| `hybrid_dense_weight`| `f32` | `0.5` | Dense search weight in linear fusion ($0.0 = \text{lexical only}, 1.0 = \text{dense only}$). |
| `rrf_k` | `usize` | `60` | Smoothing constant in Reciprocal Rank Fusion. |
| `semantic_cache_capacity` | `usize` | `10,000` | Maximum number of cached query embeddings in LRU cache. |
| `semantic_cache_threshold`| `f32` | `0.92` | Minimum cosine similarity required to trigger a semantic cache hit. |
| `semantic_cache_ttl_secs` | `u64` | `3600` | Time-to-live for cache entries in seconds ($0 = \text{infinite}$). |
