//! # FlashStore RAG Pipeline Example
//!
//! Demonstrates end-to-end Retrieval-Augmented Generation (RAG) using FlashStore:
//! 1. Initializing an embedded FlashStore LSM-tree database.
//! 2. Configuring hybrid search (BM25 sparse + vector dense) and semantic caching.
//! 3. Ingesting documents with automated chunking, metadata, and embeddings.
//! 4. Querying with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
//! 5. Assembling an LLM-ready prompt context with structured citations.
//! 6. Verifying semantic cache acceleration on repeated queries.

use flash_store::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

fn main() -> Result<()> {
    println!("=== FlashStore RAG & Reranker Pipeline Demo ===");

    // 1. Initialize persistent FlashStore storage engine in a temporary directory
    let dir = tempdir().map_err(FlashStoreError::Io)?;
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(2 * 1024 * 1024)
        .build();
    let db = Arc::new(FlashStore::open(options)?);
    println!("[1/6] Initialized underlying FlashStore LSM storage engine.");

    // 2. Configure RAG Engine parameters
    let rag_config = RagConfigBuilder::new()
        .embedding_dim(32)
        .similarity_metric(SimilarityMetric::Cosine)
        .hybrid_dense_weight(0.5)
        .rrf_k(60)
        .semantic_cache(100, 0.90, 3600)
        .build();

    let engine = RagEngine::with_store(rag_config, db.clone());

    // 3. Create RAG Pipeline with deterministic offline embedding provider
    let embedder = Arc::new(MockEmbeddingProvider::new(32));
    let chunking_config = ChunkingConfig {
        strategy: ChunkingStrategy::Paragraph,
        min_chunk_size: 20,
    };
    let context_config = ContextConfig {
        format: ContextFormat::Markdown,
        max_chars: 4096,
        include_scores: true,
        include_metadata: true,
        header: Some("=== Relevant FlashStore Knowledge Base Passages ===".to_string()),
        ..Default::default()
    };

    let mut pipeline = RagPipeline::new(engine)
        .with_embedder(embedder)
        .with_chunking(chunking_config)
        .with_context_config(context_config);

    println!("[2/6] Configured RAG pipeline with 32-dim embeddings and semantic cache.");

    // 4. Ingest sample documents with rich metadata
    let docs = vec![
        (
            "doc_lsm".to_string(),
            "LSM trees optimize write throughput by transforming random disk writes into \
             sequential append operations on a Write-Ahead Log (WAL) and concurrent MemTable. \
             Periodically, immutable MemTables are flushed to Level 0 SSTable files on disk."
                .to_string(),
            {
                let mut meta = DocumentMetadata::new();
                meta.insert("title", "Log-Structured Merge-Trees");
                meta.insert("author", "Database Systems Group");
                meta.insert("tag", "lsm");
                meta.insert("category", "database_internals");
                Some(meta)
            },
        ),
        (
            "doc_sstable".to_string(),
            "An SSTable (Sorted String Table) is an immutable on-disk file consisting of \
             ordered Data Blocks, an Index Block, and a probabilistic Bloom Filter. \
             Bloom filters eliminate unnecessary disk I/O by confirming non-existent keys in O(1) time."
                .to_string(),
            {
                let mut meta = DocumentMetadata::new();
                meta.insert("title", "SSTable File Format and Bloom Filters");
                meta.insert("author", "Storage Team");
                meta.insert("tag", "sstable");
                meta.insert("category", "database_internals");
                Some(meta)
            },
        ),
        (
            "doc_compaction".to_string(),
            "Leveled Compaction merges overlapping SSTables across adjacent tiers (e.g. L0 to L1). \
             Compaction purges obsolete key revisions and expired tombstones, reclaiming storage space \
             and bounding read amplification for point lookups and range queries."
                .to_string(),
            {
                let mut meta = DocumentMetadata::new();
                meta.insert("title", "Leveled Compaction Mechanics");
                meta.insert("author", "Performance Engineering");
                meta.insert("tag", "compaction");
                meta.insert("category", "maintenance");
                Some(meta)
            },
        ),
        (
            "doc_semantic_cache".to_string(),
            "A semantic vector cache stores query embeddings and their retrieval results. \
             When a semantically equivalent query arrives with cosine similarity exceeding a \
             configured threshold (e.g. 0.90), cached results are returned instantly without \
             traversing the search index."
                .to_string(),
            {
                let mut meta = DocumentMetadata::new();
                meta.insert("title", "Semantic Vector Caching");
                meta.insert("author", "AI Infra Team");
                meta.insert("tag", "cache");
                meta.insert("category", "rag");
                Some(meta)
            },
        ),
    ];

    println!(
        "[3/6] Batch ingesting {} documents into RAG engine...",
        docs.len()
    );
    let report = pipeline.batch_ingest(docs)?;
    println!(
        "  -> Successfully ingested {} docs ({} chunks, {} chars) in {} ms",
        report.docs_ingested, report.chunks_created, report.total_chars, report.elapsed_ms
    );

    // 5. Execute hybrid query with cross-encoder reranking
    let query_str = "How do SSTables and Bloom filters accelerate reads?";
    println!("\n[4/6] Executing hybrid query: \"{}\"", query_str);

    let result = pipeline.query(query_str, 3)?;

    println!("\nTop Retrieved Candidates (with Reranker Explanations):");
    for (rank, doc) in result.documents.iter().enumerate() {
        println!(
            "  {}. ID: {:<18} Combined Score: {:.4}",
            rank + 1,
            doc.id.as_str(),
            doc.score
        );
        if let (Some(dense), Some(sparse)) = (doc.dense_score, doc.sparse_score) {
            println!(
                "     Dense Score: {:.4} | Sparse (BM25): {:.4}",
                dense, sparse
            );
        }
        if let Some(ref exp) = doc.explanation {
            println!(
                "     Cross-Encoder -> Token Coverage: {:.2} | Phrase: {:.2} | Proximity: {:.2}",
                exp.token_coverage, exp.phrase_match, exp.proximity
            );
        }
    }

    println!("\n[5/6] Assembled LLM Prompt Context (Markdown layout with citations):");
    println!("------------------------------------------------------------");
    println!("{}", result.context.text);
    println!("------------------------------------------------------------");
    println!(
        "Citations generated: {} (Total characters: {}, Truncated: {})",
        result.context.citations.len(),
        result.context.total_chars,
        result.context.was_truncated
    );

    println!("\nFull Assembled LLM Prompt Preview:");
    println!("------------------------------------------------------------");
    println!("{}", result.prompt);
    println!("------------------------------------------------------------");

    // 6. Test Semantic Cache acceleration
    println!("\n[6/6] Testing Semantic Vector Cache hit on repeated query...");
    let _cached_res = pipeline.query(query_str, 3)?;

    // Print final engine telemetry statistics
    let metrics = pipeline.engine().metrics();
    println!("\n=== Engine Telemetry Summary ===");
    println!("Total queries executed:  {}", metrics.total_searches);
    println!("Semantic cache hits:    {}", metrics.cache_hits);
    println!("Semantic cache misses:  {}", metrics.cache_misses);
    println!(
        "Cache hit ratio:        {:.1}%",
        metrics.cache_hit_rate * 100.0
    );

    println!("\nDemo completed successfully!");
    Ok(())
}
