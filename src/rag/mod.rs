//! # RAG Retrieval, Cross-Encoder Reranking, and Semantic Caching Subsystem
//!
//! Extends FlashStore with hybrid dense-sparse retrieval, cross-encoder
//! reranking, semantic vector query caching, prompt context assembly,
//! offline embedding providers, and IR telemetry evaluation.
//!
//! ## Subsystem Architecture
//!
//! 1. **Document Ingestion & Chunking** ([`chunker`]): Flexible text splitting using paragraph, sentence, fixed-token, or fixed-character boundaries.
//! 2. **Sparse Lexical Search** ([`sparse`]): Okapi BM25 inverted index with tokenization, term frequency saturation, and IDF scoring.
//! 3. **Dense Vector Search** ([`dense`]): High-performance vector index supporting Cosine similarity, Dot Product, and Euclidean distance.
//! 4. **Hybrid Search Fusion** ([`hybrid`]): Combines candidate lists using Reciprocal Rank Fusion (RRF), Weighted Linear Combination, or Borda Count.
//! 5. **Cross-Encoder Reranking** ([`reranker`]): Evaluates query-document pairs using token overlap, contiguous phrase matching, and term proximity, with optional Maximal Marginal Relevance (MMR) diversity reranking.
//! 6. **Semantic Vector Cache** ([`cache`]): In-memory query cache with cosine similarity thresholding, TTL expiration, and LRU eviction.
//! 7. **Context Assembly & Prompting** ([`context`]): Assembles retrieved passages into LLM prompts respecting token budgets and formatting structured citations (`[1]`, `[2]`).
//! 8. **Storage Persistence** ([`store`]): Maps documents, vectors, metadata, and chunks into FlashStore's LSM key-space via atomic [`crate::batch::WriteBatch`] transactions.
//! 9. **End-to-End Pipeline** ([`pipeline`]): Coordinates ingestion, embedding generation, query execution, and prompt synthesis.
//!
//! ## Quick Start
//!
//! ```rust
//! use flash_store::prelude::*;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<()> {
//! let config = RagConfigBuilder::new()
//!     .embedding_dim(16)
//!     .similarity_metric(SimilarityMetric::Cosine)
//!     .build();
//!
//! let engine = RagEngine::new(config);
//! let embedder = Arc::new(MockEmbeddingProvider::new(16));
//! let mut pipeline = RagPipeline::new(engine).with_embedder(embedder);
//!
//! pipeline.ingest_text("doc1", "FlashStore provides fast writes and reads.", None)?;
//! let result = pipeline.query("FlashStore reads", 1)?;
//! assert_eq!(result.documents.len(), 1);
//! assert_eq!(result.documents[0].id.as_str(), "doc1");
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod chunker;
pub mod config;
pub mod context;
pub mod dense;
pub mod engine;
pub mod hybrid;
pub mod metrics;
pub mod pipeline;
pub mod reranker;
pub mod sparse;
pub mod store;
pub mod types;

pub use cache::{SemanticCache, SemanticCacheEntry};
pub use chunker::{chunk_document, chunk_text};
pub use config::{RagConfig, RagConfigBuilder, SimilarityMetric};
pub use context::{
    AssembledContext, Citation, ContextAssembler, ContextConfig, ContextFormat, RagPromptTemplate,
    TruncationStrategy,
};
pub use dense::{
    compute_similarity, compute_similarity_with_norm, cosine_similarity, dot_product,
    euclidean_distance, DenseIndex,
};
pub use engine::{RagEngine, PREFIX_CHUNK, PREFIX_DOC, PREFIX_META, PREFIX_SYS, PREFIX_VEC};
pub use hybrid::{
    borda_count_fusion, filter_min_score, reciprocal_rank_fusion, weighted_linear_fusion,
    z_score_normalize, SearchResult,
};
pub use metrics::{
    EvaluationSummary, QueryEvaluationSample, RagMetricsSnapshot, RagMetricsTracker,
    RetrievalEvaluator, StageLatencyTracker, StageMetricsSnapshot,
};
pub use pipeline::{
    EmbeddingProvider, IngestReport, MockEmbeddingProvider, PipelineQueryResult, RagPipeline,
    StaticEmbeddingProvider,
};
pub use reranker::{
    maximal_marginal_relevance, rerank_candidates, rerank_candidates_weighted,
    rerank_candidates_with_explanation, CrossEncoderScorer, LexicalSemanticCrossEncoder,
    RerankResult,
};
pub use sparse::{compute_idf, tokenize, tokenize_filtered, SparseIndex};
pub use store::RagStoreAdapter;
pub use types::{
    ChunkingConfig, ChunkingStrategy, Document, DocumentBuilder, DocumentChunk, DocumentId,
    DocumentMetadata, Embedding, FilterCondition, FusionStrategy, MetadataFilter, MetadataValue,
    RagQuery, RagQueryBuilder, ScoreExplanation, ScoredDocument, SemanticCacheStats,
};
