//! # RAG Retrieval, Cross-Encoder Reranking, and Semantic Caching Subsystem
//!
//! Extends FlashStore with hybrid dense-sparse retrieval, cross-encoder
//! reranking, semantic vector query caching, prompt context assembly,
//! offline embedding providers, and IR telemetry evaluation.

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
