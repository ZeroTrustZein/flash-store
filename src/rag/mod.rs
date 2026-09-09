//! # RAG Retrieval, Cross-Encoder Reranking, and Semantic Caching Subsystem
//!
//! Extends FlashStore with hybrid dense-sparse retrieval, cross-encoder
//! reranking, and semantic vector query caching.

pub mod cache;
pub mod config;
pub mod dense;
pub mod engine;
pub mod hybrid;
pub mod reranker;
pub mod sparse;
pub mod types;

pub use cache::{SemanticCache, SemanticCacheEntry};
pub use config::{RagConfig, RagConfigBuilder, SimilarityMetric};
pub use dense::{cosine_similarity, dot_product, euclidean_distance, DenseIndex};
pub use engine::{RagEngine, PREFIX_DOC, PREFIX_VEC};
pub use hybrid::{reciprocal_rank_fusion, weighted_linear_fusion, SearchResult};
pub use reranker::{
    rerank_candidates, CrossEncoderScorer, LexicalSemanticCrossEncoder, RerankResult,
};
pub use sparse::{compute_idf, tokenize, SparseIndex};
pub use types::{
    ChunkingConfig, ChunkingStrategy, Document, DocumentBuilder, DocumentChunk, DocumentId,
    DocumentMetadata, Embedding, FilterCondition, FusionStrategy, MetadataFilter, MetadataValue,
    RagQuery, RagQueryBuilder, ScoreExplanation, ScoredDocument, SemanticCacheStats,
};
