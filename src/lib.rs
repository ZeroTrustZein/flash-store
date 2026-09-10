//! # FlashStore
//!
//! An embedded, thread-safe, high-performance Log-Structured Merge-Tree (LSM-Tree)
//! key-value storage engine implemented in Rust.
//!
//! ## Architecture Overview
//!
//! - **Write-Ahead Log (WAL)**: Append-only persistent journal with CRC32 checksum framing for durability.
//! - **MemTable**: In-memory sorted index backed by concurrent skiplists for rapid writes.
//! - **SSTable**: Immutable disk-based tables with data blocks, Bloom filters, meta index blocks, and fixed-size footers.
//! - **Block Cache**: Configurable LRU cache for uncompressed data blocks.
//! - **Compaction Engine**: Multi-level tiered compaction that purges overwritten versions and tombstones.
//! - **Manifest & Versioning**: ACID version edits tracking SSTable level assignments across crash recoveries.
//! - **RAG & Reranker Subsystem**: Native hybrid BM25/vector retrieval, cross-encoder reranking, semantic caching, and LLM context assembly.
//!
//! ## Quick Start
//!
//! ```rust
//! use flash_store::prelude::*;
//!
//! # fn main() -> Result<()> {
//! # let dir = tempfile::tempdir().unwrap();
//! let options = OptionsBuilder::new()
//!     .dir(dir.path())
//!     .memtable_size(4 * 1024 * 1024)
//!     .build();
//!
//! let db = FlashStore::open(options)?;
//!
//! // Basic CRUD
//! db.put("user_1001", "Alice")?;
//! if let Some(value) = db.get("user_1001")? {
//!     println!("Found user: {}", String::from_utf8_lossy(&value));
//! }
//! db.delete("user_1001")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## RAG & Vector Retrieval Quick Start
//!
//! ```rust
//! use flash_store::prelude::*;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<()> {
//! # let dir = tempfile::tempdir().unwrap();
//! let store_opts = OptionsBuilder::new().dir(dir.path()).build();
//! let db = Arc::new(FlashStore::open(store_opts)?);
//!
//! let rag_config = RagConfigBuilder::new()
//!     .embedding_dim(16)
//!     .similarity_metric(SimilarityMetric::Cosine)
//!     .build();
//!
//! let engine = RagEngine::with_store(rag_config, db);
//! let embedder = Arc::new(MockEmbeddingProvider::new(16));
//! let mut pipeline = RagPipeline::new(engine).with_embedder(embedder);
//!
//! pipeline.ingest_text(
//!     "doc1",
//!     "FlashStore is an embedded LSM-tree key-value storage engine in Rust.",
//!     None,
//! )?;
//!
//! let result = pipeline.query("FlashStore in Rust", 1)?;
//! assert_eq!(result.documents.len(), 1);
//! assert_eq!(result.documents[0].id.as_str(), "doc1");
//! # Ok(())
//! # }
//! ```

pub mod batch;
pub mod cache;
pub mod cli;
pub mod compaction;
pub mod config;
pub mod engine;
pub mod error;
pub mod iterator;
pub mod manifest;
pub mod memtable;
pub mod rag;
pub mod sstable;
pub mod types;
pub mod wal;

pub use batch::WriteBatch;
pub use cache::{BlockCache, LruBlockCache};
pub use cli::{open_rag_pipeline, Cmd, GlobalOpts, OutputFormat, RagCmd};
pub use compaction::{CompactionTask, Compactor};
pub use config::{Options, OptionsBuilder};
pub use engine::{FlashStore, Stats};
pub use error::{FlashStoreError, Result};
pub use iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
pub use rag::{
    borda_count_fusion, chunk_document, chunk_text, cosine_similarity, dot_product,
    euclidean_distance, filter_min_score, maximal_marginal_relevance, reciprocal_rank_fusion,
    rerank_candidates, rerank_candidates_weighted, rerank_candidates_with_explanation, tokenize,
    tokenize_filtered, weighted_linear_fusion, z_score_normalize, AssembledContext, ChunkingConfig,
    ChunkingStrategy, Citation, ContextAssembler, ContextConfig, ContextFormat, CrossEncoderScorer,
    DenseIndex, Document, DocumentBuilder, DocumentChunk, DocumentId, DocumentMetadata, Embedding,
    EmbeddingProvider, EvaluationSummary, FilterCondition, FusionStrategy, IngestReport,
    LexicalSemanticCrossEncoder, MetadataFilter, MetadataValue, MockEmbeddingProvider,
    PipelineQueryResult, QueryEvaluationSample, RagConfig, RagConfigBuilder, RagEngine,
    RagMetricsSnapshot, RagMetricsTracker, RagPipeline, RagPromptTemplate, RagQuery,
    RagQueryBuilder, RagStoreAdapter, RerankResult, RetrievalEvaluator, ScoreExplanation,
    ScoredDocument, SearchResult, SemanticCache, SemanticCacheEntry, SemanticCacheStats,
    SimilarityMetric, SparseIndex, StageLatencyTracker, StageMetricsSnapshot,
    StaticEmbeddingProvider, TruncationStrategy, PREFIX_CHUNK, PREFIX_DOC, PREFIX_META, PREFIX_SYS,
    PREFIX_VEC,
};
pub use sstable::{table_file_name, table_path};
pub use types::{
    ChecksumType, CompressionType, DefaultUserKeyComparator, Entry, InternalKey, IntoBytes, Key,
    KeyRange, SequenceNumber, UserKey, UserKeyComparator, Value, ValueType, MAX_SEQUENCE_NUMBER,
    MIN_SEQUENCE_NUMBER,
};

pub mod prelude {
    pub use crate::batch::WriteBatch;
    pub use crate::cache::{BlockCache, LruBlockCache};
    pub use crate::cli::{open_rag_pipeline, Cmd, GlobalOpts, OutputFormat, RagCmd};
    pub use crate::compaction::{CompactionTask, Compactor};
    pub use crate::config::{Options, OptionsBuilder};
    pub use crate::engine::{FlashStore, Stats};
    pub use crate::error::{FlashStoreError, Result};
    pub use crate::iterator::{
        MemtableIterator, MergingIterator, SSTableIterator, StorageIterator,
    };
    pub use crate::rag::{
        borda_count_fusion, chunk_document, chunk_text, cosine_similarity, dot_product,
        euclidean_distance, filter_min_score, maximal_marginal_relevance, reciprocal_rank_fusion,
        rerank_candidates, rerank_candidates_weighted, rerank_candidates_with_explanation,
        tokenize, tokenize_filtered, weighted_linear_fusion, z_score_normalize, AssembledContext,
        ChunkingConfig, ChunkingStrategy, Citation, ContextAssembler, ContextConfig, ContextFormat,
        CrossEncoderScorer, DenseIndex, Document, DocumentBuilder, DocumentChunk, DocumentId,
        DocumentMetadata, Embedding, EmbeddingProvider, EvaluationSummary, FilterCondition,
        FusionStrategy, IngestReport, LexicalSemanticCrossEncoder, MetadataFilter, MetadataValue,
        MockEmbeddingProvider, PipelineQueryResult, QueryEvaluationSample, RagConfig,
        RagConfigBuilder, RagEngine, RagMetricsSnapshot, RagMetricsTracker, RagPipeline,
        RagPromptTemplate, RagQuery, RagQueryBuilder, RagStoreAdapter, RerankResult,
        RetrievalEvaluator, ScoreExplanation, ScoredDocument, SearchResult, SemanticCache,
        SemanticCacheEntry, SemanticCacheStats, SimilarityMetric, SparseIndex, StageLatencyTracker,
        StageMetricsSnapshot, StaticEmbeddingProvider, TruncationStrategy, PREFIX_CHUNK,
        PREFIX_DOC, PREFIX_META, PREFIX_SYS, PREFIX_VEC,
    };
    pub use crate::sstable::{table_file_name, table_path};
    pub use crate::types::{
        ChecksumType, CompressionType, DefaultUserKeyComparator, Entry, InternalKey, IntoBytes,
        Key, KeyRange, SequenceNumber, UserKey, UserKeyComparator, Value, ValueType,
        MAX_SEQUENCE_NUMBER, MIN_SEQUENCE_NUMBER,
    };
}
