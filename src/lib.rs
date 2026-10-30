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
//! - **Multi-Agent Workflow Subsystem**: Autonomous task graph orchestration, adaptive routing, stateful reflection loops, dynamic tool calling, episodic memory, and telemetry.
//! - **Model Context Protocol (MCP) Subsystem**: Native JSON-RPC 2.0 endpoints exposing LSM storage, RAG search, and agent swarm tools/resources/prompts to LLM host clients.
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
//!
//! ## Autonomous Multi-Agent Swarm Quick Start
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
//! let mut pipeline = AgentPipeline::new("swarm-quickstart").with_store(db);
//!
//! let agent = AgentDefinition::builder("worker-01", AgentRole::Coder)
//!     .capability(AgentCapability::TextGeneration)
//!     .capability(AgentCapability::StorageOps)
//!     .build();
//! pipeline.register_agent(agent)?;
//!
//! let task = AgentTask::builder("task-1", "Initialize Swarm Config")
//!     .priority(TaskPriority::High)
//!     .require_capability(AgentCapability::TextGeneration)
//!     .build();
//! pipeline.submit_task(task, &[])?;
//!
//! let summary = pipeline.run()?;
//! assert_eq!(summary.completed_tasks, 1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Model Context Protocol (MCP) Server Quick Start
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
//! let mcp_config = McpServerConfigBuilder::new()
//!     .server_name("flash-embedded-mcp")
//!     .enable_tools(true)
//!     .enable_resources(true)
//!     .enable_prompts(true)
//!     .build();
//!
//! let subsystem = McpSubsystem::build(mcp_config, db, None, None);
//!
//! let (client, server_transport) = MockMcpTransport::create_channel_pair();
//! let server = subsystem.server().clone();
//! let server_transport_clone = server_transport.clone();
//! let handle = std::thread::spawn(move || {
//!     let _ = server.serve(&*server_transport_clone);
//! });
//!
//! let init_req = JsonRpcRequest::new(
//!     "req-1",
//!     "initialize",
//!     Some(serde_json::to_value(InitializeParams {
//!         protocol_version: MCP_PROTOCOL_VERSION.to_string(),
//!         capabilities: Default::default(),
//!         client_info: Implementation {
//!             name: "mcp-quickstart".to_string(),
//!             version: "1.0.0".to_string(),
//!         },
//!     })?),
//! );
//! client.send_request(init_req)?;
//! let init_resp = client.receive_response()?;
//! assert!(init_resp.result.is_some());
//!
//! drop(client);
//! server_transport.close();
//! let _ = handle.join();
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod batch;
pub mod cache;
pub mod cli;
pub mod compaction;
pub mod config;
pub mod engine;
pub mod error;
pub mod iterator;
pub mod manifest;
pub mod mcp;
pub mod memtable;
pub mod rag;
pub mod sstable;
pub mod types;
pub mod wal;

pub use agent::{
    compute_aggregate_score, create_standard_tool_registry, format_revision_prompt,
    score_capability_match, score_priority_tier, score_role_affinity, score_workload,
    verify_critical_invariants, AdaptiveRoutingWeights, AdaptiveTaskRouter, AgentCapability,
    AgentDefinition, AgentDefinitionBuilder, AgentEngine, AgentEvent, AgentEventHandler, AgentId,
    AgentMemoryStore, AgentMessage, AgentMetricsSnapshot, AgentMetricsTracker, AgentPipeline,
    AgentPipelineConfig, AgentRole, AgentStorageKey, AgentStoreAdapter, AgentTask,
    AgentTaskBuilder, CalculatorTool, CallbackEventHandler, CollectingEventHandler,
    CompositeReflector, ConvergenceConfig, ConvergenceEvaluator, DefaultTaskRouter,
    DurationTracker, EchoTool, EpisodicMemoryEntry, EvaluationCriterion, EventBroadcaster,
    HeuristicReflector, KvDeleteTool, KvGetTool, KvPutTool, KvScanTool, LatencySummary,
    LoadBalancingRouter, MemoryScoringWeights, MessageRole, MetricsEventHandler,
    PriorityPreemptiveRouter, ReflectionCritique, ReflectionEngine, ReflectionState,
    ReflectionStep, ReflectionVerdict, Reflector, RoutingDecision, RoutingStrategy,
    RuleBasedReflector, ScoredMemory, TaskExecutionReport, TaskGraph, TaskId, TaskPriority,
    TaskRouter, TaskStatus, Tool, ToolCall, ToolDefinition, ToolDispatcher, ToolOutput,
    ToolParamType, ToolParameter, ToolRegistry, WorkflowState, WorkflowStatus, WorkflowSummary,
};
pub use batch::WriteBatch;
pub use cache::{BlockCache, LruBlockCache};
pub use cli::{
    open_agent_pipeline, open_mcp_subsystem, open_rag_pipeline, AgentCmd, Cmd, GlobalOpts, McpCmd,
    OutputFormat, RagCmd,
};
pub use compaction::{CompactionTask, Compactor};
pub use config::{Options, OptionsBuilder};
pub use engine::{FlashStore, Stats};
pub use error::{FlashStoreError, Result};
pub use iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
pub use mcp::{
    mcp_error_codes, ClientCapabilities, CollectingNotificationObserver,
    FlashAgentOrchestratePrompt, FlashAgentResourceProvider, FlashAgentRunTool, FlashDeleteTool,
    FlashGetTool, FlashPutTool, FlashRagQueryTool, FlashRagResourceProvider,
    FlashRagSynthesizePrompt, FlashScanTool, FlashStatsTool, FlashStorageResourceProvider,
    FlashTroubleshootPrompt, Implementation, InitializeParams, InitializeResult, JsonRpcError,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, LoggingCapability, LoggingMessageParams,
    McpContent, McpGetPromptParams, McpGetPromptResult, McpListPromptsResult,
    McpListResourcesResult, McpListToolsResult, McpLogLevel, McpNotificationBroadcaster,
    McpNotificationObserver, McpPromptArgument, McpPromptDefinition, McpPromptHandler,
    McpPromptMessage, McpPromptRegistry, McpReadResourceResult, McpResourceContent,
    McpResourceDefinition, McpResourceProvider, McpResourceRegistry, McpServer, McpServerConfig,
    McpServerConfigBuilder, McpSubsystem, McpToolCallParams, McpToolDefinition, McpToolHandler,
    McpToolRegistry, McpToolResult, McpTransport, MockMcpClientSide, MockMcpTransport,
    PromptsCapability, RequestId, ResourceUpdatedParams, ResourcesCapability, ServerCapabilities,
    StandardToolSchemas, StdioTransport, ToolsCapability, DEFAULT_SERVER_NAME,
    DEFAULT_SERVER_VERSION, MCP_PROTOCOL_VERSION,
};
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
    pub use crate::agent::{
        compute_aggregate_score, create_standard_tool_registry, format_revision_prompt,
        score_capability_match, score_priority_tier, score_role_affinity, score_workload,
        verify_critical_invariants, AdaptiveRoutingWeights, AdaptiveTaskRouter, AgentCapability,
        AgentDefinition, AgentDefinitionBuilder, AgentEngine, AgentEvent, AgentEventHandler,
        AgentId, AgentMemoryStore, AgentMessage, AgentMetricsSnapshot, AgentMetricsTracker,
        AgentPipeline, AgentPipelineConfig, AgentRole, AgentStorageKey, AgentStoreAdapter,
        AgentTask, AgentTaskBuilder, CalculatorTool, CallbackEventHandler, CollectingEventHandler,
        CompositeReflector, ConvergenceConfig, ConvergenceEvaluator, DefaultTaskRouter,
        DurationTracker, EchoTool, EpisodicMemoryEntry, EvaluationCriterion, EventBroadcaster,
        HeuristicReflector, KvDeleteTool, KvGetTool, KvPutTool, KvScanTool, LatencySummary,
        LoadBalancingRouter, MemoryScoringWeights, MessageRole, MetricsEventHandler,
        PriorityPreemptiveRouter, ReflectionCritique, ReflectionEngine, ReflectionState,
        ReflectionStep, ReflectionVerdict, Reflector, RoutingDecision, RoutingStrategy,
        RuleBasedReflector, ScoredMemory, TaskExecutionReport, TaskGraph, TaskId, TaskPriority,
        TaskRouter, TaskStatus, Tool, ToolCall, ToolDefinition, ToolDispatcher, ToolOutput,
        ToolParamType, ToolParameter, ToolRegistry, WorkflowState, WorkflowStatus, WorkflowSummary,
    };
    pub use crate::batch::WriteBatch;
    pub use crate::cache::{BlockCache, LruBlockCache};
    pub use crate::cli::{
        open_agent_pipeline, open_mcp_subsystem, open_rag_pipeline, AgentCmd, Cmd, GlobalOpts,
        McpCmd, OutputFormat, RagCmd,
    };
    pub use crate::compaction::{CompactionTask, Compactor};
    pub use crate::config::{Options, OptionsBuilder};
    pub use crate::engine::{FlashStore, Stats};
    pub use crate::error::{FlashStoreError, Result};
    pub use crate::iterator::{
        MemtableIterator, MergingIterator, SSTableIterator, StorageIterator,
    };
    pub use crate::mcp::{
        mcp_error_codes, ClientCapabilities, CollectingNotificationObserver,
        FlashAgentOrchestratePrompt, FlashAgentResourceProvider, FlashAgentRunTool,
        FlashDeleteTool, FlashGetTool, FlashPutTool, FlashRagQueryTool, FlashRagResourceProvider,
        FlashRagSynthesizePrompt, FlashScanTool, FlashStatsTool, FlashStorageResourceProvider,
        FlashTroubleshootPrompt, Implementation, InitializeParams, InitializeResult, JsonRpcError,
        JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, LoggingCapability,
        LoggingMessageParams, McpContent, McpGetPromptParams, McpGetPromptResult,
        McpListPromptsResult, McpListResourcesResult, McpListToolsResult, McpLogLevel,
        McpNotificationBroadcaster, McpNotificationObserver, McpPromptArgument,
        McpPromptDefinition, McpPromptHandler, McpPromptMessage, McpPromptRegistry,
        McpReadResourceResult, McpResourceContent, McpResourceDefinition, McpResourceProvider,
        McpResourceRegistry, McpServer, McpServerConfig, McpServerConfigBuilder, McpSubsystem,
        McpToolCallParams, McpToolDefinition, McpToolHandler, McpToolRegistry, McpToolResult,
        McpTransport, MockMcpClientSide, MockMcpTransport, PromptsCapability, RequestId,
        ResourceUpdatedParams, ResourcesCapability, ServerCapabilities, StandardToolSchemas,
        StdioTransport, ToolsCapability, DEFAULT_SERVER_NAME, DEFAULT_SERVER_VERSION,
        MCP_PROTOCOL_VERSION,
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
