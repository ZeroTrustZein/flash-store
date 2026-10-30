//! # MCP Integration Handlers and Subsystem Providers
//!
//! Provides resource providers, tool bindings, and subsystem bridge components
//! that integrate the core storage engine, RAG subsystem, and autonomous multi-agent
//! workflow engine into the Model Context Protocol layer.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;

use crate::agent::AgentEngine;
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use crate::mcp::config::McpServerConfig;
use crate::mcp::prompts::McpPromptRegistry;
use crate::mcp::resources::{
    FlashStorageResourceProvider, McpResourceContent, McpResourceDefinition, McpResourceProvider,
    McpResourceRegistry,
};
use crate::mcp::server::McpServer;
use crate::mcp::tools::{
    FlashAgentRunTool, FlashDeleteTool, FlashGetTool, FlashPutTool, FlashRagQueryTool,
    FlashScanTool, FlashStatsTool, McpToolRegistry,
};
use crate::rag::RagEngine;

/// Resource provider exposing live RAG retrieval statistics and configuration.
pub struct FlashRagResourceProvider {
    rag_engine: Arc<RwLock<RagEngine>>,
}

impl FlashRagResourceProvider {
    /// Creates a new `FlashRagResourceProvider` wrapping a shared `RagEngine`.
    pub fn new(rag_engine: Arc<RwLock<RagEngine>>) -> Self {
        Self { rag_engine }
    }
}

impl McpResourceProvider for FlashRagResourceProvider {
    fn list_resources(&self) -> Vec<McpResourceDefinition> {
        vec![
            McpResourceDefinition::new(
                "flash://rag/stats",
                "FlashStore RAG Telemetry",
                "Real-time index statistics: document counts, dense vector embeddings, and sparse tokens",
                "application/json",
            ),
            McpResourceDefinition::new(
                "flash://rag/config",
                "FlashStore RAG Configuration",
                "Active RAG hyper-parameters, vector dimensionality, and fusion strategies",
                "application/json",
            ),
        ]
    }

    fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let guard = self.rag_engine.read();
        match uri {
            "flash://rag/stats" => {
                let metrics = guard.metrics();
                let payload = json!({
                    "total_documents": guard.doc_count(),
                    "dense_vectors": guard.dense_vector_count(),
                    "sparse_terms": guard.sparse_term_count(),
                    "total_searches": metrics.total_searches,
                    "cache_hits": metrics.cache_hits,
                    "cache_misses": metrics.cache_misses,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "mean_latency_ms": metrics.search_latency.avg_micros / 1000.0,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://rag/stats",
                    "application/json",
                    text,
                )])
            }
            "flash://rag/config" => {
                let cfg = guard.config();
                let payload = json!({
                    "embedding_dim": cfg.embedding_dim,
                    "similarity_metric": format!("{:?}", cfg.similarity_metric),
                    "hybrid_dense_weight": cfg.hybrid_dense_weight,
                    "bm25_k1": cfg.bm25_k1,
                    "bm25_b": cfg.bm25_b,
                    "rrf_k": cfg.rrf_k,
                    "semantic_cache_capacity": cfg.semantic_cache_capacity,
                    "semantic_cache_threshold": cfg.semantic_cache_threshold,
                    "semantic_cache_ttl_secs": cfg.semantic_cache_ttl_secs,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://rag/config",
                    "application/json",
                    text,
                )])
            }
            _ => Err(FlashStoreError::InvalidArgument(format!(
                "RAG resource not found: {}",
                uri
            ))),
        }
    }
}

/// Resource provider exposing multi-agent swarm state and task execution telemetry.
pub struct FlashAgentResourceProvider {
    agent_engine: Arc<RwLock<AgentEngine>>,
}

impl FlashAgentResourceProvider {
    /// Creates a new `FlashAgentResourceProvider` wrapping a shared `AgentEngine`.
    pub fn new(agent_engine: Arc<RwLock<AgentEngine>>) -> Self {
        Self { agent_engine }
    }
}

impl McpResourceProvider for FlashAgentResourceProvider {
    fn list_resources(&self) -> Vec<McpResourceDefinition> {
        vec![
            McpResourceDefinition::new(
                "flash://agent/state",
                "FlashStore Agent Workflow State",
                "Current state of agent workflows, pending and completed tasks, and swarm status",
                "application/json",
            ),
            McpResourceDefinition::new(
                "flash://agent/agents",
                "FlashStore Registered Agents",
                "Roster of registered agent roles, capabilities, and affinity mappings",
                "application/json",
            ),
        ]
    }

    fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let guard = self.agent_engine.read();
        let state = guard.state();
        match uri {
            "flash://agent/state" => {
                let payload = json!({
                    "workflow_id": state.workflow_id,
                    "status": format!("{:?}", state.status),
                    "tasks_count": state.tasks.len(),
                    "active_agents_count": state.active_agents.len(),
                    "messages_count": state.message_history.len(),
                    "created_at": state.created_at,
                    "updated_at": state.updated_at,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://agent/state",
                    "application/json",
                    text,
                )])
            }
            "flash://agent/agents" => {
                let agents: Vec<serde_json::Value> = state
                    .active_agents
                    .values()
                    .map(|a| {
                        json!({
                            "id": a.id.as_str(),
                            "role": format!("{:?}", a.role),
                            "capabilities": a.capabilities.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>(),
                            "max_iterations": a.max_iterations,
                        })
                    })
                    .collect();
                let payload = json!({
                    "count": agents.len(),
                    "agents": agents,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://agent/agents",
                    "application/json",
                    text,
                )])
            }
            _ => Err(FlashStoreError::InvalidArgument(format!(
                "Agent resource not found: {}",
                uri
            ))),
        }
    }
}

/// Integrated MCP subsystem orchestrating tools, resources, and prompt templates.
pub struct McpSubsystem {
    server: Arc<McpServer>,
    store: Arc<FlashStore>,
    rag_engine: Option<Arc<RwLock<RagEngine>>>,
    agent_engine: Option<Arc<RwLock<AgentEngine>>>,
}

impl McpSubsystem {
    /// Builds a full MCP subsystem wired to storage, optional RAG, and optional Multi-Agent engines.
    pub fn build(
        config: McpServerConfig,
        store: Arc<FlashStore>,
        rag_engine: Option<Arc<RwLock<RagEngine>>>,
        agent_engine: Option<Arc<RwLock<AgentEngine>>>,
    ) -> Self {
        // Build tool registry with core CRUD tools
        let tools = Arc::new(McpToolRegistry::new());
        tools.register(Arc::new(FlashGetTool::new(store.clone())));
        tools.register(Arc::new(FlashPutTool::new(store.clone())));
        tools.register(Arc::new(FlashDeleteTool::new(store.clone())));
        tools.register(Arc::new(FlashScanTool::new(store.clone())));
        tools.register(Arc::new(FlashStatsTool::new(store.clone())));

        // Wire RAG tool
        tools.register(Arc::new(FlashRagQueryTool::new(rag_engine.clone())));

        // Wire Agent tool
        tools.register(Arc::new(FlashAgentRunTool::new(agent_engine.clone())));

        // Build resource registry
        let resources = Arc::new(McpResourceRegistry::new());
        resources.register(
            "flash",
            Arc::new(FlashStorageResourceProvider::new(store.clone())),
        );

        if let Some(rag) = &rag_engine {
            resources.register(
                "flash",
                Arc::new(FlashRagResourceProvider::new(rag.clone())),
            );
        }

        if let Some(agent) = &agent_engine {
            resources.register(
                "flash",
                Arc::new(FlashAgentResourceProvider::new(agent.clone())),
            );
        }

        // Build prompt registry
        let prompts = Arc::new(McpPromptRegistry::with_standard_prompts());

        let server = Arc::new(McpServer::with_all(config, tools, resources, prompts));

        Self {
            server,
            store,
            rag_engine,
            agent_engine,
        }
    }

    /// Accesses the underlying MCP server.
    pub fn server(&self) -> &Arc<McpServer> {
        &self.server
    }

    /// Accesses the underlying FlashStore engine.
    pub fn store(&self) -> &Arc<FlashStore> {
        &self.store
    }

    /// Accesses the optional RAG engine.
    pub fn rag_engine(&self) -> Option<&Arc<RwLock<RagEngine>>> {
        self.rag_engine.as_ref()
    }

    /// Accesses the optional Agent engine.
    pub fn agent_engine(&self) -> Option<&Arc<RwLock<AgentEngine>>> {
        self.agent_engine.as_ref()
    }
}
