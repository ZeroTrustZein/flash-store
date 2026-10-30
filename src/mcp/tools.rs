//! # Model Context Protocol (MCP) Structured Tools and Schemas
//!
//! Exposes structured JSON Schema tool definitions, argument validation, and invocation handlers
//! for FlashStore LSM-tree storage operations, RAG pipelines, and agent swarm workflows.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use crate::mcp::types::McpContent;

/// Structured tool specification exposed to MCP clients and LLMs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Unique name identifier for the tool (e.g. `flash_get`).
    pub name: String,
    /// Detailed human-readable description of what the tool does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing expected parameters and required properties.
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpToolDefinition {
    /// Creates a new tool definition with a JSON schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
        }
    }
}

/// Request parameters for invoking an MCP tool (`tools/call`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolCallParams {
    /// The name of the tool to run.
    pub name: String,
    /// Arguments provided by the LLM client, matching `inputSchema`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// Result payload returned from tool execution (`tools/call`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolResult {
    /// Ordered list of content items (text messages, images, etc.).
    pub content: Vec<McpContent>,
    /// Whether the tool execution resulted in an error.
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl McpToolResult {
    /// Constructs a successful single-text tool result.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::text(text)],
            is_error: Some(false),
        }
    }

    /// Constructs an error tool result.
    pub fn error(error_message: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::text(error_message)],
            is_error: Some(true),
        }
    }

    /// Constructs a result from multiple content blocks.
    pub fn multi(content: Vec<McpContent>, is_error: bool) -> Self {
        Self {
            content,
            is_error: Some(is_error),
        }
    }
}

/// Payload returned by the `tools/list` endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpListToolsResult {
    /// List of available tools registered on this MCP server.
    pub tools: Vec<McpToolDefinition>,
    /// Optional pagination cursor if more tools exist.
    #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Trait implemented by executable MCP tool handlers.
pub trait McpToolHandler: Send + Sync {
    /// Returns the schema and definition of this tool.
    fn definition(&self) -> McpToolDefinition;

    /// Executes the tool with the given JSON arguments.
    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult>;
}

/// Registry managing available MCP tools and dispatching execution.
#[derive(Default)]
pub struct McpToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn McpToolHandler>>>,
}

impl McpToolRegistry {
    /// Creates a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Constructs a tool registry pre-populated with standard FlashStore storage tools.
    pub fn with_flash_store(store: Arc<FlashStore>) -> Self {
        let registry = Self::new();
        registry.register_storage_tools(store);
        registry
    }

    /// Registers the standard FlashStore storage tools (`flash_get`, `flash_put`, `flash_delete`, `flash_scan`, `flash_stats`).
    pub fn register_storage_tools(&self, store: Arc<FlashStore>) {
        self.register(Arc::new(FlashGetTool::new(store.clone())));
        self.register(Arc::new(FlashPutTool::new(store.clone())));
        self.register(Arc::new(FlashDeleteTool::new(store.clone())));
        self.register(Arc::new(FlashScanTool::new(store.clone())));
        self.register(Arc::new(FlashStatsTool::new(store)));
    }

    /// Registers a tool handler under its definition name.
    pub fn register(&self, handler: Arc<dyn McpToolHandler>) {
        let name = handler.definition().name;
        self.tools.write().insert(name, handler);
    }

    /// Lists definitions of all registered tools.
    pub fn list(&self) -> Vec<McpToolDefinition> {
        let guard = self.tools.read();
        let mut defs: Vec<_> = guard.values().map(|t| t.definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Dispatches a tool execution call by tool name.
    pub fn call(&self, name: &str, arguments: Option<Value>) -> Result<McpToolResult> {
        let handler = {
            let guard = self.tools.read();
            guard.get(name).cloned()
        };

        match handler {
            Some(h) => h.execute(arguments),
            None => Err(FlashStoreError::InvalidArgument(format!(
                "Tool not found: {}",
                name
            ))),
        }
    }

    /// Returns whether a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.read().contains_key(name)
    }

    /// Returns the total count of registered tools.
    pub fn count(&self) -> usize {
        self.tools.read().len()
    }
}

// ============================================================================
// Built-in Storage Tool Handlers
// ============================================================================

/// Tool handler for retrieving values by key via `flash_get`.
pub struct FlashGetTool {
    store: Arc<FlashStore>,
}

impl FlashGetTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpToolHandler for FlashGetTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_get()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let key = arguments
            .as_ref()
            .and_then(|a| a.get("key"))
            .and_then(|k| k.as_str())
            .ok_or_else(|| {
                FlashStoreError::InvalidArgument("Missing required 'key' argument".into())
            })?;

        match self.store.get(key)? {
            Some(bytes) => {
                let text = String::from_utf8(bytes.to_vec())
                    .unwrap_or_else(|_| format!("<binary {} bytes>", bytes.len()));
                let response = json!({
                    "found": true,
                    "key": key,
                    "value": text,
                });
                Ok(McpToolResult::success(response.to_string()))
            }
            None => {
                let response = json!({
                    "found": false,
                    "key": key,
                });
                Ok(McpToolResult::success(response.to_string()))
            }
        }
    }
}

/// Tool handler for writing key-value pairs via `flash_put`.
pub struct FlashPutTool {
    store: Arc<FlashStore>,
}

impl FlashPutTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpToolHandler for FlashPutTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_put()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let args = arguments.ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required arguments object".into())
        })?;

        let key = args.get("key").and_then(|k| k.as_str()).ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required 'key' argument".into())
        })?;

        let value = args.get("value").and_then(|v| v.as_str()).ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required 'value' argument".into())
        })?;

        self.store.put(key, value)?;

        let response = json!({
            "success": true,
            "key": key,
            "bytes_written": value.len(),
        });
        Ok(McpToolResult::success(response.to_string()))
    }
}

/// Tool handler for deleting keys via `flash_delete`.
pub struct FlashDeleteTool {
    store: Arc<FlashStore>,
}

impl FlashDeleteTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpToolHandler for FlashDeleteTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_delete()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let key = arguments
            .as_ref()
            .and_then(|a| a.get("key"))
            .and_then(|k| k.as_str())
            .ok_or_else(|| {
                FlashStoreError::InvalidArgument("Missing required 'key' argument".into())
            })?;

        self.store.delete(key)?;

        let response = json!({
            "success": true,
            "deleted_key": key,
        });
        Ok(McpToolResult::success(response.to_string()))
    }
}

/// Tool handler for scanning lexicographical key ranges via `flash_scan`.
pub struct FlashScanTool {
    store: Arc<FlashStore>,
}

impl FlashScanTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpToolHandler for FlashScanTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_scan()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let start_key = arguments
            .as_ref()
            .and_then(|a| a.get("start_key"))
            .and_then(|k| k.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| bytes::Bytes::copy_from_slice(s.as_bytes()));

        let end_key = arguments
            .as_ref()
            .and_then(|a| a.get("end_key"))
            .and_then(|k| k.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| bytes::Bytes::copy_from_slice(s.as_bytes()));

        let limit = arguments
            .as_ref()
            .and_then(|a| a.get("limit"))
            .and_then(|l| l.as_u64())
            .unwrap_or(50) as usize;

        let results = self.store.scan(start_key, end_key)?;
        let mut items = Vec::new();

        for (k, v) in results.into_iter().take(limit) {
            let key_str = String::from_utf8_lossy(&k).to_string();
            let val_str = String::from_utf8_lossy(&v).to_string();
            items.push(json!({
                "key": key_str,
                "value": val_str,
            }));
        }

        let response = json!({
            "count": items.len(),
            "items": items,
        });
        Ok(McpToolResult::success(response.to_string()))
    }
}

/// Tool handler for inspecting engine telemetry and stats via `flash_stats`.
pub struct FlashStatsTool {
    store: Arc<FlashStore>,
}

impl FlashStatsTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpToolHandler for FlashStatsTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_stats()
    }

    fn execute(&self, _arguments: Option<Value>) -> Result<McpToolResult> {
        let stats = self.store.stats();
        let response = json!({
            "active_memtable_size_bytes": stats.active_memtable_size,
            "immutable_memtables_count": stats.immutable_memtables_count,
            "levels_file_count": stats.levels_file_count,
        });
        Ok(McpToolResult::success(response.to_string()))
    }
}

/// Tool handler for executing hybrid dense-sparse vector queries via `flash_rag_query`.
pub struct FlashRagQueryTool {
    rag_engine: Option<Arc<RwLock<crate::rag::RagEngine>>>,
}

impl FlashRagQueryTool {
    pub fn new(rag_engine: Option<Arc<RwLock<crate::rag::RagEngine>>>) -> Self {
        Self { rag_engine }
    }
}

impl McpToolHandler for FlashRagQueryTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_rag_query()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let args = arguments.ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required arguments object".into())
        })?;

        let query = args.get("query").and_then(|q| q.as_str()).ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required 'query' argument".into())
        })?;

        let top_k = args.get("top_k").and_then(|t| t.as_u64()).unwrap_or(5) as usize;

        if let Some(engine) = &self.rag_engine {
            let mut guard = engine.write();
            let hits = guard.search(query, None, top_k)?;
            let results: Vec<Value> = hits
                .into_iter()
                .map(|h| {
                    json!({
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "dense_score": h.dense_score,
                        "sparse_score": h.sparse_score,
                    })
                })
                .collect();

            let response = json!({
                "query": query,
                "count": results.len(),
                "results": results,
            });
            Ok(McpToolResult::success(response.to_string()))
        } else {
            let response = json!({
                "query": query,
                "count": 0,
                "results": [],
                "note": "RAG engine not configured or unattached",
            });
            Ok(McpToolResult::success(response.to_string()))
        }
    }
}

/// Tool handler for running agent tasks via `flash_agent_run`.
pub struct FlashAgentRunTool {
    agent_engine: Option<Arc<RwLock<crate::agent::AgentEngine>>>,
}

impl FlashAgentRunTool {
    pub fn new(agent_engine: Option<Arc<RwLock<crate::agent::AgentEngine>>>) -> Self {
        Self { agent_engine }
    }
}

impl McpToolHandler for FlashAgentRunTool {
    fn definition(&self) -> McpToolDefinition {
        StandardToolSchemas::flash_agent_run()
    }

    fn execute(&self, arguments: Option<Value>) -> Result<McpToolResult> {
        let args = arguments.ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required arguments object".into())
        })?;

        let task_id = args
            .get("task_id")
            .and_then(|t| t.as_str())
            .ok_or_else(|| {
                FlashStoreError::InvalidArgument("Missing required 'task_id' argument".into())
            })?;

        let prompt = args.get("prompt").and_then(|p| p.as_str()).ok_or_else(|| {
            FlashStoreError::InvalidArgument("Missing required 'prompt' argument".into())
        })?;

        let priority_str = args
            .get("priority")
            .and_then(|p| p.as_str())
            .unwrap_or("normal");

        let priority = match priority_str.to_lowercase().as_str() {
            "low" => crate::agent::TaskPriority::Low,
            "high" => crate::agent::TaskPriority::High,
            "critical" => crate::agent::TaskPriority::Critical,
            _ => crate::agent::TaskPriority::Medium,
        };

        if let Some(engine) = &self.agent_engine {
            let mut guard = engine.write();
            let task = crate::agent::AgentTask::builder(task_id, prompt)
                .priority(priority)
                .build();
            guard.submit_task(task, &[])?;
            let report = guard.execute_task(&crate::agent::TaskId::from(task_id))?;

            let response = json!({
                "task_id": task_id,
                "status": "completed",
                "assigned_agent": report.assigned_agent.as_str(),
                "verdict": format!("{:?}", report.reflection_verdict),
                "iterations": report.reflection_iterations,
            });
            Ok(McpToolResult::success(response.to_string()))
        } else {
            let mut standalone = crate::agent::AgentEngine::new(format!("mcp-{}", task_id));
            let agent = crate::agent::AgentDefinition::builder(
                "mcp-worker",
                crate::agent::AgentRole::Coder,
            )
            .capability(crate::agent::AgentCapability::TextGeneration)
            .capability(crate::agent::AgentCapability::ToolCalling)
            .build();
            standalone.register_agent(agent);

            let task = crate::agent::AgentTask::builder(task_id, prompt)
                .priority(priority)
                .build();
            standalone.submit_task(task, &[])?;
            let summary = standalone.run_workflow()?;

            let response = json!({
                "task_id": task_id,
                "status": format!("{:?}", summary.status),
                "completed_tasks": summary.completed_tasks,
                "total_tasks": summary.total_tasks,
            });
            Ok(McpToolResult::success(response.to_string()))
        }
    }
}

// ============================================================================
// Built-in Standard Schemas
// ============================================================================

/// Helper for constructing standard FlashStore tool schemas.
pub struct StandardToolSchemas;

impl StandardToolSchemas {
    /// JSON schema for `flash_get`.
    pub fn flash_get() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_get",
            "Retrieves the value for a given key from FlashStore LSM-tree engine.",
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The exact key to look up"
                    }
                },
                "required": ["key"]
            }),
        )
    }

    /// JSON schema for `flash_put`.
    pub fn flash_put() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_put",
            "Inserts or updates a key-value pair in FlashStore with WAL persistence.",
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The target key"
                    },
                    "value": {
                        "type": "string",
                        "description": "The string or binary-encoded value to store"
                    }
                },
                "required": ["key", "value"]
            }),
        )
    }

    /// JSON schema for `flash_delete`.
    pub fn flash_delete() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_delete",
            "Appends a tombstone to remove a key from FlashStore.",
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to delete"
                    }
                },
                "required": ["key"]
            }),
        )
    }

    /// JSON schema for `flash_scan`.
    pub fn flash_scan() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_scan",
            "Scans key-value pairs in lexicographical range [start_key, end_key).",
            json!({
                "type": "object",
                "properties": {
                    "start_key": {
                        "type": "string",
                        "description": "Inclusive range start key (or empty for begin)"
                    },
                    "end_key": {
                        "type": "string",
                        "description": "Exclusive range end key (or empty for unbounded)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of items to return (default: 50)"
                    }
                }
            }),
        )
    }

    /// JSON schema for `flash_stats`.
    pub fn flash_stats() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_stats",
            "Fetches live LSM-tree storage telemetry, active memtables, and SSTable levels.",
            json!({
                "type": "object",
                "properties": {}
            }),
        )
    }

    /// JSON schema for `flash_rag_query`.
    pub fn flash_rag_query() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_rag_query",
            "Performs hybrid dense-sparse vector and lexical search with cross-encoder reranking.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top ranked candidate chunks to return"
                    }
                },
                "required": ["query"]
            }),
        )
    }

    /// JSON schema for `flash_agent_run`.
    pub fn flash_agent_run() -> McpToolDefinition {
        McpToolDefinition::new(
            "flash_agent_run",
            "Submits a task to the autonomous multi-agent task graph with reflection loops.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Unique task identifier"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Goal specification and prompt instructions"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "description": "Task priority level"
                    }
                },
                "required": ["task_id", "prompt"]
            }),
        )
    }
}
