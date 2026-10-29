//! # Model Context Protocol (MCP) Structured Tools and Schemas
//!
//! Exposes structured JSON Schema tool definitions, argument validation, and invocation handlers
//! for FlashStore LSM-tree storage operations, RAG pipelines, and agent swarm workflows.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
