//! # Model Context Protocol (MCP) Subsystem
//!
//! Provides native JSON-RPC 2.0 endpoints, structured tool definitions, resource handlers,
//! and transport mechanisms allowing FlashStore to serve as an embedded MCP server
//! for autonomous coding agents, LLM tool dispatchers, and external orchestration layers.

pub mod config;
pub mod protocol;
pub mod resources;
pub mod server;
pub mod tools;
pub mod transport;
pub mod types;

// Re-exports for clean API ergonomics
pub use config::{
    McpServerConfig, McpServerConfigBuilder, DEFAULT_SERVER_NAME, DEFAULT_SERVER_VERSION,
    MCP_PROTOCOL_VERSION,
};
pub use protocol::{
    error_codes as mcp_error_codes, JsonRpcError, JsonRpcNotification, JsonRpcRequest,
    JsonRpcResponse, RequestId,
};
pub use resources::{
    McpListResourcesResult, McpReadResourceResult, McpResourceContent, McpResourceDefinition,
    McpResourceProvider, McpResourceRegistry,
};
pub use server::McpServer;
pub use tools::{
    McpListToolsResult, McpToolCallParams, McpToolDefinition, McpToolHandler, McpToolRegistry,
    McpToolResult, StandardToolSchemas,
};
pub use transport::{McpTransport, MockMcpClientSide, MockMcpTransport, StdioTransport};
pub use types::{
    ClientCapabilities, Implementation, InitializeParams, InitializeResult, McpContent,
    McpLogLevel, ServerCapabilities, ToolsCapability,
};
