//! # Model Context Protocol (MCP) Domain Types and Schemas
//!
//! Defines schemas and data contracts specified by the Model Context Protocol specification.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Information describing the client or server implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Implementation {
    /// Human-readable name of the software application.
    pub name: String,
    /// Semantic or build version of the application.
    pub version: String,
}

impl Implementation {
    /// Creates a new implementation descriptor.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Tools capability descriptor advertising tool support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether the server emits listChanged notifications when tools are added/removed.
    #[serde(rename = "listChanged", skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resources capability descriptor advertising resource support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcesCapability {
    /// Whether the server supports subscription to resource change notifications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Whether the server emits listChanged notifications when resources change.
    #[serde(rename = "listChanged", skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompts capability descriptor advertising prompt templates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptsCapability {
    /// Whether the server emits listChanged notifications when prompt templates change.
    #[serde(rename = "listChanged", skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Logging capability descriptor advertising logging support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Server capabilities advertised during handshake.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tool support capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Resource support capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Prompt template support capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Logging support capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
    /// Experimental capabilities dictionary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
}

/// Client capabilities communicated during handshake.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Root directory list capabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<Value>,
    /// Sampling capabilities for LLM agent completion loops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Value>,
    /// Experimental capabilities dictionary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
}

/// Parameters sent by the client in the `initialize` method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeParams {
    /// The MCP protocol version the client wishes to speak.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client-supported capabilities.
    pub capabilities: ClientCapabilities,
    /// Client software name and version.
    #[serde(rename = "clientInfo")]
    pub client_info: Implementation,
}

/// Result returned by the server for the `initialize` method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeResult {
    /// The MCP protocol version the server has negotiated.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server-supported capabilities.
    pub capabilities: ServerCapabilities,
    /// Server software name and version.
    #[serde(rename = "serverInfo")]
    pub server_info: Implementation,
    /// Optional instructions or system guidance for the client / LLM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Single content item returned inside tool results or prompt messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Base64-encoded image content.
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Embedded resource content.
    #[serde(rename = "resource")]
    Resource { resource: Value },
}

impl McpContent {
    /// Creates a text content item.
    pub fn text(text: impl Into<String>) -> Self {
        McpContent::Text { text: text.into() }
    }

    /// Creates an image content item.
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        McpContent::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

/// Severity level for MCP logging notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpLogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

impl std::fmt::Display for McpLogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpLogLevel::Debug => write!(f, "debug"),
            McpLogLevel::Info => write!(f, "info"),
            McpLogLevel::Notice => write!(f, "notice"),
            McpLogLevel::Warning => write!(f, "warning"),
            McpLogLevel::Error => write!(f, "error"),
            McpLogLevel::Critical => write!(f, "critical"),
            McpLogLevel::Alert => write!(f, "alert"),
            McpLogLevel::Emergency => write!(f, "emergency"),
        }
    }
}
