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

    /// Creates an embedded resource content item.
    pub fn resource(resource: Value) -> Self {
        McpContent::Resource { resource }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_implementation_creation() {
        let impl_desc = Implementation::new("test-server", "1.2.3");
        assert_eq!(impl_desc.name, "test-server");
        assert_eq!(impl_desc.version, "1.2.3");

        let serialized = serde_json::to_string(&impl_desc).unwrap();
        let deserialized: Implementation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, impl_desc);
    }

    #[test]
    fn test_server_capabilities_serialization() {
        let mut caps = ServerCapabilities::default();
        caps.tools = Some(ToolsCapability {
            list_changed: Some(true),
        });
        caps.resources = Some(ResourcesCapability {
            subscribe: Some(true),
            list_changed: Some(false),
        });
        caps.prompts = Some(PromptsCapability {
            list_changed: Some(true),
        });
        caps.logging = Some(LoggingCapability {});

        let val = serde_json::to_value(&caps).unwrap();
        assert!(val.get("tools").is_some());
        assert_eq!(val["tools"]["listChanged"], true);
        assert_eq!(val["resources"]["subscribe"], true);
        assert_eq!(val["resources"]["listChanged"], false);
        assert!(val.get("logging").is_some());

        let roundtrip: ServerCapabilities = serde_json::from_value(val).unwrap();
        assert_eq!(roundtrip, caps);
    }

    #[test]
    fn test_initialize_roundtrip() {
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation::new("test-client", "0.1.0"),
        };
        let params_val = serde_json::to_value(&params).unwrap();
        assert_eq!(params_val["protocolVersion"], "2024-11-05");
        assert_eq!(params_val["clientInfo"]["name"], "test-client");

        let res = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ServerCapabilities::default(),
            server_info: Implementation::new("flash-server", "1.0.0"),
            instructions: Some("Do not panic".to_string()),
        };
        let res_val = serde_json::to_value(&res).unwrap();
        assert_eq!(res_val["serverInfo"]["version"], "1.0.0");
        assert_eq!(res_val["instructions"], "Do not panic");
    }

    #[test]
    fn test_mcp_content_variants() {
        let txt = McpContent::text("hello world");
        let txt_val = serde_json::to_value(&txt).unwrap();
        assert_eq!(txt_val["type"], "text");
        assert_eq!(txt_val["text"], "hello world");

        let img = McpContent::image("aW1hZ2U=", "image/png");
        let img_val = serde_json::to_value(&img).unwrap();
        assert_eq!(img_val["type"], "image");
        assert_eq!(img_val["data"], "aW1hZ2U=");
        assert_eq!(img_val["mimeType"], "image/png");

        let res = McpContent::resource(json!({"uri": "flash://doc"}));
        let res_val = serde_json::to_value(&res).unwrap();
        assert_eq!(res_val["type"], "resource");
        assert_eq!(res_val["resource"]["uri"], "flash://doc");
    }

    #[test]
    fn test_log_levels_display_and_serde() {
        let levels = vec![
            (McpLogLevel::Debug, "debug"),
            (McpLogLevel::Info, "info"),
            (McpLogLevel::Notice, "notice"),
            (McpLogLevel::Warning, "warning"),
            (McpLogLevel::Error, "error"),
            (McpLogLevel::Critical, "critical"),
            (McpLogLevel::Alert, "alert"),
            (McpLogLevel::Emergency, "emergency"),
        ];

        for (lvl, expected_str) in levels {
            assert_eq!(lvl.to_string(), expected_str);
            let ser = serde_json::to_string(&lvl).unwrap();
            assert_eq!(ser, format!("\"{}\"", expected_str));
            let de: McpLogLevel = serde_json::from_str(&ser).unwrap();
            assert_eq!(de, lvl);
        }
    }
}
