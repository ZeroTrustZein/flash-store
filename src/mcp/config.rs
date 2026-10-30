//! # MCP Server Configuration and Settings
//!
//! Provides configuration structs and builders for the Model Context Protocol (MCP) server.

use std::path::{Path, PathBuf};

/// Default MCP protocol version supported by FlashStore MCP server.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Default server name identifier.
pub const DEFAULT_SERVER_NAME: &str = "flash-store-mcp";

/// Default server version.
pub const DEFAULT_SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Configuration for the FlashStore MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Identifier name of the MCP server.
    pub server_name: String,
    /// Semantic version of the MCP server.
    pub server_version: String,
    /// Protocol version string negotiated with clients.
    pub protocol_version: String,
    /// Path to the underlying FlashStore database directory.
    pub store_path: PathBuf,
    /// Whether tool execution capability is enabled.
    pub enable_tools: bool,
    /// Whether resource reading capability is enabled.
    pub enable_resources: bool,
    /// Whether prompt templates capability is enabled.
    pub enable_prompts: bool,
    /// Whether logging notifications are emitted over MCP.
    pub enable_logging: bool,
    /// Maximum payload size in bytes for JSON-RPC messages (default: 16 MB).
    pub max_message_size: usize,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            server_name: DEFAULT_SERVER_NAME.to_string(),
            server_version: DEFAULT_SERVER_VERSION.to_string(),
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            store_path: PathBuf::from("./data/flash-store"),
            enable_tools: true,
            enable_resources: true,
            enable_prompts: true,
            enable_logging: true,
            max_message_size: 16 * 1024 * 1024,
        }
    }
}

/// Builder for constructing [`McpServerConfig`] instances.
#[derive(Debug, Clone, Default)]
pub struct McpServerConfigBuilder {
    config: McpServerConfig,
}

impl McpServerConfigBuilder {
    /// Creates a new configuration builder with default values.
    pub fn new() -> Self {
        Self {
            config: McpServerConfig::default(),
        }
    }

    /// Sets the server name identifier.
    pub fn server_name(mut self, name: impl Into<String>) -> Self {
        self.config.server_name = name.into();
        self
    }

    /// Sets the server semantic version.
    pub fn server_version(mut self, version: impl Into<String>) -> Self {
        self.config.server_version = version.into();
        self
    }

    /// Sets the protocol version string.
    pub fn protocol_version(mut self, version: impl Into<String>) -> Self {
        self.config.protocol_version = version.into();
        self
    }

    /// Sets the database directory path.
    pub fn store_path(mut self, path: impl AsRef<Path>) -> Self {
        self.config.store_path = path.as_ref().to_path_buf();
        self
    }

    /// Enables or disables tool execution capability.
    pub fn enable_tools(mut self, enable: bool) -> Self {
        self.config.enable_tools = enable;
        self
    }

    /// Enables or disables resource reading capability.
    pub fn enable_resources(mut self, enable: bool) -> Self {
        self.config.enable_resources = enable;
        self
    }

    /// Enables or disables prompt templates capability.
    pub fn enable_prompts(mut self, enable: bool) -> Self {
        self.config.enable_prompts = enable;
        self
    }

    /// Enables or disables logging notifications over MCP.
    pub fn enable_logging(mut self, enable: bool) -> Self {
        self.config.enable_logging = enable;
        self
    }

    /// Sets the maximum payload size in bytes.
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.config.max_message_size = size;
        self
    }

    /// Builds and returns the configured [`McpServerConfig`].
    pub fn build(self) -> McpServerConfig {
        self.config
    }
}
