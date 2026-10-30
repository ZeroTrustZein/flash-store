//! # Model Context Protocol (MCP) Server Implementation
//!
//! Provides the primary request router, method dispatcher, and lifecycle loop
//! implementing the Model Context Protocol specification for FlashStore.

use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use crate::mcp::config::McpServerConfig;
use crate::mcp::notifications::McpNotificationBroadcaster;
use crate::mcp::prompts::{
    McpGetPromptParams, McpListPromptsResult, McpPromptHandler, McpPromptRegistry,
};
use crate::mcp::protocol::{error_codes, JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestId};
use crate::mcp::resources::{
    McpListResourcesResult, McpReadResourceResult, McpResourceProvider, McpResourceRegistry,
};
use crate::mcp::tools::{McpListToolsResult, McpToolCallParams, McpToolHandler, McpToolRegistry};
use crate::mcp::transport::McpTransport;
use crate::mcp::types::{
    Implementation, InitializeParams, InitializeResult, LoggingCapability, McpLogLevel,
    PromptsCapability, ResourcesCapability, ServerCapabilities, ToolsCapability,
};

/// High-level MCP server orchestrating endpoints, tool execution, resource access, and prompt templates.
pub struct McpServer {
    config: McpServerConfig,
    tools: Arc<McpToolRegistry>,
    resources: Arc<McpResourceRegistry>,
    prompts: Arc<McpPromptRegistry>,
    notifications: Arc<McpNotificationBroadcaster>,
}

impl McpServer {
    /// Creates a new `McpServer` with the specified configuration and registries.
    pub fn new(
        config: McpServerConfig,
        tools: Arc<McpToolRegistry>,
        resources: Arc<McpResourceRegistry>,
    ) -> Self {
        Self {
            config,
            tools,
            resources,
            prompts: Arc::new(McpPromptRegistry::new()),
            notifications: Arc::new(McpNotificationBroadcaster::new()),
        }
    }

    /// Creates a new `McpServer` with all registries and notification broadcaster explicitly provided.
    pub fn with_all(
        config: McpServerConfig,
        tools: Arc<McpToolRegistry>,
        resources: Arc<McpResourceRegistry>,
        prompts: Arc<McpPromptRegistry>,
    ) -> Self {
        Self {
            config,
            tools,
            resources,
            prompts,
            notifications: Arc::new(McpNotificationBroadcaster::new()),
        }
    }

    /// Creates an `McpServer` initialized with standard tools, resources, and prompt templates backed by `FlashStore`.
    pub fn with_store(config: McpServerConfig, store: Arc<FlashStore>) -> Self {
        let tools = Arc::new(McpToolRegistry::with_flash_store(store.clone()));
        let resources = Arc::new(McpResourceRegistry::with_flash_store(store));
        let prompts = Arc::new(McpPromptRegistry::with_standard_prompts());
        Self::with_all(config, tools, resources, prompts)
    }

    /// Registers a tool handler into the active tool registry.
    pub fn register_tool(&self, handler: Arc<dyn McpToolHandler>) {
        self.tools.register(handler);
        self.notifications.notify_tools_changed();
    }

    /// Registers a resource provider into the active resource registry.
    pub fn register_resource_provider(
        &self,
        prefix: impl Into<String>,
        provider: Arc<dyn McpResourceProvider>,
    ) {
        self.resources.register(prefix, provider);
    }

    /// Registers a prompt handler into the active prompt registry.
    pub fn register_prompt(&self, handler: Arc<dyn McpPromptHandler>) {
        self.prompts.register(handler);
        self.notifications.notify_prompts_changed();
    }

    /// Accesses the server configuration.
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Accesses the tool registry.
    pub fn tools(&self) -> &Arc<McpToolRegistry> {
        &self.tools
    }

    /// Accesses the resource registry.
    pub fn resources(&self) -> &Arc<McpResourceRegistry> {
        &self.resources
    }

    /// Accesses the prompt registry.
    pub fn prompts(&self) -> &Arc<McpPromptRegistry> {
        &self.prompts
    }

    /// Accesses the notification broadcaster.
    pub fn notifications(&self) -> &Arc<McpNotificationBroadcaster> {
        &self.notifications
    }

    /// Handles a single JSON-RPC 2.0 request and produces an appropriate response.
    pub fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone();
        match req.method.as_str() {
            "initialize" => self.handle_initialize(id, req.params),
            "ping" => JsonRpcResponse::success(id, json!({})),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, req.params),
            "resources/list" => self.handle_resources_list(id),
            "resources/read" => self.handle_resources_read(id, req.params),
            "prompts/list" => self.handle_prompts_list(id),
            "prompts/get" => self.handle_prompts_get(id, req.params),
            "logging/setLevel" => self.handle_logging_set_level(id, req.params),
            "notifications/initialized" => {
                // Client confirmed initialization; no response needed for notifications
                JsonRpcResponse::success(id, json!({}))
            }
            unknown => JsonRpcResponse::error(Some(id), JsonRpcError::method_not_found(unknown)),
        }
    }

    fn handle_initialize(&self, id: RequestId, params: Option<Value>) -> JsonRpcResponse {
        let init_params: InitializeParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    return JsonRpcResponse::error(
                        Some(id),
                        JsonRpcError::invalid_params(format!(
                            "Failed parsing initialize params: {}",
                            e
                        )),
                    );
                }
            },
            None => {
                return JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::invalid_params("Missing initialize parameters"),
                );
            }
        };

        debug!(
            "MCP Client connected: {} (version {}) protocol: {}",
            init_params.client_info.name,
            init_params.client_info.version,
            init_params.protocol_version
        );

        let capabilities = ServerCapabilities {
            tools: if self.config.enable_tools {
                Some(ToolsCapability {
                    list_changed: Some(true),
                })
            } else {
                None
            },
            resources: if self.config.enable_resources {
                Some(ResourcesCapability {
                    subscribe: Some(false),
                    list_changed: Some(true),
                })
            } else {
                None
            },
            prompts: if self.config.enable_prompts {
                Some(PromptsCapability {
                    list_changed: Some(true),
                })
            } else {
                None
            },
            logging: if self.config.enable_logging {
                Some(LoggingCapability {})
            } else {
                None
            },
            experimental: None,
        };

        let result = InitializeResult {
            protocol_version: self.config.protocol_version.clone(),
            capabilities,
            server_info: Implementation::new(
                self.config.server_name.clone(),
                self.config.server_version.clone(),
            ),
            instructions: Some(
                "FlashStore MCP Server provides direct access to embedded LSM-tree key-value storage, \
                 hybrid RAG vector retrieval, autonomous multi-agent orchestration, and prompt templates.".to_string(),
            ),
        };

        match serde_json::to_value(result) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
            ),
        }
    }

    fn handle_tools_list(&self, id: RequestId) -> JsonRpcResponse {
        if !self.config.enable_tools {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Tools capability is disabled",
                    None,
                ),
            );
        }

        let tools = self.tools.list();
        let result = McpListToolsResult {
            tools,
            next_cursor: None,
        };

        match serde_json::to_value(result) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
            ),
        }
    }

    fn handle_tools_call(&self, id: RequestId, params: Option<Value>) -> JsonRpcResponse {
        if !self.config.enable_tools {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Tools capability is disabled",
                    None,
                ),
            );
        }

        let call_params: McpToolCallParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(cp) => cp,
                Err(e) => {
                    return JsonRpcResponse::error(
                        Some(id),
                        JsonRpcError::invalid_params(format!(
                            "Invalid tools/call parameters: {}",
                            e
                        )),
                    );
                }
            },
            None => {
                return JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::invalid_params("Missing tools/call parameters"),
                );
            }
        };

        match self.tools.call(&call_params.name, call_params.arguments) {
            Ok(tool_result) => match serde_json::to_value(tool_result) {
                Ok(v) => JsonRpcResponse::success(id, v),
                Err(e) => JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
                ),
            },
            Err(err) => {
                let err_msg = format!("Tool execution failed: {}", err);
                JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::new(error_codes::INTERNAL_ERROR, err_msg, None),
                )
            }
        }
    }

    fn handle_resources_list(&self, id: RequestId) -> JsonRpcResponse {
        if !self.config.enable_resources {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Resources capability is disabled",
                    None,
                ),
            );
        }

        let resources = self.resources.list();
        let result = McpListResourcesResult {
            resources,
            next_cursor: None,
        };

        match serde_json::to_value(result) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
            ),
        }
    }

    fn handle_resources_read(&self, id: RequestId, params: Option<Value>) -> JsonRpcResponse {
        if !self.config.enable_resources {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Resources capability is disabled",
                    None,
                ),
            );
        }

        let uri = match params.and_then(|p| p.get("uri").and_then(|u| u.as_str().map(String::from)))
        {
            Some(u) => u,
            None => {
                return JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::invalid_params("Missing required 'uri' parameter"),
                );
            }
        };

        match self.resources.read(&uri) {
            Ok(contents) => {
                let result = McpReadResourceResult { contents };
                match serde_json::to_value(result) {
                    Ok(v) => JsonRpcResponse::success(id, v),
                    Err(e) => JsonRpcResponse::error(
                        Some(id),
                        JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
                    ),
                }
            }
            Err(err) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::INVALID_PARAMS,
                    format!("Resource read error: {}", err),
                    None,
                ),
            ),
        }
    }

    fn handle_prompts_list(&self, id: RequestId) -> JsonRpcResponse {
        if !self.config.enable_prompts {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Prompts capability is disabled",
                    None,
                ),
            );
        }

        let prompts = self.prompts.list();
        let result = McpListPromptsResult {
            prompts,
            next_cursor: None,
        };

        match serde_json::to_value(result) {
            Ok(v) => JsonRpcResponse::success(id, v),
            Err(e) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
            ),
        }
    }

    fn handle_prompts_get(&self, id: RequestId, params: Option<Value>) -> JsonRpcResponse {
        if !self.config.enable_prompts {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Prompts capability is disabled",
                    None,
                ),
            );
        }

        let get_params: McpGetPromptParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(gp) => gp,
                Err(e) => {
                    return JsonRpcResponse::error(
                        Some(id),
                        JsonRpcError::invalid_params(format!(
                            "Invalid prompts/get parameters: {}",
                            e
                        )),
                    );
                }
            },
            None => {
                return JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::invalid_params("Missing prompts/get parameters"),
                );
            }
        };

        match self
            .prompts
            .render(&get_params.name, get_params.arguments.as_ref())
        {
            Ok(prompt_result) => match serde_json::to_value(prompt_result) {
                Ok(v) => JsonRpcResponse::success(id, v),
                Err(e) => JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::internal_error(format!("Serialization failure: {}", e)),
                ),
            },
            Err(err) => JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::INVALID_PARAMS,
                    format!("Prompt render error: {}", err),
                    None,
                ),
            ),
        }
    }

    fn handle_logging_set_level(&self, id: RequestId, params: Option<Value>) -> JsonRpcResponse {
        if !self.config.enable_logging {
            return JsonRpcResponse::error(
                Some(id),
                JsonRpcError::new(
                    error_codes::SERVER_ERROR_START,
                    "Logging capability is disabled",
                    None,
                ),
            );
        }

        let level_str =
            match params.and_then(|p| p.get("level").and_then(|l| l.as_str().map(String::from))) {
                Some(l) => l,
                None => {
                    return JsonRpcResponse::error(
                        Some(id),
                        JsonRpcError::invalid_params("Missing required 'level' parameter"),
                    );
                }
            };

        let level = match serde_json::from_value::<McpLogLevel>(Value::String(level_str.clone())) {
            Ok(lvl) => lvl,
            Err(_) => {
                return JsonRpcResponse::error(
                    Some(id),
                    JsonRpcError::invalid_params(format!("Unrecognized log level: {}", level_str)),
                );
            }
        };

        self.notifications.set_min_log_level(level);
        JsonRpcResponse::success(id, json!({ "status": "ok", "level": level }))
    }

    /// Runs the server event loop processing requests from the transport until EOF or error.
    pub fn serve<T: McpTransport>(&self, transport: &T) -> Result<()> {
        info!("FlashStore MCP server starting event loop...");
        loop {
            match transport.read_request() {
                Ok(Some(request)) => {
                    let response = self.handle_request(request);
                    if let Err(e) = transport.write_response(&response) {
                        error!("Failed writing MCP response: {}", e);
                        return Err(e);
                    }
                }
                Ok(None) => {
                    debug!("Transport stream closed (EOF). Terminating server loop.");
                    break;
                }
                Err(FlashStoreError::Io(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    warn!("Encountered parse/read error in MCP stream: {}", e);
                    let err_resp = JsonRpcResponse::error(
                        None,
                        JsonRpcError::parse_error(format!("Stream read error: {}", e)),
                    );
                    let _ = transport.write_response(&err_resp);
                }
            }
        }
        info!("FlashStore MCP server event loop finished.");
        Ok(())
    }
}
