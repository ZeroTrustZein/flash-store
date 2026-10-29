//! # Comprehensive Tests for Model Context Protocol (MCP) Subsystem
//!
//! Validates:
//! - MCP server configuration builders and default parameter validation.
//! - JSON-RPC 2.0 request/response/error/notification framing and serialization.
//! - Protocol version negotiation and client/server capability discovery.
//! - Tool definitions, structured JSON Schemas, parameter validation, and execution.
//! - Resource definitions, URI routing, and inspectable resource reading.
//! - Mock channel transport and bidirectional end-to-end request-response loops.

use std::sync::Arc;

use serde_json::json;

use flash_store::error::Result;
use flash_store::mcp::{
    mcp_error_codes, ClientCapabilities, Implementation, InitializeParams, JsonRpcError,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpContent, McpLogLevel,
    McpResourceContent, McpResourceDefinition, McpResourceProvider, McpResourceRegistry,
    McpServer, McpServerConfig, McpServerConfigBuilder, McpToolDefinition, McpToolHandler,
    McpToolRegistry, McpToolResult, MockMcpTransport, RequestId, StandardToolSchemas,
    DEFAULT_SERVER_NAME, DEFAULT_SERVER_VERSION, MCP_PROTOCOL_VERSION,
};

#[test]
fn test_mcp_config_defaults_and_builder() {
    let default_config = McpServerConfig::default();
    assert_eq!(default_config.server_name, DEFAULT_SERVER_NAME);
    assert_eq!(default_config.server_version, DEFAULT_SERVER_VERSION);
    assert_eq!(default_config.protocol_version, MCP_PROTOCOL_VERSION);
    assert!(default_config.enable_tools);
    assert!(default_config.enable_resources);
    assert!(default_config.enable_prompts);
    assert!(default_config.enable_logging);

    let custom_config = McpServerConfigBuilder::new()
        .server_name("custom-flash-mcp")
        .server_version("2.5.0")
        .protocol_version("2024-11-05")
        .enable_tools(false)
        .enable_resources(true)
        .max_message_size(1024 * 1024)
        .build();

    assert_eq!(custom_config.server_name, "custom-flash-mcp");
    assert_eq!(custom_config.server_version, "2.5.0");
    assert!(!custom_config.enable_tools);
    assert!(custom_config.enable_resources);
    assert_eq!(custom_config.max_message_size, 1024 * 1024);
}

#[test]
fn test_jsonrpc_request_response_serialization() {
    let req = JsonRpcRequest::new(
        "req-101",
        "tools/call",
        Some(json!({
            "name": "flash_get",
            "arguments": {"key": "alpha"}
        })),
    );

    let serialized = serde_json::to_string(&req).expect("Serialization failed");
    let deserialized: JsonRpcRequest =
        serde_json::from_str(&serialized).expect("Deserialization failed");

    assert_eq!(deserialized.jsonrpc, "2.0");
    assert_eq!(deserialized.id, RequestId::String("req-101".to_string()));
    assert_eq!(deserialized.method, "tools/call");

    let resp = JsonRpcResponse::success("req-101", json!({"status": "ok"}));
    let resp_str = serde_json::to_string(&resp).unwrap();
    assert!(resp_str.contains("\"result\":{\"status\":\"ok\"}"));

    let err_resp = JsonRpcResponse::error(
        Some("req-102".into()),
        JsonRpcError::method_not_found("unknown/action"),
    );
    assert_eq!(err_resp.error.as_ref().unwrap().code, mcp_error_codes::METHOD_NOT_FOUND);

    let notif = JsonRpcNotification::new("notifications/message", Some(json!({"level": "info"})));
    assert_eq!(notif.method, "notifications/message");
}

#[test]
fn test_request_id_display_and_equality() {
    let id_str = RequestId::from("query_abc");
    let id_num = RequestId::from(42);

    assert_eq!(format!("{}", id_str), "query_abc");
    assert_eq!(format!("{}", id_num), "42");
    assert_ne!(id_str, id_num);
}

#[test]
fn test_standard_tool_schemas() {
    let get_def = StandardToolSchemas::flash_get();
    assert_eq!(get_def.name, "flash_get");
    assert!(get_def.input_schema.get("properties").unwrap().get("key").is_some());

    let put_def = StandardToolSchemas::flash_put();
    assert_eq!(put_def.name, "flash_put");
    assert!(put_def.input_schema.get("required").unwrap().as_array().unwrap().contains(&json!("key")));

    let scan_def = StandardToolSchemas::flash_scan();
    assert_eq!(scan_def.name, "flash_scan");

    let rag_def = StandardToolSchemas::flash_rag_query();
    assert_eq!(rag_def.name, "flash_rag_query");

    let agent_def = StandardToolSchemas::flash_agent_run();
    assert_eq!(agent_def.name, "flash_agent_run");
}

struct DummyEchoTool;

impl McpToolHandler for DummyEchoTool {
    fn definition(&self) -> McpToolDefinition {
        McpToolDefinition::new(
            "test_echo",
            "Echoes input text",
            json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }),
        )
    }

    fn execute(&self, arguments: Option<serde_json::Value>) -> Result<McpToolResult> {
        let text = arguments
            .and_then(|a| a.get("text").and_then(|t| t.as_str().map(String::from)))
            .unwrap_or_else(|| "no text".to_string());
        Ok(McpToolResult::success(format!("Echo: {}", text)))
    }
}

#[test]
fn test_tool_registry_and_dispatch() {
    let registry = McpToolRegistry::new();
    assert_eq!(registry.count(), 0);

    registry.register(Arc::new(DummyEchoTool));
    assert_eq!(registry.count(), 1);
    assert!(registry.contains("test_echo"));

    let list = registry.list();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "test_echo");

    let result = registry
        .call("test_echo", Some(json!({"text": "Hello FlashStore"})))
        .expect("Execution should succeed");

    assert_eq!(result.is_error, Some(false));
    assert_eq!(result.content.len(), 1);
    match &result.content[0] {
        McpContent::Text { text } => assert_eq!(text, "Echo: Hello FlashStore"),
        _ => panic!("Expected text content"),
    }

    let not_found = registry.call("non_existent", None);
    assert!(not_found.is_err());
}

struct DummyStatsResourceProvider;

impl McpResourceProvider for DummyStatsResourceProvider {
    fn list_resources(&self) -> Vec<McpResourceDefinition> {
        vec![McpResourceDefinition::new(
            "flash://stats",
            "Engine Statistics",
            "Live LSM-Tree and MemTable statistics",
            "application/json",
        )]
    }

    fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        if uri == "flash://stats" {
            Ok(vec![McpResourceContent::text(
                "flash://stats",
                "application/json",
                json!({"active_memtables": 1, "sstable_count": 4}).to_string(),
            )])
        } else {
            Ok(vec![])
        }
    }
}

#[test]
fn test_resource_registry_and_read() {
    let registry = McpResourceRegistry::new();
    registry.register("flash", Arc::new(DummyStatsResourceProvider));

    let list = registry.list();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].uri, "flash://stats");

    let contents = registry.read("flash://stats").expect("Read should succeed");
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0].uri, "flash://stats");
    assert!(contents[0].text.as_ref().unwrap().contains("active_memtables"));

    let err = registry.read("flash://invalid_uri");
    assert!(err.is_err());
}

#[test]
fn test_mcp_server_initialize_and_ping() {
    let config = McpServerConfigBuilder::new().build();
    let tools = Arc::new(McpToolRegistry::new());
    let resources = Arc::new(McpResourceRegistry::new());
    let server = McpServer::new(config, tools, resources);

    // 1. Ping
    let ping_req = JsonRpcRequest::new(1, "ping", None);
    let ping_resp = server.handle_request(ping_req);
    assert_eq!(ping_resp.id, Some(RequestId::Number(1)));
    assert_eq!(ping_resp.result, Some(json!({})));

    // 2. Initialize
    let init_params = InitializeParams {
        protocol_version: "2024-11-05".to_string(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation::new("test-client", "1.0.0"),
    };

    let init_req = JsonRpcRequest::new(
        2,
        "initialize",
        Some(serde_json::to_value(init_params).unwrap()),
    );
    let init_resp = server.handle_request(init_req);
    assert!(init_resp.error.is_none());
    let init_res_val = init_resp.result.unwrap();
    assert_eq!(init_res_val.get("protocolVersion").unwrap(), "2024-11-05");
    assert_eq!(
        init_res_val.get("serverInfo").unwrap().get("name").unwrap(),
        DEFAULT_SERVER_NAME
    );
}

#[test]
fn test_mcp_server_mock_transport_end_to_end() {
    let config = McpServerConfigBuilder::new().build();
    let tools = Arc::new(McpToolRegistry::new());
    tools.register(Arc::new(DummyEchoTool));
    let resources = Arc::new(McpResourceRegistry::new());
    let server = Arc::new(McpServer::new(config, tools, resources));

    let (client, transport) = MockMcpTransport::create_channel_pair();

    // Spawn server in thread
    let server_handle = std::thread::spawn({
        let server = Arc::clone(&server);
        let transport = Arc::clone(&transport);
        move || {
            server.serve(&*transport).unwrap();
        }
    });

    // Send tools/list request
    let list_req = JsonRpcRequest::new(10, "tools/list", None);
    client.send_request(list_req).unwrap();

    let list_resp = client.receive_response().unwrap();
    assert_eq!(list_resp.id, Some(RequestId::Number(10)));
    let result = list_resp.result.unwrap();
    let tool_list = result.get("tools").unwrap().as_array().unwrap();
    assert_eq!(tool_list.len(), 1);
    assert_eq!(tool_list[0].get("name").unwrap(), "test_echo");

    // Send tools/call request
    let call_req = JsonRpcRequest::new(
        11,
        "tools/call",
        Some(json!({
            "name": "test_echo",
            "arguments": {"text": "End-to-end MCP works!"}
        })),
    );
    client.send_request(call_req).unwrap();

    let call_resp = client.receive_response().unwrap();
    assert_eq!(call_resp.id, Some(RequestId::Number(11)));
    let call_res_val = call_resp.result.unwrap();
    let content = call_res_val.get("content").unwrap().as_array().unwrap();
    assert_eq!(
        content[0].get("text").unwrap(),
        "Echo: End-to-end MCP works!"
    );

    // Close transport and join server thread
    transport.close();
    server_handle.join().unwrap();
}

#[test]
fn test_log_level_formatting() {
    assert_eq!(format!("{}", McpLogLevel::Debug), "debug");
    assert_eq!(format!("{}", McpLogLevel::Info), "info");
    assert_eq!(format!("{}", McpLogLevel::Warning), "warning");
    assert_eq!(format!("{}", McpLogLevel::Error), "error");
}
