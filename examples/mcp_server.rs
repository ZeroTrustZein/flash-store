//! # FlashStore Model Context Protocol (MCP) Server Example
//!
//! Demonstrates using FlashStore as an embedded Model Context Protocol (MCP) server:
//! 1. Initializing an embedded FlashStore engine.
//! 2. Creating an `McpSubsystem` with storage, RAG, and agent capabilities.
//! 3. Initializing a mock bidirectional transport loop simulating an MCP host/client (e.g. Claude Desktop).
//! 4. Performing `initialize` protocol handshake and discovering server capabilities.
//! 5. Discovering and invoking tools (`flash_put`, `flash_get`, `flash_scan`).
//! 6. Querying inspectable resources (`flash://stats`, `flash://manifest`).
//! 7. Rendering diagnostic prompt templates (`flash_troubleshoot`).

use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;

use serde_json::json;

use flash_store::error::Result;
use flash_store::mcp::{
    InitializeParams, JsonRpcRequest, McpGetPromptParams, McpServerConfigBuilder, McpSubsystem,
    McpToolCallParams, MockMcpTransport, DEFAULT_SERVER_NAME, DEFAULT_SERVER_VERSION,
    MCP_PROTOCOL_VERSION,
};
use flash_store::{FlashStore, OptionsBuilder};

fn main() -> Result<()> {
    println!("=== FlashStore Model Context Protocol (MCP) Server Demo ===\n");

    // 1. Initialize embedded FlashStore storage engine
    let dir = tempdir().map_err(flash_store::FlashStoreError::Io)?;
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = Arc::new(FlashStore::open(options)?);
    println!("[1/6] Opened FlashStore LSM engine at {:?}", dir.path());

    // 2. Build the MCP Subsystem
    let config = McpServerConfigBuilder::new()
        .server_name(DEFAULT_SERVER_NAME)
        .server_version(DEFAULT_SERVER_VERSION)
        .store_path(dir.path())
        .enable_tools(true)
        .enable_resources(true)
        .enable_prompts(true)
        .build();

    let subsystem = McpSubsystem::build(config, db, None, None);
    println!("[2/6] Initialized McpSubsystem with storage tools, resources, and prompt templates.");

    // 3. Set up bidirectional in-memory mock transport
    let (client, server_transport) = MockMcpTransport::create_channel_pair();
    let server = subsystem.server().clone();

    let server_transport_clone = server_transport.clone();
    // Run server event-loop in background thread
    let server_handle = std::thread::spawn(move || {
        let _ = server.serve(&*server_transport_clone);
    });

    // 4. Client Handshake: `initialize`
    let init_req = JsonRpcRequest::new(
        "req-init-1",
        "initialize",
        Some(serde_json::to_value(InitializeParams {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: Default::default(),
            client_info: flash_store::mcp::Implementation {
                name: "mcp-demo-client".to_string(),
                version: "1.0.0".to_string(),
            },
        })?),
    );
    client.send_request(init_req)?;
    let init_resp = client.receive_response()?;
    println!(
        "[3/6] Protocol Handshake Succeeded! Server capabilities: {}",
        init_resp.result.unwrap_or_default()
    );

    // 5. Tool Invocations: `tools/call` for `flash_put` and `flash_get`
    let put_req = JsonRpcRequest::new(
        "req-put-1",
        "tools/call",
        Some(serde_json::to_value(McpToolCallParams {
            name: "flash_put".to_string(),
            arguments: Some(json!({
                "key": "cluster:node-1",
                "value": "online"
            })),
        })?),
    );
    client.send_request(put_req)?;
    let put_resp = client.receive_response()?;
    println!("[4/6] Invoked 'flash_put' tool: {:?}", put_resp.result);

    let get_req = JsonRpcRequest::new(
        "req-get-1",
        "tools/call",
        Some(serde_json::to_value(McpToolCallParams {
            name: "flash_get".to_string(),
            arguments: Some(json!({
                "key": "cluster:node-1"
            })),
        })?),
    );
    client.send_request(get_req)?;
    let get_resp = client.receive_response()?;
    println!("      Invoked 'flash_get' tool: {:?}", get_resp.result);

    // 6. Inspectable Resources: `resources/read`
    let res_req = JsonRpcRequest::new(
        "req-res-1",
        "resources/read",
        Some(json!({ "uri": "flash://stats" })),
    );
    client.send_request(res_req)?;
    let res_resp = client.receive_response()?;
    println!(
        "[5/6] Read MCP Resource 'flash://stats': {:?}",
        res_resp.result
    );

    // 7. Prompt Rendering: `prompts/get`
    let mut args = HashMap::new();
    args.insert(
        "issue".to_string(),
        "Write stall during high memtable ingestion".to_string(),
    );
    args.insert("component".to_string(), "memtable".to_string());

    let prompt_req = JsonRpcRequest::new(
        "req-prompt-1",
        "prompts/get",
        Some(serde_json::to_value(McpGetPromptParams {
            name: "flash_troubleshoot".to_string(),
            arguments: Some(args),
        })?),
    );
    client.send_request(prompt_req)?;
    let prompt_resp = client.receive_response()?;
    println!(
        "[6/6] Rendered Prompt Template 'flash_troubleshoot': {:?}",
        prompt_resp.result
    );

    drop(client);
    server_transport.close();
    let _ = server_handle.join();
    println!("\n=== MCP Server Demo Complete ===");
    Ok(())
}
