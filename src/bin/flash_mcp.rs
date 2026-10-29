//! # FlashStore MCP Server Binary Entrypoint
//!
//! Exposes FlashStore LSM-tree key-value storage, RAG retrieval, and agent workflows
//! over standard I/O (stdin/stdout) via the Model Context Protocol (MCP).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

use flash_store::mcp::{McpServer, McpServerConfigBuilder, StdioTransport};

#[derive(Parser, Debug)]
#[command(name = "flash-mcp")]
#[command(about = "Model Context Protocol (MCP) server for FlashStore LSM-tree engine")]
#[command(version)]
struct Args {
    /// Path to FlashStore database directory
    #[arg(short, long, default_value = "./data/flash-store")]
    dir: PathBuf,

    /// Identifier name for this MCP server instance
    #[arg(long, default_value = "flash-store-mcp")]
    name: String,

    /// Disable tool execution endpoint
    #[arg(long, default_value_t = false)]
    disable_tools: bool,

    /// Disable resource inspect endpoint
    #[arg(long, default_value_t = false)]
    disable_resources: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config = McpServerConfigBuilder::new()
        .server_name(args.name)
        .store_path(&args.dir)
        .enable_tools(!args.disable_tools)
        .enable_resources(!args.disable_resources)
        .build();

    let store_opts = flash_store::OptionsBuilder::new()
        .dir(&args.dir)
        .create_if_missing(true)
        .build();
    let store = Arc::new(flash_store::FlashStore::open(store_opts)?);

    let server = McpServer::with_store(config, store);
    let transport = StdioTransport::new(io::stdin(), io::stdout());

    server.serve(&transport)?;

    Ok(())
}
