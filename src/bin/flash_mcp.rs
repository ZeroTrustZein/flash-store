//! # FlashStore MCP Server Binary Entrypoint
//!
//! Exposes FlashStore LSM-tree key-value storage, RAG retrieval, and agent workflows
//! over standard I/O (stdin/stdout) via the Model Context Protocol (MCP).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use parking_lot::RwLock;

use flash_store::mcp::{McpServerConfigBuilder, McpSubsystem, StdioTransport};
use flash_store::rag::{RagConfig, RagEngine};

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

    /// Semantic version of the MCP server
    #[arg(long)]
    server_version: Option<String>,

    /// Disable tool execution endpoint
    #[arg(long, default_value_t = false)]
    disable_tools: bool,

    /// Disable resource inspect endpoint
    #[arg(long, default_value_t = false)]
    disable_resources: bool,

    /// Disable prompt templates endpoint
    #[arg(long, default_value_t = false)]
    disable_prompts: bool,

    /// Enable RAG retrieval tools and resources
    #[arg(long, default_value_t = false)]
    rag: bool,

    /// Enable autonomous multi-agent workflow tools and resources
    #[arg(long, default_value_t = false)]
    agent: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut builder = McpServerConfigBuilder::new()
        .server_name(args.name)
        .store_path(&args.dir)
        .enable_tools(!args.disable_tools)
        .enable_resources(!args.disable_resources)
        .enable_prompts(!args.disable_prompts);

    if let Some(v) = args.server_version {
        builder = builder.server_version(v);
    }
    let config = builder.build();

    let store_opts = flash_store::OptionsBuilder::new()
        .dir(&args.dir)
        .create_if_missing(true)
        .build();
    let store = Arc::new(flash_store::FlashStore::open(store_opts)?);

    let rag_engine = if args.rag {
        let rag_cfg = RagConfig::default();
        let engine = RagEngine::recover(rag_cfg, store.clone())?;
        Some(Arc::new(RwLock::new(engine)))
    } else {
        None
    };

    let agent_engine = if args.agent {
        let mut engine =
            flash_store::agent::AgentEngine::new("flash_mcp").with_store(store.clone());
        flash_store::cli::mcp::seed_standard_agents_in_engine(&mut engine);
        Some(Arc::new(RwLock::new(engine)))
    } else {
        None
    };

    let subsystem = McpSubsystem::build(config, store, rag_engine, agent_engine);
    let transport = StdioTransport::new(io::stdin(), io::stdout());

    subsystem.server().serve(&transport)?;

    Ok(())
}
