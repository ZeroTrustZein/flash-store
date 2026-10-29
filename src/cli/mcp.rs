//! # Model Context Protocol (MCP) CLI Handler
//!
//! Exposes FlashStore's Model Context Protocol (MCP) capabilities via command-line subcommands:
//! serving over stdio, inspecting and executing tools, querying resources, rendering prompt
//! templates, inspecting server info, and executing raw JSON-RPC 2.0 requests.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Subcommand;
use parking_lot::RwLock;
use serde_json::Value;

use crate::agent::AgentPipeline;
use crate::cli::agent::seed_standard_agents;
use crate::cli::formatter;
use crate::cli::{open_db, GlobalOpts, OutputFormat};
use crate::error::{FlashStoreError, Result};
use crate::mcp::{
    JsonRpcRequest, McpGetPromptParams, McpServerConfigBuilder, McpSubsystem, McpToolCallParams,
    StdioTransport, DEFAULT_SERVER_NAME, DEFAULT_SERVER_VERSION, MCP_PROTOCOL_VERSION,
};
use crate::rag::{RagConfig, RagEngine};

/// All MCP subcommands for `flash-cli mcp`.
#[derive(Debug, Subcommand)]
pub enum McpCmd {
    /// Start the Model Context Protocol (MCP) server over standard I/O (stdio)
    Serve {
        /// Server identifier name
        #[arg(long, default_value = "flash-store-mcp")]
        name: String,

        /// Disable tool execution capability
        #[arg(long, default_value_t = false)]
        disable_tools: bool,

        /// Disable resource inspection capability
        #[arg(long, default_value_t = false)]
        disable_resources: bool,

        /// Disable prompt templates capability
        #[arg(long, default_value_t = false)]
        disable_prompts: bool,
    },

    /// List all registered Model Context Protocol (MCP) tools
    Tools {
        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Call an MCP tool directly with JSON arguments or argument file
    Call {
        /// Tool name to invoke (e.g. flash_get, flash_put, flash_rag_query, flash_scan)
        #[arg(value_name = "TOOL")]
        name: String,

        /// Arguments as JSON string: '{"key": "user:1", "value": "alice"}'
        #[arg(short, long)]
        args: Option<String>,

        /// Path to JSON file containing arguments
        #[arg(long)]
        file: Option<PathBuf>,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List all available Model Context Protocol (MCP) resources
    Resources {
        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Read an MCP resource by URI (e.g. flash://storage/stats, flash://rag/stats, flash://agent/state)
    Read {
        /// URI of resource to inspect
        #[arg(value_name = "URI")]
        uri: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List all registered Model Context Protocol (MCP) prompt templates
    Prompts {
        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Render and retrieve an MCP prompt template
    #[command(alias = "get-prompt")]
    GetPrompt {
        /// Prompt template name (e.g. troubleshoot_lsm, rag_synthesize, agent_orchestrate)
        #[arg(value_name = "NAME")]
        name: String,

        /// Template arguments as JSON string: '{"error_message": "Checksum mismatch"}'
        #[arg(short, long)]
        args: Option<String>,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Display MCP server capabilities, protocol version, and telemetry
    #[command(alias = "status", alias = "ping")]
    Info {
        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Send a raw JSON-RPC 2.0 request string or file to the MCP server
    Request {
        /// Raw JSON-RPC 2.0 string
        #[arg(short, long)]
        json: Option<String>,

        /// Path to file containing raw JSON-RPC 2.0 request
        #[arg(long)]
        file: Option<PathBuf>,

        /// Output format: json, text
        #[arg(long, default_value = "json")]
        format: String,
    },
}

/// Open an initialized `McpSubsystem` backed by persistent `FlashStore`, RAG, and Agent engines.
pub fn open_mcp_subsystem(opts: &GlobalOpts) -> Result<McpSubsystem> {
    let db = Arc::new(open_db(opts)?);

    // Attempt to recover RAG engine if state exists or initialize default
    let rag_engine = {
        let config = RagConfig::default();
        match RagEngine::recover(config, db.clone()) {
            Ok(engine) => Some(Arc::new(RwLock::new(engine))),
            Err(_) => None,
        }
    };

    // Attempt to restore or seed agent pipeline
    let agent_engine = {
        let workflow_id = "default";
        match AgentPipeline::restore_from_store(db.clone(), workflow_id) {
            Ok(pipeline) => Some(pipeline.engine().clone()),
            Err(_) => {
                let mut pipeline = AgentPipeline::new(workflow_id).with_store(db.clone());
                let _ = seed_standard_agents(&mut pipeline);
                Some(pipeline.engine().clone())
            }
        }
    };

    let config = McpServerConfigBuilder::new()
        .server_name(DEFAULT_SERVER_NAME)
        .store_path(&opts.path)
        .build();

    Ok(McpSubsystem::build(config, db, rag_engine, agent_engine))
}

/// Parse format string into `OutputFormat`.
pub fn parse_output_format(s: &str, json_flag: bool) -> OutputFormat {
    if json_flag || s.eq_ignore_ascii_case("json") {
        OutputFormat::Json
    } else if s.eq_ignore_ascii_case("tsv") {
        OutputFormat::Tsv
    } else {
        OutputFormat::Text
    }
}

/// Parse JSON argument string or file into `serde_json::Value`.
pub fn parse_json_args(raw: Option<String>, file: Option<PathBuf>) -> Result<Value> {
    if let Some(f) = file {
        let content = fs::read_to_string(&f).map_err(|e| {
            FlashStoreError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed reading args file '{}': {}", f.display(), e),
            ))
        })?;
        serde_json::from_str(&content).map_err(|e| {
            FlashStoreError::InvalidArgument(format!("Invalid JSON in file '{}': {}", f.display(), e))
        })
    } else if let Some(s) = raw {
        if s.trim().is_empty() {
            Ok(serde_json::json!({}))
        } else {
            serde_json::from_str(&s).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Invalid JSON arguments '{}': {}", s, e))
            })
        }
    } else {
        Ok(serde_json::json!({}))
    }
}

/// Execute an MCP subcommand with global options.
pub fn run_mcp_command(cmd: McpCmd, opts: &GlobalOpts) -> Result<()> {
    match cmd {
        McpCmd::Serve {
            name,
            disable_tools,
            disable_resources,
            disable_prompts,
        } => {
            let db = Arc::new(open_db(opts)?);
            let rag_engine = {
                let config = RagConfig::default();
                RagEngine::recover(config, db.clone()).ok().map(|e| Arc::new(RwLock::new(e)))
            };
            let agent_engine = {
                let mut pipeline = AgentPipeline::new("mcp_server").with_store(db.clone());
                let _ = seed_standard_agents(&mut pipeline);
                Some(pipeline.engine().clone())
            };

            let config = McpServerConfigBuilder::new()
                .server_name(name)
                .store_path(&opts.path)
                .enable_tools(!disable_tools)
                .enable_resources(!disable_resources)
                .enable_prompts(!disable_prompts)
                .build();

            let subsystem = McpSubsystem::build(config, db, rag_engine, agent_engine);
            let transport = StdioTransport::new(io::stdin(), io::stdout());
            subsystem.server().serve(&transport)?;
            Ok(())
        }

        McpCmd::Tools { format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let tools = subsystem.server().tools().list_tools();
            formatter::format_mcp_tools(&tools, fmt);
            Ok(())
        }

        McpCmd::Call {
            name,
            args,
            file,
            format,
        } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let arguments = parse_json_args(args, file)?;

            let params = McpToolCallParams {
                name: name.clone(),
                arguments: Some(arguments),
            };

            let res = subsystem.server().tools().call(&params.name, params.arguments)?;
            formatter::format_mcp_tool_result(&res, fmt);
            Ok(())
        }

        McpCmd::Resources { format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let resources = subsystem.server().resources().list_resources();
            formatter::format_mcp_resources(&resources, fmt);
            Ok(())
        }

        McpCmd::Read { uri, format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let contents = subsystem.server().resources().read_resource(&uri)?;
            formatter::format_mcp_resource_contents(&contents, fmt);
            Ok(())
        }

        McpCmd::Prompts { format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let prompts = subsystem.server().prompts().list_prompts();
            formatter::format_mcp_prompts(&prompts, fmt);
            Ok(())
        }

        McpCmd::GetPrompt { name, args, format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let arguments = parse_json_args(args, None)?;

            let params = McpGetPromptParams {
                name: name.clone(),
                arguments: if arguments.as_object().map(|o| o.is_empty()).unwrap_or(true) {
                    None
                } else {
                    Some(arguments)
                },
            };

            let res = subsystem
                .server()
                .prompts()
                .get_prompt(&params.name, params.arguments)?;
            formatter::format_mcp_prompt_result(&res, fmt);
            Ok(())
        }

        McpCmd::Info { format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let fmt = parse_output_format(&format, opts.json);

            let tools_count = subsystem.server().tools().list_tools().len();
            let resources_count = subsystem.server().resources().list_resources().len();
            let prompts_count = subsystem.server().prompts().list_prompts().len();

            let info = serde_json::json!({
                "server_name": DEFAULT_SERVER_NAME,
                "server_version": DEFAULT_SERVER_VERSION,
                "protocol_version": MCP_PROTOCOL_VERSION,
                "tools_count": tools_count,
                "resources_count": resources_count,
                "prompts_count": prompts_count,
                "tools_enabled": subsystem.server().config().enable_tools,
                "resources_enabled": subsystem.server().config().enable_resources,
                "prompts_enabled": subsystem.server().config().enable_prompts,
                "logging_enabled": subsystem.server().config().enable_logging,
            });

            formatter::format_mcp_server_info(&info, fmt);
            Ok(())
        }

        McpCmd::Request { json, file, format } => {
            let subsystem = open_mcp_subsystem(opts)?;
            let raw_req = if let Some(f) = file {
                fs::read_to_string(&f).map_err(|e| {
                    FlashStoreError::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed reading request file '{}': {}", f.display(), e),
                    ))
                })?
            } else if let Some(j) = json {
                j
            } else {
                return Err(FlashStoreError::InvalidArgument(
                    "Either --json or --file must be provided for MCP Request".into(),
                ));
            };

            let req: JsonRpcRequest = serde_json::from_str(&raw_req).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Failed parsing JSON-RPC request: {}", e))
            })?;

            let resp = subsystem.server().handle_request(req);
            let fmt = parse_output_format(&format, opts.json);
            match fmt {
                OutputFormat::Json | OutputFormat::Tsv => {
                    println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                }
                OutputFormat::Text => {
                    if let Some(res) = &resp.result {
                        println!("Result: {}", serde_json::to_string_pretty(res).unwrap());
                    } else if let Some(err) = &resp.error {
                        println!("Error [{}]: {}", err.code, err.message);
                    } else {
                        println!("Empty response");
                    }
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_mcp_args_helpers() {
        let empty = parse_json_args(None, None).unwrap();
        assert_eq!(empty, serde_json::json!({}));

        let parsed = parse_json_args(Some(r#"{"key": "test_val"}"#.into()), None).unwrap();
        assert_eq!(parsed["key"], "test_val");

        let err = parse_json_args(Some("not json".into()), None);
        assert!(err.is_err());

        assert_eq!(parse_output_format("json", false), OutputFormat::Json);
        assert_eq!(parse_output_format("text", true), OutputFormat::Json);
        assert_eq!(parse_output_format("tsv", false), OutputFormat::Tsv);
        assert_eq!(parse_output_format("other", false), OutputFormat::Text);
    }

    #[test]
    fn test_open_mcp_subsystem_and_tools() -> Result<()> {
        let dir = tempdir().unwrap();
        let opts = GlobalOpts {
            path: dir.path().to_path_buf(),
            quiet: true,
            ..Default::default()
        };

        let subsystem = open_mcp_subsystem(&opts)?;
        let tools = subsystem.server().tools().list_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "flash_put"));
        assert!(tools.iter().any(|t| t.name == "flash_get"));
        assert!(tools.iter().any(|t| t.name == "flash_rag_query"));

        let resources = subsystem.server().resources().list_resources();
        assert!(!resources.is_empty());

        let prompts = subsystem.server().prompts().list_prompts();
        assert!(!prompts.is_empty());
        assert!(prompts.iter().any(|p| p.name == "troubleshoot_lsm"));

        Ok(())
    }
}
