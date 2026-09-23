use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Subcommand;
use serde::Deserialize;

use crate::agent::{
    create_standard_tool_registry, AgentCapability, AgentDefinition, AgentId, AgentPipeline,
    AgentRole, AgentTask, EpisodicMemoryEntry, TaskId, TaskPriority, TaskStatus, ToolOutput,
};
use crate::cli::formatter;
use crate::cli::{open_db, GlobalOpts, OutputFormat};
use crate::error::{FlashStoreError, Result};

/// All Agent swarm subcommands.
#[derive(Debug, Subcommand)]
pub enum AgentCmd {
    /// Submit and execute an autonomous task
    Run {
        /// Task title or prompt (positional)
        #[arg(value_name = "PROMPT")]
        prompt: Option<String>,

        /// Task prompt via flag
        #[arg(short, long = "prompt")]
        prompt_flag: Option<String>,

        /// Optional task ID
        #[arg(long)]
        id: Option<String>,

        /// Task priority: low, medium, high, critical
        #[arg(long, default_value = "medium")]
        priority: String,

        /// Explicitly assign to agent ID
        #[arg(long)]
        agent: Option<String>,

        /// Required capabilities (comma-separated: text_generation, tool_calling, code_execution, storage_ops, memory_retrieval, reflection)
        #[arg(long)]
        capabilities: Option<String>,

        /// Input context as JSON string or raw text
        #[arg(long)]
        input: Option<String>,

        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Execute a multi-task workflow graph defined in a JSON file
    Workflow {
        /// Path to JSON workflow definition file
        #[arg(short, long)]
        file: PathBuf,

        /// Workflow ID override
        #[arg(long)]
        id: Option<String>,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Register a new agent definition in the swarm
    Register {
        /// Unique agent ID
        #[arg(short, long)]
        id: String,

        /// Agent display name
        #[arg(short, long)]
        name: Option<String>,

        /// Agent role: planner, router, coder, critic, verifier, tool_executor, memory_agent, or custom
        #[arg(short, long, default_value = "coder")]
        role: String,

        /// Comma-separated capabilities
        #[arg(long)]
        capabilities: Option<String>,

        /// System prompt instruction for the agent
        #[arg(long)]
        prompt: Option<String>,

        /// Maximum reflection/execution iterations
        #[arg(long, default_value_t = 10)]
        max_iterations: usize,

        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,
    },

    /// List active agents in the swarm
    Agents {
        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List workflow tasks
    Tasks {
        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Filter by task status: pending, assigned, in_progress, in_reflection, completed, failed, cancelled
        #[arg(long)]
        status: Option<String>,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Inspect a task by ID
    GetTask {
        /// Task identifier
        #[arg(value_name = "ID")]
        id: String,

        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List all registered swarm tools and schemas
    Tools {
        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Invoke a tool directly with JSON arguments
    Call {
        /// Tool name (e.g. calculator, echo, kv_get, kv_put, kv_scan, kv_delete)
        #[arg(value_name = "TOOL")]
        tool: String,

        /// JSON arguments payload string (e.g. '{"expression": "2 + 2"}')
        #[arg(value_name = "ARGS", default_value = "{}")]
        args: String,

        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Search or record episodic memory
    Memory {
        /// Memory query string for semantic recall
        #[arg(short, long)]
        query: Option<String>,

        /// Record a new episodic memory content string
        #[arg(long)]
        record: Option<String>,

        /// Agent ID for recorded memory
        #[arg(long, default_value = "default")]
        agent: String,

        /// Importance score for recorded memory (0.0 - 1.0)
        #[arg(long, default_value_t = 0.5)]
        importance: f32,

        /// Limit number of recalled memories
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Display multi-agent swarm telemetry, latency breakdown, and execution statistics
    Stats {
        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Save workflow state checkpoint to FlashStore
    Checkpoint {
        /// Workflow namespace / ID
        #[arg(long, default_value = "default")]
        workflow: String,
    },
}

/// JSON input structure for multi-task workflow batch files.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchWorkflowInput {
    pub workflow_id: Option<String>,
    pub tasks: Vec<BatchTaskItem>,
}

/// Individual task specification in workflow batch file.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchTaskItem {
    pub id: Option<String>,
    pub title: Option<String>,
    pub description: String,
    pub priority: Option<String>,
    pub agent: Option<String>,
    pub capabilities: Option<Vec<String>>,
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub depends_on: Vec<String>,
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

/// Parse priority string into `TaskPriority`.
pub fn parse_priority(s: &str) -> TaskPriority {
    match s.trim().to_lowercase().as_str() {
        "low" => TaskPriority::Low,
        "high" => TaskPriority::High,
        "critical" => TaskPriority::Critical,
        _ => TaskPriority::Medium,
    }
}

/// Parse role string into `AgentRole`.
pub fn parse_role(s: &str) -> AgentRole {
    match s.trim().to_lowercase().as_str() {
        "planner" => AgentRole::Planner,
        "router" => AgentRole::Router,
        "coder" => AgentRole::Coder,
        "critic" => AgentRole::Critic,
        "verifier" => AgentRole::Verifier,
        "executor" | "tool_executor" => AgentRole::ToolExecutor,
        "memory" | "memory_agent" => AgentRole::MemoryAgent,
        other => AgentRole::Custom(other.to_string()),
    }
}

/// Parse capabilities comma-separated string into a `HashSet<AgentCapability>`.
pub fn parse_capabilities(s: &str) -> HashSet<AgentCapability> {
    let mut caps = HashSet::new();
    for part in s.split(',') {
        let trimmed = part.trim().to_lowercase();
        match trimmed.as_str() {
            "text" | "text_generation" => {
                caps.insert(AgentCapability::TextGeneration);
            }
            "tool" | "tool_calling" => {
                caps.insert(AgentCapability::ToolCalling);
            }
            "memory" | "memory_retrieval" => {
                caps.insert(AgentCapability::MemoryRetrieval);
            }
            "reflection" => {
                caps.insert(AgentCapability::Reflection);
            }
            "code" | "code_execution" => {
                caps.insert(AgentCapability::CodeExecution);
            }
            "storage" | "storage_ops" => {
                caps.insert(AgentCapability::StorageOps);
            }
            _ => {}
        }
    }
    caps
}

/// Seed standard starter agents if swarm registry is empty.
pub fn seed_standard_agents(pipeline: &mut AgentPipeline) -> Result<()> {
    if !pipeline.engine().state().active_agents.is_empty() {
        return Ok(());
    }

    let planner = AgentDefinition::builder("planner", AgentRole::Planner)
        .name("Swarm Planner")
        .capability(AgentCapability::TextGeneration)
        .capability(AgentCapability::Reflection)
        .system_prompt("Analyze objectives and generate step-by-step execution plans.")
        .build();

    let coder = AgentDefinition::builder("coder", AgentRole::Coder)
        .name("Code Synthesis Specialist")
        .capability(AgentCapability::CodeExecution)
        .capability(AgentCapability::ToolCalling)
        .capability(AgentCapability::TextGeneration)
        .system_prompt("Synthesize correct, robust, and clean code.")
        .build();

    let executor = AgentDefinition::builder("executor", AgentRole::ToolExecutor)
        .name("Tool Execution Worker")
        .capability(AgentCapability::ToolCalling)
        .capability(AgentCapability::StorageOps)
        .system_prompt("Execute deterministic tools and storage operations.")
        .build();

    let critic = AgentDefinition::builder("critic", AgentRole::Critic)
        .name("Verification & Reflection Critic")
        .capability(AgentCapability::Reflection)
        .capability(AgentCapability::TextGeneration)
        .system_prompt("Critique task execution results and enforce invariants.")
        .build();

    let memory = AgentDefinition::builder("memory", AgentRole::MemoryAgent)
        .name("Episodic Memory Agent")
        .capability(AgentCapability::MemoryRetrieval)
        .capability(AgentCapability::StorageOps)
        .system_prompt("Maintain blackboard state and recall episodic memory.")
        .build();

    pipeline.register_agent(planner)?;
    pipeline.register_agent(coder)?;
    pipeline.register_agent(executor)?;
    pipeline.register_agent(critic)?;
    pipeline.register_agent(memory)?;

    Ok(())
}

/// Open an `AgentPipeline` configured with persistent `FlashStore` and standard agents.
pub fn open_agent_pipeline(opts: &GlobalOpts, workflow_id: &str) -> Result<AgentPipeline> {
    let db = Arc::new(open_db(opts)?);
    let mut pipeline = match AgentPipeline::restore_from_store(db.clone(), workflow_id) {
        Ok(p) => p,
        Err(_) => AgentPipeline::new(workflow_id).with_store(db),
    };
    seed_standard_agents(&mut pipeline)?;
    Ok(pipeline)
}

/// Execute an agent subcommand with the given global options.
pub fn run_agent_command(cmd: AgentCmd, opts: &GlobalOpts) -> Result<()> {
    match cmd {
        AgentCmd::Run {
            prompt,
            prompt_flag,
            id,
            priority,
            agent,
            capabilities,
            input,
            workflow,
            format,
        } => {
            let mut pipeline = open_agent_pipeline(opts, &workflow)?;
            let prompt_text = prompt.or(prompt_flag).ok_or_else(|| {
                FlashStoreError::InvalidArgument("Task prompt or description must be provided".into())
            })?;
            let fmt = parse_output_format(&format, opts.json);

            let task_id = id.unwrap_or_else(|| {
                format!(
                    "task_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_micros()
                )
            });

            let prio = parse_priority(&priority);
            let mut builder = AgentTask::builder(task_id.as_str(), prompt_text.as_str())
                .description(&prompt_text)
                .priority(prio);

            if let Some(caps_str) = capabilities {
                for cap in parse_capabilities(&caps_str) {
                    builder = builder.require_capability(cap);
                }
            }

            if let Some(inp) = input {
                let parsed_val: serde_json::Value = serde_json::from_str(&inp)
                    .unwrap_or_else(|_| serde_json::Value::String(inp));
                builder = builder.input_context(parsed_val);
            }

            let mut task = builder.build();
            if let Some(agent_id) = agent {
                task.assign(AgentId::new(agent_id));
            }

            let report = pipeline.run_single_task(task)?;
            let _ = pipeline.checkpoint();
            formatter::format_task_report(&report, fmt);
        }

        AgentCmd::Workflow { file, id, format } => {
            let content = fs::read_to_string(&file).map_err(FlashStoreError::Io)?;
            let batch: BatchWorkflowInput = serde_json::from_str(&content).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Invalid workflow JSON: {}", e))
            })?;

            let wf_id = id
                .or(batch.workflow_id)
                .unwrap_or_else(|| "batch_workflow".into());
            let mut pipeline = open_agent_pipeline(opts, &wf_id)?;
            let fmt = parse_output_format(&format, opts.json);

            for (idx, item) in batch.tasks.into_iter().enumerate() {
                let tid = item.id.unwrap_or_else(|| format!("task_{}", idx + 1));
                let title = item.title.unwrap_or_else(|| tid.clone());
                let prio = item
                    .priority
                    .as_deref()
                    .map(parse_priority)
                    .unwrap_or(TaskPriority::Medium);

                let mut task_builder = AgentTask::builder(tid.as_str(), title.as_str())
                    .description(&item.description)
                    .priority(prio);

                if let Some(caps) = item.capabilities {
                    for c in caps {
                        for parsed in parse_capabilities(&c) {
                            task_builder = task_builder.require_capability(parsed);
                        }
                    }
                }

                if let Some(inp) = item.input {
                    task_builder = task_builder.input_context(inp);
                }

                let mut task = task_builder.build();
                if let Some(ag) = item.agent {
                    task.assign(AgentId::new(ag));
                }

                let deps: Vec<TaskId> = item.depends_on.into_iter().map(TaskId::new).collect();
                pipeline.submit_task(task, &deps)?;
            }

            let summary = pipeline.run()?;
            let _ = pipeline.checkpoint();
            formatter::format_workflow_summary(&summary, fmt);
        }

        AgentCmd::Register {
            id,
            name,
            role,
            capabilities,
            prompt,
            max_iterations,
            workflow,
        } => {
            let mut pipeline = open_agent_pipeline(opts, &workflow)?;
            let parsed_role = parse_role(&role);
            let mut builder = AgentDefinition::builder(id.as_str(), parsed_role)
                .max_iterations(max_iterations);

            if let Some(n) = name {
                builder = builder.name(n);
            }
            if let Some(p) = prompt {
                builder = builder.system_prompt(p);
            }
            if let Some(caps_str) = capabilities {
                for cap in parse_capabilities(&caps_str) {
                    builder = builder.capability(cap);
                }
            }

            let agent_def = builder.build();
            pipeline.register_agent(agent_def)?;
            let _ = pipeline.checkpoint();

            if !opts.quiet {
                println!("OK: Registered agent '{}'", id);
            }
        }

        AgentCmd::Agents { workflow, format } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            let fmt = parse_output_format(&format, opts.json);
            let agents: Vec<AgentDefinition> = pipeline
                .engine()
                .state()
                .active_agents
                .values()
                .cloned()
                .collect();
            formatter::format_agent_list(&agents, fmt);
        }

        AgentCmd::Tasks {
            workflow,
            status,
            format,
        } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            let fmt = parse_output_format(&format, opts.json);
            let filter_status = status.map(|s| match s.to_lowercase().as_str() {
                "pending" => TaskStatus::Pending,
                "assigned" => TaskStatus::Assigned,
                "in_progress" => TaskStatus::InProgress,
                "in_reflection" => TaskStatus::InReflection,
                "completed" => TaskStatus::Completed,
                "failed" => TaskStatus::Failed,
                "cancelled" => TaskStatus::Cancelled,
                _ => TaskStatus::Pending,
            });

            let tasks: Vec<AgentTask> = pipeline
                .engine()
                .state()
                .tasks
                .values()
                .filter(|t| {
                    if let Some(st) = filter_status {
                        t.status == st
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();

            formatter::format_task_list(&tasks, fmt);
        }

        AgentCmd::GetTask {
            id,
            workflow,
            format,
        } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            let fmt = parse_output_format(&format, opts.json);
            let tid = TaskId::new(id.clone());

            match pipeline.engine().state().tasks.get(&tid) {
                Some(task) => formatter::format_task_detail(task, fmt),
                None => {
                    if fmt == OutputFormat::Json {
                        println!("null");
                    } else {
                        println!("Task not found: {}", id);
                    }
                }
            }
        }

        AgentCmd::Tools { workflow: _workflow, format } => {
            let fmt = parse_output_format(&format, opts.json);
            let db = Arc::new(open_db(opts)?);
            let registry = create_standard_tool_registry(Some(db));
            let defs = registry.list_definitions();
            formatter::format_tools_list(&defs, fmt);
        }

        AgentCmd::Call {
            tool,
            args,
            workflow: _workflow,
            format,
        } => {
            let fmt = parse_output_format(&format, opts.json);
            let db = Arc::new(open_db(opts)?);
            let registry = create_standard_tool_registry(Some(db));
            let tool_inst = registry.get(&tool).ok_or_else(|| {
                FlashStoreError::Agent(format!("Tool '{}' not found in registry", tool))
            })?;

            let parsed_args: serde_json::Value = serde_json::from_str(&args).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Invalid tool arguments JSON: {}", e))
            })?;

            let start = Instant::now();
            let result = tool_inst.execute(&parsed_args);
            let elapsed = start.elapsed().as_millis() as u64;

            let output = match result {
                Ok(val) => ToolOutput::success("cli_call", &tool, val, elapsed),
                Err(e) => ToolOutput::error("cli_call", &tool, e.to_string(), elapsed),
            };

            formatter::format_tool_output(&output, fmt);
        }

        AgentCmd::Memory {
            query,
            record,
            agent,
            importance,
            limit,
            workflow,
            format,
        } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            let fmt = parse_output_format(&format, opts.json);

            if let Some(content) = record {
                let entry = EpisodicMemoryEntry::new(AgentId::new(agent), content, importance);
                pipeline.memory().record_episode(entry)?;
                let _ = pipeline.checkpoint();
                if !opts.quiet {
                    println!("OK: Recorded episodic memory");
                }
            } else if let Some(q) = query {
                let recalled = pipeline.memory().recall(&q, limit)?;
                formatter::format_agent_memories(&recalled, fmt);
            } else {
                return Err(FlashStoreError::InvalidArgument(
                    "Must specify --query or --record for memory command".into(),
                ));
            }
        }

        AgentCmd::Stats { workflow, format } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            let fmt = parse_output_format(&format, opts.json);
            let snapshot = pipeline.metrics_snapshot();
            formatter::format_agent_metrics(&snapshot, fmt);
        }

        AgentCmd::Checkpoint { workflow } => {
            let pipeline = open_agent_pipeline(opts, &workflow)?;
            pipeline.checkpoint()?;
            if !opts.quiet {
                println!("OK: Workflow checkpoint '{}' saved", workflow);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_agent_cli_helpers() {
        assert_eq!(parse_priority("low"), TaskPriority::Low);
        assert_eq!(parse_priority("high"), TaskPriority::High);
        assert_eq!(parse_priority("critical"), TaskPriority::Critical);
        assert_eq!(parse_priority("other"), TaskPriority::Medium);

        assert_eq!(parse_role("planner"), AgentRole::Planner);
        assert_eq!(parse_role("coder"), AgentRole::Coder);
        assert_eq!(parse_role("critic"), AgentRole::Critic);
        assert_eq!(parse_role("custom_bot"), AgentRole::Custom("custom_bot".into()));

        let caps = parse_capabilities("code_execution,tool_calling,storage_ops");
        assert!(caps.contains(&AgentCapability::CodeExecution));
        assert!(caps.contains(&AgentCapability::ToolCalling));
        assert!(caps.contains(&AgentCapability::StorageOps));
    }

    #[test]
    fn test_agent_cli_end_to_end() -> Result<()> {
        let dir = tempdir().unwrap();
        let opts = GlobalOpts {
            path: dir.path().to_path_buf(),
            quiet: true,
            ..Default::default()
        };

        // Register agent
        run_agent_command(
            AgentCmd::Register {
                id: "test_coder".into(),
                name: Some("Test Coder".into()),
                role: "coder".into(),
                capabilities: Some("code_execution,tool_calling".into()),
                prompt: Some("You write Rust code".into()),
                max_iterations: 5,
                workflow: "test_wf".into(),
            },
            &opts,
        )?;

        // List agents
        run_agent_command(
            AgentCmd::Agents {
                workflow: "test_wf".into(),
                format: "json".into(),
            },
            &opts,
        )?;

        // Run single task
        run_agent_command(
            AgentCmd::Run {
                prompt: Some("Calculate 10 + 20 using calculator tool".into()),
                prompt_flag: None,
                id: Some("t_calc".into()),
                priority: "high".into(),
                agent: None,
                capabilities: Some("tool_calling".into()),
                input: Some("{\"expr\": \"10 + 20\"}".into()),
                workflow: "test_wf".into(),
                format: "text".into(),
            },
            &opts,
        )?;

        // Inspect task
        run_agent_command(
            AgentCmd::GetTask {
                id: "t_calc".into(),
                workflow: "test_wf".into(),
                format: "json".into(),
            },
            &opts,
        )?;

        // List tasks
        run_agent_command(
            AgentCmd::Tasks {
                workflow: "test_wf".into(),
                status: None,
                format: "tsv".into(),
            },
            &opts,
        )?;

        // Call tool directly
        run_agent_command(
            AgentCmd::Call {
                tool: "calculator".into(),
                args: "{\"expression\": \"15 + 27\"}".into(),
                workflow: "test_wf".into(),
                format: "text".into(),
            },
            &opts,
        )?;

        // List tools
        run_agent_command(
            AgentCmd::Tools {
                workflow: "test_wf".into(),
                format: "text".into(),
            },
            &opts,
        )?;

        // Record & recall memory
        run_agent_command(
            AgentCmd::Memory {
                query: None,
                record: Some("Solved arithmetic problem with calculator tool".into()),
                agent: "test_coder".into(),
                importance: 0.9,
                limit: 5,
                workflow: "test_wf".into(),
                format: "text".into(),
            },
            &opts,
        )?;

        run_agent_command(
            AgentCmd::Memory {
                query: Some("arithmetic problem".into()),
                record: None,
                agent: "test_coder".into(),
                importance: 0.5,
                limit: 5,
                workflow: "test_wf".into(),
                format: "text".into(),
            },
            &opts,
        )?;

        // Stats
        run_agent_command(
            AgentCmd::Stats {
                workflow: "test_wf".into(),
                format: "json".into(),
            },
            &opts,
        )?;

        // Checkpoint
        run_agent_command(
            AgentCmd::Checkpoint {
                workflow: "test_wf".into(),
            },
            &opts,
        )?;

        Ok(())
    }
}
