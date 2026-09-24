//! # FlashStore Autonomous Multi-Agent Swarm Workflow Example
//!
//! Demonstrates end-to-end multi-agent orchestration directly on top of FlashStore:
//! 1. Initializing an embedded FlashStore LSM-tree engine for durable swarm state.
//! 2. Creating an `AgentPipeline` with persistent storage, episodic memory, and telemetry.
//! 3. Registering specialized agents (Planner, Storage Coder, Verifier).
//! 4. Defining a Directed Acyclic Graph (DAG) of interdependent tasks.
//! 5. Executing the workflow with automated tool calling (`kv_put`) and stateful reflection loops.
//! 6. Querying episodic memory with associative recall scoring.
//! 7. Inspecting execution metrics, latency percentiles, and restoring from a FlashStore checkpoint.

use flash_store::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

fn main() -> Result<()> {
    println!("=== FlashStore Autonomous Multi-Agent Swarm Demo ===\n");

    // 1. Initialize persistent FlashStore storage engine
    let dir = tempdir().map_err(FlashStoreError::Io)?;
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(2 * 1024 * 1024)
        .build();
    let db = Arc::new(FlashStore::open(options)?);
    println!(
        "[1/7] Initialized underlying FlashStore LSM storage engine at {:?}",
        dir.path()
    );

    // 2. Create the Agent Pipeline attached to FlashStore
    let workflow_id = "metrics-ingest-swarm";
    let mut pipeline = AgentPipeline::new(workflow_id).with_store(db.clone());
    println!(
        "[2/7] Created AgentPipeline for workflow '{}'.",
        workflow_id
    );

    // 3. Register specialized swarm agents
    let planner = AgentDefinition::builder("agent-planner", AgentRole::Planner)
        .name("Architecture Planner")
        .description("Decomposes requirements into structured technical plans")
        .capability(AgentCapability::TextGeneration)
        .build();

    let storage_coder = AgentDefinition::builder("agent-coder", AgentRole::Coder)
        .name("FlashStore Storage Engineer")
        .description("Executes storage operations and tool calling")
        .capability(AgentCapability::CodeExecution)
        .capability(AgentCapability::StorageOps)
        .capability(AgentCapability::ToolCalling)
        .build();

    let verifier = AgentDefinition::builder("agent-verifier", AgentRole::Verifier)
        .name("Quality & Invariant Verifier")
        .description("Validates outputs, checks domain invariants, and drives reflection")
        .capability(AgentCapability::Reflection)
        .capability(AgentCapability::TextGeneration)
        .max_iterations(4)
        .build();

    pipeline.register_agent(planner)?;
    pipeline.register_agent(storage_coder)?;
    pipeline.register_agent(verifier)?;
    println!("[3/7] Registered 3 swarm agents: planner, storage_coder, verifier.");

    // 4. Define and submit tasks forming a DAG
    // Task 1: Planner designs the schema
    let task_plan = AgentTask::builder("task-1-plan", "Design Metrics Storage Schema")
        .description("Formulate key-value schema for telemetry time-series points")
        .priority(TaskPriority::High)
        .require_capability(AgentCapability::TextGeneration)
        .build();

    // Task 2: Coder writes a metric using kv_put tool (depends on Task 1)
    let task_write = AgentTask::builder("task-2-write", "Store Telemetry Metric Point")
        .description("Insert CPU utilization sample using FlashStore kv_put tool")
        .priority(TaskPriority::Critical)
        .require_capability(AgentCapability::StorageOps)
        .require_capability(AgentCapability::ToolCalling)
        .input_context(serde_json::json!({
            "required_tool": "kv_put",
            "tool_args": {
                "key": "telemetry:cpu:node_alpha",
                "value": "{\"load_pct\": 78.4, \"cores\": 16, \"timestamp\": 1700000000}"
            }
        }))
        .build();

    // Task 3: Verifier checks output and verifies reflection invariants (depends on Task 2)
    let task_verify = AgentTask::builder("task-3-verify", "Verify Metric Integrity")
        .description("Review stored telemetry point and confirm invariant compliance")
        .priority(TaskPriority::Medium)
        .require_capability(AgentCapability::Reflection)
        .require_capability(AgentCapability::TextGeneration)
        .build();

    pipeline.submit_task(task_plan, &[])?;
    pipeline.submit_task(task_write, &["task-1-plan".into()])?;
    pipeline.submit_task(task_verify, &["task-2-write".into()])?;
    println!("[4/7] Submitted 3 interdependent tasks with DAG constraints.");

    // 5. Execute workflow to completion
    println!("\n[5/7] Executing autonomous multi-agent workflow...");
    let summary = pipeline.run()?;

    println!("      Workflow ID:       {}", summary.workflow_id);
    println!("      Completed Tasks:   {}", summary.completed_tasks);
    println!("      Failed Tasks:      {}", summary.failed_tasks);
    println!("      Execution Time:    {} ms", summary.total_duration_ms);

    for (idx, report) in summary.reports.iter().enumerate() {
        println!(
            "      -> Step {}: Task '{}' executed by '{}' in {} ms (Status: {:?})",
            idx + 1,
            report.task_id,
            report.assigned_agent,
            report.execution_time_ms,
            report.status
        );
    }

    // Verify underlying FlashStore directly received data from the tool
    let stored_metric = db.get("telemetry:cpu:node_alpha")?;
    assert!(stored_metric.is_some());
    println!(
        "      -> Direct FlashStore LSM read: telemetry:cpu:node_alpha = {}",
        String::from_utf8_lossy(&stored_metric.unwrap())
    );

    // 6. Demonstrate Episodic Memory and Associative Recall
    println!("\n[6/7] Associative Episodic Memory Search:");
    let memory_hits = pipeline
        .memory()
        .recall("Telemetry schema and storage", 5)?;
    for (i, hit) in memory_hits.iter().enumerate() {
        println!(
            "      Hit #{}: Score {:.3} (Agent: {}) -> \"{}\"",
            i + 1,
            hit.score,
            hit.memory.agent_id,
            hit.memory.content
        );
    }

    // 7. Telemetry Metrics & FlashStore Checkpoint Recovery
    println!("\n[7/7] Telemetry and Checkpoint Recovery:");
    let stats = pipeline.metrics_snapshot();
    println!("      Total Tasks Run:   {}", stats.tasks_completed);
    println!(
        "      Task Success Rate: {:.1}%",
        stats.task_success_rate * 100.0
    );
    println!(
        "      Avg Task Latency:  {:.2} ms",
        stats.task_execution_latency.avg_ms
    );

    // Reconstitute pipeline from FlashStore persistence checkpoint
    let restored = AgentPipeline::restore_from_store(db.clone(), workflow_id)?;
    println!(
        "      Successfully reconstituted workflow '{}' from FlashStore: {} tasks verified.",
        workflow_id,
        restored.engine().state().tasks.len()
    );

    println!("\n=== Multi-Agent Swarm Demo Completed Successfully ===");
    Ok(())
}
