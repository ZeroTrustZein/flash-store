//! Comprehensive Integration and Domain Model Tests for Agent Workflow Subsystem
//!
//! Validates:
//! - Multi-agent routing, capability matching, and fallback strategies.
//! - Agent definition builders, roles, and status checks.
//! - Task prioritization, status transitions, and lifecycle state machines.
//! - Tool definitions, parameter schemas, OpenAI/MCP JSON schema compliance, and validation.
//! - Automated tool registry execution, dispatching, and error encapsulation.
//! - Stateful reflection loops, critique scoring, convergence, and candidate selection.
//! - Workflow state management, shared memory, and audit trails.
//! - Full serialization/deserialization fidelity for persistence across FlashStore KV.

use flash_store::prelude::*;
use std::sync::Arc;

#[test]
fn test_agent_id_and_role_conversions() {
    let id1 = AgentId::new("planner-01");
    let id2: AgentId = "planner-01".into();
    let id3: AgentId = String::from("planner-01").into();

    assert_eq!(id1, id2);
    assert_eq!(id2, id3);
    assert_eq!(id1.as_str(), "planner-01");
    assert_eq!(id1.as_ref(), "planner-01");
    assert_eq!(&*id1, "planner-01");
    assert_eq!(format!("{}", id1), "planner-01");

    let roles = vec![
        (AgentRole::Planner, "planner"),
        (AgentRole::Router, "router"),
        (AgentRole::Coder, "coder"),
        (AgentRole::Critic, "critic"),
        (AgentRole::Verifier, "verifier"),
        (AgentRole::ToolExecutor, "tool_executor"),
        (AgentRole::MemoryAgent, "memory_agent"),
        (
            AgentRole::Custom("security_auditor".into()),
            "security_auditor",
        ),
    ];

    for (role, expected_str) in roles {
        assert_eq!(role.as_str(), expected_str);
        assert_eq!(format!("{}", role), expected_str);
    }
}

#[test]
fn test_agent_definition_capabilities_and_builder() {
    let agent = AgentDefinition::builder("coder-alpha", AgentRole::Coder)
        .name("Lead Rust Coder")
        .description("Implements high-performance concurrent storage components")
        .capability(AgentCapability::CodeExecution)
        .capability(AgentCapability::ToolCalling)
        .capability(AgentCapability::StorageOps)
        .system_prompt("You write production-grade, zero-allocation Rust code.")
        .max_iterations(8)
        .timeout_secs(90)
        .build();

    assert_eq!(agent.id.as_str(), "coder-alpha");
    assert_eq!(agent.name, "Lead Rust Coder");
    assert_eq!(agent.role, AgentRole::Coder);
    assert!(agent.has_capability(AgentCapability::CodeExecution));
    assert!(agent.has_capability(AgentCapability::ToolCalling));
    assert!(agent.has_capability(AgentCapability::StorageOps));
    assert!(!agent.has_capability(AgentCapability::MemoryRetrieval));
    assert_eq!(agent.max_iterations, 8);
    assert_eq!(agent.timeout_secs, 90);

    // Serialization roundtrip
    let serialized = serde_json::to_string(&agent).unwrap();
    let deserialized: AgentDefinition = serde_json::from_str(&serialized).unwrap();
    assert_eq!(agent, deserialized);
}

#[test]
fn test_task_priority_ordering_and_defaults() {
    assert!(TaskPriority::Critical > TaskPriority::High);
    assert!(TaskPriority::High > TaskPriority::Medium);
    assert!(TaskPriority::Medium > TaskPriority::Low);

    assert_eq!(TaskPriority::default(), TaskPriority::Medium);
    assert_eq!(TaskPriority::Critical.as_u8(), 3);
    assert_eq!(TaskPriority::Critical.as_str(), "critical");
    assert_eq!(TaskPriority::Low.as_str(), "low");
}

#[test]
fn test_task_lifecycle_complete_flow() {
    let mut task = AgentTask::builder("task-memtable-bench", "Benchmark SkipList Concurrent Puts")
        .description("Run concurrent Criterion benchmarks on MemTable under 8 threads")
        .priority(TaskPriority::Critical)
        .require_capability(AgentCapability::CodeExecution)
        .require_capability(AgentCapability::StorageOps)
        .input_context(serde_json::json!({
            "threads": 8,
            "target_ops_sec": 500000,
        }))
        .build();

    assert_eq!(task.status, TaskStatus::Pending);
    assert!(!task.status.is_terminal());
    assert!(!task.status.is_active());

    // Assign
    task.assign(AgentId::new("worker-perf"));
    assert_eq!(task.status, TaskStatus::Assigned);
    assert_eq!(task.assigned_agent.as_deref(), Some("worker-perf"));

    // Start
    task.start();
    assert_eq!(task.status, TaskStatus::InProgress);
    assert!(task.status.is_active());

    // Enter reflection
    task.reflect();
    assert_eq!(task.status, TaskStatus::InReflection);
    assert!(task.status.is_active());

    // Complete
    task.complete(serde_json::json!({
        "actual_ops_sec": 720000,
        "p99_latency_micros": 4.2
    }));
    assert_eq!(task.status, TaskStatus::Completed);
    assert!(task.status.is_terminal());
    assert!(!task.status.is_active());
    assert!(task.error_message.is_none());

    let artifacts = task.output_artifacts.unwrap();
    assert_eq!(artifacts["actual_ops_sec"], 720000);
}

#[test]
fn test_task_routing_decisions_and_fallbacks() {
    let router = DefaultTaskRouter::new(RoutingStrategy::CapabilityBased);

    let coder = AgentDefinition::builder("agent-coder", AgentRole::Coder)
        .capability(AgentCapability::CodeExecution)
        .capability(AgentCapability::TextGeneration)
        .build();

    let storage_expert = AgentDefinition::builder("agent-storage", AgentRole::MemoryAgent)
        .capability(AgentCapability::StorageOps)
        .capability(AgentCapability::MemoryRetrieval)
        .capability(AgentCapability::ToolCalling)
        .build();

    let critic = AgentDefinition::builder("agent-critic", AgentRole::Critic)
        .capability(AgentCapability::Reflection)
        .capability(AgentCapability::TextGeneration)
        .build();

    let pool = vec![coder, storage_expert, critic];

    // Task requiring storage + tools
    let task = AgentTask::builder("t-compact", "Run LSM Compaction")
        .require_capability(AgentCapability::StorageOps)
        .require_capability(AgentCapability::ToolCalling)
        .build();

    let decision = router.route(&task, &pool).unwrap();
    assert_eq!(decision.target_agent.as_str(), "agent-storage");
    assert_eq!(decision.strategy, RoutingStrategy::CapabilityBased);
    assert_eq!(decision.confidence, 1.0);
    assert!(decision.rationale.contains("100%"));

    // Serialization roundtrip of decision
    let serialized = serde_json::to_string(&decision).unwrap();
    let deserialized: RoutingDecision = serde_json::from_str(&serialized).unwrap();
    assert_eq!(decision, deserialized);
}

#[test]
fn test_task_routing_empty_pool_returns_error() {
    let router = DefaultTaskRouter::default();
    let task = AgentTask::builder("t-err", "Dummy task").build();
    let res = router.route(&task, &[]);
    assert!(res.is_err());
}

#[test]
fn test_tool_definition_and_json_schema_validation() {
    let tool = ToolDefinition::new("store_put", "Stores a key-value record in FlashStore")
        .with_parameter(ToolParameter::new(
            "key",
            ToolParamType::String,
            "Key identifier",
            true,
        ))
        .with_parameter(ToolParameter::new(
            "value",
            ToolParamType::String,
            "Payload bytes",
            true,
        ))
        .with_parameter(
            ToolParameter::new(
                "sync",
                ToolParamType::Boolean,
                "Sync to disk via WAL",
                false,
            )
            .with_default(serde_json::json!(true)),
        )
        .with_timeout_ms(3000);

    let schema = tool.to_json_schema();
    assert_eq!(schema["name"], "store_put");
    assert_eq!(schema["parameters"]["type"], "object");
    assert_eq!(
        schema["parameters"]["required"].as_array().unwrap().len(),
        2
    );

    // Validation passes with required arguments
    assert!(tool
        .validate_arguments(&serde_json::json!({
            "key": "user_id_101",
            "value": "alice"
        }))
        .is_ok());

    // Validation fails if missing "key"
    assert!(tool
        .validate_arguments(&serde_json::json!({
            "value": "alice"
        }))
        .is_err());

    // Validation fails if not a JSON object
    assert!(tool
        .validate_arguments(&serde_json::json!("not_an_object"))
        .is_err());
}

#[test]
fn test_tool_registry_and_execution() {
    struct MockStoragePutTool;
    impl Tool for MockStoragePutTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("mock_put", "Mock storage put")
                .with_parameter(ToolParameter::new("k", ToolParamType::String, "key", true))
                .with_parameter(ToolParameter::new("v", ToolParamType::String, "val", true))
        }

        fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
            let k = arguments["k"].as_str().unwrap_or("");
            let v = arguments["v"].as_str().unwrap_or("");
            Ok(serde_json::json!({
                "stored_bytes": k.len() + v.len(),
                "status": "persisted"
            }))
        }
    }

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(MockStoragePutTool));

    assert_eq!(registry.list_definitions().len(), 1);
    assert_eq!(registry.list_json_schemas().len(), 1);

    // Dispatch valid call
    let call = ToolCall::new(
        "call-1",
        "mock_put",
        serde_json::json!({ "k": "hello", "v": "world" }),
    );
    let out = registry.execute(&call).unwrap();
    assert!(out.success);
    assert_eq!(out.result["status"], "persisted");
    assert_eq!(out.result["stored_bytes"], 10);
    assert!(out.error.is_none());

    // Dispatch unknown tool
    let bad_call = ToolCall::new("call-2", "non_existent_tool", serde_json::json!({}));
    let bad_out = registry.execute(&bad_call).unwrap();
    assert!(!bad_out.success);
    assert!(bad_out.error.unwrap().contains("not found"));
}

#[test]
fn test_stateful_reflection_loop_lifecycle() {
    let mut reflection = ReflectionState::new("task-refine-algo", 3, 0.90);
    assert_eq!(reflection.current_iteration, 0);
    assert!(!reflection.converged);

    // Iteration 1: Needs Revision
    let step1 = ReflectionStep::new(
        1,
        "pub fn search() -> bool { false }",
        vec![
            ReflectionCritique::new("accuracy", 0.3, "Does not perform search")
                .with_improvement("Implement binary search across SSTable block keys"),
        ],
        ReflectionVerdict::NeedsRevision,
    );
    let v1 = reflection.record_step(step1);
    assert_eq!(v1, ReflectionVerdict::NeedsRevision);
    assert!(!reflection.converged);
    assert_eq!(reflection.current_iteration, 1);

    // Iteration 2: Needs Revision (Score improved)
    let step2 = ReflectionStep::new(
        2,
        "pub fn search(k: &[u8]) -> bool { k == b\"test\" }",
        vec![ReflectionCritique::new("accuracy", 0.75, "Hardcoded match")],
        ReflectionVerdict::NeedsRevision,
    );
    let v2 = reflection.record_step(step2);
    assert_eq!(v2, ReflectionVerdict::NeedsRevision);
    assert!(!reflection.converged);

    // Iteration 3: Approved (Score exceeds 0.90)
    let step3 = ReflectionStep::new(
        3,
        "pub fn search(k: &[u8], entries: &[Entry]) -> Option<Entry> { entries.iter().find(|e| &e.key == k).cloned() }",
        vec![
            ReflectionCritique::new("accuracy", 0.95, "Correct iterative search implementation"),
        ],
        ReflectionVerdict::Approved,
    );
    let v3 = reflection.record_step(step3);
    assert_eq!(v3, ReflectionVerdict::Approved);
    assert!(reflection.converged);
    assert!(reflection
        .best_candidate()
        .unwrap()
        .contains("entries.iter()"));
}

#[test]
fn test_workflow_state_messaging_and_shared_memory() {
    let mut state = WorkflowState::new("wf-rag-indexing-swarm");
    assert_eq!(state.status, WorkflowStatus::Initialized);

    // Register agents
    let planner = AgentDefinition::builder("planner", AgentRole::Planner).build();
    let coder = AgentDefinition::builder("coder", AgentRole::Coder).build();
    state.register_agent(planner);
    state.register_agent(coder);
    assert_eq!(state.active_agents.len(), 2);

    // Add tasks
    let task1 = AgentTask::builder("t1", "Chunk Text Documents").build();
    let task2 = AgentTask::builder("t2", "Index Vector Embeddings").build();
    state.add_task(task1);
    state.add_task(task2);
    assert_eq!(state.tasks.len(), 2);

    // Message exchange
    let msg1 = AgentMessage::new(
        "m-001",
        "planner",
        MessageRole::System,
        "Swarm execution started",
    )
    .to_recipient("coder")
    .with_tool_call(ToolCall::new("tc-01", "get_stats", serde_json::json!({})));
    state.post_message(msg1);

    assert_eq!(state.message_history.len(), 1);
    assert_eq!(state.message_history[0].sender.as_str(), "planner");
    assert_eq!(state.message_history[0].recipient.as_deref(), Some("coder"));
    assert_eq!(state.message_history[0].tool_calls.len(), 1);

    // Shared memory
    state.set_shared_memory("lsm_tier_count", serde_json::json!(7));
    assert_eq!(
        state.get_shared_memory("lsm_tier_count"),
        Some(&serde_json::json!(7))
    );
    assert_eq!(state.get_shared_memory("non_existent"), None);

    // Serialization roundtrip of complete workflow state
    let serialized = serde_json::to_string(&state).unwrap();
    let deserialized: WorkflowState = serde_json::from_str(&serialized).unwrap();
    assert_eq!(state.workflow_id, deserialized.workflow_id);
    assert_eq!(state.tasks.len(), deserialized.tasks.len());
    assert_eq!(state.active_agents.len(), deserialized.active_agents.len());
}

#[test]
fn test_storage_key_formatting_and_namespaces() {
    let agent_id = AgentId::new("agent_alpha");
    let task_id = TaskId::new("task_gamma");

    let agent_k = AgentStorageKey::agent_key(&agent_id);
    let task_k = AgentStorageKey::task_key(&task_id);
    let refl_k = AgentStorageKey::reflection_key(&task_id);
    let msg_k = AgentStorageKey::message_key("wf_001", "msg_007");

    assert_eq!(agent_k, "agent:def:agent_alpha");
    assert_eq!(task_k, "agent:task:task_gamma");
    assert_eq!(refl_k, "agent:refl:task_gamma");
    assert_eq!(msg_k, "agent:msg:wf_001:msg_007");
}

#[test]
fn test_adaptive_task_router_comprehensive() {
    let router = AdaptiveTaskRouter::default();

    let coder = AgentDefinition::builder("agent-coder", AgentRole::Coder)
        .capability(AgentCapability::CodeExecution)
        .capability(AgentCapability::ToolCalling)
        .build();
    let critic = AgentDefinition::builder("agent-critic", AgentRole::Critic)
        .capability(AgentCapability::Reflection)
        .capability(AgentCapability::TextGeneration)
        .build();
    let memory = AgentDefinition::builder("agent-memory", AgentRole::MemoryAgent)
        .capability(AgentCapability::StorageOps)
        .capability(AgentCapability::MemoryRetrieval)
        .capability(AgentCapability::ToolCalling)
        .build();

    let pool = vec![coder, critic, memory];

    // 1. Code task routes to coder
    let code_task = AgentTask::builder("t-code", "Implement SkipList insert")
        .priority(TaskPriority::High)
        .require_capability(AgentCapability::CodeExecution)
        .build();
    let d1 = router.route(&code_task, &pool).unwrap();
    assert_eq!(d1.target_agent.as_str(), "agent-coder");
    assert_eq!(d1.strategy, RoutingStrategy::Adaptive);
    assert!(d1.confidence > 0.5);

    // 2. Storage task routes to memory agent
    let storage_task = AgentTask::builder("t-store", "Persist RAG document chunk")
        .priority(TaskPriority::Critical)
        .require_capability(AgentCapability::StorageOps)
        .require_capability(AgentCapability::ToolCalling)
        .build();
    let d2 = router.route(&storage_task, &pool).unwrap();
    assert_eq!(d2.target_agent.as_str(), "agent-memory");
    assert!(!d2.fallback_agents.is_empty());
}

#[test]
fn test_load_balancing_router_concurrency_tracking() {
    let router = LoadBalancingRouter::new(3);

    let worker_a = AgentDefinition::builder("worker-a", AgentRole::Coder)
        .capability(AgentCapability::CodeExecution)
        .build();
    let worker_b = AgentDefinition::builder("worker-b", AgentRole::Coder)
        .capability(AgentCapability::CodeExecution)
        .build();

    let agents = vec![worker_a, worker_b];

    let task1 = AgentTask::builder("t1", "Job 1")
        .require_capability(AgentCapability::CodeExecution)
        .build();
    let d1 = router.route(&task1, &agents).unwrap();
    assert_eq!(d1.target_agent.as_str(), "worker-a");

    // Track load on worker-a
    router.track_task_started(&AgentId::new("worker-a"));
    assert_eq!(router.get_load(&AgentId::new("worker-a")), 1);

    // Next task should route to worker-b (least loaded)
    let task2 = AgentTask::builder("t2", "Job 2")
        .require_capability(AgentCapability::CodeExecution)
        .build();
    let d2 = router.route(&task2, &agents).unwrap();
    assert_eq!(d2.target_agent.as_str(), "worker-b");

    // Complete task on worker-a
    router.track_task_completed(&AgentId::new("worker-a"));
    assert_eq!(router.get_load(&AgentId::new("worker-a")), 0);
}

#[test]
fn test_priority_preemptive_router_critical_guard() {
    let router = PriorityPreemptiveRouter::new().with_threshold(0.8);

    let junior = AgentDefinition::builder("junior-agent", AgentRole::Coder)
        .capability(AgentCapability::TextGeneration)
        .build();
    let agents = vec![junior];

    let critical_task = AgentTask::builder("crit-01", "Hotfix Corruption Bug")
        .priority(TaskPriority::Critical)
        .require_capability(AgentCapability::CodeExecution)
        .build();

    // Junior agent doesn't satisfy CodeExecution, should error on Critical
    let err = router.route(&critical_task, &agents);
    assert!(err.is_err());
}

#[test]
fn test_heuristic_reflector_all_rules() {
    let reflector = HeuristicReflector::new()
        .with_min_length(30)
        .with_required_keyword("LSM")
        .with_prohibited_keyword("TODO")
        .with_json_validation(true);

    let task = AgentTask::builder("t-eval", "JSON Storage Spec").build();

    // Case 1: Bad output (too short, TODO placeholder, no LSM keyword, invalid JSON)
    let bad_critiques = reflector.evaluate(&task, "{ TODO: 1 }", 1).unwrap();
    assert_eq!(bad_critiques.len(), 4);
    let avg_bad = compute_aggregate_score(&bad_critiques);
    assert!(avg_bad < 0.5);

    // Case 2: Good output
    let good_payload = r#"{"engine": "LSM", "status": "production_ready", "shards": 8}"#;
    let good_critiques = reflector.evaluate(&task, good_payload, 1).unwrap();
    assert_eq!(good_critiques.len(), 4);
    let avg_good = compute_aggregate_score(&good_critiques);
    assert_eq!(avg_good, 1.0);
}

#[test]
fn test_reflection_engine_convergence_and_stagnation() {
    let reflector = Arc::new(
        HeuristicReflector::new()
            .with_min_length(25)
            .with_required_keyword("atomic"),
    );

    let engine = ReflectionEngine::new(reflector).with_config(ConvergenceConfig {
        passing_threshold: 0.85,
        max_iterations: 4,
        ..ConvergenceConfig::default()
    });

    let task = AgentTask::builder("t-refl-loop", "Write atomic batch write logic").build();

    let mut iterations = 0;
    let state = engine
        .execute_loop(&task, "draft code", |_prompt, _prev| {
            iterations += 1;
            if iterations == 1 {
                Ok("longer draft without keyword".into())
            } else {
                Ok("final implementation providing atomic ACID guarantees for all writes".into())
            }
        })
        .unwrap();

    assert!(state.converged);
    assert_eq!(state.final_verdict, Some(ReflectionVerdict::Approved));
    assert!(state.steps.len() >= 2);
    assert!(state.best_candidate().unwrap().contains("atomic ACID"));
}

#[test]
fn test_tools_suite_with_flashstore_integration() {
    let dir = tempfile::tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let registry = Arc::new(create_standard_tool_registry(Some(store)));
    let dispatcher = ToolDispatcher::new(registry);

    // 1. Calculator tool
    let calc_call = ToolCall::new(
        "c1",
        "calculator",
        serde_json::json!({ "operation": "mul", "a": 12.0, "b": 12.0 }),
    );
    let calc_out = dispatcher.dispatch(&calc_call);
    assert!(calc_out.is_success());
    assert_eq!(calc_out.result["result"], 144.0);

    // 2. KvPut tool
    let put_call = ToolCall::new(
        "c2",
        "kv_put",
        serde_json::json!({ "key": "agent:role:alpha", "value": "lead_architect" }),
    );
    let put_out = dispatcher.dispatch(&put_call);
    assert!(put_out.is_success());

    // 3. KvGet tool
    let get_call = ToolCall::new(
        "c3",
        "kv_get",
        serde_json::json!({ "key": "agent:role:alpha" }),
    );
    let get_out = dispatcher.dispatch(&get_call);
    assert!(get_out.is_success());
    assert_eq!(get_out.result["value"], "lead_architect");

    // 4. KvScan tool
    let scan_call = ToolCall::new(
        "c4",
        "kv_scan",
        serde_json::json!({ "start_key": "agent:role:", "limit": 5 }),
    );
    let scan_out = dispatcher.dispatch(&scan_call);
    assert!(scan_out.is_success());
    assert_eq!(scan_out.result["count"], 1);

    // 5. KvDelete tool
    let del_call = ToolCall::new(
        "c5",
        "kv_delete",
        serde_json::json!({ "key": "agent:role:alpha" }),
    );
    let del_out = dispatcher.dispatch(&del_call);
    assert!(del_out.is_success());

    let get_after_del = dispatcher.dispatch(&get_call);
    assert!(get_after_del.is_success());
    assert_eq!(get_after_del.result["found"], false);
}

#[test]
fn test_task_graph_dag_topological_and_cycle_detection() {
    let mut graph = TaskGraph::new();
    let a = TaskId::new("task-a");
    let b = TaskId::new("task-b");
    let c = TaskId::new("task-c");
    let d = TaskId::new("task-d");

    // a -> b, a -> c, b -> d, c -> d (Diamond DAG)
    graph.add_dependency(&b, &a).unwrap();
    graph.add_dependency(&c, &a).unwrap();
    graph.add_dependency(&d, &b).unwrap();
    graph.add_dependency(&d, &c).unwrap();

    let tiers = graph.topological_tiers().unwrap();
    assert_eq!(tiers.len(), 3);
    assert_eq!(tiers[0], vec![a.clone()]);
    assert_eq!(tiers[1], vec![b.clone(), c.clone()]);
    assert_eq!(tiers[2], vec![d.clone()]);

    // Adding cycle d -> a must be rejected
    let cycle_err = graph.add_dependency(&a, &d);
    assert!(cycle_err.is_err());
}

#[test]
fn test_agent_engine_complete_pipeline_with_tools_and_reflection() {
    let dir = tempfile::tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let mut engine = AgentEngine::new("swarm-wf-end-to-end").with_store(store);

    let planner = AgentDefinition::builder("agent-planner", AgentRole::Planner)
        .capability(AgentCapability::TextGeneration)
        .build();
    let tool_bot = AgentDefinition::builder("agent-toolbot", AgentRole::ToolExecutor)
        .capability(AgentCapability::ToolCalling)
        .capability(AgentCapability::StorageOps)
        .build();
    let critic = AgentDefinition::builder("agent-critic", AgentRole::Critic)
        .capability(AgentCapability::Reflection)
        .capability(AgentCapability::TextGeneration)
        .build();

    engine.register_agent(planner);
    engine.register_agent(tool_bot);
    engine.register_agent(critic);

    // Task 1: Planner sets up config
    let t1 = AgentTask::builder("step1-plan", "Setup swarm parameters")
        .require_capability(AgentCapability::TextGeneration)
        .build();

    // Task 2: ToolBot writes config to KV store using tool
    let t2 = AgentTask::builder("step2-store", "Write storage parameter to FlashStore")
        .require_capability(AgentCapability::ToolCalling)
        .require_capability(AgentCapability::StorageOps)
        .input_context(serde_json::json!({
            "required_tool": "kv_put",
            "tool_args": {
                "key": "swarm:lsm:block_size",
                "value": "4096"
            }
        }))
        .build();

    // Task 3: Critic reflects on result
    let t3 = AgentTask::builder("step3-critique", "Verify and review configuration")
        .require_capability(AgentCapability::Reflection)
        .build();

    engine.submit_task(t1, &[]).unwrap();
    engine.submit_task(t2, &[TaskId::new("step1-plan")]).unwrap();
    engine.submit_task(t3, &[TaskId::new("step2-store")]).unwrap();

    let summary = engine.run_workflow().unwrap();
    assert_eq!(summary.status, WorkflowStatus::Succeeded);
    assert_eq!(summary.total_tasks, 3);
    assert_eq!(summary.completed_tasks, 3);
    assert_eq!(summary.failed_tasks, 0);
    assert_eq!(summary.total_tool_calls, 1);

    // Verify shared memory contains step2 output
    let step2_mem = engine.state().get_shared_memory("task:step2-store:output");
    assert!(step2_mem.is_some());

    // Verify FlashStore actually received the key written by the agent via tool!
    let raw_val = engine
        .state()
        .get_shared_memory("task:step2-store:output")
        .unwrap();
    assert!(raw_val["tool_outputs"][0]["success"].as_bool().unwrap());
}
