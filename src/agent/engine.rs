//! # Multi-Agent Swarm Orchestrator and Workflow Execution Engine
//!
//! Provides the primary runtime execution logic for autonomous multi-agent workflows:
//! - Directed Acyclic Graph (DAG) task dependency resolution, cycle detection, and topological tiering.
//! - Context synthesis across inter-agent messages and shared blackboard memory.
//! - Autonomous agent step execution coordinating routing, tool execution, and reflection loops.
//! - End-to-end multi-agent workflow lifecycle orchestration (`WorkflowStatus` state machine).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::agent::reflection::{ConvergenceConfig, HeuristicReflector, ReflectionEngine};
use crate::agent::router::{AdaptiveRoutingWeights, AdaptiveTaskRouter};
use crate::agent::tools::create_standard_tool_registry;
use crate::agent::types::{
    AgentCapability, AgentDefinition, AgentId, AgentMessage, AgentTask, MessageRole,
    ReflectionVerdict, Reflector, RoutingDecision, TaskId, TaskRouter, TaskStatus, ToolCall,
    ToolRegistry, WorkflowState, WorkflowStatus,
};
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};

// ============================================================================
// DAG Task Dependency Graph
// ============================================================================

/// Directed Acyclic Graph tracking task dependency relationships.
#[derive(Debug, Clone, Default)]
pub struct TaskGraph {
    /// Mapping from task ID to the set of task IDs it directly depends on.
    dependencies: HashMap<TaskId, HashSet<TaskId>>,
    /// Reverse mapping: task ID to the set of task IDs that depend on it.
    dependents: HashMap<TaskId, HashSet<TaskId>>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
        }
    }

    /// Registers a task in the graph.
    pub fn add_task(&mut self, task_id: TaskId) {
        self.dependencies.entry(task_id.clone()).or_default();
        self.dependents.entry(task_id).or_default();
    }

    /// Adds a dependency: `task` cannot be executed until `depends_on` has completed.
    pub fn add_dependency(&mut self, task: &TaskId, depends_on: &TaskId) -> Result<()> {
        if task == depends_on {
            return Err(FlashStoreError::Agent(format!(
                "Task '{}' cannot depend on itself",
                task
            )));
        }

        self.add_task(task.clone());
        self.add_task(depends_on.clone());

        self.dependencies
            .get_mut(task)
            .unwrap()
            .insert(depends_on.clone());

        self.dependents
            .get_mut(depends_on)
            .unwrap()
            .insert(task.clone());

        // Validate cycle freedom immediately
        self.detect_cycles()?;
        Ok(())
    }

    /// Returns direct dependencies for a given task.
    pub fn get_dependencies(&self, task: &TaskId) -> Option<&HashSet<TaskId>> {
        self.dependencies.get(task)
    }

    /// Checks if all dependencies for a task have completed.
    pub fn is_ready(&self, task: &TaskId, completed_tasks: &HashSet<TaskId>) -> bool {
        if let Some(deps) = self.dependencies.get(task) {
            deps.iter().all(|dep| completed_tasks.contains(dep))
        } else {
            true
        }
    }

    /// Detects cycles using Kahn's topological sort algorithm.
    pub fn detect_cycles(&self) -> Result<()> {
        let mut in_degrees: HashMap<TaskId, usize> = HashMap::new();
        for (task, deps) in &self.dependencies {
            in_degrees.insert(task.clone(), deps.len());
        }

        let mut queue = VecDeque::new();
        for (task, &deg) in &in_degrees {
            if deg == 0 {
                queue.push_back(task.clone());
            }
        }

        let mut visited_count = 0;
        while let Some(task) = queue.pop_front() {
            visited_count += 1;
            if let Some(deps) = self.dependents.get(&task) {
                for dependent in deps {
                    if let Some(deg) = in_degrees.get_mut(dependent) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }

        if visited_count != self.dependencies.len() {
            Err(FlashStoreError::Agent(
                "Cycle detected in task dependency graph".into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Computes parallel execution tiers where tasks within each tier can execute concurrently.
    pub fn topological_tiers(&self) -> Result<Vec<Vec<TaskId>>> {
        self.detect_cycles()?;

        let mut in_degrees: HashMap<TaskId, usize> = HashMap::new();
        for (task, deps) in &self.dependencies {
            in_degrees.insert(task.clone(), deps.len());
        }

        let mut current_tier: Vec<TaskId> = in_degrees
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(task, _)| task.clone())
            .collect();

        current_tier.sort(); // Deterministic order

        let mut tiers = Vec::new();
        let mut remaining = self.dependencies.len();

        while !current_tier.is_empty() {
            remaining -= current_tier.len();
            let mut next_tier = Vec::new();

            for task in &current_tier {
                if let Some(deps) = self.dependents.get(task) {
                    for dependent in deps {
                        if let Some(deg) = in_degrees.get_mut(dependent) {
                            *deg -= 1;
                            if *deg == 0 {
                                next_tier.push(dependent.clone());
                            }
                        }
                    }
                }
            }

            tiers.push(current_tier);
            next_tier.sort();
            current_tier = next_tier;
        }

        if remaining > 0 {
            return Err(FlashStoreError::Agent(
                "Failed to partition tasks into topological tiers due to unresolved cycle".into(),
            ));
        }

        Ok(tiers)
    }
}

// ============================================================================
// Execution Reports & Summaries
// ============================================================================

/// Execution result for a single task pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionReport {
    pub task_id: TaskId,
    pub assigned_agent: AgentId,
    pub status: TaskStatus,
    pub routing_decision: RoutingDecision,
    pub tool_calls_executed: usize,
    pub reflection_iterations: usize,
    pub reflection_verdict: Option<ReflectionVerdict>,
    pub output: Option<serde_json::Value>,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// End-to-end workflow execution summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSummary {
    pub workflow_id: String,
    pub status: WorkflowStatus,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub total_tool_calls: usize,
    pub total_messages: usize,
    pub total_duration_ms: u64,
    pub reports: Vec<TaskExecutionReport>,
}

// ============================================================================
// Swarm Execution Engine
// ============================================================================

/// High-performance autonomous multi-agent swarm orchestrator.
pub struct AgentEngine {
    state: WorkflowState,
    graph: TaskGraph,
    router: Arc<dyn TaskRouter>,
    tools: Arc<ToolRegistry>,
    reflector: Arc<dyn Reflector>,
    flash_store: Option<Arc<FlashStore>>,
}

impl AgentEngine {
    /// Creates a new `AgentEngine` with default components.
    pub fn new(workflow_id: impl Into<String>) -> Self {
        let registry = Arc::new(create_standard_tool_registry(None));
        let reflector = Arc::new(HeuristicReflector::default());
        let router = Arc::new(AdaptiveTaskRouter::new(AdaptiveRoutingWeights::default(), 4));

        Self {
            state: WorkflowState::new(workflow_id),
            graph: TaskGraph::new(),
            router,
            tools: registry,
            reflector,
            flash_store: None,
        }
    }

    /// Attaches an optional persistent `FlashStore` backend.
    pub fn with_store(mut self, store: Arc<FlashStore>) -> Self {
        self.tools = Arc::new(create_standard_tool_registry(Some(store.clone())));
        self.flash_store = Some(store);
        self
    }

    /// Customizes the task router.
    pub fn with_router(mut self, router: Arc<dyn TaskRouter>) -> Self {
        self.router = router;
        self
    }

    /// Customizes the reflection critique engine.
    pub fn with_reflector(mut self, reflector: Arc<dyn Reflector>) -> Self {
        self.reflector = reflector;
        self
    }

    /// Customizes the tool registry.
    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    /// Returns a reference to the active workflow state.
    pub fn state(&self) -> &WorkflowState {
        &self.state
    }

    /// Returns a mutable reference to the active workflow state.
    pub fn state_mut(&mut self) -> &mut WorkflowState {
        &mut self.state
    }

    /// Registers an agent definition with the swarm.
    pub fn register_agent(&mut self, agent: AgentDefinition) {
        self.state.register_agent(agent);
    }

    /// Submits a task to the workflow, with optional dependencies on preceding tasks.
    pub fn submit_task(&mut self, task: AgentTask, depends_on: &[TaskId]) -> Result<()> {
        let task_id = task.id.clone();
        self.graph.add_task(task_id.clone());

        for dep in depends_on {
            self.graph.add_dependency(&task_id, dep)?;
        }

        self.state.add_task(task);
        Ok(())
    }

    /// Executes a single task step through routing, tool calling, and reflection.
    pub fn execute_task(&mut self, task_id: &TaskId) -> Result<TaskExecutionReport> {
        let start = Instant::now();

        // 1. Fetch task
        let mut task = self
            .state
            .tasks
            .get(task_id)
            .cloned()
            .ok_or_else(|| FlashStoreError::Agent(format!("Task '{}' not found", task_id)))?;

        // 2. Verify dependencies
        let completed: HashSet<TaskId> = self
            .state
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Completed)
            .map(|t| t.id.clone())
            .collect();

        if !self.graph.is_ready(task_id, &completed) {
            return Err(FlashStoreError::Agent(format!(
                "Task '{}' dependencies are not yet satisfied",
                task_id
            )));
        }

        // 3. Route task if unassigned
        let available_agents: Vec<AgentDefinition> =
            self.state.active_agents.values().cloned().collect();

        let routing_decision = self.router.route(&task, &available_agents)?;
        task.assign(routing_decision.target_agent.clone());
        task.start();

        let agent = self
            .state
            .active_agents
            .get(&routing_decision.target_agent)
            .cloned()
            .ok_or_else(|| {
                FlashStoreError::Agent(format!(
                    "Assigned agent '{}' missing from active registry",
                    routing_decision.target_agent
                ))
            })?;

        // Post routing message
        self.state.post_message(AgentMessage::new(
            format!("msg-route-{}", task.id),
            "router",
            MessageRole::System,
            format!(
                "Task '{}' routed to agent '{}': {}",
                task.id, agent.id, routing_decision.rationale
            ),
        ));

        // 4. Autonomous Agent Step & Tool Execution
        let mut tool_calls_executed = 0;
        let mut tool_outputs = Vec::new();

        // If task or agent requires tools, execute required tool operations
        if agent.has_capability(AgentCapability::ToolCalling) {
            if let Some(tool_name) = task.input_context.get("required_tool").and_then(|t| t.as_str()) {
                let args = task.input_context.get("tool_args").cloned().unwrap_or(json!({}));
                let call = ToolCall::new(format!("call-{}-1", task.id), tool_name, args);
                
                let output = self.tools.execute(&call)?;
                tool_outputs.push(output);
                tool_calls_executed += 1;
            }
        }

        // 5. Candidate Generation
        let initial_output = format!(
            "Execution completed by {} for task '{}'.\nDescription: {}\nTool outcomes: {:?}",
            agent.name, task.title, task.description, tool_outputs
        );

        // 6. Reflection Loop Execution
        let mut reflection_verdict = None;
        let mut reflection_iterations = 0;
        let mut final_text = initial_output;

        if agent.has_capability(AgentCapability::Reflection)
            || task.required_capabilities.contains(&AgentCapability::Reflection)
        {
            task.reflect();
            let refl_engine = ReflectionEngine::new(self.reflector.clone())
                .with_config(ConvergenceConfig {
                    max_iterations: agent.max_iterations.min(5),
                    ..ConvergenceConfig::default()
                });

            let refl_state = refl_engine.execute_loop(
                &task,
                &final_text,
                |_prompt, prev| Ok(format!("{}\nVerified and refined by {}.", prev, agent.id)),
            )?;

            reflection_iterations = refl_state.steps.len();
            reflection_verdict = refl_state.latest_step().map(|s| s.verdict);
            if let Some(best) = refl_state.best_candidate() {
                final_text = best.to_string();
            }

            self.state
                .reflection_states
                .insert(task.id.clone(), refl_state);
        }

        // 7. Complete Task
        let output_artifacts = json!({
            "agent": agent.id.as_str(),
            "result_text": final_text,
            "tool_outputs": tool_outputs,
        });

        task.complete(output_artifacts.clone());

        // Update shared memory
        self.state.set_shared_memory(
            format!("task:{}:output", task.id),
            output_artifacts.clone(),
        );

        // Record assistant response message
        self.state.post_message(AgentMessage::new(
            format!("msg-done-{}", task.id),
            agent.id.clone(),
            MessageRole::Assistant,
            final_text,
        ));

        // Save updated task back into state
        self.state.tasks.insert(task.id.clone(), task.clone());

        let duration = start.elapsed().as_millis() as u64;

        Ok(TaskExecutionReport {
            task_id: task.id,
            assigned_agent: agent.id,
            status: task.status,
            routing_decision,
            tool_calls_executed,
            reflection_iterations,
            reflection_verdict,
            output: Some(output_artifacts),
            execution_time_ms: duration,
            error: None,
        })
    }

    /// Runs all tasks in the workflow following DAG dependency ordering.
    pub fn run_workflow(&mut self) -> Result<WorkflowSummary> {
        let start = Instant::now();
        self.state.status = WorkflowStatus::Running;

        let tiers = self.graph.topological_tiers()?;
        let mut reports = Vec::new();
        let mut completed_count = 0;
        let mut failed_count = 0;
        let mut total_tools = 0;

        for tier in tiers {
            for task_id in tier {
                match self.execute_task(&task_id) {
                    Ok(report) => {
                        total_tools += report.tool_calls_executed;
                        if report.status == TaskStatus::Completed {
                            completed_count += 1;
                        } else {
                            failed_count += 1;
                        }
                        reports.push(report);
                    }
                    Err(e) => {
                        self.state.status = WorkflowStatus::Failed;
                        if let Some(t) = self.state.tasks.get_mut(&task_id) {
                            t.fail(e.to_string());
                        }
                        return Err(e);
                    }
                }
            }
        }

        self.state.status = if failed_count == 0 {
            WorkflowStatus::Succeeded
        } else {
            WorkflowStatus::Failed
        };

        let duration = start.elapsed().as_millis() as u64;

        Ok(WorkflowSummary {
            workflow_id: self.state.workflow_id.clone(),
            status: self.state.status,
            total_tasks: self.state.tasks.len(),
            completed_tasks: completed_count,
            failed_tasks: failed_count,
            total_tool_calls: total_tools,
            total_messages: self.state.message_history.len(),
            total_duration_ms: duration,
            reports,
        })
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::AgentRole;

    #[test]
    fn test_task_graph_dependency_and_tiers() {
        let mut graph = TaskGraph::new();
        let t1 = TaskId::new("t1");
        let t2 = TaskId::new("t2");
        let t3 = TaskId::new("t3");
        let t4 = TaskId::new("t4");

        // t1 has no deps
        // t2 depends on t1
        // t3 depends on t1
        // t4 depends on t2 and t3
        graph.add_dependency(&t2, &t1).unwrap();
        graph.add_dependency(&t3, &t1).unwrap();
        graph.add_dependency(&t4, &t2).unwrap();
        graph.add_dependency(&t4, &t3).unwrap();

        let tiers = graph.topological_tiers().unwrap();
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0], vec![t1.clone()]);
        assert_eq!(tiers[1], vec![t2.clone(), t3.clone()]);
        assert_eq!(tiers[2], vec![t4.clone()]);

        let mut completed = HashSet::new();
        assert!(graph.is_ready(&t1, &completed));
        assert!(!graph.is_ready(&t2, &completed));

        completed.insert(t1.clone());
        assert!(graph.is_ready(&t2, &completed));
        assert!(graph.is_ready(&t3, &completed));
        assert!(!graph.is_ready(&t4, &completed));

        completed.insert(t2.clone());
        completed.insert(t3.clone());
        assert!(graph.is_ready(&t4, &completed));
    }

    #[test]
    fn test_task_graph_cycle_rejection() {
        let mut graph = TaskGraph::new();
        let t1 = TaskId::new("t1");
        let t2 = TaskId::new("t2");

        graph.add_dependency(&t1, &t2).unwrap();
        let res = graph.add_dependency(&t2, &t1);
        assert!(res.is_err());
    }

    #[test]
    fn test_agent_engine_end_to_end_workflow() {
        let mut engine = AgentEngine::new("wf-demo-01");

        // Register agents
        let planner = AgentDefinition::builder("planner-01", AgentRole::Planner)
            .capability(AgentCapability::TextGeneration)
            .build();
        let coder = AgentDefinition::builder("coder-01", AgentRole::Coder)
            .capability(AgentCapability::CodeExecution)
            .capability(AgentCapability::ToolCalling)
            .capability(AgentCapability::Reflection)
            .build();

        engine.register_agent(planner);
        engine.register_agent(coder);

        // Submit tasks with dependencies: task 1 -> task 2
        let task1 = AgentTask::builder("task-plan", "Create Architecture Plan")
            .require_capability(AgentCapability::TextGeneration)
            .build();
        let task2 = AgentTask::builder("task-calc", "Calculate Concurrency Target")
            .require_capability(AgentCapability::CodeExecution)
            .require_capability(AgentCapability::ToolCalling)
            .require_capability(AgentCapability::Reflection)
            .input_context(json!({
                "required_tool": "calculator",
                "tool_args": { "operation": "mul", "a": 16.0, "b": 1024.0 }
            }))
            .build();

        engine.submit_task(task1, &[]).unwrap();
        engine.submit_task(task2, &[TaskId::new("task-plan")]).unwrap();

        let summary = engine.run_workflow().unwrap();
        assert_eq!(summary.status, WorkflowStatus::Succeeded);
        assert_eq!(summary.total_tasks, 2);
        assert_eq!(summary.completed_tasks, 2);
        assert_eq!(summary.failed_tasks, 0);
        assert_eq!(summary.total_tool_calls, 1);
        assert!(summary.total_messages >= 4);

        // Verify shared memory populated
        let out = engine.state().get_shared_memory("task:task-calc:output");
        assert!(out.is_some());
    }
}
