//! # High-Level Autonomous Swarm Pipeline
//!
//! Provides the primary integration coordinator orchestrating multi-agent workflow
//! lifecycles, episodic memory, telemetry metrics, and FlashStore persistence.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::agent::engine::{AgentEngine, TaskExecutionReport, WorkflowSummary};
use crate::agent::events::{AgentEvent, AgentEventHandler, EventBroadcaster, MetricsEventHandler};
use crate::agent::memory::{AgentMemoryStore, EpisodicMemoryEntry};
use crate::agent::metrics::{AgentMetricsSnapshot, AgentMetricsTracker};
use crate::agent::store::AgentStoreAdapter;
use crate::agent::types::{AgentDefinition, AgentTask, TaskId, TaskStatus};
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};

/// Configuration parameters for the autonomous multi-agent pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPipelineConfig {
    /// Whether to automatically create an atomic FlashStore checkpoint upon workflow completion.
    pub auto_checkpoint: bool,
    /// Whether to record completed task solutions into episodic memory.
    pub record_episodes_on_completion: bool,
    /// Whether to synchronize memory and blackboard state to FlashStore on completion.
    pub sync_memory_on_finish: bool,
    /// Namespace identifier for memory and checkpoint keys.
    pub namespace: String,
}

impl Default for AgentPipelineConfig {
    fn default() -> Self {
        Self {
            auto_checkpoint: true,
            record_episodes_on_completion: true,
            sync_memory_on_finish: true,
            namespace: "default".into(),
        }
    }
}

/// Primary orchestrator and integration coordinator for autonomous multi-agent workflows.
pub struct AgentPipeline {
    engine: AgentEngine,
    store: Option<AgentStoreAdapter>,
    memory: Arc<AgentMemoryStore>,
    metrics: Arc<AgentMetricsTracker>,
    broadcaster: Arc<EventBroadcaster>,
    config: AgentPipelineConfig,
}

impl AgentPipeline {
    /// Creates a new `AgentPipeline` with a specific workflow ID.
    pub fn new(workflow_id: impl Into<String>) -> Self {
        let wf_id = workflow_id.into();
        let metrics = Arc::new(AgentMetricsTracker::new());
        let broadcaster = Arc::new(EventBroadcaster::new());

        // Connect metrics event handler by default
        broadcaster.add_handler(Arc::new(MetricsEventHandler::new(metrics.clone())));

        Self {
            engine: AgentEngine::new(&wf_id),
            store: None,
            memory: Arc::new(AgentMemoryStore::default()),
            metrics,
            broadcaster,
            config: AgentPipelineConfig {
                namespace: wf_id,
                ..Default::default()
            },
        }
    }

    /// Attaches persistent `FlashStore` storage and configures `AgentStoreAdapter`.
    pub fn with_store(mut self, store: Arc<FlashStore>) -> Self {
        self.engine = self.engine.with_store(store.clone());
        self.store = Some(AgentStoreAdapter::new(store));
        self
    }

    /// Attaches an existing `AgentStoreAdapter`.
    pub fn with_store_adapter(mut self, adapter: AgentStoreAdapter) -> Self {
        self.engine = self.engine.with_store(adapter.store().clone());
        self.store = Some(adapter);
        self
    }

    /// Attaches shared episodic memory.
    pub fn with_memory(mut self, memory: Arc<AgentMemoryStore>) -> Self {
        self.memory = memory;
        self
    }

    /// Attaches a custom metrics tracker.
    pub fn with_metrics(mut self, metrics: Arc<AgentMetricsTracker>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Customizes pipeline operational configuration.
    pub fn with_config(mut self, config: AgentPipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Registers a custom event handler on the pipeline broadcaster.
    pub fn add_event_handler(&self, handler: Arc<dyn AgentEventHandler>) {
        self.broadcaster.add_handler(handler);
    }

    // ========================================================================
    // Registration & Task Submission
    // ========================================================================

    /// Registers an agent definition with the pipeline and underlying engine.
    pub fn register_agent(&mut self, agent: AgentDefinition) -> Result<()> {
        if let Some(ref store) = self.store {
            store.persist_agent(&agent)?;
        }
        self.engine.register_agent(agent);
        Ok(())
    }

    /// Registers multiple agent definitions at once.
    pub fn register_agents(&mut self, agents: &[AgentDefinition]) -> Result<()> {
        for agent in agents {
            self.register_agent(agent.clone())?;
        }
        Ok(())
    }

    /// Submits a task to the pipeline with optional dependency constraints.
    pub fn submit_task(&mut self, task: AgentTask, depends_on: &[TaskId]) -> Result<()> {
        let event = AgentEvent::TaskSubmitted {
            task_id: task.id.clone(),
            priority: task.priority,
            timestamp: AgentEvent::current_timestamp(),
        };
        self.broadcaster.broadcast(&event);

        if let Some(ref store) = self.store {
            store.persist_task(&task)?;
        }

        self.engine.submit_task(task, depends_on)
    }

    // ========================================================================
    // Workflow Execution
    // ========================================================================

    /// Runs all submitted workflow tasks to completion following DAG order.
    pub fn run(&mut self) -> Result<WorkflowSummary> {
        let start = Instant::now();
        let total_tasks = self.engine.state().tasks.len();

        self.broadcaster.broadcast(&AgentEvent::WorkflowStarted {
            workflow_id: self.engine.state().workflow_id.clone(),
            task_count: total_tasks,
            timestamp: AgentEvent::current_timestamp(),
        });

        // Execute workflow in engine
        let summary = match self.engine.run_workflow() {
            Ok(s) => s,
            Err(e) => {
                let dur = start.elapsed().as_millis() as u64;
                self.broadcaster
                    .broadcast(&AgentEvent::WorkflowCompleted {
                        workflow_id: self.engine.state().workflow_id.clone(),
                        completed_tasks: 0,
                        failed_tasks: total_tasks,
                        duration_ms: dur,
                        timestamp: AgentEvent::current_timestamp(),
                    });
                return Err(e);
            }
        };

        // Post-execution processing: events, memory recording, and persistence
        for report in &summary.reports {
            if report.status == TaskStatus::Completed {
                self.broadcaster.broadcast(&AgentEvent::TaskCompleted {
                    task_id: report.task_id.clone(),
                    agent_id: report.assigned_agent.clone(),
                    duration_ms: report.execution_time_ms,
                    timestamp: AgentEvent::current_timestamp(),
                });

                if self.config.record_episodes_on_completion {
                    let text = report
                        .output
                        .as_ref()
                        .and_then(|o| o.get("result_text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let entry = EpisodicMemoryEntry::new(
                        report.assigned_agent.clone(),
                        format!("Task {}: {}", report.task_id, text),
                        0.8,
                    )
                    .with_task_id(report.task_id.clone())
                    .with_tags(vec!["task_output".into(), report.assigned_agent.to_string()]);

                    let _ = self.memory.record_episode(entry);
                }
            } else {
                self.broadcaster.broadcast(&AgentEvent::TaskFailed {
                    task_id: report.task_id.clone(),
                    error: report
                        .error
                        .clone()
                        .unwrap_or_else(|| "Unknown failure".into()),
                    duration_ms: report.execution_time_ms,
                    timestamp: AgentEvent::current_timestamp(),
                });
            }
        }

        let dur = start.elapsed().as_millis() as u64;
        self.broadcaster.broadcast(&AgentEvent::WorkflowCompleted {
            workflow_id: summary.workflow_id.clone(),
            completed_tasks: summary.completed_tasks,
            failed_tasks: summary.failed_tasks,
            duration_ms: dur,
            timestamp: AgentEvent::current_timestamp(),
        });

        // FlashStore Checkpointing
        if let Some(ref store) = self.store {
            if self.config.auto_checkpoint {
                store.checkpoint_full_workflow(self.engine.state())?;
            }
            if self.config.sync_memory_on_finish {
                self.memory
                    .persist_to_store(store, &self.config.namespace)?;
            }
        }

        Ok(summary)
    }

    /// Submits and immediately executes a single task.
    pub fn run_single_task(&mut self, task: AgentTask) -> Result<TaskExecutionReport> {
        let task_id = task.id.clone();
        self.submit_task(task, &[])?;
        let report = self.engine.execute_task(&task_id)?;

        if report.status == TaskStatus::Completed {
            self.broadcaster.broadcast(&AgentEvent::TaskCompleted {
                task_id: report.task_id.clone(),
                agent_id: report.assigned_agent.clone(),
                duration_ms: report.execution_time_ms,
                timestamp: AgentEvent::current_timestamp(),
            });

            if self.config.record_episodes_on_completion {
                let text = report
                    .output
                    .as_ref()
                    .and_then(|o| o.get("result_text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                let entry = EpisodicMemoryEntry::new(
                    report.assigned_agent.clone(),
                    format!("Single task {}: {}", report.task_id, text),
                    0.8,
                )
                .with_task_id(report.task_id.clone());
                let _ = self.memory.record_episode(entry);
            }
        }

        Ok(report)
    }

    /// Checkpoints the current workflow and memory state to FlashStore.
    pub fn checkpoint(&self) -> Result<()> {
        let store = self.store.as_ref().ok_or_else(|| {
            FlashStoreError::Agent("Cannot checkpoint pipeline without persistent store".into())
        })?;

        store.checkpoint_full_workflow(self.engine.state())?;
        self.memory
            .persist_to_store(store, &self.config.namespace)?;
        Ok(())
    }

    /// Restores a pipeline from an existing checkpoint in FlashStore.
    pub fn restore_from_store(store: Arc<FlashStore>, workflow_id: &str) -> Result<Self> {
        let adapter = AgentStoreAdapter::new(store.clone());
        let wf_state = adapter
            .load_workflow_state(workflow_id)?
            .ok_or_else(|| {
                FlashStoreError::Agent(format!("Workflow checkpoint '{}' not found", workflow_id))
            })?;

        let mut pipeline = AgentPipeline::new(workflow_id)
            .with_store(store)
            .with_store_adapter(adapter.clone());

        // Restore tasks into engine
        for task in wf_state.tasks.values() {
            pipeline.engine.submit_task(task.clone(), &[])?;
        }

        // Restore active agents
        for agent in wf_state.active_agents.values() {
            pipeline.engine.register_agent(agent.clone());
        }

        // Restore memory
        pipeline
            .memory
            .load_from_store(&adapter, &pipeline.config.namespace)?;

        Ok(pipeline)
    }

    // ========================================================================
    // Subsystem Accessors
    // ========================================================================

    pub fn engine(&self) -> &AgentEngine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut AgentEngine {
        &mut self.engine
    }

    pub fn memory(&self) -> &Arc<AgentMemoryStore> {
        &self.memory
    }

    pub fn metrics(&self) -> &Arc<AgentMetricsTracker> {
        &self.metrics
    }

    pub fn metrics_snapshot(&self) -> AgentMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn store(&self) -> Option<&AgentStoreAdapter> {
        self.store.as_ref()
    }

    pub fn broadcaster(&self) -> &Arc<EventBroadcaster> {
        &self.broadcaster
    }
}
