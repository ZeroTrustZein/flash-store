//! # Multi-Agent Lifecycle Events and Integration Handlers
//!
//! Provides an event-driven architecture and lifecycle hooks for monitoring,
//! auditing, metrics recording, and real-time streaming of multi-agent execution events.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::agent::metrics::AgentMetricsTracker;
use crate::agent::types::{AgentId, ReflectionVerdict, TaskId, TaskPriority};

/// Lifecycle events emitted during multi-agent workflow execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    WorkflowStarted {
        workflow_id: String,
        task_count: usize,
        timestamp: u64,
    },
    TaskSubmitted {
        task_id: TaskId,
        priority: TaskPriority,
        timestamp: u64,
    },
    TaskRouted {
        task_id: TaskId,
        agent_id: AgentId,
        score: f32,
        preempted: bool,
        timestamp: u64,
    },
    TaskStarted {
        task_id: TaskId,
        agent_id: AgentId,
        timestamp: u64,
    },
    ToolCallStarted {
        task_id: TaskId,
        agent_id: AgentId,
        tool_name: String,
        timestamp: u64,
    },
    ToolCallCompleted {
        task_id: TaskId,
        agent_id: AgentId,
        tool_name: String,
        success: bool,
        duration_ms: u64,
        timestamp: u64,
    },
    ReflectionEvaluated {
        task_id: TaskId,
        iteration: usize,
        verdict: ReflectionVerdict,
        score: f32,
        timestamp: u64,
    },
    TaskCompleted {
        task_id: TaskId,
        agent_id: AgentId,
        duration_ms: u64,
        timestamp: u64,
    },
    TaskFailed {
        task_id: TaskId,
        error: String,
        duration_ms: u64,
        timestamp: u64,
    },
    WorkflowCompleted {
        workflow_id: String,
        completed_tasks: usize,
        failed_tasks: usize,
        duration_ms: u64,
        timestamp: u64,
    },
}

impl AgentEvent {
    /// Returns the Unix timestamp of the event in seconds.
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::WorkflowStarted { timestamp, .. } => *timestamp,
            Self::TaskSubmitted { timestamp, .. } => *timestamp,
            Self::TaskRouted { timestamp, .. } => *timestamp,
            Self::TaskStarted { timestamp, .. } => *timestamp,
            Self::ToolCallStarted { timestamp, .. } => *timestamp,
            Self::ToolCallCompleted { timestamp, .. } => *timestamp,
            Self::ReflectionEvaluated { timestamp, .. } => *timestamp,
            Self::TaskCompleted { timestamp, .. } => *timestamp,
            Self::TaskFailed { timestamp, .. } => *timestamp,
            Self::WorkflowCompleted { timestamp, .. } => *timestamp,
        }
    }

    /// Helper for current epoch seconds.
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Trait for handling multi-agent lifecycle events.
pub trait AgentEventHandler: Send + Sync {
    /// Dispatches an event to the handler.
    fn handle(&self, event: &AgentEvent);
}

/// Event handler that collects all received events in a thread-safe vector.
#[derive(Debug, Default)]
pub struct CollectingEventHandler {
    events: RwLock<Vec<AgentEvent>>,
    limit: usize,
}

impl CollectingEventHandler {
    pub fn new(limit: usize) -> Self {
        Self {
            events: RwLock::new(Vec::new()),
            limit: limit.max(10),
        }
    }

    pub fn events(&self) -> Vec<AgentEvent> {
        self.events.read().clone()
    }

    pub fn count(&self) -> usize {
        self.events.read().len()
    }

    pub fn clear(&self) {
        self.events.write().clear();
    }
}

impl AgentEventHandler for CollectingEventHandler {
    fn handle(&self, event: &AgentEvent) {
        let mut events = self.events.write();
        if events.len() >= self.limit {
            events.remove(0);
        }
        events.push(event.clone());
    }
}

/// Event handler that forwards lifecycle events directly into an `AgentMetricsTracker`.
pub struct MetricsEventHandler {
    metrics: Arc<AgentMetricsTracker>,
}

impl MetricsEventHandler {
    pub fn new(metrics: Arc<AgentMetricsTracker>) -> Self {
        Self { metrics }
    }
}

impl AgentEventHandler for MetricsEventHandler {
    fn handle(&self, event: &AgentEvent) {
        match event {
            AgentEvent::TaskSubmitted { .. } => {
                self.metrics.record_task_submitted();
            }
            AgentEvent::TaskRouted { preempted, .. } => {
                if *preempted {
                    self.metrics.record_task_preempted();
                }
            }
            AgentEvent::ToolCallCompleted {
                tool_name,
                success,
                duration_ms,
                ..
            } => {
                self.metrics.record_tool_call(
                    tool_name,
                    *success,
                    std::time::Duration::from_millis(*duration_ms),
                );
            }
            AgentEvent::ReflectionEvaluated {
                iteration, verdict, ..
            } => {
                self.metrics.record_reflection_pass(*iteration, *verdict);
            }
            AgentEvent::TaskCompleted { duration_ms, .. } => {
                self.metrics
                    .record_task_completed(std::time::Duration::from_millis(*duration_ms));
            }
            AgentEvent::TaskFailed { duration_ms, .. } => {
                self.metrics
                    .record_task_failed(std::time::Duration::from_millis(*duration_ms));
            }
            _ => {}
        }
    }
}

/// Event handler backed by a closure callback.
pub struct CallbackEventHandler<F>
where
    F: Fn(&AgentEvent) + Send + Sync,
{
    callback: F,
}

impl<F> CallbackEventHandler<F>
where
    F: Fn(&AgentEvent) + Send + Sync,
{
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> AgentEventHandler for CallbackEventHandler<F>
where
    F: Fn(&AgentEvent) + Send + Sync,
{
    fn handle(&self, event: &AgentEvent) {
        (self.callback)(event);
    }
}

/// Multi-caster broadcaster distributing lifecycle events to registered handlers.
pub struct EventBroadcaster {
    handlers: RwLock<Vec<Arc<dyn AgentEventHandler>>>,
    dispatched_count: AtomicUsize,
}

impl Default for EventBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBroadcaster {
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(Vec::new()),
            dispatched_count: AtomicUsize::new(0),
        }
    }

    /// Registers a new event handler.
    pub fn add_handler(&self, handler: Arc<dyn AgentEventHandler>) {
        self.handlers.write().push(handler);
    }

    /// Broadcasts an event to all subscribed handlers.
    pub fn broadcast(&self, event: &AgentEvent) {
        let handlers = self.handlers.read();
        for h in handlers.iter() {
            h.handle(event);
        }
        self.dispatched_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of events dispatched.
    pub fn dispatched_count(&self) -> usize {
        self.dispatched_count.load(Ordering::Relaxed)
    }

    /// Returns the number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.read().len()
    }
}
