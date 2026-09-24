//! # Multi-Agent Telemetry, Metrics, and Observability Subsystem
//!
//! Provides thread-safe metrics collection and performance snapshots across
//! task routing, tool execution, reflection convergence, and memory operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::agent::types::{ReflectionVerdict, RoutingDecision};

/// Snapshot of stage latency metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: u64,
    pub total_ms: u64,
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: f64,
}

impl Default for LatencySummary {
    fn default() -> Self {
        Self {
            count: 0,
            total_ms: 0,
            min_ms: 0,
            max_ms: 0,
            avg_ms: 0.0,
        }
    }
}

/// Thread-safe duration and count tracker.
#[derive(Debug)]
pub struct DurationTracker {
    count: AtomicU64,
    total_micros: AtomicU64,
    min_micros: AtomicU64,
    max_micros: AtomicU64,
}

impl Default for DurationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DurationTracker {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_micros: AtomicU64::new(0),
            min_micros: AtomicU64::new(u64::MAX),
            max_micros: AtomicU64::new(0),
        }
    }

    pub fn record(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_micros.fetch_add(micros, Ordering::Relaxed);

        let mut curr_min = self.min_micros.load(Ordering::Relaxed);
        while micros < curr_min {
            match self.min_micros.compare_exchange_weak(
                curr_min,
                micros,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => curr_min = actual,
            }
        }

        let mut curr_max = self.max_micros.load(Ordering::Relaxed);
        while micros > curr_max {
            match self.max_micros.compare_exchange_weak(
                curr_max,
                micros,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => curr_max = actual,
            }
        }
    }

    pub fn snapshot(&self) -> LatencySummary {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return LatencySummary::default();
        }

        let total_us = self.total_micros.load(Ordering::Relaxed);
        let min_us = self.min_micros.load(Ordering::Relaxed);
        let max_us = self.max_micros.load(Ordering::Relaxed);

        let total_ms = total_us / 1000;
        let min_ms = if min_us == u64::MAX { 0 } else { min_us / 1000 };
        let max_ms = max_us / 1000;
        let avg_ms = (total_us as f64) / (count as f64 * 1000.0);

        LatencySummary {
            count,
            total_ms,
            min_ms,
            max_ms,
            avg_ms,
        }
    }

    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_micros.store(0, Ordering::Relaxed);
        self.min_micros.store(u64::MAX, Ordering::Relaxed);
        self.max_micros.store(0, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of multi-agent swarm operational telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetricsSnapshot {
    // Tasks
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_preempted: u64,
    pub task_success_rate: f64,
    pub task_execution_latency: LatencySummary,

    // Routing
    pub total_routing_decisions: u64,
    pub preemptive_routings: u64,
    pub fallback_routings: u64,
    pub avg_routing_confidence: f64,

    // Tools
    pub total_tool_calls: u64,
    pub successful_tool_calls: u64,
    pub failed_tool_calls: u64,
    pub tool_success_rate: f64,
    pub tool_execution_latency: LatencySummary,
    pub tool_usage_by_name: HashMap<String, u64>,

    // Reflection
    pub total_reflection_passes: u64,
    pub approved_reflections: u64,
    pub revision_reflections: u64,
    pub rejected_reflections: u64,
    pub escalated_reflections: u64,
    pub reflection_approval_rate: f64,
    pub avg_reflection_iterations: f64,

    // Memory
    pub blackboard_reads: u64,
    pub blackboard_writes: u64,
    pub episodic_records: u64,
    pub episodic_recalls: u64,
}

/// Thread-safe telemetry accumulator for multi-agent swarm operations.
pub struct AgentMetricsTracker {
    // Tasks
    tasks_submitted: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    tasks_preempted: AtomicU64,
    task_latency: DurationTracker,

    // Routing
    total_routing_decisions: AtomicU64,
    preemptive_routings: AtomicU64,
    fallback_routings: AtomicU64,
    routing_confidence_sum: AtomicU64, // Scaled by 1000

    // Tools
    total_tool_calls: AtomicU64,
    successful_tool_calls: AtomicU64,
    failed_tool_calls: AtomicU64,
    tool_latency: DurationTracker,
    tool_counts: RwLock<HashMap<String, u64>>,

    // Reflection
    total_reflection_passes: AtomicU64,
    approved_reflections: AtomicU64,
    revision_reflections: AtomicU64,
    rejected_reflections: AtomicU64,
    escalated_reflections: AtomicU64,
    total_reflection_iterations: AtomicU64,

    // Memory
    blackboard_reads: AtomicU64,
    blackboard_writes: AtomicU64,
    episodic_records: AtomicU64,
    episodic_recalls: AtomicU64,
}

impl Default for AgentMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentMetricsTracker {
    pub fn new() -> Self {
        Self {
            tasks_submitted: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            tasks_preempted: AtomicU64::new(0),
            task_latency: DurationTracker::new(),

            total_routing_decisions: AtomicU64::new(0),
            preemptive_routings: AtomicU64::new(0),
            fallback_routings: AtomicU64::new(0),
            routing_confidence_sum: AtomicU64::new(0),

            total_tool_calls: AtomicU64::new(0),
            successful_tool_calls: AtomicU64::new(0),
            failed_tool_calls: AtomicU64::new(0),
            tool_latency: DurationTracker::new(),
            tool_counts: RwLock::new(HashMap::new()),

            total_reflection_passes: AtomicU64::new(0),
            approved_reflections: AtomicU64::new(0),
            revision_reflections: AtomicU64::new(0),
            rejected_reflections: AtomicU64::new(0),
            escalated_reflections: AtomicU64::new(0),
            total_reflection_iterations: AtomicU64::new(0),

            blackboard_reads: AtomicU64::new(0),
            blackboard_writes: AtomicU64::new(0),
            episodic_records: AtomicU64::new(0),
            episodic_recalls: AtomicU64::new(0),
        }
    }

    // ========================================================================
    // Task Event Recorders
    // ========================================================================

    pub fn record_task_submitted(&self) {
        self.tasks_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_task_completed(&self, duration: Duration) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        self.task_latency.record(duration);
    }

    pub fn record_task_failed(&self, duration: Duration) {
        self.tasks_failed.fetch_add(1, Ordering::Relaxed);
        self.task_latency.record(duration);
    }

    pub fn record_task_preempted(&self) {
        self.tasks_preempted.fetch_add(1, Ordering::Relaxed);
    }

    // ========================================================================
    // Routing Event Recorders
    // ========================================================================

    pub fn record_routing_decision(&self, decision: &RoutingDecision) {
        self.total_routing_decisions.fetch_add(1, Ordering::Relaxed);
        if decision.strategy == crate::agent::types::RoutingStrategy::PriorityTiered {
            self.preemptive_routings.fetch_add(1, Ordering::Relaxed);
        }
        if !decision.fallback_agents.is_empty() {
            self.fallback_routings.fetch_add(1, Ordering::Relaxed);
        }
        let score_millis = (decision.confidence.clamp(0.0, 1.0) * 1000.0) as u64;
        self.routing_confidence_sum
            .fetch_add(score_millis, Ordering::Relaxed);
    }

    // ========================================================================
    // Tool Event Recorders
    // ========================================================================

    pub fn record_tool_call(&self, tool_name: &str, success: bool, duration: Duration) {
        self.total_tool_calls.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_tool_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_tool_calls.fetch_add(1, Ordering::Relaxed);
        }
        self.tool_latency.record(duration);

        let mut counts = self.tool_counts.write();
        *counts.entry(tool_name.to_string()).or_insert(0) += 1;
    }

    // ========================================================================
    // Reflection Event Recorders
    // ========================================================================

    pub fn record_reflection_pass(&self, iterations: usize, verdict: ReflectionVerdict) {
        self.total_reflection_passes.fetch_add(1, Ordering::Relaxed);
        self.total_reflection_iterations
            .fetch_add(iterations as u64, Ordering::Relaxed);
        match verdict {
            ReflectionVerdict::Approved => {
                self.approved_reflections.fetch_add(1, Ordering::Relaxed);
            }
            ReflectionVerdict::NeedsRevision => {
                self.revision_reflections.fetch_add(1, Ordering::Relaxed);
            }
            ReflectionVerdict::Rejected => {
                self.rejected_reflections.fetch_add(1, Ordering::Relaxed);
            }
            ReflectionVerdict::Escalated => {
                self.escalated_reflections.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // ========================================================================
    // Memory Event Recorders
    // ========================================================================

    pub fn record_blackboard_read(&self) {
        self.blackboard_reads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_blackboard_write(&self) {
        self.blackboard_writes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_episodic_record(&self) {
        self.episodic_records.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_episodic_recall(&self) {
        self.episodic_recalls.fetch_add(1, Ordering::Relaxed);
    }

    // ========================================================================
    // Snapshot & Reset
    // ========================================================================

    pub fn snapshot(&self) -> AgentMetricsSnapshot {
        let tasks_sub = self.tasks_submitted.load(Ordering::Relaxed);
        let tasks_comp = self.tasks_completed.load(Ordering::Relaxed);
        let tasks_fail = self.tasks_failed.load(Ordering::Relaxed);
        let tasks_pre = self.tasks_preempted.load(Ordering::Relaxed);

        let total_finished = tasks_comp + tasks_fail;
        let task_success_rate = if total_finished > 0 {
            tasks_comp as f64 / total_finished as f64
        } else {
            0.0
        };

        let total_routes = self.total_routing_decisions.load(Ordering::Relaxed);
        let preempt_routes = self.preemptive_routings.load(Ordering::Relaxed);
        let fallback_routes = self.fallback_routings.load(Ordering::Relaxed);
        let conf_sum = self.routing_confidence_sum.load(Ordering::Relaxed);
        let avg_conf = if total_routes > 0 {
            (conf_sum as f64) / (total_routes as f64 * 1000.0)
        } else {
            0.0
        };

        let tool_total = self.total_tool_calls.load(Ordering::Relaxed);
        let tool_success = self.successful_tool_calls.load(Ordering::Relaxed);
        let tool_fail = self.failed_tool_calls.load(Ordering::Relaxed);
        let tool_success_rate = if tool_total > 0 {
            tool_success as f64 / tool_total as f64
        } else {
            0.0
        };

        let refl_passes = self.total_reflection_passes.load(Ordering::Relaxed);
        let refl_appr = self.approved_reflections.load(Ordering::Relaxed);
        let refl_rev = self.revision_reflections.load(Ordering::Relaxed);
        let refl_rej = self.rejected_reflections.load(Ordering::Relaxed);
        let refl_esc = self.escalated_reflections.load(Ordering::Relaxed);
        let refl_iters = self.total_reflection_iterations.load(Ordering::Relaxed);

        let refl_appr_rate = if refl_passes > 0 {
            refl_appr as f64 / refl_passes as f64
        } else {
            0.0
        };
        let avg_refl_iters = if refl_passes > 0 {
            refl_iters as f64 / refl_passes as f64
        } else {
            0.0
        };

        AgentMetricsSnapshot {
            tasks_submitted: tasks_sub,
            tasks_completed: tasks_comp,
            tasks_failed: tasks_fail,
            tasks_preempted: tasks_pre,
            task_success_rate,
            task_execution_latency: self.task_latency.snapshot(),

            total_routing_decisions: total_routes,
            preemptive_routings: preempt_routes,
            fallback_routings: fallback_routes,
            avg_routing_confidence: avg_conf,

            total_tool_calls: tool_total,
            successful_tool_calls: tool_success,
            failed_tool_calls: tool_fail,
            tool_success_rate,
            tool_execution_latency: self.tool_latency.snapshot(),
            tool_usage_by_name: self.tool_counts.read().clone(),

            total_reflection_passes: refl_passes,
            approved_reflections: refl_appr,
            revision_reflections: refl_rev,
            rejected_reflections: refl_rej,
            escalated_reflections: refl_esc,
            reflection_approval_rate: refl_appr_rate,
            avg_reflection_iterations: avg_refl_iters,

            blackboard_reads: self.blackboard_reads.load(Ordering::Relaxed),
            blackboard_writes: self.blackboard_writes.load(Ordering::Relaxed),
            episodic_records: self.episodic_records.load(Ordering::Relaxed),
            episodic_recalls: self.episodic_recalls.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.tasks_submitted.store(0, Ordering::Relaxed);
        self.tasks_completed.store(0, Ordering::Relaxed);
        self.tasks_failed.store(0, Ordering::Relaxed);
        self.tasks_preempted.store(0, Ordering::Relaxed);
        self.task_latency.reset();

        self.total_routing_decisions.store(0, Ordering::Relaxed);
        self.preemptive_routings.store(0, Ordering::Relaxed);
        self.fallback_routings.store(0, Ordering::Relaxed);
        self.routing_confidence_sum.store(0, Ordering::Relaxed);

        self.total_tool_calls.store(0, Ordering::Relaxed);
        self.successful_tool_calls.store(0, Ordering::Relaxed);
        self.failed_tool_calls.store(0, Ordering::Relaxed);
        self.tool_latency.reset();
        self.tool_counts.write().clear();

        self.total_reflection_passes.store(0, Ordering::Relaxed);
        self.approved_reflections.store(0, Ordering::Relaxed);
        self.revision_reflections.store(0, Ordering::Relaxed);
        self.rejected_reflections.store(0, Ordering::Relaxed);
        self.escalated_reflections.store(0, Ordering::Relaxed);
        self.total_reflection_iterations.store(0, Ordering::Relaxed);

        self.blackboard_reads.store(0, Ordering::Relaxed);
        self.blackboard_writes.store(0, Ordering::Relaxed);
        self.episodic_records.store(0, Ordering::Relaxed);
        self.episodic_recalls.store(0, Ordering::Relaxed);
    }
}
