//! # Advanced Task Routing Algorithms and Strategies
//!
//! Autonomous routing engine matching agent tasks to optimal agent instances based on:
//! - Multi-factor capability scoring and role specialization affinities.
//! - Real-time concurrency tracking and least-loaded load balancing.
//! - Priority-preemptive scheduling reserving specialist agents for critical workloads.
//! - Adaptive hybrid ranking combining capability, role, load, and priority weights.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::agent::types::{
    AgentCapability, AgentDefinition, AgentId, AgentRole, AgentTask, RoutingDecision,
    RoutingStrategy, TaskPriority, TaskRouter,
};
use crate::error::{FlashStoreError, Result};

// ============================================================================
// Scoring Functions
// ============================================================================

/// Computes the capability match score between task requirements and an agent's capability set.
/// Returns a score in `[0.0, 1.0]`. If no capabilities are required, returns `1.0`.
#[inline]
pub fn score_capability_match(task: &AgentTask, agent: &AgentDefinition) -> f32 {
    task.capability_match_score(agent)
}

/// Computes role affinity score between a task and an agent's architectural role.
/// Returns a score in `[0.0, 1.0]`.
pub fn score_role_affinity(task: &AgentTask, agent: &AgentDefinition) -> f32 {
    let mut affinity: f32 = 0.5; // neutral baseline

    // Role bonuses based on required capabilities
    for cap in &task.required_capabilities {
        match (cap, &agent.role) {
            (AgentCapability::CodeExecution, AgentRole::Coder) => affinity += 0.25,
            (AgentCapability::Reflection, AgentRole::Critic) => affinity += 0.25,
            (AgentCapability::ToolCalling, AgentRole::ToolExecutor) => affinity += 0.2,
            (AgentCapability::StorageOps, AgentRole::MemoryAgent) => affinity += 0.25,
            (AgentCapability::MemoryRetrieval, AgentRole::MemoryAgent) => affinity += 0.25,
            (AgentCapability::TextGeneration, AgentRole::Planner) => affinity += 0.15,
            _ => {}
        }
    }

    // Role bonus based on task title/description keywords
    let text = format!("{} {}", task.title, task.description).to_lowercase();
    match &agent.role {
        AgentRole::Coder
            if text.contains("code") || text.contains("implement") || text.contains("refactor") =>
        {
            affinity += 0.2;
        }
        AgentRole::Critic
            if text.contains("critique") || text.contains("review") || text.contains("inspect") =>
        {
            affinity += 0.2;
        }
        AgentRole::Verifier
            if text.contains("test") || text.contains("verify") || text.contains("bench") =>
        {
            affinity += 0.2;
        }
        AgentRole::Planner
            if text.contains("plan") || text.contains("decompose") || text.contains("design") =>
        {
            affinity += 0.2;
        }
        AgentRole::MemoryAgent
            if text.contains("store") || text.contains("memory") || text.contains("rag") =>
        {
            affinity += 0.2;
        }
        _ => {}
    }

    affinity.clamp(0.0, 1.0)
}

/// Computes a load score where 1.0 means completely idle and 0.0 means fully loaded.
pub fn score_workload(active_tasks: usize, max_concurrency: usize) -> f32 {
    if max_concurrency == 0 {
        return 0.0;
    }
    let ratio = active_tasks as f32 / max_concurrency as f32;
    (1.0 - ratio).clamp(0.0, 1.0)
}

/// Computes priority alignment factor.
/// High and Critical tasks receive a bonus when matched to highly capable agents.
pub fn score_priority_tier(task_priority: TaskPriority, capability_score: f32) -> f32 {
    match task_priority {
        TaskPriority::Critical => {
            // Strictly favor top capability
            if capability_score >= 0.99 {
                1.0
            } else {
                0.2
            }
        }
        TaskPriority::High => {
            if capability_score >= 0.75 {
                0.9
            } else {
                0.4
            }
        }
        TaskPriority::Medium => 0.6,
        TaskPriority::Low => 0.5,
    }
}

// ============================================================================
// Load-Balancing Task Router
// ============================================================================

/// Router that balances workload across agents while satisfying capability requirements.
#[derive(Debug, Clone, Default)]
pub struct LoadBalancingRouter {
    active_loads: Arc<RwLock<HashMap<AgentId, usize>>>,
    max_concurrency_per_agent: usize,
}

impl LoadBalancingRouter {
    /// Creates a new `LoadBalancingRouter` with a default concurrency cap.
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            active_loads: Arc::new(RwLock::new(HashMap::new())),
            max_concurrency_per_agent: if max_concurrency == 0 {
                4
            } else {
                max_concurrency
            },
        }
    }

    /// Records the start of a task on an agent.
    pub fn track_task_started(&self, agent_id: &AgentId) {
        if let Ok(mut loads) = self.active_loads.write() {
            *loads.entry(agent_id.clone()).or_insert(0) += 1;
        }
    }

    /// Records the completion of a task on an agent.
    pub fn track_task_completed(&self, agent_id: &AgentId) {
        if let Ok(mut loads) = self.active_loads.write() {
            if let Some(count) = loads.get_mut(agent_id) {
                if *count > 0 {
                    *count -= 1;
                }
            }
        }
    }

    /// Gets current active load count for an agent.
    pub fn get_load(&self, agent_id: &AgentId) -> usize {
        self.active_loads
            .read()
            .ok()
            .and_then(|loads| loads.get(agent_id).copied())
            .unwrap_or(0)
    }

    /// Clears all tracked loads.
    pub fn reset_loads(&self) {
        if let Ok(mut loads) = self.active_loads.write() {
            loads.clear();
        }
    }
}

impl TaskRouter for LoadBalancingRouter {
    fn route(
        &self,
        task: &AgentTask,
        available_agents: &[AgentDefinition],
    ) -> Result<RoutingDecision> {
        if available_agents.is_empty() {
            return Err(FlashStoreError::Agent(
                "No available agents registered in the swarm".into(),
            ));
        }

        let loads = self
            .active_loads
            .read()
            .map_err(|e| FlashStoreError::LockError(e.to_string()))?;

        // Filter agents that satisfy minimum required capabilities (or highest partial if none 100%)
        let mut candidates: Vec<(&AgentDefinition, f32, usize)> = available_agents
            .iter()
            .map(|agent| {
                let cap_score = score_capability_match(task, agent);
                let current_load = loads.get(&agent.id).copied().unwrap_or(0);
                (agent, cap_score, current_load)
            })
            .collect();

        // Sort by capability score descending, then by current load ascending
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        // Filter to those with highest capability match
        let top_cap = candidates[0].1;
        let eligible: Vec<&(&AgentDefinition, f32, usize)> = candidates
            .iter()
            .filter(|c| (c.1 - top_cap).abs() < f32::EPSILON)
            .collect();

        // Pick the least loaded among highest capability
        let best = eligible
            .iter()
            .min_by_key(|c| c.2)
            .copied()
            .unwrap_or(&candidates[0]);

        let load_score = score_workload(best.2, self.max_concurrency_per_agent);
        let confidence = (best.1 * 0.7 + load_score * 0.3).clamp(0.0, 1.0);

        let mut decision = RoutingDecision::new(
            task.id.clone(),
            best.0.id.clone(),
            RoutingStrategy::LeastLoaded,
            confidence,
            format!(
                "Assigned to {} with capability {:.0}% and active load {}/{}",
                best.0.id,
                best.1 * 100.0,
                best.2,
                self.max_concurrency_per_agent
            ),
        );

        for (fallback, _, _) in candidates.iter().skip(1).take(2) {
            decision = decision.with_fallback(fallback.id.clone());
        }

        Ok(decision)
    }
}

// ============================================================================
// Priority-Preemptive Router
// ============================================================================

/// Router that prioritizes high-tier specialist agents for critical tasks.
#[derive(Debug, Clone, Default)]
pub struct PriorityPreemptiveRouter {
    critical_threshold: f32,
}

impl PriorityPreemptiveRouter {
    pub fn new() -> Self {
        Self {
            critical_threshold: 0.8,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.critical_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl TaskRouter for PriorityPreemptiveRouter {
    fn route(
        &self,
        task: &AgentTask,
        available_agents: &[AgentDefinition],
    ) -> Result<RoutingDecision> {
        if available_agents.is_empty() {
            return Err(FlashStoreError::Agent(
                "No available agents registered in the swarm".into(),
            ));
        }

        let mut candidates: Vec<(&AgentDefinition, f32, f32)> = available_agents
            .iter()
            .map(|agent| {
                let cap_score = score_capability_match(task, agent);
                let role_score = score_role_affinity(task, agent);
                (agent, cap_score, role_score)
            })
            .collect();

        // Sort by composite capability & role score
        candidates.sort_by(|a, b| {
            let score_a = a.1 * 0.6 + a.2 * 0.4;
            let score_b = b.1 * 0.6 + b.2 * 0.4;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (best_agent, top_cap, top_role) = candidates[0];

        // Verify critical tasks meet capability standards
        if task.priority == TaskPriority::Critical
            && top_cap < self.critical_threshold
            && !task.required_capabilities.is_empty()
        {
            return Err(FlashStoreError::Agent(format!(
                "Critical task '{}' cannot be safely routed: top candidate {} only satisfies {:.0}% of capabilities (required {:.0}%)",
                task.id,
                best_agent.id,
                top_cap * 100.0,
                self.critical_threshold * 100.0
            )));
        }

        let confidence = (top_cap * 0.6 + top_role * 0.4).clamp(0.0, 1.0);

        let mut decision = RoutingDecision::new(
            task.id.clone(),
            best_agent.id.clone(),
            RoutingStrategy::PriorityTiered,
            confidence,
            format!(
                "Priority {:?} routed to {} (cap: {:.0}%, role: {:.0}%)",
                task.priority,
                best_agent.id,
                top_cap * 100.0,
                top_role * 100.0
            ),
        );

        for (fallback, _, _) in candidates.iter().skip(1).take(2) {
            decision = decision.with_fallback(fallback.id.clone());
        }

        Ok(decision)
    }
}

// ============================================================================
// Adaptive Hybrid Task Router
// ============================================================================

/// Configurable multi-criteria weights for adaptive routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveRoutingWeights {
    pub capability: f32,
    pub role: f32,
    pub workload: f32,
    pub priority: f32,
}

impl Default for AdaptiveRoutingWeights {
    fn default() -> Self {
        Self {
            capability: 0.45,
            role: 0.25,
            workload: 0.20,
            priority: 0.10,
        }
    }
}

/// Adaptive hybrid router balancing capability matching, role affinity, live concurrency, and priority tiers.
#[derive(Debug, Clone)]
pub struct AdaptiveTaskRouter {
    weights: AdaptiveRoutingWeights,
    load_balancer: LoadBalancingRouter,
}

impl Default for AdaptiveTaskRouter {
    fn default() -> Self {
        Self::new(AdaptiveRoutingWeights::default(), 4)
    }
}

impl AdaptiveTaskRouter {
    /// Creates a new `AdaptiveTaskRouter`.
    pub fn new(weights: AdaptiveRoutingWeights, max_concurrency: usize) -> Self {
        Self {
            weights,
            load_balancer: LoadBalancingRouter::new(max_concurrency),
        }
    }

    /// Accesses the underlying load balancer to track task lifecycles.
    pub fn load_balancer(&self) -> &LoadBalancingRouter {
        &self.load_balancer
    }

    /// Computes full multi-factor score for an agent.
    pub fn evaluate_agent(&self, task: &AgentTask, agent: &AgentDefinition) -> (f32, String) {
        let cap_score = score_capability_match(task, agent);
        let role_score = score_role_affinity(task, agent);
        let current_load = self.load_balancer.get_load(&agent.id);
        let load_score = score_workload(current_load, self.load_balancer.max_concurrency_per_agent);
        let prio_score = score_priority_tier(task.priority, cap_score);

        let composite = self.weights.capability * cap_score
            + self.weights.role * role_score
            + self.weights.workload * load_score
            + self.weights.priority * prio_score;

        let rationale = format!(
            "cap: {:.0}%, role: {:.0}%, load: {}/{}, prio: {:.1}",
            cap_score * 100.0,
            role_score * 100.0,
            current_load,
            self.load_balancer.max_concurrency_per_agent,
            prio_score
        );

        (composite.clamp(0.0, 1.0), rationale)
    }
}

impl TaskRouter for AdaptiveTaskRouter {
    fn route(
        &self,
        task: &AgentTask,
        available_agents: &[AgentDefinition],
    ) -> Result<RoutingDecision> {
        if available_agents.is_empty() {
            return Err(FlashStoreError::Agent(
                "No available agents registered in the swarm".into(),
            ));
        }

        let mut scored: Vec<(&AgentDefinition, f32, String)> = available_agents
            .iter()
            .map(|agent| {
                let (score, rationale) = self.evaluate_agent(task, agent);
                (agent, score, rationale)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_agent, best_score, rationale) = &scored[0];

        let mut decision = RoutingDecision::new(
            task.id.clone(),
            best_agent.id.clone(),
            RoutingStrategy::Adaptive,
            *best_score,
            format!(
                "Adaptive choice {} (fitness: {:.2}): {}",
                best_agent.id, best_score, rationale
            ),
        );

        for (fallback, _, _) in scored.iter().skip(1).take(2) {
            decision = decision.with_fallback(fallback.id.clone());
        }

        Ok(decision)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_agents() -> Vec<AgentDefinition> {
        vec![
            AgentDefinition::builder("agent-coder", AgentRole::Coder)
                .capability(AgentCapability::CodeExecution)
                .capability(AgentCapability::ToolCalling)
                .build(),
            AgentDefinition::builder("agent-critic", AgentRole::Critic)
                .capability(AgentCapability::Reflection)
                .capability(AgentCapability::TextGeneration)
                .build(),
            AgentDefinition::builder("agent-storage", AgentRole::MemoryAgent)
                .capability(AgentCapability::StorageOps)
                .capability(AgentCapability::MemoryRetrieval)
                .capability(AgentCapability::ToolCalling)
                .build(),
        ]
    }

    #[test]
    fn test_score_capability_match() {
        let task = AgentTask::builder("t1", "Write code")
            .require_capability(AgentCapability::CodeExecution)
            .require_capability(AgentCapability::ToolCalling)
            .build();

        let agents = sample_agents();
        assert_eq!(score_capability_match(&task, &agents[0]), 1.0); // coder has both
        assert_eq!(score_capability_match(&task, &agents[1]), 0.0); // critic has neither
        assert_eq!(score_capability_match(&task, &agents[2]), 0.5); // storage has 1 of 2
    }

    #[test]
    fn test_load_balancing_router() {
        let router = LoadBalancingRouter::new(2);
        let agents = sample_agents();

        let task1 = AgentTask::builder("t1", "Storage 1")
            .require_capability(AgentCapability::StorageOps)
            .build();

        let decision1 = router.route(&task1, &agents).unwrap();
        assert_eq!(decision1.target_agent.as_str(), "agent-storage");

        // Simulate agent-storage becoming loaded
        router.track_task_started(&AgentId::new("agent-storage"));
        assert_eq!(router.get_load(&AgentId::new("agent-storage")), 1);

        router.track_task_completed(&AgentId::new("agent-storage"));
        assert_eq!(router.get_load(&AgentId::new("agent-storage")), 0);
    }

    #[test]
    fn test_priority_preemptive_router() {
        let router = PriorityPreemptiveRouter::new();
        let agents = sample_agents();

        let critical_task = AgentTask::builder("t-crit", "Fix crash in MemTable")
            .priority(TaskPriority::Critical)
            .require_capability(AgentCapability::CodeExecution)
            .build();

        let decision = router.route(&critical_task, &agents).unwrap();
        assert_eq!(decision.target_agent.as_str(), "agent-coder");
        assert!(decision.confidence >= 0.8);
    }

    #[test]
    fn test_adaptive_router_ranking_and_fallbacks() {
        let router = AdaptiveTaskRouter::default();
        let agents = sample_agents();

        let task = AgentTask::builder("t-hybrid", "Implement Storage Tooling")
            .require_capability(AgentCapability::StorageOps)
            .require_capability(AgentCapability::ToolCalling)
            .build();

        let decision = router.route(&task, &agents).unwrap();
        assert_eq!(decision.target_agent.as_str(), "agent-storage");
        assert!(!decision.fallback_agents.is_empty());
    }
}
