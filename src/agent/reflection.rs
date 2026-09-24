//! # Stateful Reflection Loops and Self-Correction Engine
//!
//! Provides core algorithms for:
//! - Multi-dimensional critique evaluation and weighted quality scoring.
//! - Critical invariant vetoes preventing unsafe or incomplete outputs from approval.
//! - Convergence detection (threshold passing, stagnation delta, regression detection).
//! - Iteration history tracking and best-candidate preservation across refinement passes.
//! - Actionable prompt synthesis for targeted self-correction.

use std::sync::Arc;

use crate::agent::types::{
    AgentTask, ReflectionCritique, ReflectionState, ReflectionStep, ReflectionVerdict, Reflector,
};
use crate::error::Result;

// ============================================================================
// Multi-Dimensional Criteria & Scoring
// ============================================================================

/// Specification of an evaluation criterion for reflection.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationCriterion {
    pub name: String,
    pub weight: f32,
    pub is_critical: bool,
    pub min_threshold: f32,
}

impl EvaluationCriterion {
    pub fn new(name: impl Into<String>, weight: f32) -> Self {
        Self {
            name: name.into(),
            weight: weight.max(0.0),
            is_critical: false,
            min_threshold: 0.5,
        }
    }

    pub fn critical(mut self, min_threshold: f32) -> Self {
        self.is_critical = true;
        self.min_threshold = min_threshold.clamp(0.0, 1.0);
        self
    }
}

/// Computes the weighted aggregate score across critiques.
/// Returns a score in `[0.0, 1.0]`. If critiques is empty, returns `0.0`.
pub fn compute_aggregate_score(critiques: &[ReflectionCritique]) -> f32 {
    if critiques.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;
    for c in critiques {
        total_score += c.score.clamp(0.0, 1.0);
    }

    total_score / critiques.len() as f32
}

/// Checks whether any critical criteria failed their minimum threshold.
/// Returns `true` if all critical criteria passed, `false` if any failed.
pub fn verify_critical_invariants(
    critiques: &[ReflectionCritique],
    critical_threshold: f32,
) -> (bool, Option<String>) {
    for c in critiques {
        // If criterion is safety, security, or correctness, treat as critical
        let is_critical_name =
            c.dimension == "safety" || c.dimension == "security" || c.dimension == "correctness";

        if is_critical_name && c.score < critical_threshold {
            return (
                false,
                Some(format!(
                    "Critical dimension '{}' failed with score {:.2} (required {:.2}): {}",
                    c.dimension, c.score, critical_threshold, c.feedback
                )),
            );
        }
    }

    (true, None)
}

// ============================================================================
// Convergence Evaluation Algorithm
// ============================================================================

/// Configuration for convergence detection in reflection loops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceConfig {
    /// Minimum score required to approve a candidate output (default 0.85).
    pub passing_threshold: f32,
    /// Absolute score difference between consecutive iterations indicating stagnation (default 0.02).
    pub stagnation_epsilon: f32,
    /// Score drop threshold between consecutive iterations indicating degradation (default 0.15).
    pub regression_tolerance: f32,
    /// Minimum critical invariant threshold (default 0.60).
    pub critical_threshold: f32,
    /// Maximum allowed reflection iterations (default 5).
    pub max_iterations: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            passing_threshold: 0.85,
            stagnation_epsilon: 0.02,
            regression_tolerance: 0.15,
            critical_threshold: 0.60,
            max_iterations: 5,
        }
    }
}

/// Evaluator that determines whether a reflection loop has converged, regressed, or needs revision.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceEvaluator {
    pub config: ConvergenceConfig,
}

impl ConvergenceEvaluator {
    pub fn new(config: ConvergenceConfig) -> Self {
        Self { config }
    }

    /// Evaluates candidate output critiques against historical reflection steps.
    pub fn evaluate(
        &self,
        iteration: usize,
        current_score: f32,
        critiques: &[ReflectionCritique],
        history: &[ReflectionStep],
    ) -> (ReflectionVerdict, Option<String>) {
        // 1. Verify critical invariants (safety, security, correctness)
        let (critical_passed, critical_reason) =
            verify_critical_invariants(critiques, self.config.critical_threshold);
        if !critical_passed {
            if iteration >= self.config.max_iterations {
                return (ReflectionVerdict::Rejected, critical_reason);
            } else {
                return (ReflectionVerdict::NeedsRevision, critical_reason);
            }
        }

        // 2. Check if passing threshold is met
        if current_score >= self.config.passing_threshold {
            return (
                ReflectionVerdict::Approved,
                Some(format!(
                    "Candidate achieved score {:.2} >= threshold {:.2}",
                    current_score, self.config.passing_threshold
                )),
            );
        }

        // 3. Check for max iterations exhaustion
        if iteration >= self.config.max_iterations {
            if current_score >= 0.70 {
                return (
                    ReflectionVerdict::Escalated,
                    Some(format!(
                        "Max iterations ({}) reached with promising but sub-threshold score {:.2}",
                        self.config.max_iterations, current_score
                    )),
                );
            } else {
                return (
                    ReflectionVerdict::Rejected,
                    Some(format!(
                        "Max iterations ({}) exhausted without achieving passing threshold (final score {:.2})",
                        self.config.max_iterations, current_score
                    )),
                );
            }
        }

        // 4. Check for regression or stagnation against previous step
        if let Some(prev) = history.last() {
            let prev_score = prev.aggregate_score;

            // Regression check
            if current_score < prev_score - self.config.regression_tolerance {
                return (
                    ReflectionVerdict::NeedsRevision,
                    Some(format!(
                        "Score regressed from {:.2} to {:.2}. Revert regressions and focus on improvements.",
                        prev_score, current_score
                    )),
                );
            }

            // Stagnation check over consecutive iterations
            if history.len() >= 2 {
                let prev_prev_score = history[history.len() - 2].aggregate_score;
                let delta1 = (current_score - prev_score).abs();
                let delta2 = (prev_score - prev_prev_score).abs();

                if delta1 < self.config.stagnation_epsilon
                    && delta2 < self.config.stagnation_epsilon
                {
                    if current_score >= self.config.passing_threshold - 0.05 {
                        // Close enough, converge
                        return (
                            ReflectionVerdict::Approved,
                            Some(format!(
                                "Converged with stagnation at {:.2} within tolerance",
                                current_score
                            )),
                        );
                    } else {
                        return (
                            ReflectionVerdict::Escalated,
                            Some(format!(
                                "Stagnated at score {:.2} across multiple iterations without reaching threshold",
                                current_score
                            )),
                        );
                    }
                }
            }
        }

        (
            ReflectionVerdict::NeedsRevision,
            Some(format!(
                "Iteration {} score {:.2} below threshold {:.2}; revision required",
                iteration, current_score, self.config.passing_threshold
            )),
        )
    }
}

// ============================================================================
// Actionable Revision Prompt Synthesis
// ============================================================================

/// Synthesizes an actionable prompt instructing the agent on concrete revisions needed.
pub fn format_revision_prompt(
    step: &ReflectionStep,
    task: &AgentTask,
    candidate_output: &str,
) -> String {
    let mut prompt = format!(
        "### Reflection Feedback (Iteration {} - Score: {:.2})\n\n",
        step.iteration, step.aggregate_score
    );

    prompt.push_str(&format!("**Task Goal:** {}\n\n", task.title));
    prompt.push_str("#### Identified Issues & Critiques:\n");

    for critique in &step.critiques {
        let status = if critique.score >= 0.7 {
            "PASS"
        } else {
            "FAIL"
        };
        prompt.push_str(&format!(
            "- `[{}]` **{}** (score: {:.2}): {}\n",
            status, critique.dimension, critique.score, critique.feedback
        ));

        for improvement in &critique.actionable_improvements {
            prompt.push_str(&format!("  * Action: {}\n", improvement));
        }
    }

    prompt.push_str("\n#### Instructions for Next Iteration:\n");
    prompt.push_str("1. Address every critique marked [FAIL] above.\n");
    prompt.push_str("2. Preserve all aspects of the output that scored well.\n");
    prompt.push_str("3. Provide the full revised candidate output.\n\n");
    prompt.push_str("#### Current Candidate Output:\n");
    prompt.push_str("```\n");
    prompt.push_str(candidate_output);
    prompt.push_str("\n```\n");

    prompt
}

// ============================================================================
// Reflection Loop Engine
// ============================================================================

/// Coordinates iterative candidate refinement, critique generation, and convergence tracking.
pub struct ReflectionEngine {
    reflector: Arc<dyn Reflector>,
    evaluator: ConvergenceEvaluator,
}

impl ReflectionEngine {
    /// Creates a new `ReflectionEngine` with the given reflector and default convergence config.
    pub fn new(reflector: Arc<dyn Reflector>) -> Self {
        Self {
            reflector,
            evaluator: ConvergenceEvaluator::default(),
        }
    }

    /// Customizes convergence configuration.
    pub fn with_config(mut self, config: ConvergenceConfig) -> Self {
        self.evaluator = ConvergenceEvaluator::new(config);
        self
    }

    /// Evaluates a candidate output, appends a new reflection step to the state, and returns the verdict.
    pub fn step(
        &self,
        task: &AgentTask,
        state: &mut ReflectionState,
        candidate_output: &str,
    ) -> Result<ReflectionVerdict> {
        let iteration = state.current_iteration + 1;
        let critiques = self.reflector.evaluate(task, candidate_output, iteration)?;
        let overall_score = compute_aggregate_score(&critiques);

        let (verdict, explanation) =
            self.evaluator
                .evaluate(iteration, overall_score, &critiques, &state.steps);

        let step = ReflectionStep::new(iteration, candidate_output, critiques, verdict);

        if let Some(exp) = explanation {
            tracing::debug!(iteration = iteration, verdict = ?verdict, score = overall_score, "{exp}");
        }

        state.record_step(step);
        Ok(verdict)
    }

    /// Runs the full reflection loop on a candidate output until convergence, revision exhaustion, or rejection.
    /// In an autonomous workflow, `generator_fn` produces the revised output given revision prompts.
    pub fn execute_loop<F>(
        &self,
        task: &AgentTask,
        initial_candidate: &str,
        mut generator_fn: F,
    ) -> Result<ReflectionState>
    where
        F: FnMut(&str, &str) -> Result<String>, // (revision_prompt, previous_candidate) -> new_candidate
    {
        let mut state = ReflectionState::new(
            task.id.clone(),
            self.evaluator.config.max_iterations,
            self.evaluator.config.passing_threshold,
        );
        let mut current_candidate = initial_candidate.to_string();

        loop {
            let verdict = self.step(task, &mut state, &current_candidate)?;

            match verdict {
                ReflectionVerdict::Approved => {
                    break;
                }
                ReflectionVerdict::NeedsRevision => {
                    if let Some(last_step) = state.steps.last() {
                        let prompt = format_revision_prompt(last_step, task, &current_candidate);
                        current_candidate = generator_fn(&prompt, &current_candidate)?;
                    } else {
                        break;
                    }
                }
                ReflectionVerdict::Rejected | ReflectionVerdict::Escalated => {
                    break;
                }
            }
        }

        Ok(state)
    }
}

// ============================================================================
// Heuristic & Composite Reflectors
// ============================================================================

/// Flexible rule-based evaluator verifying length, keywords, formatting, and safety.
pub struct HeuristicReflector {
    pub min_length: usize,
    pub max_length: Option<usize>,
    pub required_keywords: Vec<String>,
    pub prohibited_keywords: Vec<String>,
    pub require_json_syntax: bool,
    pub require_code_fence: bool,
}

impl Default for HeuristicReflector {
    fn default() -> Self {
        Self {
            min_length: 20,
            max_length: None,
            required_keywords: Vec::new(),
            prohibited_keywords: vec![
                "TODO".into(),
                "FIXME".into(),
                "unimplemented!()".into(),
                "panic!()".into(),
            ],
            require_json_syntax: false,
            require_code_fence: false,
        }
    }
}

impl HeuristicReflector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_min_length(mut self, min: usize) -> Self {
        self.min_length = min;
        self
    }

    pub fn with_required_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.required_keywords.push(keyword.into());
        self
    }

    pub fn with_prohibited_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.prohibited_keywords.push(keyword.into());
        self
    }

    pub fn with_json_validation(mut self, require: bool) -> Self {
        self.require_json_syntax = require;
        self
    }
}

impl Reflector for HeuristicReflector {
    fn evaluate(
        &self,
        _task: &AgentTask,
        candidate_output: &str,
        _iteration: usize,
    ) -> Result<Vec<ReflectionCritique>> {
        let mut critiques = Vec::new();

        // 1. Length check
        if candidate_output.len() < self.min_length {
            let score = (candidate_output.len() as f32 / self.min_length as f32).max(0.1);
            critiques.push(
                ReflectionCritique::new("completeness", score, "Output is too short")
                    .with_improvement(format!(
                        "Elaborate to at least {} characters",
                        self.min_length
                    )),
            );
        } else {
            critiques.push(ReflectionCritique::new(
                "completeness",
                1.0,
                "Length satisfies minimum requirement",
            ));
        }

        // 2. Safety / Prohibited keywords
        let mut found_prohibited = Vec::new();
        for p in &self.prohibited_keywords {
            if candidate_output.contains(p) {
                found_prohibited.push(p.clone());
            }
        }

        if !found_prohibited.is_empty() {
            critiques.push(
                ReflectionCritique::new(
                    "safety",
                    0.2,
                    format!(
                        "Contains prohibited placeholder tokens: {:?}",
                        found_prohibited
                    ),
                )
                .with_improvement("Remove all placeholders and provide production-ready code"),
            );
        } else {
            critiques.push(ReflectionCritique::new(
                "safety",
                1.0,
                "Zero prohibited placeholder tokens detected",
            ));
        }

        // 3. Required keywords
        if !self.required_keywords.is_empty() {
            let mut missing = Vec::new();
            for req in &self.required_keywords {
                if !candidate_output.contains(req) {
                    missing.push(req.clone());
                }
            }

            if !missing.is_empty() {
                let score = 1.0 - (missing.len() as f32 / self.required_keywords.len() as f32);
                critiques.push(
                    ReflectionCritique::new(
                        "correctness",
                        score,
                        format!("Missing required terms: {:?}", missing),
                    )
                    .with_improvement(format!("Ensure inclusion of: {:?}", missing)),
                );
            } else {
                critiques.push(ReflectionCritique::new(
                    "correctness",
                    1.0,
                    "All required key concepts are addressed",
                ));
            }
        }

        // 4. JSON syntax validation
        if self.require_json_syntax {
            match serde_json::from_str::<serde_json::Value>(candidate_output) {
                Ok(_) => {
                    critiques.push(ReflectionCritique::new(
                        "format_compliance",
                        1.0,
                        "Valid JSON payload",
                    ));
                }
                Err(e) => {
                    critiques.push(
                        ReflectionCritique::new(
                            "format_compliance",
                            0.0,
                            format!("Invalid JSON syntax: {}", e),
                        )
                        .with_improvement("Format output as strictly valid JSON"),
                    );
                }
            }
        }

        Ok(critiques)
    }
}

/// Composite reflector aggregating critiques from multiple reflectors.
#[derive(Default)]
pub struct CompositeReflector {
    reflectors: Vec<Arc<dyn Reflector>>,
}

impl CompositeReflector {
    pub fn new() -> Self {
        Self {
            reflectors: Vec::new(),
        }
    }

    pub fn add(&mut self, reflector: Arc<dyn Reflector>) {
        self.reflectors.push(reflector);
    }
}

impl Reflector for CompositeReflector {
    fn evaluate(
        &self,
        task: &AgentTask,
        candidate_output: &str,
        iteration: usize,
    ) -> Result<Vec<ReflectionCritique>> {
        let mut all_critiques = Vec::new();
        for r in &self.reflectors {
            let sub = r.evaluate(task, candidate_output, iteration)?;
            all_critiques.extend(sub);
        }
        Ok(all_critiques)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_evaluator_approval() {
        let evaluator = ConvergenceEvaluator::default();
        let critiques = vec![
            ReflectionCritique::new("correctness", 0.95, "Accurate logic"),
            ReflectionCritique::new("safety", 1.0, "Secure"),
            ReflectionCritique::new("completeness", 0.90, "Thorough"),
        ];

        let score = compute_aggregate_score(&critiques);
        assert!(score >= 0.85);

        let (verdict, explanation) = evaluator.evaluate(1, score, &critiques, &[]);
        assert_eq!(verdict, ReflectionVerdict::Approved);
        assert!(explanation.is_some());
    }

    #[test]
    fn test_critical_invariant_veto() {
        let evaluator = ConvergenceEvaluator::default();
        let critiques = vec![
            ReflectionCritique::new("completeness", 1.0, "Great length"),
            ReflectionCritique::new("conciseness", 1.0, "Concise"),
            ReflectionCritique::new(
                "safety",
                0.3,
                "Critical SQL injection vulnerability detected",
            ),
        ];

        let score = compute_aggregate_score(&critiques);
        // Average is (1.0 + 1.0 + 0.3)/3 = 0.766, but safety failed critical threshold 0.60
        let (verdict, reason) = evaluator.evaluate(1, score, &critiques, &[]);
        assert_eq!(verdict, ReflectionVerdict::NeedsRevision);
        assert!(reason.unwrap().contains("Critical dimension 'safety'"));
    }

    #[test]
    fn test_reflection_engine_autonomous_loop() {
        let reflector = Arc::new(
            HeuristicReflector::new()
                .with_min_length(15)
                .with_required_keyword("FlashStore"),
        );

        let engine = ReflectionEngine::new(reflector);
        let task = AgentTask::builder("t-ref", "Describe FlashStore").build();

        // Simulate multi-turn refinement:
        // Turn 1: Too short -> Needs revision
        // Turn 2: Long enough and has FlashStore -> Approved
        let mut turn = 0;
        let final_state = engine
            .execute_loop(&task, "Short text", |_prompt, _prev| {
                turn += 1;
                if turn == 1 {
                    Ok("FlashStore is a high performance concurrent LSM storage engine.".into())
                } else {
                    Ok("Done".into())
                }
            })
            .unwrap();

        assert_eq!(final_state.steps.len(), 2);
        assert_eq!(
            final_state.steps[0].verdict,
            ReflectionVerdict::NeedsRevision
        );
        assert_eq!(final_state.steps[1].verdict, ReflectionVerdict::Approved);
        assert_eq!(
            final_state.best_candidate().unwrap(),
            "FlashStore is a high performance concurrent LSM storage engine."
        );
    }
}
