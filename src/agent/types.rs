//! # Domain Models and Type Annotations for Autonomous Multi-Agent Workflows
//!
//! Provides type-safe domain models for:
//! - Multi-agent routing, agent identities, role definitions, and capability matching.
//! - Task definitions, priority tiers, lifecycle states, and execution tracking.
//! - Automated tool definitions, parameter schemas, function calls, and output results.
//! - Stateful reflection loops, critiques, convergence evaluation, and self-correction iterations.
//! - Workflow state machines, inter-agent messaging, shared memory, and storage namespace keys.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{FlashStoreError, Result};

// ============================================================================
// Agent Identification, Roles & Capabilities
// ============================================================================

/// Strongly typed identifier for an agent instance or profile.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    /// Creates a new `AgentId`.
    #[inline]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns a string slice of the ID.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for AgentId {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for AgentId {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for AgentId {
    #[inline]
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AgentId {
    #[inline]
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Specialized architectural roles assigned to swarm agents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Primary orchestrator directing workflow decomposition and task assignment.
    Planner,
    /// Intelligent dispatcher matching tasks to agent capabilities and load.
    Router,
    /// Core software engineering and logic implementation agent.
    Coder,
    /// Quality assurance and static evaluation critic for reflection loops.
    Critic,
    /// Dynamic test execution and behavioral validation verifier.
    Verifier,
    /// Deterministic tool executor executing functions and sandbox actions.
    ToolExecutor,
    /// Long-term memory and retrieval agent managing KV & RAG indexes.
    MemoryAgent,
    /// Custom domain-specific agent role.
    Custom(String),
}

impl AgentRole {
    /// Returns default human-readable role name.
    pub fn as_str(&self) -> &str {
        match self {
            AgentRole::Planner => "planner",
            AgentRole::Router => "router",
            AgentRole::Coder => "coder",
            AgentRole::Critic => "critic",
            AgentRole::Verifier => "verifier",
            AgentRole::ToolExecutor => "tool_executor",
            AgentRole::MemoryAgent => "memory_agent",
            AgentRole::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for AgentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Atomic capabilities supported by an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCapability {
    /// Natural language synthesis and reasoning.
    TextGeneration,
    /// Autonomous tool invocation and function parameter extraction.
    ToolCalling,
    /// Read/write interaction with RAG & vector stores.
    MemoryRetrieval,
    /// Self-critique, error analysis, and reflection loops.
    Reflection,
    /// Code analysis, compilation, and sandbox execution.
    CodeExecution,
    /// Direct key-value storage engine operations.
    StorageOps,
}

impl fmt::Display for AgentCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            AgentCapability::TextGeneration => "text_generation",
            AgentCapability::ToolCalling => "tool_calling",
            AgentCapability::MemoryRetrieval => "memory_retrieval",
            AgentCapability::Reflection => "reflection",
            AgentCapability::CodeExecution => "code_execution",
            AgentCapability::StorageOps => "storage_ops",
        };
        write!(f, "{}", name)
    }
}

/// Metadata and configuration specification for an active agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDefinition {
    pub id: AgentId,
    pub name: String,
    pub role: AgentRole,
    pub description: String,
    pub capabilities: HashSet<AgentCapability>,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub timeout_secs: u64,
}

impl AgentDefinition {
    /// Constructs a new agent definition builder.
    pub fn builder(id: impl Into<AgentId>, role: AgentRole) -> AgentDefinitionBuilder {
        AgentDefinitionBuilder::new(id, role)
    }

    /// Checks whether the agent supports a required capability.
    #[inline]
    pub fn has_capability(&self, cap: AgentCapability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Fluent builder for [`AgentDefinition`].
pub struct AgentDefinitionBuilder {
    id: AgentId,
    name: String,
    role: AgentRole,
    description: String,
    capabilities: HashSet<AgentCapability>,
    system_prompt: String,
    max_iterations: usize,
    timeout_secs: u64,
}

impl AgentDefinitionBuilder {
    pub fn new(id: impl Into<AgentId>, role: AgentRole) -> Self {
        let agent_id = id.into();
        let default_name = format!("{}-agent", agent_id.as_str());
        Self {
            id: agent_id,
            name: default_name,
            role,
            description: String::new(),
            capabilities: HashSet::new(),
            system_prompt: String::new(),
            max_iterations: 10,
            timeout_secs: 60,
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn capability(mut self, cap: AgentCapability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    pub fn capabilities<I: IntoIterator<Item = AgentCapability>>(mut self, caps: I) -> Self {
        self.capabilities.extend(caps);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    pub fn build(self) -> AgentDefinition {
        AgentDefinition {
            id: self.id,
            name: self.name,
            role: self.role,
            description: self.description,
            capabilities: self.capabilities,
            system_prompt: self.system_prompt,
            max_iterations: self.max_iterations,
            timeout_secs: self.timeout_secs,
        }
    }
}

// ============================================================================
// Task Specification, Priorities & Routing
// ============================================================================

/// Strongly typed identifier for tasks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskId(String);

impl TaskId {
    #[inline]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for TaskId {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for TaskId {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for TaskId {
    #[inline]
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TaskId {
    #[inline]
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Task urgency and scheduling priority.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskPriority {
    Low = 0,
    #[default]
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl TaskPriority {
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TaskPriority::Low => "low",
            TaskPriority::Medium => "medium",
            TaskPriority::High => "high",
            TaskPriority::Critical => "critical",
        }
    }
}

/// Execution lifecycle status of an agent task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Assigned,
    InProgress,
    InReflection,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    /// Returns true if status represents terminal completion or abort.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
        )
    }

    /// Returns true if the task is actively executing or reflecting.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, TaskStatus::InProgress | TaskStatus::InReflection)
    }
}

/// Concrete unit of work dispatched through the multi-agent system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: TaskId,
    pub title: String,
    pub description: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub assigned_agent: Option<AgentId>,
    pub required_capabilities: HashSet<AgentCapability>,
    pub input_context: serde_json::Value,
    pub output_artifacts: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl AgentTask {
    /// Creates a new builder for an `AgentTask`.
    pub fn builder(id: impl Into<TaskId>, title: impl Into<String>) -> AgentTaskBuilder {
        AgentTaskBuilder::new(id, title)
    }

    /// Marks the task as assigned to an agent.
    pub fn assign(&mut self, agent: AgentId) {
        self.assigned_agent = Some(agent);
        self.status = TaskStatus::Assigned;
        self.updated_at = current_timestamp();
    }

    /// Transitions status to InProgress.
    pub fn start(&mut self) {
        self.status = TaskStatus::InProgress;
        self.updated_at = current_timestamp();
    }

    /// Transitions status to InReflection.
    pub fn reflect(&mut self) {
        self.status = TaskStatus::InReflection;
        self.updated_at = current_timestamp();
    }

    /// Completes the task with resulting output artifacts.
    pub fn complete(&mut self, artifacts: serde_json::Value) {
        self.output_artifacts = Some(artifacts);
        self.status = TaskStatus::Completed;
        self.error_message = None;
        self.updated_at = current_timestamp();
    }

    /// Fails the task with an explanatory error.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = TaskStatus::Failed;
        self.error_message = Some(reason.into());
        self.updated_at = current_timestamp();
    }
}

/// Builder for [`AgentTask`].
pub struct AgentTaskBuilder {
    id: TaskId,
    title: String,
    description: String,
    priority: TaskPriority,
    required_capabilities: HashSet<AgentCapability>,
    input_context: serde_json::Value,
}

impl AgentTaskBuilder {
    pub fn new(id: impl Into<TaskId>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            priority: TaskPriority::Medium,
            required_capabilities: HashSet::new(),
            input_context: serde_json::Value::Null,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn require_capability(mut self, cap: AgentCapability) -> Self {
        self.required_capabilities.insert(cap);
        self
    }

    pub fn input_context(mut self, ctx: serde_json::Value) -> Self {
        self.input_context = ctx;
        self
    }

    pub fn build(self) -> AgentTask {
        let now = current_timestamp();
        AgentTask {
            id: self.id,
            title: self.title,
            description: self.description,
            priority: self.priority,
            status: TaskStatus::Pending,
            assigned_agent: None,
            required_capabilities: self.required_capabilities,
            input_context: self.input_context,
            output_artifacts: None,
            error_message: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Strategy used by the TaskRouter to select an optimal agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Match explicit target agent roles.
    RoleBased,
    /// Match required capability sets to agent capability profiles.
    #[default]
    CapabilityBased,
    /// Fair-share round-robin among eligible candidates.
    RoundRobin,
    /// Direct routing prioritized by task urgency and agent tier.
    PriorityTiered,
    /// Reflection-guided routing adapting based on prior critique history.
    ReflectionGuided,
}

/// The outcome and rationale of routing a task to an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub task_id: TaskId,
    pub target_agent: AgentId,
    pub strategy: RoutingStrategy,
    pub confidence: f32,
    pub rationale: String,
    pub fallback_agents: Vec<AgentId>,
}

impl RoutingDecision {
    pub fn new(
        task_id: impl Into<TaskId>,
        target_agent: impl Into<AgentId>,
        strategy: RoutingStrategy,
        confidence: f32,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            target_agent: target_agent.into(),
            strategy,
            confidence: confidence.clamp(0.0, 1.0),
            rationale: rationale.into(),
            fallback_agents: Vec::new(),
        }
    }

    pub fn with_fallback(mut self, fallback: impl Into<AgentId>) -> Self {
        self.fallback_agents.push(fallback.into());
        self
    }
}

// ============================================================================
// Automated Tool Calling & Schemas
// ============================================================================

/// JSON schema parameter type descriptor for tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParamType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

impl fmt::Display for ToolParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ToolParamType::String => "string",
            ToolParamType::Integer => "integer",
            ToolParamType::Float => "float",
            ToolParamType::Boolean => "boolean",
            ToolParamType::Array => "array",
            ToolParamType::Object => "object",
        };
        write!(f, "{}", s)
    }
}

/// Specification of a single parameter accepted by a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: ToolParamType,
    pub description: String,
    pub required: bool,
    pub default_value: Option<serde_json::Value>,
}

impl ToolParameter {
    pub fn new(
        name: impl Into<String>,
        param_type: ToolParamType,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            description: description.into(),
            required,
            default_value: None,
        }
    }

    pub fn with_default(mut self, default_val: serde_json::Value) -> Self {
        self.default_value = Some(default_val);
        self
    }
}

/// Structured definition of a tool available for agent execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub is_deterministic: bool,
    pub timeout_ms: u64,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            is_deterministic: true,
            timeout_ms: 5000,
        }
    }

    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.is_deterministic = deterministic;
        self
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Converts this tool definition into OpenAI/MCP compatible JSON function schema.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for p in &self.parameters {
            let mut prop_obj = serde_json::Map::new();
            prop_obj.insert(
                "type".to_string(),
                serde_json::Value::String(p.param_type.to_string()),
            );
            prop_obj.insert(
                "description".to_string(),
                serde_json::Value::String(p.description.clone()),
            );
            if let Some(ref def) = p.default_value {
                prop_obj.insert("default".to_string(), def.clone());
            }
            properties.insert(p.name.clone(), serde_json::Value::Object(prop_obj));
            if p.required {
                required.push(serde_json::Value::String(p.name.clone()));
            }
        }

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    }

    /// Validates an incoming argument map against the tool schema.
    pub fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<()> {
        let obj = arguments
            .as_object()
            .ok_or_else(|| FlashStoreError::ToolError {
                tool: self.name.clone(),
                reason: "Arguments must be a valid JSON object".into(),
            })?;

        for param in &self.parameters {
            if param.required && !obj.contains_key(&param.name) {
                return Err(FlashStoreError::ToolError {
                    tool: self.name.clone(),
                    reason: format!("Missing required parameter '{}'", param.name),
                });
            }
        }

        Ok(())
    }
}

/// Structured invocation request generated by an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
        }
    }
}

/// Execution outcome of an automated tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub call_id: String,
    pub tool_name: String,
    pub success: bool,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

impl ToolOutput {
    /// Creates a successful tool execution output.
    pub fn success(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: serde_json::Value,
        execution_time_ms: u64,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            success: true,
            result,
            execution_time_ms,
            error: None,
        }
    }

    /// Creates an error tool execution output.
    pub fn error(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        reason: impl Into<String>,
        execution_time_ms: u64,
    ) -> Self {
        let err_str = reason.into();
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            success: false,
            result: serde_json::Value::Null,
            execution_time_ms,
            error: Some(err_str),
        }
    }
}

// ============================================================================
// Stateful Reflection Loops & Self-Correction
// ============================================================================

/// Final decision produced by a reflection evaluation iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReflectionVerdict {
    /// Output passes all quality, safety, and correctness checks.
    Approved,
    /// Output requires another refinement pass with suggested improvements.
    NeedsRevision,
    /// Output cannot be salvaged or failed critical security invariants.
    Rejected,
    /// Requires human intervention or higher-tier model adjudication.
    Escalated,
}

impl ReflectionVerdict {
    #[inline]
    pub fn is_approved(&self) -> bool {
        matches!(self, ReflectionVerdict::Approved)
    }

    #[inline]
    pub fn should_continue_loop(&self) -> bool {
        matches!(self, ReflectionVerdict::NeedsRevision)
    }
}

/// Multi-dimensional critique generated by the Critic agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectionCritique {
    pub dimension: String,
    pub score: f32, // 0.0 to 1.0
    pub feedback: String,
    pub actionable_improvements: Vec<String>,
}

impl ReflectionCritique {
    pub fn new(dimension: impl Into<String>, score: f32, feedback: impl Into<String>) -> Self {
        Self {
            dimension: dimension.into(),
            score: score.clamp(0.0, 1.0),
            feedback: feedback.into(),
            actionable_improvements: Vec::new(),
        }
    }

    pub fn with_improvement(mut self, improvement: impl Into<String>) -> Self {
        self.actionable_improvements.push(improvement.into());
        self
    }
}

/// A snapshot of an individual iteration in the reflection loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectionStep {
    pub iteration: usize,
    pub candidate_output: String,
    pub critiques: Vec<ReflectionCritique>,
    pub aggregate_score: f32,
    pub verdict: ReflectionVerdict,
    pub timestamp: u64,
}

impl ReflectionStep {
    pub fn new(
        iteration: usize,
        candidate_output: impl Into<String>,
        critiques: Vec<ReflectionCritique>,
        verdict: ReflectionVerdict,
    ) -> Self {
        let aggregate_score = if critiques.is_empty() {
            if verdict == ReflectionVerdict::Approved {
                1.0
            } else {
                0.0
            }
        } else {
            let sum: f32 = critiques.iter().map(|c| c.score).sum();
            sum / (critiques.len() as f32)
        };

        Self {
            iteration,
            candidate_output: candidate_output.into(),
            critiques,
            aggregate_score,
            verdict,
            timestamp: current_timestamp(),
        }
    }
}

/// Stateful history and convergence status of an ongoing reflection loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectionState {
    pub task_id: TaskId,
    pub current_iteration: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub steps: Vec<ReflectionStep>,
    pub converged: bool,
    pub final_verdict: Option<ReflectionVerdict>,
}

impl ReflectionState {
    /// Creates a new reflection state tracker for a task.
    pub fn new(task_id: impl Into<TaskId>, max_iterations: usize, threshold: f32) -> Self {
        Self {
            task_id: task_id.into(),
            current_iteration: 0,
            max_iterations,
            convergence_threshold: threshold.clamp(0.0, 1.0),
            steps: Vec::new(),
            converged: false,
            final_verdict: None,
        }
    }

    /// Records a new step and evaluates convergence.
    pub fn record_step(&mut self, step: ReflectionStep) -> ReflectionVerdict {
        self.current_iteration = step.iteration;
        let verdict = step.verdict;
        let score = step.aggregate_score;

        self.steps.push(step);

        if verdict == ReflectionVerdict::Approved || score >= self.convergence_threshold {
            self.converged = true;
            self.final_verdict = Some(ReflectionVerdict::Approved);
            return ReflectionVerdict::Approved;
        }

        if self.current_iteration >= self.max_iterations {
            self.converged = false;
            let final_v = if verdict == ReflectionVerdict::NeedsRevision {
                ReflectionVerdict::Escalated
            } else {
                verdict
            };
            self.final_verdict = Some(final_v);
            return final_v;
        }

        self.final_verdict = Some(verdict);
        verdict
    }

    /// Returns the best candidate output across all recorded steps.
    pub fn best_candidate(&self) -> Option<&str> {
        self.steps
            .iter()
            .max_by(|a, b| {
                a.aggregate_score
                    .partial_cmp(&b.aggregate_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.candidate_output.as_str())
    }
}

// ============================================================================
// Workflow State, Inter-Agent Messaging & Audits
// ============================================================================

/// Role classification for message senders within a workflow dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
    Critic,
}

/// An immutable message exchanged between agents or recorded in session context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub message_id: String,
    pub sender: AgentId,
    pub recipient: Option<AgentId>,
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_outputs: Vec<ToolOutput>,
    pub timestamp: u64,
}

impl AgentMessage {
    pub fn new(
        message_id: impl Into<String>,
        sender: impl Into<AgentId>,
        role: MessageRole,
        content: impl Into<String>,
    ) -> Self {
        Self {
            message_id: message_id.into(),
            sender: sender.into(),
            recipient: None,
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_outputs: Vec::new(),
            timestamp: current_timestamp(),
        }
    }

    pub fn to_recipient(mut self, recipient: impl Into<AgentId>) -> Self {
        self.recipient = Some(recipient.into());
        self
    }

    pub fn with_tool_call(mut self, call: ToolCall) -> Self {
        self.tool_calls.push(call);
        self
    }

    pub fn with_tool_output(mut self, output: ToolOutput) -> Self {
        self.tool_outputs.push(output);
        self
    }
}

/// Global lifecycle state of an orchestrated multi-agent workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    Initialized,
    Running,
    Paused,
    Succeeded,
    Failed,
}

/// Comprehensive state snapshot of an entire multi-agent workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowState {
    pub workflow_id: String,
    pub status: WorkflowStatus,
    pub tasks: HashMap<TaskId, AgentTask>,
    pub active_agents: HashMap<AgentId, AgentDefinition>,
    pub message_history: Vec<AgentMessage>,
    pub reflection_states: HashMap<TaskId, ReflectionState>,
    pub shared_memory: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl WorkflowState {
    pub fn new(workflow_id: impl Into<String>) -> Self {
        let now = current_timestamp();
        Self {
            workflow_id: workflow_id.into(),
            status: WorkflowStatus::Initialized,
            tasks: HashMap::new(),
            active_agents: HashMap::new(),
            message_history: Vec::new(),
            reflection_states: HashMap::new(),
            shared_memory: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn register_agent(&mut self, agent: AgentDefinition) {
        self.active_agents.insert(agent.id.clone(), agent);
        self.updated_at = current_timestamp();
    }

    pub fn add_task(&mut self, task: AgentTask) {
        self.tasks.insert(task.id.clone(), task);
        self.updated_at = current_timestamp();
    }

    pub fn post_message(&mut self, msg: AgentMessage) {
        self.message_history.push(msg);
        self.updated_at = current_timestamp();
    }

    pub fn set_shared_memory(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.shared_memory.insert(key.into(), value);
        self.updated_at = current_timestamp();
    }

    pub fn get_shared_memory(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared_memory.get(key)
    }
}

// ============================================================================
// Storage Namespace Keys for FlashStore Persistence
// ============================================================================

/// Key namespace generator and parser for persisting agent state into FlashStore.
pub struct AgentStorageKey;

impl AgentStorageKey {
    pub const PREFIX_AGENT: &'static str = "agent:def:";
    pub const PREFIX_TASK: &'static str = "agent:task:";
    pub const PREFIX_REFLECTION: &'static str = "agent:refl:";
    pub const PREFIX_MESSAGE: &'static str = "agent:msg:";

    #[inline]
    pub fn agent_key(agent_id: &AgentId) -> String {
        format!("{}{}", Self::PREFIX_AGENT, agent_id.as_str())
    }

    #[inline]
    pub fn task_key(task_id: &TaskId) -> String {
        format!("{}{}", Self::PREFIX_TASK, task_id.as_str())
    }

    #[inline]
    pub fn reflection_key(task_id: &TaskId) -> String {
        format!("{}{}", Self::PREFIX_REFLECTION, task_id.as_str())
    }

    #[inline]
    pub fn message_key(workflow_id: &str, msg_id: &str) -> String {
        format!("{}{}:{}", Self::PREFIX_MESSAGE, workflow_id, msg_id)
    }
}

// ============================================================================
// Core Swarm Traits & Contracts
// ============================================================================

/// Contract for autonomous task routing engines.
pub trait TaskRouter: Send + Sync {
    /// Evaluates an agent task against available agent definitions and returns a routing decision.
    fn route(
        &self,
        task: &AgentTask,
        available_agents: &[AgentDefinition],
    ) -> Result<RoutingDecision>;
}

/// Default implementation of [`TaskRouter`] using capability matching and priority weighting.
#[derive(Debug, Clone, Default)]
pub struct DefaultTaskRouter {
    pub default_strategy: RoutingStrategy,
}

impl DefaultTaskRouter {
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            default_strategy: strategy,
        }
    }
}

impl TaskRouter for DefaultTaskRouter {
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

        // If task requires specific capabilities, filter and score candidates
        if !task.required_capabilities.is_empty() {
            let mut candidates: Vec<(&AgentDefinition, f32)> = available_agents
                .iter()
                .map(|agent| {
                    let matched = task
                        .required_capabilities
                        .iter()
                        .filter(|c| agent.has_capability(**c))
                        .count();
                    let score = matched as f32 / task.required_capabilities.len() as f32;
                    (agent, score)
                })
                .collect();

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let (best_agent, top_score) = candidates[0];
            if top_score > 0.0 {
                let mut decision = RoutingDecision::new(
                    task.id.clone(),
                    best_agent.id.clone(),
                    RoutingStrategy::CapabilityBased,
                    top_score,
                    format!("Matched {:.0}% of required capabilities", top_score * 100.0),
                );
                for (fallback, _) in candidates.iter().skip(1).take(2) {
                    decision = decision.with_fallback(fallback.id.clone());
                }
                return Ok(decision);
            }
        }

        // Fallback: Default to first agent or round-robin
        let default_agent = &available_agents[0];
        Ok(RoutingDecision::new(
            task.id.clone(),
            default_agent.id.clone(),
            self.default_strategy,
            0.5,
            format!("Default assignment to agent {}", default_agent.id),
        ))
    }
}

/// Contract for executable tools callable by swarm agents.
pub trait Tool: Send + Sync {
    /// Returns the tool metadata and schema descriptor.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool given arguments parsed as JSON Value.
    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value>;
}

/// In-memory registry for tools callable by swarm agents.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.definition().name;
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn list_json_schemas(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|t| t.definition().to_json_schema())
            .collect()
    }

    pub fn execute(&self, call: &ToolCall) -> Result<ToolOutput> {
        let start = SystemTime::now();
        let tool = match self.get(&call.tool_name) {
            Some(t) => t,
            None => {
                let duration = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
                return Ok(ToolOutput::error(
                    call.call_id.clone(),
                    call.tool_name.clone(),
                    format!("Tool '{}' not found in registry", call.tool_name),
                    duration,
                ));
            }
        };

        // Validate arguments
        if let Err(e) = tool.definition().validate_arguments(&call.arguments) {
            let duration = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
            return Ok(ToolOutput::error(
                call.call_id.clone(),
                call.tool_name.clone(),
                e.to_string(),
                duration,
            ));
        }

        match tool.execute(&call.arguments) {
            Ok(result) => {
                let duration = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
                Ok(ToolOutput::success(
                    call.call_id.clone(),
                    call.tool_name.clone(),
                    result,
                    duration,
                ))
            }
            Err(e) => {
                let duration = start.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
                Ok(ToolOutput::error(
                    call.call_id.clone(),
                    call.tool_name.clone(),
                    e.to_string(),
                    duration,
                ))
            }
        }
    }
}

/// Contract for critique generators in stateful reflection loops.
pub trait Reflector: Send + Sync {
    /// Critiques an agent's candidate output for a given task and iteration.
    fn evaluate(
        &self,
        task: &AgentTask,
        candidate_output: &str,
        iteration: usize,
    ) -> Result<Vec<ReflectionCritique>>;
}

/// Deterministic rule-based reflector for fast-loop evaluation.
pub struct RuleBasedReflector {
    pub min_length: usize,
    pub prohibited_substrings: Vec<String>,
}

impl RuleBasedReflector {
    pub fn new(min_length: usize, prohibited: Vec<String>) -> Self {
        Self {
            min_length,
            prohibited_substrings: prohibited,
        }
    }
}

impl Default for RuleBasedReflector {
    fn default() -> Self {
        Self {
            min_length: 10,
            prohibited_substrings: vec!["TODO".into(), "FIXME".into(), "unimplemented!()".into()],
        }
    }
}

impl Reflector for RuleBasedReflector {
    fn evaluate(
        &self,
        _task: &AgentTask,
        candidate_output: &str,
        _iteration: usize,
    ) -> Result<Vec<ReflectionCritique>> {
        let mut critiques = Vec::new();

        // Check length
        if candidate_output.len() < self.min_length {
            critiques.push(
                ReflectionCritique::new("completeness", 0.4, "Output is too brief")
                    .with_improvement(format!(
                        "Provide at least {} characters of substance",
                        self.min_length
                    )),
            );
        } else {
            critiques.push(ReflectionCritique::new(
                "completeness",
                1.0,
                "Length satisfies threshold",
            ));
        }

        // Check prohibited keywords
        let mut found_prohibited = Vec::new();
        for pattern in &self.prohibited_substrings {
            if candidate_output.contains(pattern) {
                found_prohibited.push(pattern.clone());
            }
        }

        if !found_prohibited.is_empty() {
            critiques.push(
                ReflectionCritique::new(
                    "production_ready",
                    0.2,
                    format!("Contains placeholders: {:?}", found_prohibited),
                )
                .with_improvement("Replace all placeholder TODOs with concrete implementations"),
            );
        } else {
            critiques.push(ReflectionCritique::new(
                "production_ready",
                1.0,
                "Zero placeholders detected",
            ));
        }

        Ok(critiques)
    }
}

// ============================================================================
// Internal Utilities
// ============================================================================

#[inline]
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_definition_builder() {
        let agent = AgentDefinition::builder("agent-007", AgentRole::Coder)
            .name("James Code")
            .description("Lead developer agent")
            .capability(AgentCapability::CodeExecution)
            .capability(AgentCapability::ToolCalling)
            .system_prompt("You write elegant Rust code.")
            .max_iterations(5)
            .timeout_secs(120)
            .build();

        assert_eq!(agent.id.as_str(), "agent-007");
        assert_eq!(agent.role, AgentRole::Coder);
        assert!(agent.has_capability(AgentCapability::CodeExecution));
        assert!(agent.has_capability(AgentCapability::ToolCalling));
        assert!(!agent.has_capability(AgentCapability::StorageOps));
        assert_eq!(agent.max_iterations, 5);
        assert_eq!(agent.timeout_secs, 120);
    }

    #[test]
    fn test_task_lifecycle_transitions() {
        let mut task = AgentTask::builder("task-101", "Implement WriteBatch")
            .description("Add atomic multi-key batch writes")
            .priority(TaskPriority::High)
            .require_capability(AgentCapability::CodeExecution)
            .input_context(serde_json::json!({ "feature": "write_batch" }))
            .build();

        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.priority, TaskPriority::High);

        task.assign(AgentId::new("coder-1"));
        assert_eq!(task.status, TaskStatus::Assigned);
        assert_eq!(task.assigned_agent.as_deref(), Some("coder-1"));

        task.start();
        assert_eq!(task.status, TaskStatus::InProgress);
        assert!(task.status.is_active());

        task.reflect();
        assert_eq!(task.status, TaskStatus::InReflection);

        task.complete(serde_json::json!({ "diff_lines": 42 }));
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.status.is_terminal());
        assert!(task.output_artifacts.is_some());
    }

    #[test]
    fn test_tool_definition_json_schema() {
        let tool = ToolDefinition::new("get_kv", "Retrieves a value from FlashStore")
            .with_parameter(ToolParameter::new(
                "key",
                ToolParamType::String,
                "Key to look up",
                true,
            ))
            .with_parameter(
                ToolParameter::new(
                    "include_metadata",
                    ToolParamType::Boolean,
                    "Whether to return timestamps and version",
                    false,
                )
                .with_default(serde_json::json!(false)),
            )
            .with_deterministic(true)
            .with_timeout_ms(2500);

        let schema = tool.to_json_schema();
        assert_eq!(schema["name"], "get_kv");
        assert_eq!(schema["parameters"]["type"], "object");
        assert!(schema["parameters"]["properties"]["key"].is_object());
        assert_eq!(schema["parameters"]["required"][0], "key");

        // Validate arguments
        assert!(tool
            .validate_arguments(&serde_json::json!({"key": "user_1"}))
            .is_ok());
        assert!(tool.validate_arguments(&serde_json::json!({})).is_err());
    }

    #[test]
    fn test_tool_registry_and_dispatch() {
        struct EchoTool;
        impl Tool for EchoTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new("echo", "Echoes input text").with_parameter(ToolParameter::new(
                    "msg",
                    ToolParamType::String,
                    "message",
                    true,
                ))
            }
            fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
                let msg = arguments["msg"].as_str().unwrap_or("");
                Ok(serde_json::json!({ "echo": msg }))
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));

        assert!(registry.get("echo").is_some());
        assert!(registry.get("unknown").is_none());

        // Successful call
        let call = ToolCall::new("c-1", "echo", serde_json::json!({"msg": "hello"}));
        let output = registry.execute(&call).unwrap();
        assert!(output.success);
        assert_eq!(output.result["echo"], "hello");

        // Missing param validation error
        let call_err = ToolCall::new("c-2", "echo", serde_json::json!({}));
        let output_err = registry.execute(&call_err).unwrap();
        assert!(!output_err.success);
        assert!(output_err.error.is_some());
    }

    #[test]
    fn test_default_task_router() {
        let router = DefaultTaskRouter::default();

        let agent1 = AgentDefinition::builder("agent-coder", AgentRole::Coder)
            .capability(AgentCapability::CodeExecution)
            .build();

        let agent2 = AgentDefinition::builder("agent-db", AgentRole::MemoryAgent)
            .capability(AgentCapability::StorageOps)
            .capability(AgentCapability::MemoryRetrieval)
            .build();

        let agents = vec![agent1, agent2];

        // Route storage task
        let storage_task = AgentTask::builder("t-1", "Flush SSTable")
            .require_capability(AgentCapability::StorageOps)
            .build();

        let decision = router.route(&storage_task, &agents).unwrap();
        assert_eq!(decision.target_agent.as_str(), "agent-db");
        assert_eq!(decision.strategy, RoutingStrategy::CapabilityBased);
    }

    #[test]
    fn test_rule_based_reflector() {
        let reflector = RuleBasedReflector::default();
        let task = AgentTask::builder("t-1", "Implement logic").build();

        // With TODO
        let bad_output = "fn foo() { // TODO: implement\n }";
        let critiques = reflector.evaluate(&task, bad_output, 1).unwrap();
        assert!(critiques.iter().any(|c| c.score < 0.5));

        // Clean output
        let good_output = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let good_critiques = reflector.evaluate(&task, good_output, 1).unwrap();
        assert!(good_critiques.iter().all(|c| c.score == 1.0));
    }

    #[test]
    fn test_storage_key_generation() {
        let agent_id = AgentId::new("agent-1");
        let task_id = TaskId::new("task-42");

        assert_eq!(AgentStorageKey::agent_key(&agent_id), "agent:def:agent-1");
        assert_eq!(AgentStorageKey::task_key(&task_id), "agent:task:task-42");
        assert_eq!(
            AgentStorageKey::reflection_key(&task_id),
            "agent:refl:task-42"
        );
    }
}
