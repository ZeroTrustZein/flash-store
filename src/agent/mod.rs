//! # Multi-Agent Workflow Engine
//!
//! Autonomous multi-agent task routing, stateful reflection loops, and automated tool-calling.

pub mod engine;
pub mod reflection;
pub mod router;
pub mod tools;
pub mod types;

pub use engine::{AgentEngine, TaskExecutionReport, TaskGraph, WorkflowSummary};
pub use reflection::{
    compute_aggregate_score, format_revision_prompt, verify_critical_invariants,
    CompositeReflector, ConvergenceConfig, ConvergenceEvaluator, EvaluationCriterion,
    HeuristicReflector, ReflectionEngine,
};
pub use router::{
    score_capability_match, score_priority_tier, score_role_affinity, score_workload,
    AdaptiveRoutingWeights, AdaptiveTaskRouter, LoadBalancingRouter, PriorityPreemptiveRouter,
};
pub use tools::{
    create_standard_tool_registry, CalculatorTool, EchoTool, KvDeleteTool, KvGetTool, KvPutTool,
    KvScanTool, ToolDispatcher,
};
pub use types::{
    AgentCapability, AgentDefinition, AgentDefinitionBuilder, AgentId, AgentMessage, AgentRole,
    AgentStorageKey, AgentTask, AgentTaskBuilder, DefaultTaskRouter, MessageRole,
    ReflectionCritique, ReflectionState, ReflectionStep, ReflectionVerdict, Reflector,
    RoutingDecision, RoutingStrategy, RuleBasedReflector, TaskId, TaskPriority, TaskRouter,
    TaskStatus, Tool, ToolCall, ToolDefinition, ToolOutput, ToolParamType, ToolParameter,
    ToolRegistry, WorkflowState, WorkflowStatus,
};
