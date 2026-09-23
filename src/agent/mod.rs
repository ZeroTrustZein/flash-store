//! # Multi-Agent Workflow Engine
//!
//! Autonomous multi-agent task routing, stateful reflection loops, and automated tool-calling.

pub mod types;

pub use types::{
    AgentCapability, AgentDefinition, AgentDefinitionBuilder, AgentId, AgentMessage, AgentRole,
    AgentStorageKey, AgentTask, AgentTaskBuilder, DefaultTaskRouter, MessageRole,
    ReflectionCritique, ReflectionState, ReflectionStep, ReflectionVerdict, Reflector,
    RoutingDecision, RoutingStrategy, RuleBasedReflector, TaskId, TaskPriority, TaskRouter,
    TaskStatus, Tool, ToolCall, ToolDefinition, ToolOutput, ToolParamType, ToolParameter,
    ToolRegistry, WorkflowState, WorkflowStatus,
};
