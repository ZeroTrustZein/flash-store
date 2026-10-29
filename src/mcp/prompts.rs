//! # Model Context Protocol (MCP) Prompts Subsystem
//!
//! Provides structured prompt definitions, argument schemas, dynamic template evaluation,
//! and prompt registry management per the Model Context Protocol specification.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{FlashStoreError, Result};
use crate::mcp::types::McpContent;

/// Definition of an argument accepted by a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptArgument {
    /// Name of the argument.
    pub name: String,
    /// Human-readable description of the argument purpose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether this argument must be provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
}

impl McpPromptArgument {
    /// Creates a new required prompt argument.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            required: Some(true),
        }
    }

    /// Creates a new optional prompt argument.
    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            required: Some(false),
        }
    }
}

/// Definition of a prompt exposed via the MCP protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptDefinition {
    /// Unique identifier / name of the prompt.
    pub name: String,
    /// Human-readable description of what the prompt accomplishes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// List of arguments accepted by this prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<McpPromptArgument>>,
}

impl McpPromptDefinition {
    /// Creates a new prompt definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        arguments: Option<Vec<McpPromptArgument>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            arguments,
        }
    }
}

/// A single message in a prompt conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptMessage {
    /// Message role (typically "user" or "assistant").
    pub role: String,
    /// Message content.
    pub content: McpContent,
}

impl McpPromptMessage {
    /// Creates a user message containing text.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: McpContent::text(text),
        }
    }

    /// Creates an assistant message containing text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: McpContent::text(text),
        }
    }
}

/// Result returned by `prompts/list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpListPromptsResult {
    /// Available prompt definitions.
    pub prompts: Vec<McpPromptDefinition>,
    /// Optional cursor for pagination.
    #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Result returned by `prompts/get`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpGetPromptResult {
    /// Optional description of the rendered prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Formatted messages comprising the prompt.
    pub messages: Vec<McpPromptMessage>,
}

/// Parameters for calling `prompts/get`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpGetPromptParams {
    /// Name of the prompt to retrieve.
    pub name: String,
    /// Arguments provided for template interpolation.
    #[serde(default)]
    pub arguments: Option<HashMap<String, String>>,
}

/// Trait implemented by dynamic prompt handlers.
pub trait McpPromptHandler: Send + Sync {
    /// Returns the prompt schema definition.
    fn definition(&self) -> McpPromptDefinition;

    /// Renders the prompt given caller-supplied arguments.
    fn render(&self, arguments: Option<&HashMap<String, String>>) -> Result<McpGetPromptResult>;
}

/// Built-in prompt handler for diagnosing and troubleshooting FlashStore KV operations.
pub struct FlashTroubleshootPrompt;

impl McpPromptHandler for FlashTroubleshootPrompt {
    fn definition(&self) -> McpPromptDefinition {
        McpPromptDefinition::new(
            "flash_troubleshoot",
            "Generates an expert diagnostic analysis and mitigation plan for FlashStore LSM-tree state.",
            Some(vec![
                McpPromptArgument::required("issue", "Observed issue or error description"),
                McpPromptArgument::optional("component", "Subsystem component: memtable, wal, sstable, or compaction"),
            ]),
        )
    }

    fn render(&self, arguments: Option<&HashMap<String, String>>) -> Result<McpGetPromptResult> {
        let issue = arguments
            .and_then(|a| a.get("issue"))
            .map(|s| s.as_str())
            .unwrap_or("Unknown storage issue");
        let component = arguments
            .and_then(|a| a.get("component"))
            .map(|s| s.as_str())
            .unwrap_or("all");

        let prompt_text = format!(
            "You are a systems engineer diagnosing FlashStore, an embedded LSM-tree engine.\n\
             Component: {}\n\
             Observed issue: {}\n\n\
             Please analyze the root cause, identify whether WAL replay, Bloom filter saturation, or compaction write stall is implicated, and provide step-by-step remediation.",
            component, issue
        );

        Ok(McpGetPromptResult {
            description: Some(format!("Troubleshooting prompt for '{}'", issue)),
            messages: vec![McpPromptMessage::user(prompt_text)],
        })
    }
}

/// Built-in prompt handler for RAG synthesis over retrieved context chunks.
pub struct FlashRagSynthesizePrompt;

impl McpPromptHandler for FlashRagSynthesizePrompt {
    fn definition(&self) -> McpPromptDefinition {
        McpPromptDefinition::new(
            "flash_rag_synthesize",
            "Synthesizes an authoritative answer to a user query using FlashStore RAG context chunks.",
            Some(vec![
                McpPromptArgument::required("query", "The original user question"),
                McpPromptArgument::required("context", "Retrieved document chunks and citations"),
            ]),
        )
    }

    fn render(&self, arguments: Option<&HashMap<String, String>>) -> Result<McpGetPromptResult> {
        let query = arguments
            .and_then(|a| a.get("query"))
            .map(|s| s.as_str())
            .ok_or_else(|| {
                FlashStoreError::InvalidArgument("Missing required 'query' argument".into())
            })?;
        let context = arguments
            .and_then(|a| a.get("context"))
            .map(|s| s.as_str())
            .unwrap_or("");

        let prompt_text = format!(
            "Answer the query accurately based strictly on the provided context.\n\n\
             === CONTEXT ===\n{}\n\n\
             === QUERY ===\n{}\n\n\
             Include inline citations where appropriate.",
            context, query
        );

        Ok(McpGetPromptResult {
            description: Some(format!("RAG synthesis prompt for query: '{}'", query)),
            messages: vec![McpPromptMessage::user(prompt_text)],
        })
    }
}

/// Built-in prompt handler for orchestrating multi-agent tasks.
pub struct FlashAgentOrchestratePrompt;

impl McpPromptHandler for FlashAgentOrchestratePrompt {
    fn definition(&self) -> McpPromptDefinition {
        McpPromptDefinition::new(
            "flash_agent_orchestrate",
            "Decomposes a complex objective into an acyclic task graph across autonomous agent roles.",
            Some(vec![
                McpPromptArgument::required("goal", "High-level goal or problem statement"),
                McpPromptArgument::optional("roles", "Comma-separated list of agent roles available"),
            ]),
        )
    }

    fn render(&self, arguments: Option<&HashMap<String, String>>) -> Result<McpGetPromptResult> {
        let goal = arguments
            .and_then(|a| a.get("goal"))
            .map(|s| s.as_str())
            .ok_or_else(|| {
                FlashStoreError::InvalidArgument("Missing required 'goal' argument".into())
            })?;
        let roles = arguments
            .and_then(|a| a.get("roles"))
            .map(|s| s.as_str())
            .unwrap_or("Coder, Reviewer, Tester, Architect");

        let prompt_text = format!(
            "You are the Lead Swarm Orchestrator for FlashStore agents.\n\
             Target Goal: {}\n\
             Available Roles: {}\n\n\
             Decompose this goal into discrete subtasks with explicit dependencies, required capabilities, and convergence criteria.",
            goal, roles
        );

        Ok(McpGetPromptResult {
            description: Some("Multi-agent task decomposition prompt".to_string()),
            messages: vec![McpPromptMessage::user(prompt_text)],
        })
    }
}

/// Thread-safe registry storing and dispatching MCP prompts.
#[derive(Default)]
pub struct McpPromptRegistry {
    prompts: RwLock<HashMap<String, Arc<dyn McpPromptHandler>>>,
}

impl McpPromptRegistry {
    /// Creates a new empty prompt registry.
    pub fn new() -> Self {
        Self {
            prompts: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a prompt registry pre-populated with standard FlashStore prompt handlers.
    pub fn with_standard_prompts() -> Self {
        let registry = Self::new();
        registry.register(Arc::new(FlashTroubleshootPrompt));
        registry.register(Arc::new(FlashRagSynthesizePrompt));
        registry.register(Arc::new(FlashAgentOrchestratePrompt));
        registry
    }

    /// Registers a prompt handler into the registry.
    pub fn register(&self, handler: Arc<dyn McpPromptHandler>) {
        let def = handler.definition();
        self.prompts.write().insert(def.name, handler);
    }

    /// Checks if a prompt with the given name exists in the registry.
    pub fn contains(&self, name: &str) -> bool {
        self.prompts.read().contains_key(name)
    }

    /// Returns the number of registered prompts.
    pub fn count(&self) -> usize {
        self.prompts.read().len()
    }

    /// Returns all registered prompt definitions.
    pub fn list(&self) -> Vec<McpPromptDefinition> {
        let guard = self.prompts.read();
        let mut list: Vec<McpPromptDefinition> = guard.values().map(|h| h.definition()).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    /// Renders a prompt by name with the given arguments.
    pub fn render(
        &self,
        name: &str,
        arguments: Option<&HashMap<String, String>>,
    ) -> Result<McpGetPromptResult> {
        let guard = self.prompts.read();
        let handler = guard
            .get(name)
            .ok_or_else(|| FlashStoreError::InvalidArgument(format!("Prompt not found: {}", name)))?;
        handler.render(arguments)
    }
}
