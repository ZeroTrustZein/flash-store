//! # Storage Adapter for Autonomous Multi-Agent Swarms
//!
//! Provides persistence, hydration, batch writes, and state recovery between the
//! in-memory agent workflow subsystem and the persistent FlashStore LSM-tree engine.

use std::sync::Arc;

use bytes::Bytes;
use serde_json;

use crate::agent::types::{
    AgentDefinition, AgentId, AgentMessage, AgentStorageKey, AgentTask, ReflectionCritique, TaskId,
    TaskStatus, WorkflowState,
};
use crate::batch::WriteBatch;
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};

/// Storage adapter providing atomic persistence, recovery, and hydration
/// between the in-memory multi-agent workflow subsystem and FlashStore.
#[derive(Clone)]
pub struct AgentStoreAdapter {
    store: Arc<FlashStore>,
}

impl AgentStoreAdapter {
    /// Creates a new `AgentStoreAdapter` wrapping the given `FlashStore` engine.
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying `FlashStore` engine.
    pub fn store(&self) -> &Arc<FlashStore> {
        &self.store
    }

    // ========================================================================
    // Agent Definitions
    // ========================================================================

    /// Persists an agent definition to storage.
    pub fn persist_agent(&self, agent: &AgentDefinition) -> Result<()> {
        let key = AgentStorageKey::agent_key(&agent.id);
        let val = serde_json::to_vec(agent)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize agent: {}", e)))?;
        self.store.put(key.into_bytes(), val)
    }

    /// Persists a batch of agent definitions atomically.
    pub fn persist_agents_batch(&self, agents: &[AgentDefinition]) -> Result<()> {
        if agents.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new();
        for agent in agents {
            let key = AgentStorageKey::agent_key(&agent.id);
            let val = serde_json::to_vec(agent)
                .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize agent: {}", e)))?;
            batch.put(key.into_bytes(), val);
        }
        self.store.write_batch(batch)
    }

    /// Loads an agent definition by its ID.
    pub fn load_agent(&self, agent_id: &AgentId) -> Result<Option<AgentDefinition>> {
        let key = AgentStorageKey::agent_key(agent_id);
        match self.store.get(key.as_bytes())? {
            Some(bytes) => {
                let agent: AgentDefinition = serde_json::from_slice(&bytes).map_err(|e| {
                    FlashStoreError::Agent(format!("Failed to deserialize agent: {}", e))
                })?;
                Ok(Some(agent))
            }
            None => Ok(None),
        }
    }

    /// Lists all persisted agent definitions.
    pub fn list_agents(&self) -> Result<Vec<AgentDefinition>> {
        let prefix = AgentStorageKey::PREFIX_AGENT.as_bytes();
        let entries = self
            .store
            .scan(Some(Bytes::copy_from_slice(prefix)), None)?;

        let mut agents = Vec::new();
        for (k, v) in entries {
            if !k.starts_with(prefix) {
                break;
            }
            let agent: AgentDefinition = serde_json::from_slice(&v).map_err(|e| {
                FlashStoreError::Agent(format!("Failed to deserialize agent: {}", e))
            })?;
            agents.push(agent);
        }
        Ok(agents)
    }

    /// Deletes an agent definition from storage. Returns `true` if it existed.
    pub fn delete_agent(&self, agent_id: &AgentId) -> Result<bool> {
        let key = AgentStorageKey::agent_key(agent_id);
        let exists = self.store.get(key.as_bytes())?.is_some();
        if exists {
            self.store.delete(key.into_bytes())?;
        }
        Ok(exists)
    }

    // ========================================================================
    // Tasks
    // ========================================================================

    /// Persists a single agent task to storage.
    pub fn persist_task(&self, task: &AgentTask) -> Result<()> {
        let key = AgentStorageKey::task_key(&task.id);
        let val = serde_json::to_vec(task)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize task: {}", e)))?;
        self.store.put(key.into_bytes(), val)
    }

    /// Persists multiple tasks in a single atomic batch.
    pub fn persist_tasks_batch(&self, tasks: &[AgentTask]) -> Result<()> {
        if tasks.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new();
        for task in tasks {
            let key = AgentStorageKey::task_key(&task.id);
            let val = serde_json::to_vec(task)
                .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize task: {}", e)))?;
            batch.put(key.into_bytes(), val);
        }
        self.store.write_batch(batch)
    }

    /// Loads a task by its ID.
    pub fn load_task(&self, task_id: &TaskId) -> Result<Option<AgentTask>> {
        let key = AgentStorageKey::task_key(task_id);
        match self.store.get(key.as_bytes())? {
            Some(bytes) => {
                let task: AgentTask = serde_json::from_slice(&bytes).map_err(|e| {
                    FlashStoreError::Agent(format!("Failed to deserialize task: {}", e))
                })?;
                Ok(Some(task))
            }
            None => Ok(None),
        }
    }

    /// Lists all persisted tasks.
    pub fn list_tasks(&self) -> Result<Vec<AgentTask>> {
        let prefix = AgentStorageKey::PREFIX_TASK.as_bytes();
        let entries = self
            .store
            .scan(Some(Bytes::copy_from_slice(prefix)), None)?;

        let mut tasks = Vec::new();
        for (k, v) in entries {
            if !k.starts_with(prefix) {
                break;
            }
            let task: AgentTask = serde_json::from_slice(&v).map_err(|e| {
                FlashStoreError::Agent(format!("Failed to deserialize task: {}", e))
            })?;
            tasks.push(task);
        }
        Ok(tasks)
    }

    /// Lists tasks filtered by their status.
    pub fn list_tasks_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let tasks = self.list_tasks()?;
        Ok(tasks.into_iter().filter(|t| t.status == status).collect())
    }

    /// Deletes a task from storage. Returns `true` if it existed.
    pub fn delete_task(&self, task_id: &TaskId) -> Result<bool> {
        let key = AgentStorageKey::task_key(task_id);
        let exists = self.store.get(key.as_bytes())?.is_some();
        if exists {
            self.store.delete(key.into_bytes())?;
        }
        Ok(exists)
    }

    // ========================================================================
    // Reflection Critiques
    // ========================================================================

    /// Persists a reflection critique for a task.
    pub fn persist_reflection_critique(
        &self,
        task_id: &TaskId,
        critique: &ReflectionCritique,
    ) -> Result<()> {
        let key = AgentStorageKey::reflection_key(task_id);
        let val = serde_json::to_vec(critique).map_err(|e| {
            FlashStoreError::Agent(format!("Failed to serialize reflection critique: {}", e))
        })?;
        self.store.put(key.into_bytes(), val)
    }

    /// Loads a reflection critique for a task.
    pub fn load_reflection_critique(&self, task_id: &TaskId) -> Result<Option<ReflectionCritique>> {
        let key = AgentStorageKey::reflection_key(task_id);
        match self.store.get(key.as_bytes())? {
            Some(bytes) => {
                let critique: ReflectionCritique = serde_json::from_slice(&bytes).map_err(|e| {
                    FlashStoreError::Agent(format!("Failed to deserialize critique: {}", e))
                })?;
                Ok(Some(critique))
            }
            None => Ok(None),
        }
    }

    // ========================================================================
    // Inter-Agent Messages
    // ========================================================================

    /// Persists an inter-agent message within a workflow namespace.
    pub fn persist_message(&self, workflow_id: &str, message: &AgentMessage) -> Result<()> {
        let key = AgentStorageKey::message_key(workflow_id, &message.message_id);
        let val = serde_json::to_vec(message)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize message: {}", e)))?;
        self.store.put(key.into_bytes(), val)
    }

    /// Loads all messages for a specific workflow, sorted in chronological order.
    pub fn load_messages(&self, workflow_id: &str) -> Result<Vec<AgentMessage>> {
        let prefix = format!("{}{}:", AgentStorageKey::PREFIX_MESSAGE, workflow_id);
        let prefix_bytes = prefix.as_bytes();
        let entries = self
            .store
            .scan(Some(Bytes::copy_from_slice(prefix_bytes)), None)?;

        let mut messages = Vec::new();
        for (k, v) in entries {
            if !k.starts_with(prefix_bytes) {
                break;
            }
            let msg: AgentMessage = serde_json::from_slice(&v).map_err(|e| {
                FlashStoreError::Agent(format!("Failed to deserialize message: {}", e))
            })?;
            messages.push(msg);
        }
        messages.sort_by_key(|m| m.timestamp);
        Ok(messages)
    }

    // ========================================================================
    // Workflow State Checkpoints
    // ========================================================================

    /// Persists a snapshot/checkpoint of the workflow state.
    pub fn persist_workflow_state(&self, state: &WorkflowState) -> Result<()> {
        let key = AgentStorageKey::workflow_key(&state.workflow_id);
        let val = serde_json::to_vec(state).map_err(|e| {
            FlashStoreError::Agent(format!("Failed to serialize workflow state: {}", e))
        })?;
        self.store.put(key.into_bytes(), val)
    }

    /// Loads a workflow checkpoint by workflow ID.
    pub fn load_workflow_state(&self, workflow_id: &str) -> Result<Option<WorkflowState>> {
        let key = AgentStorageKey::workflow_key(workflow_id);
        match self.store.get(key.as_bytes())? {
            Some(bytes) => {
                let state: WorkflowState = serde_json::from_slice(&bytes).map_err(|e| {
                    FlashStoreError::Agent(format!("Failed to deserialize workflow state: {}", e))
                })?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    /// Lists all stored workflow checkpoints.
    pub fn list_workflows(&self) -> Result<Vec<WorkflowState>> {
        let prefix = AgentStorageKey::PREFIX_WORKFLOW.as_bytes();
        let entries = self
            .store
            .scan(Some(Bytes::copy_from_slice(prefix)), None)?;

        let mut workflows = Vec::new();
        for (k, v) in entries {
            if !k.starts_with(prefix) {
                break;
            }
            let wf: WorkflowState = serde_json::from_slice(&v).map_err(|e| {
                FlashStoreError::Agent(format!("Failed to deserialize workflow state: {}", e))
            })?;
            workflows.push(wf);
        }
        Ok(workflows)
    }

    /// Deletes a workflow and all associated state checkpoints.
    pub fn delete_workflow_state(&self, workflow_id: &str) -> Result<bool> {
        let key = AgentStorageKey::workflow_key(workflow_id);
        let exists = self.store.get(key.as_bytes())?.is_some();
        if exists {
            self.store.delete(key.into_bytes())?;
        }
        Ok(exists)
    }

    /// Checkpoints an entire workflow atomically: state, all active tasks, and messages.
    pub fn checkpoint_full_workflow(&self, state: &WorkflowState) -> Result<()> {
        let mut batch = WriteBatch::new();

        // 1. Workflow state
        let wf_key = AgentStorageKey::workflow_key(&state.workflow_id);
        let wf_val = serde_json::to_vec(state).map_err(|e| {
            FlashStoreError::Agent(format!("Failed to serialize workflow state: {}", e))
        })?;
        batch.put(wf_key.into_bytes(), wf_val);

        // 2. All tasks in workflow
        for task in state.tasks.values() {
            let task_key = AgentStorageKey::task_key(&task.id);
            let task_val = serde_json::to_vec(task)
                .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize task: {}", e)))?;
            batch.put(task_key.into_bytes(), task_val);
        }

        // 3. All messages in workflow
        for msg in &state.message_history {
            let msg_key = AgentStorageKey::message_key(&state.workflow_id, &msg.message_id);
            let msg_val = serde_json::to_vec(msg).map_err(|e| {
                FlashStoreError::Agent(format!("Failed to serialize message: {}", e))
            })?;
            batch.put(msg_key.into_bytes(), msg_val);
        }

        self.store.write_batch(batch)
    }
}
