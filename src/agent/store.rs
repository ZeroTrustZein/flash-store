//! # Storage Adapter for Autonomous Multi-Agent Swarms
//!
//! Provides persistence, hydration, batch writes, and state recovery between the
//! in-memory agent workflow subsystem and the persistent FlashStore LSM-tree engine.

use std::sync::Arc;

use bytes::Bytes;
use serde::{de::DeserializeOwned, Serialize};

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
    // Generic Storage Helpers
    // ========================================================================

    fn serialize_val<T: Serialize>(entity: &str, val: &T) -> Result<Vec<u8>> {
        serde_json::to_vec(val)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize {}: {}", entity, e)))
    }

    fn deserialize_val<T: DeserializeOwned>(entity: &str, bytes: &[u8]) -> Result<T> {
        serde_json::from_slice(bytes)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to deserialize {}: {}", entity, e)))
    }

    fn put_entity<T: Serialize>(&self, entity: &str, key: String, val: &T) -> Result<()> {
        let bytes = Self::serialize_val(entity, val)?;
        self.store.put(key.into_bytes(), bytes)
    }

    fn get_entity<T: DeserializeOwned>(&self, entity: &str, key: &str) -> Result<Option<T>> {
        match self.store.get(key.as_bytes())? {
            Some(bytes) => Ok(Some(Self::deserialize_val(entity, &bytes)?)),
            None => Ok(None),
        }
    }

    fn scan_entities<T: DeserializeOwned>(&self, entity: &str, prefix: &str) -> Result<Vec<T>> {
        let prefix_bytes = prefix.as_bytes();
        let entries = self
            .store
            .scan(Some(Bytes::copy_from_slice(prefix_bytes)), None)?;

        let mut results = Vec::new();
        for (k, v) in entries {
            if !k.starts_with(prefix_bytes) {
                break;
            }
            results.push(Self::deserialize_val(entity, &v)?);
        }
        Ok(results)
    }

    fn delete_entity(&self, key: &str) -> Result<bool> {
        let exists = self.store.get(key.as_bytes())?.is_some();
        if exists {
            self.store.delete(key.as_bytes())?;
        }
        Ok(exists)
    }

    // ========================================================================
    // Agent Definitions
    // ========================================================================

    /// Persists an agent definition to storage.
    pub fn persist_agent(&self, agent: &AgentDefinition) -> Result<()> {
        self.put_entity("agent", AgentStorageKey::agent_key(&agent.id), agent)
    }

    /// Persists a batch of agent definitions atomically.
    pub fn persist_agents_batch(&self, agents: &[AgentDefinition]) -> Result<()> {
        if agents.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new();
        for agent in agents {
            let key = AgentStorageKey::agent_key(&agent.id);
            let val = Self::serialize_val("agent", agent)?;
            batch.put(key.into_bytes(), val);
        }
        self.store.write_batch(batch)
    }

    /// Loads an agent definition by its ID.
    pub fn load_agent(&self, agent_id: &AgentId) -> Result<Option<AgentDefinition>> {
        self.get_entity("agent", &AgentStorageKey::agent_key(agent_id))
    }

    /// Lists all persisted agent definitions.
    pub fn list_agents(&self) -> Result<Vec<AgentDefinition>> {
        self.scan_entities("agent", AgentStorageKey::PREFIX_AGENT)
    }

    /// Deletes an agent definition from storage. Returns `true` if it existed.
    pub fn delete_agent(&self, agent_id: &AgentId) -> Result<bool> {
        self.delete_entity(&AgentStorageKey::agent_key(agent_id))
    }

    // ========================================================================
    // Tasks
    // ========================================================================

    /// Persists a single agent task to storage.
    pub fn persist_task(&self, task: &AgentTask) -> Result<()> {
        self.put_entity("task", AgentStorageKey::task_key(&task.id), task)
    }

    /// Persists multiple tasks in a single atomic batch.
    pub fn persist_tasks_batch(&self, tasks: &[AgentTask]) -> Result<()> {
        if tasks.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new();
        for task in tasks {
            let key = AgentStorageKey::task_key(&task.id);
            let val = Self::serialize_val("task", task)?;
            batch.put(key.into_bytes(), val);
        }
        self.store.write_batch(batch)
    }

    /// Loads a task by its ID.
    pub fn load_task(&self, task_id: &TaskId) -> Result<Option<AgentTask>> {
        self.get_entity("task", &AgentStorageKey::task_key(task_id))
    }

    /// Lists all persisted tasks.
    pub fn list_tasks(&self) -> Result<Vec<AgentTask>> {
        self.scan_entities("task", AgentStorageKey::PREFIX_TASK)
    }

    /// Lists tasks filtered by their status.
    pub fn list_tasks_by_status(&self, status: TaskStatus) -> Result<Vec<AgentTask>> {
        let tasks = self.list_tasks()?;
        Ok(tasks.into_iter().filter(|t| t.status == status).collect())
    }

    /// Deletes a task from storage. Returns `true` if it existed.
    pub fn delete_task(&self, task_id: &TaskId) -> Result<bool> {
        self.delete_entity(&AgentStorageKey::task_key(task_id))
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
        self.put_entity(
            "reflection critique",
            AgentStorageKey::reflection_key(task_id),
            critique,
        )
    }

    /// Loads a reflection critique for a task.
    pub fn load_reflection_critique(&self, task_id: &TaskId) -> Result<Option<ReflectionCritique>> {
        self.get_entity(
            "reflection critique",
            &AgentStorageKey::reflection_key(task_id),
        )
    }

    // ========================================================================
    // Inter-Agent Messages
    // ========================================================================

    /// Persists an inter-agent message within a workflow namespace.
    pub fn persist_message(&self, workflow_id: &str, message: &AgentMessage) -> Result<()> {
        self.put_entity(
            "message",
            AgentStorageKey::message_key(workflow_id, &message.message_id),
            message,
        )
    }

    /// Loads all messages for a specific workflow, sorted in chronological order.
    pub fn load_messages(&self, workflow_id: &str) -> Result<Vec<AgentMessage>> {
        let prefix = format!("{}{}:", AgentStorageKey::PREFIX_MESSAGE, workflow_id);
        let mut messages: Vec<AgentMessage> = self.scan_entities("message", &prefix)?;
        messages.sort_by_key(|m| m.timestamp);
        Ok(messages)
    }

    // ========================================================================
    // Workflow State Checkpoints
    // ========================================================================

    /// Persists a snapshot/checkpoint of the workflow state.
    pub fn persist_workflow_state(&self, state: &WorkflowState) -> Result<()> {
        self.put_entity(
            "workflow state",
            AgentStorageKey::workflow_key(&state.workflow_id),
            state,
        )
    }

    /// Loads a workflow checkpoint by workflow ID.
    pub fn load_workflow_state(&self, workflow_id: &str) -> Result<Option<WorkflowState>> {
        self.get_entity(
            "workflow state",
            &AgentStorageKey::workflow_key(workflow_id),
        )
    }

    /// Lists all stored workflow checkpoints.
    pub fn list_workflows(&self) -> Result<Vec<WorkflowState>> {
        self.scan_entities("workflow state", AgentStorageKey::PREFIX_WORKFLOW)
    }

    /// Deletes a workflow and all associated state checkpoints.
    pub fn delete_workflow_state(&self, workflow_id: &str) -> Result<bool> {
        self.delete_entity(&AgentStorageKey::workflow_key(workflow_id))
    }

    /// Checkpoints an entire workflow atomically: state, all active tasks, and messages.
    pub fn checkpoint_full_workflow(&self, state: &WorkflowState) -> Result<()> {
        let mut batch = WriteBatch::new();

        // 1. Workflow state
        let wf_key = AgentStorageKey::workflow_key(&state.workflow_id);
        let wf_val = Self::serialize_val("workflow state", state)?;
        batch.put(wf_key.into_bytes(), wf_val);

        // 2. All tasks in workflow
        for task in state.tasks.values() {
            let task_key = AgentStorageKey::task_key(&task.id);
            let task_val = Self::serialize_val("task", task)?;
            batch.put(task_key.into_bytes(), task_val);
        }

        // 3. All messages in workflow
        for msg in &state.message_history {
            let msg_key = AgentStorageKey::message_key(&state.workflow_id, &msg.message_id);
            let msg_val = Self::serialize_val("message", msg)?;
            batch.put(msg_key.into_bytes(), msg_val);
        }

        self.store.write_batch(batch)
    }
}
