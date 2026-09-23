//! # Agent Memory and Episodic Knowledge Subsystem
//!
//! Provides working memory (blackboard), episodic experience memory, and semantic
//! associative recall for autonomous multi-agent swarms.
//!
//! Features:
//! - **Working Memory (Blackboard)**: Shared scratchpad for intra- and inter-agent coordination.
//! - **Episodic Memory**: Recorded agent task attempts, reflections, tool outputs, and learnings.
//! - **Semantic Recall**: Hybrid ranking over memory episodes combining semantic vector similarity,
//!   temporal recency decay, and importance weighting.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::agent::store::AgentStoreAdapter;
use crate::agent::types::{AgentId, AgentStorageKey, TaskId};
use crate::error::{FlashStoreError, Result};
use crate::rag::cosine_similarity;
use crate::rag::pipeline::EmbeddingProvider;

/// A recorded experience or episode in agent memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryEntry {
    pub id: String,
    pub agent_id: AgentId,
    pub task_id: Option<TaskId>,
    pub timestamp: u64,
    pub content: String,
    pub tags: Vec<String>,
    pub importance: f32,
    pub embedding: Option<Vec<f32>>,
}

impl EpisodicMemoryEntry {
    pub fn new(agent_id: AgentId, content: impl Into<String>, importance: f32) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let content_str = content.into();
        let id = format!("mem_{}_{:x}", ts, Self::hash_str(&content_str));

        Self {
            id,
            agent_id,
            task_id: None,
            timestamp: ts,
            content: content_str,
            tags: Vec::new(),
            importance: importance.clamp(0.0, 1.0),
            embedding: None,
        }
    }

    pub fn with_task_id(mut self, task_id: TaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    fn hash_str(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3u64);
        }
        h
    }
}

/// A recalled memory paired with its composite ranking score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: EpisodicMemoryEntry,
    pub score: f32,
    pub relevance_score: f32,
    pub recency_score: f32,
    pub importance_score: f32,
}

/// Configuration parameters for memory retrieval scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryScoringWeights {
    /// Weight assigned to semantic relevance (vector cosine similarity). Default: 0.5.
    pub relevance_weight: f32,
    /// Weight assigned to temporal recency decay. Default: 0.3.
    pub recency_weight: f32,
    /// Weight assigned to designated episode importance. Default: 0.2.
    pub importance_weight: f32,
    /// Half-life in seconds for exponential temporal recency decay. Default: 3600 (1 hour).
    pub recency_half_life_secs: f32,
}

impl Default for MemoryScoringWeights {
    fn default() -> Self {
        Self {
            relevance_weight: 0.5,
            recency_weight: 0.3,
            importance_weight: 0.2,
            recency_half_life_secs: 3600.0,
        }
    }
}

/// Thread-safe blackboard and episodic memory store for multi-agent workflows.
pub struct AgentMemoryStore {
    blackboard: RwLock<HashMap<String, serde_json::Value>>,
    episodes: RwLock<Vec<EpisodicMemoryEntry>>,
    weights: MemoryScoringWeights,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    max_episodes: usize,
    episode_counter: AtomicU64,
}

impl Default for AgentMemoryStore {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl AgentMemoryStore {
    /// Creates a new `AgentMemoryStore` with a maximum capacity of episodic entries.
    pub fn new(max_episodes: usize) -> Self {
        Self {
            blackboard: RwLock::new(HashMap::new()),
            episodes: RwLock::new(Vec::new()),
            weights: MemoryScoringWeights::default(),
            embedder: None,
            max_episodes: max_episodes.max(10),
            episode_counter: AtomicU64::new(0),
        }
    }

    /// Attaches an embedding provider for automated vector semantic retrieval.
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Customizes memory scoring weights.
    pub fn with_weights(mut self, weights: MemoryScoringWeights) -> Self {
        self.weights = weights;
        self
    }

    // ========================================================================
    // Blackboard (Working Memory)
    // ========================================================================

    /// Sets a value on the shared blackboard.
    pub fn set_blackboard(&self, key: impl Into<String>, value: serde_json::Value) {
        self.blackboard.write().insert(key.into(), value);
    }

    /// Gets a value from the shared blackboard.
    pub fn get_blackboard(&self, key: &str) -> Option<serde_json::Value> {
        self.blackboard.read().get(key).cloned()
    }

    /// Deletes a key from the shared blackboard.
    pub fn remove_blackboard(&self, key: &str) -> Option<serde_json::Value> {
        self.blackboard.write().remove(key)
    }

    /// Checks if a key exists on the blackboard.
    pub fn contains_blackboard(&self, key: &str) -> bool {
        self.blackboard.read().contains_key(key)
    }

    /// Takes a full snapshot of the working memory blackboard.
    pub fn snapshot_blackboard(&self) -> HashMap<String, serde_json::Value> {
        self.blackboard.read().clone()
    }

    /// Clears the working memory blackboard.
    pub fn clear_blackboard(&self) {
        self.blackboard.write().clear();
    }

    // ========================================================================
    // Episodic Memory (Experiences & Learnings)
    // ========================================================================

    /// Appends a new episodic memory. Automatically computes vector embeddings if an
    /// embedder is configured and the entry does not already contain an embedding.
    pub fn record_episode(&self, mut entry: EpisodicMemoryEntry) -> Result<()> {
        if entry.embedding.is_none() {
            if let Some(ref embedder) = self.embedder {
                let vec = embedder.embed_query(&entry.content)?;
                entry.embedding = Some(vec);
            }
        }

        let mut episodes = self.episodes.write();
        episodes.push(entry);
        self.episode_counter.fetch_add(1, Ordering::Relaxed);

        // Evict oldest if capacity exceeded
        if episodes.len() > self.max_episodes {
            let overflow = episodes.len() - self.max_episodes;
            episodes.drain(0..overflow);
        }

        Ok(())
    }

    /// Returns the total number of stored episodic memories.
    pub fn episode_count(&self) -> usize {
        self.episodes.read().len()
    }

    /// Retrieves all episodes recorded by a specific agent.
    pub fn get_episodes_by_agent(&self, agent_id: &AgentId) -> Vec<EpisodicMemoryEntry> {
        self.episodes
            .read()
            .iter()
            .filter(|e| &e.agent_id == agent_id)
            .cloned()
            .collect()
    }

    /// Retrieves all episodes associated with a specific task.
    pub fn get_episodes_by_task(&self, task_id: &TaskId) -> Vec<EpisodicMemoryEntry> {
        self.episodes
            .read()
            .iter()
            .filter(|e| e.task_id.as_ref() == Some(task_id))
            .cloned()
            .collect()
    }

    /// Retrieves episodes matching a specific tag.
    pub fn get_episodes_by_tag(&self, tag: &str) -> Vec<EpisodicMemoryEntry> {
        self.episodes
            .read()
            .iter()
            .filter(|e| e.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
            .cloned()
            .collect()
    }

    // ========================================================================
    // Semantic Recall & Ranking
    // ========================================================================

    /// Performs hybrid recall for a given query string:
    /// Combines vector relevance, temporal recency decay, and importance score.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<ScoredMemory>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let query_vec = if let Some(ref embedder) = self.embedder {
            Some(embedder.embed_query(query)?)
        } else {
            None
        };

        let current_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let episodes = self.episodes.read();
        let mut scored_list = Vec::with_capacity(episodes.len());

        for ep in episodes.iter() {
            // 1. Relevance score
            let relevance = if let (Some(q_vec), Some(ep_vec)) = (&query_vec, &ep.embedding) {
                cosine_similarity(q_vec, ep_vec).max(0.0)
            } else {
                // Lexical term overlap fallback if no vectors
                Self::lexical_overlap(query, &ep.content)
            };

            // 2. Recency score (exponential decay)
            let age_secs = current_ts.saturating_sub(ep.timestamp) as f32;
            let lambda = 0.693147 / self.weights.recency_half_life_secs.max(1.0);
            let recency = (-lambda * age_secs).exp().clamp(0.0, 1.0);

            // 3. Importance score
            let importance = ep.importance;

            // Composite weighted sum
            let total_score = self.weights.relevance_weight * relevance
                + self.weights.recency_weight * recency
                + self.weights.importance_weight * importance;

            scored_list.push(ScoredMemory {
                memory: ep.clone(),
                score: total_score,
                relevance_score: relevance,
                recency_score: recency,
                importance_score: importance,
            });
        }

        // Sort descending by score
        scored_list.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored_list.truncate(top_k);

        Ok(scored_list)
    }

    /// Fast lexical Jaccard/overlap fallback between query and memory content.
    fn lexical_overlap(query: &str, content: &str) -> f32 {
        let q_words: std::collections::HashSet<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let c_words: std::collections::HashSet<String> = content
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if q_words.is_empty() || c_words.is_empty() {
            return 0.0;
        }

        let intersection = q_words.intersection(&c_words).count();
        let union = q_words.union(&c_words).count();
        intersection as f32 / union as f32
    }

    // ========================================================================
    // Persistence Synchronization with FlashStore
    // ========================================================================

    /// Synchronizes in-memory working memory and episodes to FlashStore.
    pub fn persist_to_store(&self, store: &AgentStoreAdapter, namespace: &str) -> Result<()> {
        let blackboard_snap = self.snapshot_blackboard();
        let bb_key = AgentStorageKey::memory_key(namespace, "blackboard");
        let bb_val = serde_json::to_vec(&blackboard_snap)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize blackboard: {}", e)))?;
        store.store().put(bb_key.into_bytes(), bb_val)?;

        let episodes = self.episodes.read().clone();
        let ep_key = AgentStorageKey::memory_key(namespace, "episodes");
        let ep_val = serde_json::to_vec(&episodes)
            .map_err(|e| FlashStoreError::Agent(format!("Failed to serialize episodes: {}", e)))?;
        store.store().put(ep_key.into_bytes(), ep_val)?;

        Ok(())
    }

    /// Hydrates in-memory working memory and episodes from FlashStore.
    pub fn load_from_store(&self, store: &AgentStoreAdapter, namespace: &str) -> Result<bool> {
        let bb_key = AgentStorageKey::memory_key(namespace, "blackboard");
        let mut loaded = false;

        if let Some(bytes) = store.store().get(bb_key.as_bytes())? {
            let map: HashMap<String, serde_json::Value> = serde_json::from_slice(&bytes)
                .map_err(|e| FlashStoreError::Agent(format!("Failed to deserialize blackboard: {}", e)))?;
            *self.blackboard.write() = map;
            loaded = true;
        }

        let ep_key = AgentStorageKey::memory_key(namespace, "episodes");
        if let Some(bytes) = store.store().get(ep_key.as_bytes())? {
            let eps: Vec<EpisodicMemoryEntry> = serde_json::from_slice(&bytes)
                .map_err(|e| FlashStoreError::Agent(format!("Failed to deserialize episodes: {}", e)))?;
            *self.episodes.write() = eps;
            loaded = true;
        }

        Ok(loaded)
    }
}
