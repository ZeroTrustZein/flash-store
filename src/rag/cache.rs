use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{FlashStoreError, Result};
use crate::rag::dense::{dot_product, l2_norm};
use crate::rag::hybrid::SearchResult;
use crate::rag::types::SemanticCacheStats;
use serde::{Deserialize, Serialize};

/// Returns current Unix epoch timestamp in seconds.
pub fn current_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// An entry stored in the semantic cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheEntry {
    /// Original user query string.
    pub query: String,
    /// Query vector embedding.
    pub embedding: Vec<f32>,
    /// Cached retrieval results.
    pub results: Vec<SearchResult>,
    /// Creation timestamp in epoch seconds.
    pub created_at: u64,
    /// Last access timestamp in epoch seconds.
    pub last_accessed: u64,
    /// Number of cache hits served by this entry.
    pub hits: usize,
}

/// In-memory semantic query cache with similarity thresholding, TTL, and LRU eviction.
#[derive(Debug, Serialize, Deserialize)]
pub struct SemanticCache {
    capacity: usize,
    threshold: f32,
    ttl_secs: u64,
    entries: Vec<SemanticCacheEntry>,
    #[serde(skip)]
    total_hits: AtomicU64,
    #[serde(skip)]
    total_misses: AtomicU64,
}

impl SemanticCache {
    /// Creates a new semantic cache.
    pub fn new(capacity: usize, threshold: f32, ttl_secs: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            threshold: threshold.clamp(0.0, 1.0),
            ttl_secs,
            entries: Vec::new(),
            total_hits: AtomicU64::new(0),
            total_misses: AtomicU64::new(0),
        }
    }

    /// Cache capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Similarity threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Checks if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total cache hits.
    pub fn total_hits(&self) -> u64 {
        self.total_hits.load(Ordering::Relaxed)
    }

    /// Total cache misses.
    pub fn total_misses(&self) -> u64 {
        self.total_misses.load(Ordering::Relaxed)
    }

    /// Cache hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let hits = self.total_hits() as f64;
        let misses = self.total_misses() as f64;
        let total = hits + misses;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Returns telemetry statistics snapshot of the semantic cache.
    pub fn stats(&self) -> SemanticCacheStats {
        SemanticCacheStats {
            capacity: self.capacity,
            len: self.entries.len(),
            hits: self.total_hits(),
            misses: self.total_misses(),
            hit_rate: self.hit_rate(),
        }
    }

    /// Performs semantic lookup: finds a cached entry whose embedding cosine similarity
    /// meets or exceeds the similarity threshold and has not expired.
    pub fn lookup(&mut self, query_embedding: &[f32]) -> Option<Vec<SearchResult>> {
        let now = current_timestamp_secs();

        // Evict expired entries
        if self.ttl_secs > 0 {
            let ttl = self.ttl_secs;
            self.entries
                .retain(|e| now.saturating_sub(e.created_at) < ttl);
        }

        let query_norm = l2_norm(query_embedding);
        if query_norm == 0.0 {
            self.total_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let mut best_match: Option<(usize, f32)> = None;

        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.embedding.len() != query_embedding.len() {
                continue;
            }
            let entry_norm = l2_norm(&entry.embedding);
            if entry_norm == 0.0 {
                continue;
            }
            let sim = (dot_product(&entry.embedding, query_embedding) / (entry_norm * query_norm))
                .clamp(-1.0, 1.0);
            if sim >= self.threshold {
                match best_match {
                    Some((_, best_sim)) if sim > best_sim => {
                        best_match = Some((idx, sim));
                    }
                    None => {
                        best_match = Some((idx, sim));
                    }
                    _ => {}
                }
            }
        }

        if let Some((idx, _)) = best_match {
            self.total_hits.fetch_add(1, Ordering::Relaxed);
            let entry = &mut self.entries[idx];
            entry.last_accessed = now;
            entry.hits += 1;
            Some(entry.results.clone())
        } else {
            self.total_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Inserts a new semantic entry into the cache, evicting the least-recently used if full.
    pub fn insert(
        &mut self,
        query: impl Into<String>,
        embedding: Vec<f32>,
        results: Vec<SearchResult>,
    ) -> Result<()> {
        if embedding.is_empty() {
            return Err(FlashStoreError::InvalidArgument(
                "Cannot cache empty embedding".to_string(),
            ));
        }

        let now = current_timestamp_secs();

        // Evict LRU if at capacity
        if self.entries.len() >= self.capacity {
            if let Some((lru_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
            {
                self.entries.swap_remove(lru_idx);
            }
        }

        self.entries.push(SemanticCacheEntry {
            query: query.into(),
            embedding,
            results,
            created_at: now,
            last_accessed: now,
            hits: 0,
        });

        Ok(())
    }

    /// Invalidates all cached query entries that returned `doc_id` in their results.
    ///
    /// Returns the number of invalidated cache entries.
    pub fn invalidate_for_doc(&mut self, doc_id: &str) -> usize {
        let initial_len = self.entries.len();
        self.entries
            .retain(|entry| !entry.results.iter().any(|r| r.doc_id == doc_id));
        initial_len - self.entries.len()
    }

    /// Clears the semantic cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_hits.store(0, Ordering::Relaxed);
        self.total_misses.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_cache_hit_and_miss() {
        let mut cache = SemanticCache::new(5, 0.90, 3600);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let results = vec![SearchResult {
            doc_id: "doc1".to_string(),
            score: 0.95,
            dense_score: Some(0.95),
            sparse_score: None,
        }];

        let query_vec = vec![1.0, 0.0, 0.0];
        cache
            .insert("what is flash store", query_vec.clone(), results.clone())
            .unwrap();
        assert_eq!(cache.len(), 1);

        // Exact match -> hit
        let hit = cache.lookup(&[1.0, 0.0, 0.0]);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap()[0].doc_id, "doc1");
        assert_eq!(cache.total_hits(), 1);

        // Semantically very close (cos sim = 0.9998 >= 0.90) -> hit
        let near_hit = cache.lookup(&[0.99, 0.01, 0.0]);
        assert!(near_hit.is_some());
        assert_eq!(cache.total_hits(), 2);

        // Dissimilar query (cos sim = 0.0 < 0.90) -> miss
        let miss = cache.lookup(&[0.0, 1.0, 0.0]);
        assert!(miss.is_none());
        assert_eq!(cache.total_misses(), 1);

        assert!((cache.hit_rate() - (2.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_semantic_cache_lru_eviction() {
        let mut cache = SemanticCache::new(2, 0.90, 3600);
        cache.insert("q1", vec![1.0, 0.0], vec![]).unwrap();
        cache.insert("q2", vec![0.0, 1.0], vec![]).unwrap();
        assert_eq!(cache.len(), 2);

        // Insert third entry -> evicts one entry
        cache.insert("q3", vec![-1.0, 0.0], vec![]).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_semantic_cache_invalidation_for_doc() {
        let mut cache = SemanticCache::new(5, 0.90, 3600);
        let res1 = vec![SearchResult {
            doc_id: "doc_target".to_string(),
            score: 0.9,
            dense_score: None,
            sparse_score: None,
        }];
        let res2 = vec![SearchResult {
            doc_id: "doc_other".to_string(),
            score: 0.8,
            dense_score: None,
            sparse_score: None,
        }];

        cache.insert("q1", vec![1.0, 0.0], res1).unwrap();
        cache.insert("q2", vec![0.0, 1.0], res2).unwrap();
        assert_eq!(cache.len(), 2);

        let evicted = cache.invalidate_for_doc("doc_target");
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(&[1.0, 0.0]).is_none());
        assert!(cache.lookup(&[0.0, 1.0]).is_some());
    }
}
