use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Thread-safe stage latency and count accumulator.
#[derive(Debug)]
pub struct StageLatencyTracker {
    count: AtomicU64,
    total_micros: AtomicU64,
    min_micros: AtomicU64,
    max_micros: AtomicU64,
}

impl Default for StageLatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl StageLatencyTracker {
    /// Creates a new empty latency accumulator.
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_micros: AtomicU64::new(0),
            min_micros: AtomicU64::new(u64::MAX),
            max_micros: AtomicU64::new(0),
        }
    }

    /// Records an execution duration for this stage.
    pub fn record(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_micros.fetch_add(micros, Ordering::Relaxed);

        // Update min
        let mut curr_min = self.min_micros.load(Ordering::Relaxed);
        while micros < curr_min {
            match self.min_micros.compare_exchange_weak(
                curr_min,
                micros,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => curr_min = actual,
            }
        }

        // Update max
        let mut curr_max = self.max_micros.load(Ordering::Relaxed);
        while micros > curr_max {
            match self.max_micros.compare_exchange_weak(
                curr_max,
                micros,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => curr_max = actual,
            }
        }
    }

    /// Exports a read-only snapshot.
    pub fn snapshot(&self) -> StageMetricsSnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_micros.load(Ordering::Relaxed);
        let min = if count > 0 {
            let m = self.min_micros.load(Ordering::Relaxed);
            if m == u64::MAX {
                0
            } else {
                m
            }
        } else {
            0
        };
        let max = self.max_micros.load(Ordering::Relaxed);
        let avg_micros = if count > 0 {
            total as f64 / count as f64
        } else {
            0.0
        };

        StageMetricsSnapshot {
            count,
            total_micros: total,
            min_micros: min,
            max_micros: max,
            avg_micros,
        }
    }
}

/// Snapshot of a single stage's latency and throughput metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetricsSnapshot {
    pub count: u64,
    pub total_micros: u64,
    pub min_micros: u64,
    pub max_micros: u64,
    pub avg_micros: f64,
}

/// System-wide telemetry metrics for RAG retrieval subsystems.
#[derive(Debug, Default)]
pub struct RagMetricsTracker {
    pub total_searches: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub search_latency: StageLatencyTracker,
    pub sparse_latency: StageLatencyTracker,
    pub dense_latency: StageLatencyTracker,
    pub fusion_latency: StageLatencyTracker,
    pub rerank_latency: StageLatencyTracker,
}

impl RagMetricsTracker {
    /// Creates a new `RagMetricsTracker`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a cache lookup result.
    pub fn record_cache_lookup(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Produces a complete telemetry snapshot.
    pub fn snapshot(&self) -> RagMetricsSnapshot {
        let total_searches = self.total_searches.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_lookups = cache_hits + cache_misses;
        let cache_hit_rate = if total_lookups > 0 {
            cache_hits as f64 / total_lookups as f64
        } else {
            0.0
        };

        RagMetricsSnapshot {
            total_searches,
            cache_hits,
            cache_misses,
            cache_hit_rate,
            search_latency: self.search_latency.snapshot(),
            sparse_latency: self.sparse_latency.snapshot(),
            dense_latency: self.dense_latency.snapshot(),
            fusion_latency: self.fusion_latency.snapshot(),
            rerank_latency: self.rerank_latency.snapshot(),
        }
    }
}

/// Complete serialized telemetry snapshot across all RAG subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagMetricsSnapshot {
    pub total_searches: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub search_latency: StageMetricsSnapshot,
    pub sparse_latency: StageMetricsSnapshot,
    pub dense_latency: StageMetricsSnapshot,
    pub fusion_latency: StageMetricsSnapshot,
    pub rerank_latency: StageMetricsSnapshot,
}

/// A single query evaluation sample comparing retrieved document IDs against ground truth IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvaluationSample {
    pub query: String,
    pub retrieved_ids: Vec<String>,
    pub ground_truth_ids: Vec<String>,
}

/// Aggregate summary of Information Retrieval (IR) evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    /// Number of evaluated queries.
    pub sample_count: usize,
    /// Cutoff depth K evaluated.
    pub k: usize,
    /// Mean Reciprocal Rank (MRR).
    pub mrr: f64,
    /// Mean Precision@K.
    pub precision_at_k: f64,
    /// Mean Recall@K.
    pub recall_at_k: f64,
    /// Normalized Discounted Cumulative Gain (NDCG@K).
    pub ndcg_at_k: f64,
    /// Hit Rate@K (percentage of queries with at least 1 relevant hit in top K).
    pub hit_rate_at_k: f64,
}

/// Information Retrieval (IR) evaluator computing benchmark metrics
/// (MRR, NDCG@K, Precision@K, Recall@K, HitRate@K) across retrieval results.
pub struct RetrievalEvaluator;

impl RetrievalEvaluator {
    /// Evaluates a collection of query samples at rank cutoff `k`.
    pub fn evaluate(samples: &[QueryEvaluationSample], k: usize) -> EvaluationSummary {
        if samples.is_empty() || k == 0 {
            return EvaluationSummary {
                sample_count: samples.len(),
                k,
                mrr: 0.0,
                precision_at_k: 0.0,
                recall_at_k: 0.0,
                ndcg_at_k: 0.0,
                hit_rate_at_k: 0.0,
            };
        }

        let mut sum_rr = 0.0;
        let mut sum_p = 0.0;
        let mut sum_r = 0.0;
        let mut sum_ndcg = 0.0;
        let mut hits = 0;

        for sample in samples {
            let relevant: HashSet<&str> =
                sample.ground_truth_ids.iter().map(|s| s.as_str()).collect();
            let retrieved_top_k: Vec<&str> = sample
                .retrieved_ids
                .iter()
                .take(k)
                .map(|s| s.as_str())
                .collect();

            // 1. Reciprocal Rank (RR)
            let mut rr = 0.0;
            for (idx, doc_id) in retrieved_top_k.iter().enumerate() {
                if relevant.contains(*doc_id) {
                    rr = 1.0 / (idx + 1) as f64;
                    break;
                }
            }
            sum_rr += rr;

            // 2. Precision@K and Recall@K
            let mut relevant_retrieved_count = 0;
            for doc_id in &retrieved_top_k {
                if relevant.contains(*doc_id) {
                    relevant_retrieved_count += 1;
                }
            }

            let p_at_k = relevant_retrieved_count as f64 / k as f64;
            let r_at_k = if !relevant.is_empty() {
                relevant_retrieved_count as f64 / relevant.len() as f64
            } else {
                0.0
            };
            sum_p += p_at_k;
            sum_r += r_at_k;

            // 3. HitRate@K
            if relevant_retrieved_count > 0 {
                hits += 1;
            }

            // 4. NDCG@K
            let mut dcg = 0.0;
            for (idx, doc_id) in retrieved_top_k.iter().enumerate() {
                if relevant.contains(*doc_id) {
                    dcg += 1.0 / (idx as f64 + 2.0).log2();
                }
            }

            let ideal_count = relevant.len().min(k);
            let mut idcg = 0.0;
            for idx in 0..ideal_count {
                idcg += 1.0 / (idx as f64 + 2.0).log2();
            }

            let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
            sum_ndcg += ndcg;
        }

        let n = samples.len() as f64;
        EvaluationSummary {
            sample_count: samples.len(),
            k,
            mrr: sum_rr / n,
            precision_at_k: sum_p / n,
            recall_at_k: sum_r / n,
            ndcg_at_k: sum_ndcg / n,
            hit_rate_at_k: hits as f64 / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_evaluator_metrics() {
        let samples = vec![
            QueryEvaluationSample {
                query: "q1".into(),
                retrieved_ids: vec!["d1".into(), "d2".into(), "d3".into()],
                ground_truth_ids: vec!["d1".into()], // hit at rank 1 -> RR = 1.0
            },
            QueryEvaluationSample {
                query: "q2".into(),
                retrieved_ids: vec!["x1".into(), "d2".into(), "x2".into()],
                ground_truth_ids: vec!["d2".into()], // hit at rank 2 -> RR = 0.5
            },
            QueryEvaluationSample {
                query: "q3".into(),
                retrieved_ids: vec!["y1".into(), "y2".into()],
                ground_truth_ids: vec!["target".into()], // hit at rank 0 -> RR = 0.0
            },
        ];

        let summary = RetrievalEvaluator::evaluate(&samples, 3);
        assert_eq!(summary.sample_count, 3);
        assert_eq!(summary.k, 3);

        // MRR: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert!((summary.mrr - 0.5).abs() < 1e-4);

        // Hit rate: 2 out of 3 had relevant docs -> 2/3
        assert!((summary.hit_rate_at_k - (2.0 / 3.0)).abs() < 1e-4);

        // Precision@3:
        // q1: 1/3, q2: 1/3, q3: 0/3 -> avg = 2/9
        assert!((summary.precision_at_k - (2.0 / 9.0)).abs() < 1e-4);
    }

    #[test]
    fn test_metrics_tracker_latency_and_cache() {
        let tracker = RagMetricsTracker::new();
        tracker.record_cache_lookup(true);
        tracker.record_cache_lookup(false);
        tracker.record_cache_lookup(true);

        tracker.search_latency.record(Duration::from_micros(100));
        tracker.search_latency.record(Duration::from_micros(200));

        let snap = tracker.snapshot();
        assert_eq!(snap.cache_hits, 2);
        assert_eq!(snap.cache_misses, 1);
        assert!((snap.cache_hit_rate - (2.0 / 3.0)).abs() < 1e-4);
        assert_eq!(snap.search_latency.count, 2);
        assert_eq!(snap.search_latency.min_micros, 100);
        assert_eq!(snap.search_latency.max_micros, 200);
        assert_eq!(snap.search_latency.avg_micros, 150.0);
    }
}
