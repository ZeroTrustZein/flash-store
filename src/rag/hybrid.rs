use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Represents an aggregated search hit returned by retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Identifier of the matched document.
    pub doc_id: String,
    /// Combined fused score.
    pub score: f32,
    /// Raw score from dense vector search if present.
    pub dense_score: Option<f32>,
    /// Raw score from sparse BM25 search if present.
    pub sparse_score: Option<f32>,
}

/// Normalizes a list of (id, score) pairs into [0.0, 1.0] using min-max scaling.
pub fn min_max_normalize(hits: &[(String, f32)]) -> HashMap<String, f32> {
    if hits.is_empty() {
        return HashMap::new();
    }
    let min = hits.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = hits
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    let mut map = HashMap::with_capacity(hits.len());
    for (id, s) in hits {
        let norm = if range.abs() < 1e-6 {
            1.0
        } else {
            (s - min) / range
        };
        map.insert(id.clone(), norm);
    }
    map
}

/// Fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF).
///
/// Formula: RRF_score(d) = sum_{rankings} 1 / (k + rank)
pub fn reciprocal_rank_fusion(
    dense_results: &[(String, f32)],
    sparse_results: &[(String, f32)],
    rrf_k: usize,
    top_k: usize,
) -> Vec<SearchResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut dense_map: HashMap<String, f32> = HashMap::new();
    let mut sparse_map: HashMap<String, f32> = HashMap::new();

    let k_f32 = rrf_k as f32;

    for (rank, (doc_id, score)) in dense_results.iter().enumerate() {
        dense_map.insert(doc_id.clone(), *score);
        let rrf_inc = 1.0 / (k_f32 + (rank + 1) as f32);
        *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_inc;
    }

    for (rank, (doc_id, score)) in sparse_results.iter().enumerate() {
        sparse_map.insert(doc_id.clone(), *score);
        let rrf_inc = 1.0 / (k_f32 + (rank + 1) as f32);
        *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_inc;
    }

    let mut fused: Vec<SearchResult> = scores
        .into_iter()
        .map(|(doc_id, score)| SearchResult {
            dense_score: dense_map.remove(&doc_id),
            sparse_score: sparse_map.remove(&doc_id),
            doc_id,
            score,
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    fused.truncate(top_k);
    fused
}

/// Fuses dense and sparse rankings using weighted linear combination of normalized scores.
pub fn weighted_linear_fusion(
    dense_results: &[(String, f32)],
    sparse_results: &[(String, f32)],
    dense_weight: f32,
    top_k: usize,
) -> Vec<SearchResult> {
    let alpha = dense_weight.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;

    let norm_dense = min_max_normalize(dense_results);
    let norm_sparse = min_max_normalize(sparse_results);

    let mut doc_scores: HashMap<String, (Option<f32>, Option<f32>)> =
        HashMap::with_capacity(dense_results.len() + sparse_results.len());

    for (id, score) in dense_results {
        doc_scores.entry(id.clone()).or_insert((None, None)).0 = Some(*score);
    }
    for (id, score) in sparse_results {
        doc_scores.entry(id.clone()).or_insert((None, None)).1 = Some(*score);
    }

    let mut fused: Vec<SearchResult> = doc_scores
        .into_iter()
        .map(|(id, (dense_score, sparse_score))| {
            let d_norm = norm_dense.get(&id).copied().unwrap_or(0.0);
            let s_norm = norm_sparse.get(&id).copied().unwrap_or(0.0);
            let combined = alpha * d_norm + beta * s_norm;

            SearchResult {
                doc_id: id,
                score: combined,
                dense_score,
                sparse_score,
            }
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    fused.truncate(top_k);
    fused
}

/// Standardizes a list of (id, score) pairs using Z-Score normalization.
pub fn z_score_normalize(hits: &[(String, f32)]) -> HashMap<String, f32> {
    if hits.is_empty() {
        return HashMap::new();
    }
    let n = hits.len() as f32;
    let mean: f32 = hits.iter().map(|(_, s)| *s).sum::<f32>() / n;
    let variance: f32 = hits.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    let mut map = HashMap::with_capacity(hits.len());
    for (id, s) in hits {
        let z = if std_dev.abs() < 1e-6 {
            0.0
        } else {
            (s - mean) / std_dev
        };
        map.insert(id.clone(), z);
    }
    map
}

/// Fuses dense and sparse rankings using positional Borda Count voting.
///
/// In each list with N items, item at rank `r` (0-indexed) receives `N - r` points.
/// Points are summed across candidate lists.
pub fn borda_count_fusion(
    dense_results: &[(String, f32)],
    sparse_results: &[(String, f32)],
    top_k: usize,
) -> Vec<SearchResult> {
    let mut points: HashMap<String, f32> = HashMap::new();
    let dense_map: HashMap<String, f32> = dense_results.iter().cloned().collect();
    let sparse_map: HashMap<String, f32> = sparse_results.iter().cloned().collect();

    let n_dense = dense_results.len() as f32;
    for (rank, (doc_id, _)) in dense_results.iter().enumerate() {
        *points.entry(doc_id.clone()).or_insert(0.0) += n_dense - rank as f32;
    }

    let n_sparse = sparse_results.len() as f32;
    for (rank, (doc_id, _)) in sparse_results.iter().enumerate() {
        *points.entry(doc_id.clone()).or_insert(0.0) += n_sparse - rank as f32;
    }

    let mut fused: Vec<SearchResult> = points
        .into_iter()
        .map(|(doc_id, score)| SearchResult {
            dense_score: dense_map.get(&doc_id).copied(),
            sparse_score: sparse_map.get(&doc_id).copied(),
            doc_id,
            score,
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    fused.truncate(top_k);
    fused
}

/// Filters search results with scores below `min_score`.
pub fn filter_min_score(results: Vec<SearchResult>, min_score: f32) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter(|r| r.score >= min_score)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reciprocal_rank_fusion() {
        let dense = vec![
            ("docA".to_string(), 0.95),
            ("docB".to_string(), 0.85),
            ("docC".to_string(), 0.70),
        ];
        let sparse = vec![
            ("docB".to_string(), 12.0),
            ("docA".to_string(), 8.0),
            ("docD".to_string(), 4.0),
        ];

        let fused = reciprocal_rank_fusion(&dense, &sparse, 60, 3);
        assert_eq!(fused.len(), 3);
        // docA and docB both in top 2 across both lists, so should have highest RRF scores
        assert!(fused[0].doc_id == "docA" || fused[0].doc_id == "docB");
        assert!(fused[1].doc_id == "docA" || fused[1].doc_id == "docB");
        assert_eq!(fused[0].score, fused[1].score); // dense rank 1+2, sparse rank 2+1
    }

    #[test]
    fn test_weighted_linear_fusion() {
        let dense = vec![("docA".to_string(), 1.0), ("docB".to_string(), 0.0)];
        let sparse = vec![("docA".to_string(), 0.0), ("docB".to_string(), 10.0)];

        // Equal weights
        let fused = weighted_linear_fusion(&dense, &sparse, 0.5, 2);
        assert_eq!(fused.len(), 2);
        assert!((fused[0].score - 0.5).abs() < 1e-6);
        assert!((fused[1].score - 0.5).abs() < 1e-6);

        // Dense favored
        let dense_favored = weighted_linear_fusion(&dense, &sparse, 0.8, 2);
        assert_eq!(dense_favored[0].doc_id, "docA");
        assert!((dense_favored[0].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_borda_count_fusion() {
        let dense = vec![("docA".to_string(), 0.9), ("docB".to_string(), 0.5)];
        let sparse = vec![("docB".to_string(), 8.0), ("docA".to_string(), 4.0)];

        let fused = borda_count_fusion(&dense, &sparse, 2);
        assert_eq!(fused.len(), 2);
        // docA: dense rank 0 (2 pts) + sparse rank 1 (1 pt) = 3 pts
        // docB: dense rank 1 (1 pt) + sparse rank 0 (2 pts) = 3 pts
        assert_eq!(fused[0].score, 3.0);
        assert_eq!(fused[1].score, 3.0);
    }

    #[test]
    fn test_z_score_normalize_and_filter() {
        let hits = vec![
            ("docA".to_string(), 10.0),
            ("docB".to_string(), 20.0),
            ("docC".to_string(), 30.0),
        ];
        let z_scores = z_score_normalize(&hits);
        assert_eq!(z_scores.len(), 3);
        // Mean is 20.0, docB should have z ~ 0.0
        assert!(z_scores.get("docB").unwrap().abs() < 1e-5);
        assert!(z_scores.get("docA").unwrap() < &0.0);
        assert!(z_scores.get("docC").unwrap() > &0.0);

        let results = vec![
            SearchResult {
                doc_id: "doc1".to_string(),
                score: 0.8,
                dense_score: None,
                sparse_score: None,
            },
            SearchResult {
                doc_id: "doc2".to_string(),
                score: 0.3,
                dense_score: None,
                sparse_score: None,
            },
        ];
        let filtered = filter_min_score(results, 0.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].doc_id, "doc1");
    }
}
