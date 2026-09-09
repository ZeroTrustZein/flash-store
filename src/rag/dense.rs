use std::collections::HashMap;

use crate::error::{FlashStoreError, Result};
use crate::rag::config::SimilarityMetric;
use serde::{Deserialize, Serialize};

/// Computes the dot product between two slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the Euclidean L2 norm of a vector.
pub fn l2_norm(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt()
}

/// Computes cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot_product(a, b) / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Computes Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Computes similarity score according to the configured metric.
pub fn compute_similarity(metric: SimilarityMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::DotProduct => dot_product(a, b),
        SimilarityMetric::Euclidean => {
            let dist = euclidean_distance(a, b);
            1.0 / (1.0 + dist)
        }
    }
}

/// In-memory dense vector index with configurable similarity metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseIndex {
    dimension: usize,
    metric: SimilarityMetric,
    vectors: HashMap<String, Vec<f32>>,
}

impl DenseIndex {
    /// Creates a new dense vector index.
    pub fn new(dimension: usize, metric: SimilarityMetric) -> Self {
        Self {
            dimension,
            metric,
            vectors: HashMap::new(),
        }
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the similarity metric used.
    pub fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    /// Returns the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Checks if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Validates vector dimension.
    pub fn validate_dim(&self, vec: &[f32]) -> Result<()> {
        if vec.len() != self.dimension {
            return Err(FlashStoreError::VectorDimensionMismatch {
                expected: self.dimension,
                actual: vec.len(),
            });
        }
        Ok(())
    }

    /// Inserts or updates a document vector.
    pub fn insert(&mut self, doc_id: impl Into<String>, vector: Vec<f32>) -> Result<()> {
        self.validate_dim(&vector)?;
        self.vectors.insert(doc_id.into(), vector);
        Ok(())
    }

    /// Removes a document vector by doc_id.
    pub fn remove(&mut self, doc_id: &str) -> Option<Vec<f32>> {
        self.vectors.remove(doc_id)
    }

    /// Retrieves a document vector by doc_id.
    pub fn get(&self, doc_id: &str) -> Option<&Vec<f32>> {
        self.vectors.get(doc_id)
    }

    /// Searches for top_k most similar documents to the query vector.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        self.validate_dim(query)?;
        if top_k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = compute_similarity(self.metric, query, vec);
                (id.clone(), score)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Clears all vectors from the index.
    pub fn clear(&mut self) {
        self.vectors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_math() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![2.0, 0.0, 0.0];

        assert_eq!(dot_product(&a, &b), 0.0);
        assert_eq!(dot_product(&a, &c), 2.0);
        assert_eq!(l2_norm(&c), 2.0);

        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        assert_eq!(euclidean_distance(&a, &c), 1.0);
    }

    #[test]
    fn test_dense_index_crud_and_search() {
        let mut index = DenseIndex::new(3, SimilarityMetric::Cosine);
        assert_eq!(index.dimension(), 3);
        assert!(index.is_empty());

        index.insert("doc1", vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("doc2", vec![0.8, 0.6, 0.0]).unwrap();
        index.insert("doc3", vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());

        let res = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, "doc1");
        assert!((res[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(res[1].0, "doc2");

        // Dimension mismatch check
        assert!(index.search(&[1.0, 0.0], 2).is_err());
        assert!(index.insert("doc4", vec![1.0]).is_err());

        // Removal
        assert!(index.remove("doc2").is_some());
        assert_eq!(index.len(), 2);
    }
}
