use serde::{Deserialize, Serialize};

/// Distance/similarity metric for dense vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity: normalized dot product in [-1.0, 1.0]. Default.
    #[default]
    Cosine,
    /// Dot product: unnormalized dot product.
    DotProduct,
    /// Euclidean (L2) distance converted to similarity: 1 / (1 + L2).
    Euclidean,
}

/// Configuration settings for the RAG retrieval, reranking, and caching subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Dimension of dense vector embeddings (e.g., 384, 768, 1536).
    pub embedding_dim: usize,
    /// Similarity metric used for dense vector comparisons.
    pub similarity_metric: SimilarityMetric,
    /// BM25 term frequency saturation parameter (default: 1.2).
    pub bm25_k1: f32,
    /// BM25 document length normalization parameter (default: 0.75).
    pub bm25_b: f32,
    /// Weight assigned to dense scores in hybrid linear fusion (0.0 to 1.0, default: 0.5).
    pub hybrid_dense_weight: f32,
    /// Reciprocal Rank Fusion constant k (default: 60).
    pub rrf_k: usize,
    /// Maximum number of query entries in the semantic cache (default: 10,000).
    pub semantic_cache_capacity: usize,
    /// Minimum cosine similarity threshold to trigger a semantic cache hit (default: 0.92).
    pub semantic_cache_threshold: f32,
    /// Time-to-live for semantic cache entries in seconds (default: 3600 = 1 hour, 0 for infinite).
    pub semantic_cache_ttl_secs: u64,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            similarity_metric: SimilarityMetric::Cosine,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            hybrid_dense_weight: 0.5,
            rrf_k: 60,
            semantic_cache_capacity: 10_000,
            semantic_cache_threshold: 0.92,
            semantic_cache_ttl_secs: 3600,
        }
    }
}

/// Builder for [`RagConfig`].
#[derive(Debug, Default)]
pub struct RagConfigBuilder {
    config: RagConfig,
}

impl RagConfigBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: RagConfig::default(),
        }
    }

    /// Sets the expected embedding dimension.
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    /// Sets the similarity metric for vector search.
    pub fn similarity_metric(mut self, metric: SimilarityMetric) -> Self {
        self.config.similarity_metric = metric;
        self
    }

    /// Sets BM25 parameters k1 and b.
    pub fn bm25_params(mut self, k1: f32, b: f32) -> Self {
        self.config.bm25_k1 = k1;
        self.config.bm25_b = b;
        self
    }

    /// Sets hybrid retrieval dense weight (dense weight in [0.0, 1.0], sparse = 1.0 - dense).
    pub fn hybrid_dense_weight(mut self, weight: f32) -> Self {
        self.config.hybrid_dense_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Sets Reciprocal Rank Fusion constant k.
    pub fn rrf_k(mut self, k: usize) -> Self {
        self.config.rrf_k = k;
        self
    }

    /// Sets semantic cache capacity, similarity threshold, and TTL in seconds.
    pub fn semantic_cache(mut self, capacity: usize, threshold: f32, ttl_secs: u64) -> Self {
        self.config.semantic_cache_capacity = capacity;
        self.config.semantic_cache_threshold = threshold.clamp(0.0, 1.0);
        self.config.semantic_cache_ttl_secs = ttl_secs;
        self
    }

    /// Finalizes and returns the [`RagConfig`].
    pub fn build(self) -> RagConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rag_config() {
        let cfg = RagConfig::default();
        assert_eq!(cfg.embedding_dim, 384);
        assert_eq!(cfg.similarity_metric, SimilarityMetric::Cosine);
        assert_eq!(cfg.bm25_k1, 1.2);
        assert_eq!(cfg.bm25_b, 0.75);
        assert_eq!(cfg.hybrid_dense_weight, 0.5);
        assert_eq!(cfg.rrf_k, 60);
        assert_eq!(cfg.semantic_cache_capacity, 10_000);
        assert!((cfg.semantic_cache_threshold - 0.92).abs() < f32::EPSILON);
        assert_eq!(cfg.semantic_cache_ttl_secs, 3600);
    }

    #[test]
    fn test_rag_config_builder() {
        let cfg = RagConfigBuilder::new()
            .embedding_dim(768)
            .similarity_metric(SimilarityMetric::DotProduct)
            .bm25_params(1.5, 0.8)
            .hybrid_dense_weight(0.7)
            .rrf_k(40)
            .semantic_cache(5000, 0.95, 1800)
            .build();

        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.similarity_metric, SimilarityMetric::DotProduct);
        assert_eq!(cfg.bm25_k1, 1.5);
        assert_eq!(cfg.bm25_b, 0.8);
        assert_eq!(cfg.hybrid_dense_weight, 0.7);
        assert_eq!(cfg.rrf_k, 40);
        assert_eq!(cfg.semantic_cache_capacity, 5000);
        assert!((cfg.semantic_cache_threshold - 0.95).abs() < f32::EPSILON);
        assert_eq!(cfg.semantic_cache_ttl_secs, 1800);
    }
}
