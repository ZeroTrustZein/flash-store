use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::FlashStore;
use crate::error::Result;
use crate::rag::cache::SemanticCache;
use crate::rag::config::RagConfig;
use crate::rag::dense::DenseIndex;
use crate::rag::hybrid::{reciprocal_rank_fusion, weighted_linear_fusion, SearchResult};
use crate::rag::reranker::{rerank_candidates, LexicalSemanticCrossEncoder, RerankResult};
use crate::rag::sparse::SparseIndex;

/// Prefix for document raw text in FlashStore KV engine.
pub const PREFIX_DOC: &[u8] = b"rag:doc:";
/// Prefix for document vector embeddings in FlashStore KV engine.
pub const PREFIX_VEC: &[u8] = b"rag:vec:";

/// Unified RAG retrieval engine providing hybrid dense-sparse search,
/// cross-encoder reranking, and semantic query caching with FlashStore integration.
pub struct RagEngine {
    config: RagConfig,
    dense_index: DenseIndex,
    sparse_index: SparseIndex,
    semantic_cache: SemanticCache,
    cross_encoder: LexicalSemanticCrossEncoder,
    /// In-memory document text store for fast reranking and lookup.
    documents: HashMap<String, String>,
    /// Optional underlying persistent FlashStore KV engine.
    store: Option<Arc<FlashStore>>,
}

impl RagEngine {
    /// Creates a new in-memory `RagEngine` with the given configuration.
    pub fn new(config: RagConfig) -> Self {
        let dense = DenseIndex::new(config.embedding_dim, config.similarity_metric);
        let sparse = SparseIndex::new(config.bm25_k1, config.bm25_b);
        let cache = SemanticCache::new(
            config.semantic_cache_capacity,
            config.semantic_cache_threshold,
            config.semantic_cache_ttl_secs,
        );
        let cross = LexicalSemanticCrossEncoder::default();

        Self {
            config,
            dense_index: dense,
            sparse_index: sparse,
            semantic_cache: cache,
            cross_encoder: cross,
            documents: HashMap::new(),
            store: None,
        }
    }

    /// Creates a `RagEngine` backed by a persistent `FlashStore` key-value engine.
    pub fn with_store(config: RagConfig, store: Arc<FlashStore>) -> Self {
        let mut engine = Self::new(config);
        engine.store = Some(store);
        engine
    }

    /// Access the configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Retrieves document text by doc_id.
    pub fn get_document_text(&self, doc_id: &str) -> Option<&str> {
        self.documents.get(doc_id).map(|s| s.as_str())
    }

    /// Indexes a document with text and optional dense vector embedding.
    pub fn add_document(
        &mut self,
        doc_id: impl Into<String>,
        text: &str,
        embedding: Option<Vec<f32>>,
    ) -> Result<()> {
        let id = doc_id.into();

        if let Some(ref emb) = embedding {
            self.dense_index.insert(id.clone(), emb.clone())?;
        }

        self.sparse_index.add_document(id.clone(), text);
        self.documents.insert(id.clone(), text.to_string());

        // Optional persistence to FlashStore
        if let Some(ref db) = self.store {
            let doc_key = [PREFIX_DOC, id.as_bytes()].concat();
            db.put(doc_key, text.as_bytes())?;

            if let Some(ref emb) = embedding {
                let vec_key = [PREFIX_VEC, id.as_bytes()].concat();
                let encoded = bincode::serialize(emb)?;
                db.put(vec_key, encoded)?;
            }
        }

        Ok(())
    }

    /// Deletes a document from dense index, sparse index, and document store.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<bool> {
        let removed_dense = self.dense_index.remove(doc_id).is_some();
        let removed_sparse = self.sparse_index.remove_document(doc_id);
        let removed_doc = self.documents.remove(doc_id).is_some();

        if let Some(ref db) = self.store {
            let doc_key = [PREFIX_DOC, doc_id.as_bytes()].concat();
            db.delete(doc_key)?;
            let vec_key = [PREFIX_VEC, doc_id.as_bytes()].concat();
            db.delete(vec_key)?;
        }

        Ok(removed_dense || removed_sparse || removed_doc)
    }

    /// Executes hybrid retrieval for a query.
    ///
    /// 1. Checks semantic cache if `query_embedding` is provided. On cache hit, returns cached hits immediately.
    /// 2. Performs sparse BM25 retrieval.
    /// 3. Performs dense vector retrieval if `query_embedding` is provided.
    /// 4. Fuses rankings using RRF or linear combination.
    /// 5. Updates semantic cache on query_embedding presence.
    pub fn search(
        &mut self,
        query: &str,
        query_embedding: Option<&[f32]>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        // 1. Semantic Cache check
        if let Some(emb) = query_embedding {
            if let Some(cached) = self.semantic_cache.lookup(emb) {
                let mut hits = cached;
                hits.truncate(top_k);
                return Ok(hits);
            }
        }

        // 2. Sparse retrieval
        let sparse_hits = self.sparse_index.search(query, top_k * 2)?;

        // 3. Dense retrieval
        let dense_hits = if let Some(emb) = query_embedding {
            self.dense_index.search(emb, top_k * 2)?
        } else {
            Vec::new()
        };

        // 4. Hybrid fusion
        let hits = if !dense_hits.is_empty() && !sparse_hits.is_empty() {
            if self.config.rrf_k > 0 {
                reciprocal_rank_fusion(&dense_hits, &sparse_hits, self.config.rrf_k, top_k)
            } else {
                weighted_linear_fusion(
                    &dense_hits,
                    &sparse_hits,
                    self.config.hybrid_dense_weight,
                    top_k,
                )
            }
        } else if !dense_hits.is_empty() {
            dense_hits
                .into_iter()
                .take(top_k)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                })
                .collect()
        } else {
            sparse_hits
                .into_iter()
                .take(top_k)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    dense_score: None,
                    sparse_score: Some(score),
                })
                .collect()
        };

        // 5. Populate semantic cache
        if let Some(emb) = query_embedding {
            let _ = self
                .semantic_cache
                .insert(query, emb.to_vec(), hits.clone());
        }

        Ok(hits)
    }

    /// Reranks search results using the cross-encoder scoring model.
    pub fn rerank(&self, query: &str, hits: &[SearchResult], top_k: usize) -> Vec<RerankResult> {
        let candidates: Vec<(String, String, f32)> = hits
            .iter()
            .filter_map(|hit| {
                self.documents
                    .get(&hit.doc_id)
                    .map(|text| (hit.doc_id.clone(), text.clone(), hit.score))
            })
            .collect();

        rerank_candidates(&self.cross_encoder, query, &candidates, top_k)
    }

    /// Cache performance metrics: (hits, misses, hit_rate).
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        (
            self.semantic_cache.total_hits(),
            self.semantic_cache.total_misses(),
            self.semantic_cache.hit_rate(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_rag_engine_end_to_end() {
        let config = RagConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut engine = RagEngine::new(config);

        engine
            .add_document(
                "doc1",
                "FlashStore is a thread-safe embedded LSM tree",
                Some(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();

        engine
            .add_document(
                "doc2",
                "Postgres is an open source relational SQL database",
                Some(vec![0.0, 1.0, 0.0]),
            )
            .unwrap();

        assert_eq!(engine.doc_count(), 2);

        // Search with query and embedding
        let hits = engine
            .search("lsm tree embedded", Some(&[1.0, 0.0, 0.0]), 2)
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, "doc1");

        // Subsequent query with same embedding -> semantic cache hit
        let cached_hits = engine
            .search("lsm tree store", Some(&[1.0, 0.0, 0.0]), 2)
            .unwrap();
        assert_eq!(cached_hits[0].doc_id, "doc1");
        assert_eq!(engine.cache_stats().0, 1);

        // Cross-encoder rerank
        let reranked = engine.rerank("thread safe lsm tree", &hits, 2);
        assert!(!reranked.is_empty());
        assert_eq!(reranked[0].doc_id, "doc1");

        // Delete
        assert!(engine.delete_document("doc1").unwrap());
        assert_eq!(engine.doc_count(), 1);
    }

    #[test]
    fn test_rag_engine_with_flashstore_backend() {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let store = Arc::new(FlashStore::open(options).unwrap());

        let config = RagConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut engine = RagEngine::with_store(config, store.clone());

        engine
            .add_document("doc1", "rust storage engine", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();

        // Verify underlying FlashStore has raw bytes
        let raw_doc = store.get([PREFIX_DOC, b"doc1"].concat()).unwrap();
        assert!(raw_doc.is_some());
        assert_eq!(raw_doc.unwrap(), b"rust storage engine".as_slice());
    }
}
