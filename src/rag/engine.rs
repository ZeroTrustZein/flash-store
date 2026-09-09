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
use crate::rag::types::{
    Document, DocumentId, DocumentMetadata, Embedding, FusionStrategy, RagQuery, ScoredDocument,
    SemanticCacheStats,
};

/// Prefix for document raw text in FlashStore KV engine.
pub const PREFIX_DOC: &[u8] = b"rag:doc:";
/// Prefix for document vector embeddings in FlashStore KV engine.
pub const PREFIX_VEC: &[u8] = b"rag:vec:";
/// Prefix for document metadata in FlashStore KV engine.
pub const PREFIX_META: &[u8] = b"rag:meta:";

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
    /// In-memory document metadata store.
    metadata: HashMap<String, DocumentMetadata>,
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
            metadata: HashMap::new(),
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
        self.metadata.entry(id.clone()).or_default();

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

    /// Indexes a complete `Document` domain model with text, embedding, and metadata.
    pub fn add_document_model(&mut self, doc: Document) -> Result<()> {
        let id = doc.id.to_string();
        let text = doc.text;
        let embedding = doc.embedding;
        let metadata = doc.metadata;

        if let Some(ref emb) = embedding {
            self.dense_index
                .insert(id.clone(), emb.as_slice().to_vec())?;
        }

        self.sparse_index.add_document(id.clone(), &text);
        self.documents.insert(id.clone(), text.clone());
        self.metadata.insert(id.clone(), metadata.clone());

        if let Some(ref db) = self.store {
            let doc_key = [PREFIX_DOC, id.as_bytes()].concat();
            db.put(doc_key, text.as_bytes())?;

            if let Some(ref emb) = embedding {
                let vec_key = [PREFIX_VEC, id.as_bytes()].concat();
                let encoded = bincode::serialize(emb.as_slice())?;
                db.put(vec_key, encoded)?;
            }

            if !metadata.is_empty() {
                let meta_key = [PREFIX_META, id.as_bytes()].concat();
                let encoded_meta = serde_json::to_vec(&metadata)?;
                db.put(meta_key, encoded_meta)?;
            }
        }

        Ok(())
    }

    /// Deletes a document from dense index, sparse index, and document store.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<bool> {
        let removed_dense = self.dense_index.remove(doc_id).is_some();
        let removed_sparse = self.sparse_index.remove_document(doc_id);
        let removed_doc = self.documents.remove(doc_id).is_some();
        let removed_meta = self.metadata.remove(doc_id).is_some();

        if let Some(ref db) = self.store {
            let doc_key = [PREFIX_DOC, doc_id.as_bytes()].concat();
            db.delete(doc_key)?;
            let vec_key = [PREFIX_VEC, doc_id.as_bytes()].concat();
            db.delete(vec_key)?;
            let meta_key = [PREFIX_META, doc_id.as_bytes()].concat();
            db.delete(meta_key)?;
        }

        Ok(removed_dense || removed_sparse || removed_doc || removed_meta)
    }

    /// Retrieves a document by id as a domain model.
    pub fn get_document(&self, doc_id: &str) -> Option<Document> {
        let text = self.documents.get(doc_id)?;
        let metadata = self.metadata.get(doc_id).cloned().unwrap_or_default();
        let embedding = self
            .dense_index
            .get(doc_id)
            .map(|v| Embedding::from_slice(v));
        Some(Document {
            id: DocumentId::new(doc_id),
            text: text.clone(),
            embedding,
            metadata,
            chunks: Vec::new(),
            created_at: 0,
            updated_at: 0,
        })
    }

    /// Retrieves document metadata by doc_id.
    pub fn get_document_metadata(&self, doc_id: &str) -> Option<&DocumentMetadata> {
        self.metadata.get(doc_id)
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

    /// Telemetry statistics for the semantic query cache.
    pub fn semantic_cache_stats(&self) -> SemanticCacheStats {
        SemanticCacheStats {
            capacity: self.semantic_cache.capacity(),
            len: self.semantic_cache.len(),
            hits: self.semantic_cache.total_hits(),
            misses: self.semantic_cache.total_misses(),
            hit_rate: self.semantic_cache.hit_rate(),
        }
    }

    /// Executes a structured `RagQuery`, returning rich `ScoredDocument` domain models.
    pub fn search_query(&mut self, query: &RagQuery) -> Result<Vec<ScoredDocument>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }

        // 1. Semantic cache lookup if embedding is provided and no metadata filter is active
        if query.filter.is_none() {
            if let Some(ref emb) = query.embedding {
                if let Some(cached_hits) = self.semantic_cache.lookup(emb.as_slice()) {
                    let mut results: Vec<ScoredDocument> = cached_hits
                        .into_iter()
                        .take(query.top_k)
                        .map(|sr| {
                            let text = self.documents.get(&sr.doc_id).cloned();
                            let metadata = self.metadata.get(&sr.doc_id).cloned();
                            let mut sd: ScoredDocument = sr.into();
                            sd.text = text;
                            sd.metadata = metadata;
                            sd
                        })
                        .collect();
                    if let Some(min_score) = query.min_score {
                        results.retain(|r| r.score >= min_score);
                    }
                    return Ok(results);
                }
            }
        }

        let fetch_k = query.top_k * 3;

        // 2. Sparse retrieval
        let sparse_hits = if !matches!(query.fusion_strategy, FusionStrategy::DenseOnly) {
            self.sparse_index.search(&query.text, fetch_k)?
        } else {
            Vec::new()
        };

        // 3. Dense retrieval
        let dense_hits = if !matches!(query.fusion_strategy, FusionStrategy::SparseOnly) {
            if let Some(ref emb) = query.embedding {
                self.dense_index.search(emb.as_slice(), fetch_k)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // 4. Fusion according to strategy
        let mut hits = match query.fusion_strategy {
            FusionStrategy::Rrf { k } => {
                if !dense_hits.is_empty() && !sparse_hits.is_empty() {
                    reciprocal_rank_fusion(&dense_hits, &sparse_hits, k, fetch_k)
                } else if !dense_hits.is_empty() {
                    dense_hits
                        .into_iter()
                        .take(fetch_k)
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
                        .take(fetch_k)
                        .map(|(doc_id, score)| SearchResult {
                            doc_id,
                            score,
                            dense_score: None,
                            sparse_score: Some(score),
                        })
                        .collect()
                }
            }
            FusionStrategy::WeightedLinear { dense_weight } => {
                if !dense_hits.is_empty() && !sparse_hits.is_empty() {
                    weighted_linear_fusion(&dense_hits, &sparse_hits, dense_weight, fetch_k)
                } else if !dense_hits.is_empty() {
                    dense_hits
                        .into_iter()
                        .take(fetch_k)
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
                        .take(fetch_k)
                        .map(|(doc_id, score)| SearchResult {
                            doc_id,
                            score,
                            dense_score: None,
                            sparse_score: Some(score),
                        })
                        .collect()
                }
            }
            FusionStrategy::DenseOnly => dense_hits
                .into_iter()
                .take(fetch_k)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                })
                .collect(),
            FusionStrategy::SparseOnly => sparse_hits
                .into_iter()
                .take(fetch_k)
                .map(|(doc_id, score)| SearchResult {
                    doc_id,
                    score,
                    dense_score: None,
                    sparse_score: Some(score),
                })
                .collect(),
        };

        // 5. Metadata filtering
        if let Some(ref filter) = query.filter {
            hits.retain(|hit| {
                if let Some(meta) = self.metadata.get(&hit.doc_id) {
                    filter.matches(meta)
                } else {
                    false
                }
            });
        }

        // 6. Populate semantic cache if no filter
        if query.filter.is_none() {
            if let Some(ref emb) = query.embedding {
                let _ =
                    self.semantic_cache
                        .insert(&query.text, emb.as_slice().to_vec(), hits.clone());
            }
        }

        // 7. Rerank if requested
        let mut scored_docs: Vec<ScoredDocument> = if query.rerank {
            let rerank_k = query.rerank_top_k.unwrap_or(query.top_k);
            let reranked = self.rerank(&query.text, &hits, rerank_k);
            let rank_map: HashMap<String, RerankResult> = reranked
                .into_iter()
                .map(|r| (r.doc_id.clone(), r))
                .collect();

            hits.into_iter()
                .filter_map(|hit| {
                    if let Some(rr) = rank_map.get(&hit.doc_id) {
                        let text = self.documents.get(&hit.doc_id).cloned();
                        let metadata = self.metadata.get(&hit.doc_id).cloned();
                        let explanation = text
                            .as_ref()
                            .map(|t| self.cross_encoder.explain(&query.text, t));
                        Some(ScoredDocument {
                            id: DocumentId::new(&hit.doc_id),
                            score: rr.reranked_score,
                            dense_score: hit.dense_score,
                            sparse_score: hit.sparse_score,
                            text,
                            metadata,
                            explanation,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            hits.into_iter()
                .take(query.top_k)
                .map(|hit| {
                    let text = self.documents.get(&hit.doc_id).cloned();
                    let metadata = self.metadata.get(&hit.doc_id).cloned();
                    ScoredDocument {
                        id: DocumentId::new(hit.doc_id),
                        score: hit.score,
                        dense_score: hit.dense_score,
                        sparse_score: hit.sparse_score,
                        text,
                        metadata,
                        explanation: None,
                    }
                })
                .collect()
        };

        scored_docs.sort_by(|a, b| b.score.total_cmp(&a.score));

        if let Some(min_score) = query.min_score {
            scored_docs.retain(|doc| doc.score >= min_score);
        }

        scored_docs.truncate(query.top_k);
        Ok(scored_docs)
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
