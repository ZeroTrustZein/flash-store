use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::engine::FlashStore;
use crate::error::Result;
use crate::rag::cache::SemanticCache;
use crate::rag::config::RagConfig;
use crate::rag::context::{AssembledContext, ContextAssembler, ContextConfig};
use crate::rag::dense::DenseIndex;
use crate::rag::hybrid::{reciprocal_rank_fusion, weighted_linear_fusion, SearchResult};
use crate::rag::metrics::{RagMetricsSnapshot, RagMetricsTracker};
use crate::rag::reranker::{
    maximal_marginal_relevance, rerank_candidates, LexicalSemanticCrossEncoder, RerankResult,
};
use crate::rag::sparse::SparseIndex;
pub use crate::rag::store::{
    RagStoreAdapter, PREFIX_CHUNK, PREFIX_DOC, PREFIX_META, PREFIX_SYS, PREFIX_VEC,
};
use crate::rag::types::{
    ChunkingConfig, Document, DocumentId, DocumentMetadata, Embedding, FusionStrategy, RagQuery,
    ScoredDocument, SemanticCacheStats,
};

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
    /// Storage adapter handling atomic WriteBatch and recovery.
    store_adapter: Option<RagStoreAdapter>,
    /// Telemetry metrics tracker across retrieval stages.
    metrics: Arc<RagMetricsTracker>,
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
            store_adapter: None,
            metrics: Arc::new(RagMetricsTracker::new()),
        }
    }

    /// Creates a `RagEngine` backed by a persistent `FlashStore` key-value engine.
    pub fn with_store(config: RagConfig, store: Arc<FlashStore>) -> Self {
        let mut engine = Self::new(config);
        engine.store = Some(store.clone());
        engine.store_adapter = Some(RagStoreAdapter::new(store));
        engine
    }

    /// Recovers a `RagEngine` by scanning and rehydrating all indexed documents,
    /// embeddings, metadata, and chunks stored in the persistent `FlashStore`.
    pub fn recover(config: RagConfig, store: Arc<FlashStore>) -> Result<Self> {
        let adapter = RagStoreAdapter::new(store.clone());
        let docs = adapter.recover_all()?;
        let mut engine = Self::with_store(config, store);
        for doc in docs {
            engine.index_in_memory(&doc)?;
        }
        Ok(engine)
    }

    /// Internal helper to index document into in-memory structures without writing to FlashStore.
    fn index_in_memory(&mut self, doc: &Document) -> Result<()> {
        let id_str = doc.id.to_string();
        if let Some(ref emb) = doc.embedding {
            self.dense_index
                .insert(id_str.clone(), emb.as_slice().to_vec())?;
        }
        self.sparse_index.add_document(id_str.clone(), &doc.text);
        self.documents.insert(id_str.clone(), doc.text.clone());
        self.metadata.insert(id_str.clone(), doc.metadata.clone());
        self.semantic_cache.invalidate_for_doc(&id_str);
        Ok(())
    }

    /// Access the configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// Total number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Total number of dense vectors indexed.
    pub fn dense_vector_count(&self) -> usize {
        self.dense_index.len()
    }

    /// Total number of sparse vocabulary terms indexed.
    pub fn sparse_term_count(&self) -> usize {
        self.sparse_index.term_count()
    }

    /// Returns all indexed document IDs in sorted order.
    pub fn document_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.documents.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Clears the semantic vector query cache.
    pub fn clear_cache(&mut self) {
        self.semantic_cache.clear();
    }

    /// Retrieves document text by doc_id.
    pub fn get_document_text(&self, doc_id: &str) -> Option<&str> {
        self.documents.get(doc_id).map(|s| s.as_str())
    }

    /// Returns a telemetry snapshot of retrieval subsystem metrics.
    pub fn metrics(&self) -> RagMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Access the underlying store adapter if store is attached.
    pub fn store_adapter(&self) -> Option<&RagStoreAdapter> {
        self.store_adapter.as_ref()
    }

    /// Assembles context block for retrieved documents using the context assembly subsystem.
    pub fn assemble_context(
        &self,
        hits: &[ScoredDocument],
        config: &ContextConfig,
    ) -> AssembledContext {
        ContextAssembler::assemble(hits, config)
    }

    /// Indexes a document with text and optional dense vector embedding.
    pub fn add_document(
        &mut self,
        doc_id: impl Into<String>,
        text: &str,
        embedding: Option<Vec<f32>>,
    ) -> Result<()> {
        let doc = Document {
            id: DocumentId::new(doc_id),
            text: text.to_string(),
            embedding: embedding.map(Embedding::new),
            metadata: DocumentMetadata::default(),
            chunks: Vec::new(),
            created_at: 0,
            updated_at: 0,
        };
        self.add_document_model(doc)
    }

    /// Indexes a complete `Document` domain model with text, embedding, and metadata.
    pub fn add_document_model(&mut self, doc: Document) -> Result<()> {
        self.index_in_memory(&doc)?;

        if let Some(ref adapter) = self.store_adapter {
            adapter.persist_document(&doc)?;
        }

        Ok(())
    }

    /// Atomically persists and indexes a batch of documents into memory and FlashStore.
    pub fn batch_add_documents(&mut self, docs: &[Document]) -> Result<()> {
        for doc in docs {
            self.index_in_memory(doc)?;
        }
        if let Some(ref adapter) = self.store_adapter {
            adapter.persist_documents_batch(docs)?;
        }
        Ok(())
    }

    /// Decomposes a `Document` into passages according to `chunk_config` and indexes it.
    pub fn add_document_with_chunks(
        &mut self,
        doc: Document,
        chunk_config: &ChunkingConfig,
    ) -> Result<()> {
        let chunked = doc.chunk(chunk_config);
        self.add_document_model(chunked)
    }

    /// Deletes a document from dense index, sparse index, and document store.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<bool> {
        let removed_dense = self.dense_index.remove(doc_id).is_some();
        let removed_sparse = self.sparse_index.remove_document(doc_id);
        let removed_doc = self.documents.remove(doc_id).is_some();
        let removed_meta = self.metadata.remove(doc_id).is_some();
        self.semantic_cache.invalidate_for_doc(doc_id);

        if let Some(ref adapter) = self.store_adapter {
            let _ = adapter.delete_document(doc_id)?;
        }

        Ok(removed_dense || removed_sparse || removed_doc || removed_meta)
    }

    /// Retrieves a document by id as a domain model.
    pub fn get_document(&self, doc_id: &str) -> Option<Document> {
        if let Some(ref adapter) = self.store_adapter {
            if let Ok(Some(doc)) = adapter.load_document(doc_id) {
                return Some(doc);
            }
        }
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
        let start = Instant::now();
        self.metrics
            .total_searches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if top_k == 0 {
            self.metrics.search_latency.record(start.elapsed());
            return Ok(Vec::new());
        }

        // 1. Semantic Cache check
        if let Some(emb) = query_embedding {
            let cached_opt = self.semantic_cache.lookup(emb);
            self.metrics.record_cache_lookup(cached_opt.is_some());
            if let Some(mut hits) = cached_opt {
                hits.truncate(top_k);
                self.metrics.search_latency.record(start.elapsed());
                return Ok(hits);
            }
        }

        // 2. Sparse retrieval
        let t_sparse = Instant::now();
        let sparse_hits = self.sparse_index.search(query, top_k * 2)?;
        self.metrics.sparse_latency.record(t_sparse.elapsed());

        // 3. Dense retrieval
        let t_dense = Instant::now();
        let dense_hits = if let Some(emb) = query_embedding {
            let hits = self.dense_index.search(emb, top_k * 2)?;
            self.metrics.dense_latency.record(t_dense.elapsed());
            hits
        } else {
            Vec::new()
        };

        // 4. Hybrid fusion
        let t_fusion = Instant::now();
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
            dense_hits_to_search_results(dense_hits, top_k)
        } else {
            sparse_hits_to_search_results(sparse_hits, top_k)
        };
        self.metrics.fusion_latency.record(t_fusion.elapsed());

        // 5. Populate semantic cache
        if let Some(emb) = query_embedding {
            let _ = self
                .semantic_cache
                .insert(query, emb.to_vec(), hits.clone());
        }

        self.metrics.search_latency.record(start.elapsed());
        Ok(hits)
    }

    /// Reranks search results using the cross-encoder scoring model.
    pub fn rerank(&self, query: &str, hits: &[SearchResult], top_k: usize) -> Vec<RerankResult> {
        let t_rerank = Instant::now();
        let candidates: Vec<(String, String, f32)> = hits
            .iter()
            .filter_map(|hit| {
                self.documents
                    .get(&hit.doc_id)
                    .map(|text| (hit.doc_id.clone(), text.clone(), hit.score))
            })
            .collect();

        let results = rerank_candidates(&self.cross_encoder, query, &candidates, top_k);
        self.metrics.rerank_latency.record(t_rerank.elapsed());
        results
    }

    /// Selects diverse documents using Maximal Marginal Relevance (MMR).
    pub fn rerank_mmr(
        &self,
        query_embedding: &[f32],
        hits: &[SearchResult],
        lambda: f32,
        top_k: usize,
    ) -> Vec<RerankResult> {
        let candidates: Vec<(String, Vec<f32>, f32)> = hits
            .iter()
            .map(|hit| {
                let vec = self
                    .dense_index
                    .get(&hit.doc_id)
                    .cloned()
                    .unwrap_or_default();
                (hit.doc_id.clone(), vec, hit.score)
            })
            .collect();

        maximal_marginal_relevance(query_embedding, &candidates, lambda, top_k)
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
        self.semantic_cache.stats()
    }

    /// Executes a structured `RagQuery`, returning rich `ScoredDocument` domain models.
    pub fn search_query(&mut self, query: &RagQuery) -> Result<Vec<ScoredDocument>> {
        let start = Instant::now();
        self.metrics
            .total_searches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if query.top_k == 0 {
            self.metrics.search_latency.record(start.elapsed());
            return Ok(Vec::new());
        }

        // 1. Semantic cache lookup if embedding is provided and no metadata filter is active
        if query.filter.is_none() {
            if let Some(ref emb) = query.embedding {
                let cached_lookup = self.semantic_cache.lookup(emb.as_slice());
                self.metrics.record_cache_lookup(cached_lookup.is_some());
                if let Some(cached_hits) = cached_lookup {
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
                    self.metrics.search_latency.record(start.elapsed());
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
                    dense_hits_to_search_results(dense_hits, fetch_k)
                } else {
                    sparse_hits_to_search_results(sparse_hits, fetch_k)
                }
            }
            FusionStrategy::WeightedLinear { dense_weight } => {
                if !dense_hits.is_empty() && !sparse_hits.is_empty() {
                    weighted_linear_fusion(&dense_hits, &sparse_hits, dense_weight, fetch_k)
                } else if !dense_hits.is_empty() {
                    dense_hits_to_search_results(dense_hits, fetch_k)
                } else {
                    sparse_hits_to_search_results(sparse_hits, fetch_k)
                }
            }
            FusionStrategy::DenseOnly => dense_hits_to_search_results(dense_hits, fetch_k),
            FusionStrategy::SparseOnly => sparse_hits_to_search_results(sparse_hits, fetch_k),
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
        let mut scored_docs: Vec<ScoredDocument> = if let Some(lambda) = query.mmr_lambda {
            let query_emb = query
                .embedding
                .as_ref()
                .map(|e| e.as_slice())
                .unwrap_or(&[]);
            let mmr_results = self.rerank_mmr(query_emb, &hits, lambda, query.top_k);
            let mmr_map: HashMap<String, RerankResult> = mmr_results
                .into_iter()
                .map(|r| (r.doc_id.clone(), r))
                .collect();

            hits.into_iter()
                .filter_map(|hit| {
                    if let Some(rr) = mmr_map.get(&hit.doc_id) {
                        let text = self.documents.get(&hit.doc_id).cloned();
                        let metadata = self.metadata.get(&hit.doc_id).cloned();
                        Some(ScoredDocument {
                            id: DocumentId::new(&hit.doc_id),
                            score: rr.reranked_score,
                            dense_score: hit.dense_score,
                            sparse_score: hit.sparse_score,
                            text,
                            metadata,
                            explanation: None,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        } else if query.rerank {
            let rerank_k = query.rerank_top_k.unwrap_or(query.top_k);
            let candidates: Vec<(String, String, f32)> = hits
                .iter()
                .filter_map(|hit| {
                    self.documents
                        .get(&hit.doc_id)
                        .map(|text| (hit.doc_id.clone(), text.clone(), hit.score))
                })
                .collect();

            let reranked = crate::rag::reranker::rerank_candidates_with_explanation(
                &self.cross_encoder,
                &query.text,
                &candidates,
                rerank_k,
            );
            let rank_map: HashMap<String, RerankResult> = reranked
                .into_iter()
                .map(|r| (r.doc_id.clone(), r))
                .collect();

            hits.into_iter()
                .filter_map(|hit| {
                    if let Some(rr) = rank_map.get(&hit.doc_id) {
                        let text = self.documents.get(&hit.doc_id).cloned();
                        let metadata = self.metadata.get(&hit.doc_id).cloned();
                        Some(ScoredDocument {
                            id: DocumentId::new(&hit.doc_id),
                            score: rr.reranked_score,
                            dense_score: hit.dense_score,
                            sparse_score: hit.sparse_score,
                            text,
                            metadata,
                            explanation: rr.explanation.clone(),
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
        self.metrics.search_latency.record(start.elapsed());
        Ok(scored_docs)
    }
}

#[inline]
fn dense_hits_to_search_results(hits: Vec<(String, f32)>, top_k: usize) -> Vec<SearchResult> {
    hits.into_iter()
        .take(top_k)
        .map(|(doc_id, score)| SearchResult {
            doc_id,
            score,
            dense_score: Some(score),
            sparse_score: None,
        })
        .collect()
}

#[inline]
fn sparse_hits_to_search_results(hits: Vec<(String, f32)>, top_k: usize) -> Vec<SearchResult> {
    hits.into_iter()
        .take(top_k)
        .map(|(doc_id, score)| SearchResult {
            doc_id,
            score,
            dense_score: None,
            sparse_score: Some(score),
        })
        .collect()
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

    #[test]
    fn test_rag_engine_cache_invalidation_on_delete_and_update() {
        let config = RagConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut engine = RagEngine::new(config);

        engine
            .add_document("doc1", "first version text", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();

        // Search populates cache
        let hits = engine
            .search("version text", Some(&[1.0, 0.0, 0.0]), 1)
            .unwrap();
        assert_eq!(hits[0].doc_id, "doc1");
        assert_eq!(engine.cache_stats().1, 1); // 1 miss

        // Repeat search hits cache
        let hits2 = engine
            .search("version text", Some(&[1.0, 0.0, 0.0]), 1)
            .unwrap();
        assert_eq!(hits2[0].doc_id, "doc1");
        assert_eq!(engine.cache_stats().0, 1); // 1 hit

        // Updating doc1 invalidates cache
        engine
            .add_document("doc1", "updated version text", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();

        // Next search should miss cache because it was invalidated
        let _ = engine
            .search("version text", Some(&[1.0, 0.0, 0.0]), 1)
            .unwrap();
        assert_eq!(engine.cache_stats().1, 2); // 2 misses now!

        // Deleting doc1 also invalidates cache
        engine.delete_document("doc1").unwrap();
        assert_eq!(engine.doc_count(), 0);
    }

    #[test]
    fn test_rag_engine_mmr_rerank() {
        use crate::rag::types::ChunkingStrategy;

        let config = RagConfig {
            embedding_dim: 3,
            ..Default::default()
        };
        let mut engine = RagEngine::new(config);

        // doc1 and doc2 are almost identical, doc3 is diverse
        engine
            .add_document("doc1", "storage engine indexing", Some(vec![1.0, 0.0, 0.0]))
            .unwrap();
        engine
            .add_document(
                "doc2",
                "storage engine indexing fast",
                Some(vec![0.99, 0.01, 0.0]),
            )
            .unwrap();
        engine
            .add_document(
                "doc3",
                "distributed network consensus",
                Some(vec![0.0, 1.0, 0.0]),
            )
            .unwrap();

        let query = RagQuery::builder("storage distributed")
            .embedding(vec![1.0, 0.0, 0.0])
            .top_k(2)
            .mmr(0.3)
            .build();

        let results = engine.search_query(&query).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.as_str(), "doc1");
        assert_eq!(results[1].id.as_str(), "doc3");

        // Document chunking integration
        let chunk_config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedTokens {
                size: 2,
                overlap: 0,
            },
            min_chunk_size: 2,
        };
        let doc = Document::new("chunk_doc", "one two three four five six");
        engine.add_document_with_chunks(doc, &chunk_config).unwrap();
        let loaded = engine.get_document("chunk_doc").unwrap();
        assert_eq!(loaded.id.as_str(), "chunk_doc");
    }

    #[test]
    fn test_rag_engine_recovery_from_flashstore() {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let store = Arc::new(FlashStore::open(options).unwrap());

        let config = RagConfig {
            embedding_dim: 2,
            ..Default::default()
        };

        // Populate with first engine instance
        {
            let mut engine = RagEngine::with_store(config.clone(), store.clone());
            let doc1 = Document::builder("docA", "First persistent systems recovery document")
                .embedding(vec![1.0, 0.0])
                .metadata_field("topic", "systems")
                .build();
            let doc2 = Document::builder("docB", "Second persistent algorithms recovery document")
                .embedding(vec![0.0, 1.0])
                .metadata_field("topic", "algorithms")
                .build();

            engine.batch_add_documents(&[doc1, doc2]).unwrap();
            assert_eq!(engine.doc_count(), 2);
        }

        // Recover in a fresh engine instance
        let mut recovered_engine = RagEngine::recover(config, store).unwrap();
        assert_eq!(recovered_engine.doc_count(), 2);

        let hits = recovered_engine
            .search("systems recovery", Some(&[1.0, 0.0]), 1)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "docA");

        // Metrics verification
        let metrics = recovered_engine.metrics();
        assert_eq!(metrics.total_searches, 1);
        assert!(metrics.search_latency.count >= 1);

        // Context assembly verification
        let query = RagQuery::builder("systems").top_k(1).build();
        let scored = recovered_engine.search_query(&query).unwrap();
        let ctx_cfg = ContextConfig::default();
        let assembled = recovered_engine.assemble_context(&scored, &ctx_cfg);
        assert_eq!(assembled.doc_count, 1);
        assert!(assembled.text.contains("docA"));
    }
}
