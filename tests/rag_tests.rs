use flash_store::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn test_scaffold_rag_module_hierarchy() {
    let config = RagConfigBuilder::new()
        .embedding_dim(4)
        .similarity_metric(SimilarityMetric::Cosine)
        .bm25_params(1.2, 0.75)
        .hybrid_dense_weight(0.6)
        .rrf_k(50)
        .semantic_cache(100, 0.90, 600)
        .build();

    let mut engine = RagEngine::new(config);
    assert_eq!(engine.doc_count(), 0);

    // Verify empty search produces empty results gracefully
    let hits = engine.search("database", None, 5).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn test_rag_dense_and_sparse_indexing() {
    let config = RagConfigBuilder::new()
        .embedding_dim(3)
        .similarity_metric(SimilarityMetric::Cosine)
        .build();

    let mut engine = RagEngine::new(config);

    engine
        .add_document(
            "doc1",
            "FlashStore is a lightning-fast LSM storage engine in Rust",
            Some(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();

    engine
        .add_document(
            "doc2",
            "RocksDB is an LSM key-value store in C++",
            Some(vec![0.8, 0.2, 0.0]),
        )
        .unwrap();

    engine
        .add_document(
            "doc3",
            "PostgreSQL is an ACID relational database",
            Some(vec![0.0, 1.0, 0.0]),
        )
        .unwrap();

    assert_eq!(engine.doc_count(), 3);

    // 1. Text-only sparse search
    let sparse_only = engine.search("LSM storage", None, 2).unwrap();
    assert_eq!(sparse_only.len(), 2);
    assert_eq!(sparse_only[0].doc_id, "doc1");

    // 2. Hybrid search with query vector
    let hybrid_hits = engine
        .search("database", Some(&[0.0, 1.0, 0.0]), 2)
        .unwrap();
    assert_eq!(hybrid_hits.len(), 2);
    assert_eq!(hybrid_hits[0].doc_id, "doc3");

    // 3. Cross-encoder rerank
    let reranked = engine.rerank("PostgreSQL relational database", &hybrid_hits, 2);
    assert_eq!(reranked.len(), 2);
    assert_eq!(reranked[0].doc_id, "doc3");
}

#[test]
fn test_rag_semantic_cache_hit_and_eviction() {
    let config = RagConfigBuilder::new()
        .embedding_dim(2)
        .semantic_cache(2, 0.95, 3600)
        .build();

    let mut engine = RagEngine::new(config);

    engine
        .add_document(
            "doc1",
            "Rust async runtime Tokio documentation",
            Some(vec![1.0, 0.0]),
        )
        .unwrap();

    // Query 1: Cache Miss
    let res1 = engine
        .search("async runtime", Some(&[1.0, 0.0]), 1)
        .unwrap();
    assert_eq!(res1.len(), 1);
    let (hits, misses, _) = engine.cache_stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 1);

    // Query 2: Identical embedding -> Semantic Cache Hit
    let res2 = engine.search("tokio async", Some(&[1.0, 0.0]), 1).unwrap();
    assert_eq!(res2.len(), 1);
    let (hits, misses, _) = engine.cache_stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);

    // Query 3: Different embedding -> Semantic Cache Miss
    let res3 = engine.search("networking", Some(&[0.0, 1.0]), 1).unwrap();
    let (hits, misses, _) = engine.cache_stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 2);
    assert_eq!(res3.len(), 1);
    assert_eq!(res3[0].score, 0.0);
}

#[test]
fn test_rag_engine_with_flashstore_persistence() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let config = RagConfigBuilder::new()
        .embedding_dim(2)
        .similarity_metric(SimilarityMetric::Cosine)
        .build();

    let mut engine = RagEngine::with_store(config, store.clone());

    engine
        .add_document(
            "doc_persist",
            "Persistent data indexed by FlashStore LSM-tree engine",
            Some(vec![1.0, 0.0]),
        )
        .unwrap();

    // Verify key in FlashStore
    let stored_doc = store.get(b"rag:doc:doc_persist").unwrap();
    assert!(stored_doc.is_some());
    assert_eq!(
        stored_doc.unwrap(),
        b"Persistent data indexed by FlashStore LSM-tree engine".as_slice()
    );

    let stored_vec = store.get(b"rag:vec:doc_persist").unwrap();
    assert!(stored_vec.is_some());
    let decoded_vec: Vec<f32> = bincode::deserialize(&stored_vec.unwrap()).unwrap();
    assert_eq!(decoded_vec, vec![1.0, 0.0]);

    // Deletion clears persistence
    assert!(engine.delete_document("doc_persist").unwrap());
    assert!(store.get(b"rag:doc:doc_persist").unwrap().is_none());
    assert!(store.get(b"rag:vec:doc_persist").unwrap().is_none());
}
