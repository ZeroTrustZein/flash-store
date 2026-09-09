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

#[test]
fn test_rag_domain_model_and_chunks() {
    let mut chunk_meta = DocumentMetadata::new();
    chunk_meta.insert("heading", "Introduction");

    let chunk = DocumentChunk {
        chunk_id: "doc_chunk_0".to_string(),
        doc_id: DocumentId::new("doc_chunks"),
        chunk_index: 0,
        text: "Part 1 of the article about LSM-tree storage architecture.".to_string(),
        embedding: Some(Embedding::new(vec![0.5, 0.5, 0.0])),
        start_char: 0,
        end_char: 55,
        metadata: chunk_meta,
    };

    let doc = Document::builder(
        "doc_chunks",
        "Part 1 of the article about LSM-tree storage architecture. Part 2 explains compactions.",
    )
    .embedding(vec![0.4, 0.4, 0.2])
    .metadata_field("category", "engineering")
    .metadata_field("views", 1500i64)
    .chunks(vec![chunk])
    .build();

    assert_eq!(doc.id.as_str(), "doc_chunks");
    assert_eq!(doc.chunks.len(), 1);
    assert_eq!(doc.chunks[0].chunk_id, "doc_chunk_0");
    assert_eq!(doc.chunks[0].start_char, 0);
    assert_eq!(doc.chunks[0].end_char, 55);

    // Serialization roundtrip
    let json = serde_json::to_string(&doc).unwrap();
    let decoded: Document = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.id, doc.id);
    assert_eq!(decoded.text, doc.text);
    assert_eq!(decoded.chunks.len(), 1);
    assert_eq!(decoded.metadata.get_string("category"), Some("engineering"));
}

#[test]
fn test_rag_query_with_metadata_filtering() {
    let config = RagConfigBuilder::new().embedding_dim(2).build();
    let mut engine = RagEngine::new(config);

    let doc1 = Document::builder("doc1", "Fast embedded key-value database in Rust")
        .embedding(vec![1.0, 0.0])
        .metadata_field("category", "database")
        .metadata_field("rating", 4.9f64)
        .build();

    let doc2 = Document::builder("doc2", "Distributed SQL relational database engine")
        .embedding(vec![0.9, 0.1])
        .metadata_field("category", "database")
        .metadata_field("rating", 3.8f64)
        .build();

    let doc3 = Document::builder("doc3", "Frontend React UI framework tutorial")
        .embedding(vec![0.0, 1.0])
        .metadata_field("category", "frontend")
        .metadata_field("rating", 4.5f64)
        .build();

    engine.add_document_model(doc1).unwrap();
    engine.add_document_model(doc2).unwrap();
    engine.add_document_model(doc3).unwrap();

    // Query 1: Filter by category == database
    let q1 = RagQuery::builder("database")
        .top_k(10)
        .filter(MetadataFilter::condition(FilterCondition::Eq(
            "category".to_string(),
            "database".into(),
        )))
        .build();

    let res1 = engine.search_query(&q1).unwrap();
    assert_eq!(res1.len(), 2);
    for r in &res1 {
        let meta = r.metadata.as_ref().unwrap();
        assert_eq!(meta.get_string("category"), Some("database"));
    }

    // Query 2: Filter by rating >= 4.0
    let q2 = RagQuery::builder("database")
        .top_k(10)
        .filter(MetadataFilter::condition(FilterCondition::Gte(
            "rating".to_string(),
            4.0,
        )))
        .build();

    let res2 = engine.search_query(&q2).unwrap();
    assert_eq!(res2.len(), 1);
    assert_eq!(res2[0].id.as_str(), "doc1");
}

#[test]
fn test_rag_query_fusion_and_reranker_explanation() {
    let config = RagConfigBuilder::new().embedding_dim(3).build();
    let mut engine = RagEngine::new(config);

    let doc1 = Document::builder("doc1", "Log-structured merge tree storage in FlashStore")
        .embedding(vec![1.0, 0.0, 0.0])
        .build();
    let doc2 = Document::builder("doc2", "General key value storage concepts")
        .embedding(vec![0.2, 0.8, 0.0])
        .build();

    engine.add_document_model(doc1).unwrap();
    engine.add_document_model(doc2).unwrap();

    // Query with weighted linear fusion and cross-encoder rerank
    let query = RagQuery::builder("merge tree FlashStore")
        .embedding(vec![1.0, 0.0, 0.0])
        .top_k(2)
        .fusion_strategy(FusionStrategy::WeightedLinear { dense_weight: 0.6 })
        .rerank(true, Some(2))
        .build();

    let hits = engine.search_query(&query).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id.as_str(), "doc1");
    assert!(hits[0].explanation.is_some());

    let expl = hits[0].explanation.as_ref().unwrap();
    assert!(expl.token_coverage > 0.0);
    assert!(expl.combined_score > 0.0);
}

#[test]
fn test_rag_metadata_persistence_with_flashstore() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let config = RagConfigBuilder::new().embedding_dim(2).build();
    let mut engine = RagEngine::with_store(config, store.clone());

    let doc = Document::builder("doc_meta_persist", "Metadata persistent test")
        .embedding(vec![0.7, 0.7])
        .metadata_field("env", "production")
        .metadata_field("port", 8080i64)
        .build();

    engine.add_document_model(doc).unwrap();

    // Check underlying FlashStore has raw metadata JSON
    let raw_meta = store.get(b"rag:meta:doc_meta_persist").unwrap();
    assert!(raw_meta.is_some());
    let meta_decoded: DocumentMetadata = serde_json::from_slice(&raw_meta.unwrap()).unwrap();
    assert_eq!(meta_decoded.get_string("env"), Some("production"));
    assert_eq!(meta_decoded.get_i64("port"), Some(8080));

    // Delete removes metadata from store
    assert!(engine.delete_document("doc_meta_persist").unwrap());
    assert!(store.get(b"rag:meta:doc_meta_persist").unwrap().is_none());
}
