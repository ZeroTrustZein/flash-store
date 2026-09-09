use flash_store::prelude::*;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;
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

#[test]
fn test_rag_chunking_algorithms() {
    // 1. Fixed characters chunking
    let char_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::FixedChars {
            size: 25,
            overlap: 5,
        },
        min_chunk_size: 5,
    };
    let char_chunks = chunk_text("FlashStore storage engine for fast LSM indexing", &char_cfg);
    assert!(char_chunks.len() >= 2);
    assert_eq!(char_chunks[0].0, 0);

    // 2. Fixed tokens chunking with Document
    let tok_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::FixedTokens {
            size: 3,
            overlap: 1,
        },
        min_chunk_size: 3,
    };
    let doc = Document::new(
        "doc_poly",
        "rust high performance key value embedded database",
    );
    let chunked = doc.chunk(&tok_cfg);
    assert!(chunked.chunks.len() >= 3);
    assert_eq!(chunked.chunks[0].doc_id.as_str(), "doc_poly");
    assert_eq!(chunked.chunks[0].chunk_index, 0);
    assert_eq!(chunked.chunks[1].chunk_index, 1);

    // 3. Paragraph chunking
    let para_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::Paragraph,
        min_chunk_size: 10,
    };
    let para_text = "Section 1 provides an overview of the system.\n\nSection 2 describes architecture.\n\nSection 3 summarizes findings.";
    let para_chunks = chunk_text(para_text, &para_cfg);
    assert_eq!(para_chunks.len(), 3);
    assert!(para_chunks[0].2.contains("Section 1"));
    assert!(para_chunks[1].2.contains("Section 2"));
    assert!(para_chunks[2].2.contains("Section 3"));

    // 4. Sentence chunking
    let sent_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::Sentence,
        min_chunk_size: 8,
    };
    let sent_text = "First sentence is clear! Second sentence has detail. Is the third sentence present? Absolutely.";
    let sent_chunks = chunk_text(sent_text, &sent_cfg);
    assert_eq!(sent_chunks.len(), 4);
}

#[test]
fn test_rag_mmr_diversity_reranking() {
    let query_vec = vec![1.0, 0.0, 0.0];
    let candidates = vec![
        ("d1".to_string(), vec![1.0, 0.0, 0.0], 0.98),
        ("d2".to_string(), vec![0.98, 0.02, 0.0], 0.97),
        ("d3".to_string(), vec![0.0, 1.0, 0.0], 0.65),
        ("d4".to_string(), vec![0.0, 0.0, 1.0], 0.50),
    ];

    // High lambda (0.95) favors pure relevance: d1 and d2 win
    let relevance_mmr = maximal_marginal_relevance(&query_vec, &candidates, 0.95, 2);
    assert_eq!(relevance_mmr[0].doc_id, "d1");
    assert_eq!(relevance_mmr[1].doc_id, "d2");

    // Balanced lambda (0.4) penalizes redundant d2 and picks diverse d3
    let diverse_mmr = maximal_marginal_relevance(&query_vec, &candidates, 0.4, 2);
    assert_eq!(diverse_mmr[0].doc_id, "d1");
    assert_eq!(diverse_mmr[1].doc_id, "d3");
}

#[test]
fn test_rag_borda_and_z_score_fusion() {
    let dense = vec![
        ("docA".to_string(), 0.95),
        ("docB".to_string(), 0.85),
        ("docC".to_string(), 0.60),
    ];
    let sparse = vec![
        ("docC".to_string(), 15.0),
        ("docA".to_string(), 10.0),
        ("docB".to_string(), 5.0),
    ];

    let borda_hits = borda_count_fusion(&dense, &sparse, 3);
    assert_eq!(borda_hits.len(), 3);
    // docA is rank 0 in dense (3 pts) and rank 1 in sparse (2 pts) = 5 pts
    // docC is rank 2 in dense (1 pt) and rank 0 in sparse (3 pts) = 4 pts
    // docB is rank 1 in dense (2 pts) and rank 2 in sparse (1 pt) = 3 pts
    assert_eq!(borda_hits[0].doc_id, "docA");
    assert_eq!(borda_hits[0].score, 5.0);

    let z_scores = z_score_normalize(&dense);
    assert_eq!(z_scores.len(), 3);
    assert!(z_scores.get("docA").unwrap() > z_scores.get("docB").unwrap());
    assert!(z_scores.get("docB").unwrap() > z_scores.get("docC").unwrap());
}

#[test]
fn test_rag_bm25_plus_and_stopwords() {
    let mut index = SparseIndex::with_delta(1.2, 0.75, 0.5);
    index.add_document("doc1", "the quick brown fox jumps over the lazy dog");
    index.add_document("doc2", "a lazy dog sleeps under a tree");

    let results = index.search("lazy dog", 2).unwrap();
    assert_eq!(results.len(), 2);

    let tokens = tokenize_filtered("the cat is on the mat with a dog");
    assert!(!tokens.contains(&"the".to_string()));
    assert!(!tokens.contains(&"is".to_string()));
    assert!(!tokens.contains(&"on".to_string()));
    assert!(!tokens.contains(&"with".to_string()));
    assert!(!tokens.contains(&"a".to_string()));
    assert!(tokens.contains(&"cat".to_string()));
    assert!(tokens.contains(&"mat".to_string()));
    assert!(tokens.contains(&"dog".to_string()));
}

#[test]
fn test_rag_cache_invalidation_lifecycle() {
    let config = RagConfigBuilder::new()
        .embedding_dim(3)
        .semantic_cache(5, 0.90, 3600)
        .build();
    let mut engine = RagEngine::new(config);

    engine
        .add_document(
            "doc_target",
            "Initial text content",
            Some(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();

    let query_vec = [1.0, 0.0, 0.0];
    let res = engine.search("initial text", Some(&query_vec), 1).unwrap();
    assert_eq!(res[0].doc_id, "doc_target");
    assert_eq!(engine.cache_stats().0, 0); // 0 hits

    // Query 2: Cache Hit
    let res2 = engine.search("initial text", Some(&query_vec), 1).unwrap();
    assert_eq!(res2[0].doc_id, "doc_target");
    assert_eq!(engine.cache_stats().0, 1); // 1 hit

    // Update document -> cache invalidation
    engine
        .add_document(
            "doc_target",
            "Updated text content",
            Some(vec![1.0, 0.0, 0.0]),
        )
        .unwrap();

    // Query 3: Should miss cache because invalidated
    let res3 = engine.search("initial text", Some(&query_vec), 1).unwrap();
    assert_eq!(res3[0].doc_id, "doc_target");
    assert_eq!(engine.cache_stats().1, 2); // 2 misses

    // Delete document -> cache invalidation
    assert!(engine.delete_document("doc_target").unwrap());
    let res4 = engine.search("initial text", Some(&query_vec), 1).unwrap();
    assert!(res4.is_empty());
}

#[test]
fn test_subsystems_store_adapter_batch_atomic_operations() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());
    let adapter = RagStoreAdapter::new(store.clone());

    let mut meta = DocumentMetadata::new();
    meta.insert("author", "Zein");
    meta.insert("version", 2i64);

    let chunk = DocumentChunk {
        chunk_id: "doc_sub_0".to_string(),
        doc_id: DocumentId::new("doc_sub"),
        chunk_index: 0,
        text: "Passage 1 content".to_string(),
        embedding: Some(Embedding::new(vec![0.5, 0.5])),
        start_char: 0,
        end_char: 17,
        metadata: DocumentMetadata::new(),
    };

    let doc = Document::builder("doc_sub", "Complete document content for subsystems test")
        .embedding(vec![0.3, 0.4])
        .metadata(meta)
        .chunks(vec![chunk])
        .build();

    // Atomic persist
    adapter.persist_document(&doc).unwrap();

    // Verify raw keys in FlashStore
    assert!(store.get(b"rag:doc:doc_sub").unwrap().is_some());
    assert!(store.get(b"rag:vec:doc_sub").unwrap().is_some());
    assert!(store.get(b"rag:meta:doc_sub").unwrap().is_some());
    assert!(store.get(b"rag:chunk:doc_sub:0").unwrap().is_some());

    // Load via adapter
    let loaded = adapter.load_document("doc_sub").unwrap().unwrap();
    assert_eq!(loaded.id.as_str(), "doc_sub");
    assert_eq!(loaded.text, "Complete document content for subsystems test");
    assert_eq!(loaded.embedding.unwrap().as_slice(), &[0.3, 0.4]);
    assert_eq!(loaded.metadata.get_string("author"), Some("Zein"));
    assert_eq!(loaded.chunks.len(), 1);
    assert_eq!(loaded.chunks[0].chunk_id, "doc_sub_0");

    // Recover all
    let all = adapter.recover_all().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].id.as_str(), "doc_sub");

    // Atomic delete
    assert!(adapter.delete_document("doc_sub").unwrap());
    assert!(store.get(b"rag:doc:doc_sub").unwrap().is_none());
    assert!(store.get(b"rag:vec:doc_sub").unwrap().is_none());
    assert!(store.get(b"rag:meta:doc_sub").unwrap().is_none());
    assert!(store.get(b"rag:chunk:doc_sub:0").unwrap().is_none());
}

#[test]
fn test_subsystems_rag_engine_full_recovery_and_query_continuity() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let config = RagConfigBuilder::new()
        .embedding_dim(3)
        .similarity_metric(SimilarityMetric::Cosine)
        .build();

    // 1. Ingest via engine 1
    {
        let mut engine = RagEngine::with_store(config.clone(), store.clone());
        let doc1 = Document::builder("doc1", "FlashStore LSM-tree engine storage subsystem")
            .embedding(vec![1.0, 0.0, 0.0])
            .metadata_field("category", "database")
            .build();
        let doc2 = Document::builder("doc2", "Distributed consensus raft protocol")
            .embedding(vec![0.0, 1.0, 0.0])
            .metadata_field("category", "networking")
            .build();
        let doc3 = Document::builder("doc3", "Semantic query caching and vector retrieval")
            .embedding(vec![0.7, 0.7, 0.0])
            .metadata_field("category", "ai")
            .build();

        engine.batch_add_documents(&[doc1, doc2, doc3]).unwrap();
        assert_eq!(engine.doc_count(), 3);
    }

    // 2. Recover into a brand new engine instance from the store
    let mut recovered = RagEngine::recover(config, store).unwrap();
    assert_eq!(recovered.doc_count(), 3);

    // Verify sparse search works on recovered index
    let sparse_hits = recovered.search("LSM-tree storage", None, 1).unwrap();
    assert_eq!(sparse_hits.len(), 1);
    assert_eq!(sparse_hits[0].doc_id, "doc1");

    // Verify dense vector search works on recovered index
    let dense_hits = recovered
        .search("consensus", Some(&[0.0, 1.0, 0.0]), 1)
        .unwrap();
    assert_eq!(dense_hits.len(), 1);
    assert_eq!(dense_hits[0].doc_id, "doc2");

    // Verify structured query with metadata filter works
    let query = RagQuery::builder("retrieval")
        .embedding(vec![0.7, 0.7, 0.0])
        .filter(MetadataFilter::condition(FilterCondition::Eq(
            "category".to_string(),
            "ai".into(),
        )))
        .top_k(1)
        .build();

    let query_hits = recovered.search_query(&query).unwrap();
    assert_eq!(query_hits.len(), 1);
    assert_eq!(query_hits[0].id.as_str(), "doc3");

    // Verify telemetry metrics tracker on recovered engine
    let snap = recovered.metrics();
    assert_eq!(snap.total_searches, 3);
}

#[test]
fn test_subsystems_context_assembler_citations_and_prompt_template() {
    let mut meta1 = DocumentMetadata::new();
    meta1.insert("source", "docs/architecture.md");
    meta1.insert("category", "core");

    let mut meta2 = DocumentMetadata::new();
    meta2.insert("source", "docs/wal.md");

    let docs = vec![
        ScoredDocument {
            id: DocumentId::new("doc_arch"),
            score: 0.945,
            dense_score: Some(0.95),
            sparse_score: Some(0.94),
            text: Some("LSM architecture uses memtables and immutable SSTables.".to_string()),
            metadata: Some(meta1),
            explanation: None,
        },
        ScoredDocument {
            id: DocumentId::new("doc_wal"),
            score: 0.812,
            dense_score: Some(0.80),
            sparse_score: Some(0.82),
            text: Some("Write-ahead logs guarantee zero data loss on crashes.".to_string()),
            metadata: Some(meta2),
            explanation: None,
        },
    ];

    // Markdown assembly
    let md_config = ContextConfig::new()
        .format(ContextFormat::Markdown)
        .include_scores(true)
        .include_metadata(true)
        .header("### Context for question");

    let md_context = ContextAssembler::assemble(&docs, &md_config);
    assert_eq!(md_context.doc_count, 2);
    assert_eq!(md_context.citations.len(), 2);
    assert_eq!(md_context.citations[0].doc_id, "doc_arch");
    assert_eq!(
        md_context.citations[0].source,
        Some("docs/architecture.md".to_string())
    );
    assert!(md_context.text.contains("### [1] doc_arch"));
    assert!(md_context.text.contains("source: docs/architecture.md"));

    // Prompt template formatting
    let template = RagPromptTemplate::new("You are an expert storage engineer.");
    let prompt = template.format_prompt("How does FlashStore persist data?", &md_context);
    assert!(prompt.starts_with("You are an expert storage engineer."));
    assert!(prompt.contains("Context for question"));
    assert!(prompt.contains("User Query: How does FlashStore persist data?"));

    // Numbered format
    let num_config = ContextConfig::new()
        .format(ContextFormat::Numbered)
        .without_header();
    let num_context = ContextAssembler::assemble(&docs, &num_config);
    assert!(num_context.text.starts_with("[1] (doc_arch)"));
}

#[test]
fn test_subsystems_rag_pipeline_with_mock_embedder_and_chunking() {
    let config = RagConfigBuilder::new().embedding_dim(16).build();
    let engine = RagEngine::new(config);
    let embedder = Arc::new(MockEmbeddingProvider::new(16));

    let chunk_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::FixedTokens { size: 4, overlap: 1 },
        min_chunk_size: 2,
    };

    let mut pipeline = RagPipeline::new(engine)
        .with_embedder(embedder)
        .with_chunking(chunk_cfg);

    let report = pipeline
        .batch_ingest(vec![
            (
                "p1".into(),
                "FlashStore is a lightning fast embedded storage engine in Rust".into(),
                None,
            ),
            (
                "p2".into(),
                "Distributed transactions require two-phase commit consensus".into(),
                None,
            ),
        ])
        .unwrap();

    assert_eq!(report.docs_ingested, 2);
    assert!(report.chunks_created >= 2);
    assert!(report.total_chars > 0);

    let res = pipeline.query("embedded storage engine", 2).unwrap();
    assert!(!res.documents.is_empty());
    assert_eq!(res.documents[0].id.as_str(), "p1");
    assert!(res.prompt.contains("FlashStore"));
    assert_eq!(res.context.citations[0].doc_id, "p1");
}

#[test]
fn test_subsystems_telemetry_metrics_and_ir_evaluation() {
    let samples = vec![
        QueryEvaluationSample {
            query: "lsm tree".into(),
            retrieved_ids: vec!["docA".into(), "docB".into(), "docC".into()],
            ground_truth_ids: vec!["docA".into()], // Rank 1 -> RR = 1.0, Hit = 1
        },
        QueryEvaluationSample {
            query: "compaction".into(),
            retrieved_ids: vec!["docX".into(), "docB".into(), "docY".into()],
            ground_truth_ids: vec!["docB".into()], // Rank 2 -> RR = 0.5, Hit = 1
        },
        QueryEvaluationSample {
            query: "bloom filter".into(),
            retrieved_ids: vec!["docZ".into(), "docW".into()],
            ground_truth_ids: vec!["docC".into()], // Not in top 2 -> RR = 0.0, Hit = 0
        },
    ];

    let summary = RetrievalEvaluator::evaluate(&samples, 2);
    assert_eq!(summary.sample_count, 3);
    assert_eq!(summary.k, 2);
    assert!((summary.mrr - 0.5).abs() < 1e-4);
    assert!((summary.hit_rate_at_k - (2.0 / 3.0)).abs() < 1e-4);
    assert!(summary.ndcg_at_k > 0.0);
    assert!(summary.precision_at_k > 0.0);
    assert!(summary.recall_at_k > 0.0);
}

#[test]
fn test_concurrent_rag_engine_reads_and_writes() {
    let config = RagConfigBuilder::new()
        .embedding_dim(4)
        .similarity_metric(SimilarityMetric::Cosine)
        .semantic_cache(100, 0.90, 3600)
        .build();

    let engine = Arc::new(RwLock::new(RagEngine::new(config)));

    // Seed initial docs
    {
        let mut eng = engine.write().unwrap();
        for i in 0..10 {
            let doc = Document::builder(
                format!("init_doc_{}", i),
                format!("Initial document content number {} for concurrent testing", i),
            )
            .embedding(vec![1.0, 0.0, 0.0, 0.0])
            .metadata_field("index", i as i64)
            .build();
            eng.add_document_model(doc).unwrap();
        }
    }

    let mut handles = Vec::new();

    // Spawn 2 reader threads reading documents and metadata with read-lock
    for _ in 0..2 {
        let eng = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let reader = eng.read().unwrap();
                assert!(reader.doc_count() >= 5);
                if let Some(doc) = reader.get_document("init_doc_8") {
                    assert!(doc.text.contains("Initial document content"));
                }
            }
        }));
    }

    // Spawn 2 searcher threads performing search queries
    for t in 0..2 {
        let eng = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for _ in 0..30 {
                let mut searcher = eng.write().unwrap();
                let hits = searcher
                    .search(
                        "document content concurrent",
                        Some(&[1.0, 0.0, 0.0, 0.0]),
                        3,
                    )
                    .unwrap();
                assert!(!hits.is_empty(), "Thread {} got empty hits", t);
                assert!(hits.len() <= 3);
            }
        }));
    }

    // Spawn 2 writer threads adding new documents
    for t in 0..2 {
        let eng = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let doc_id = format!("writer_{}_{}", t, i);
                let doc = Document::builder(
                    &*doc_id,
                    format!("Concurrent worker {} wrote payload chunk {}", t, i),
                )
                .embedding(vec![0.0, 1.0, 0.0, 0.0])
                .metadata_field("worker", t as i64)
                .build();
                let mut writer = eng.write().unwrap();
                writer.add_document_model(doc).unwrap();
            }
        }));
    }

    // Spawn 1 writer thread deleting initial documents
    {
        let eng = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..5 {
                let doc_id = format!("init_doc_{}", i);
                thread::sleep(Duration::from_millis(5));
                let mut writer = eng.write().unwrap();
                let _ = writer.delete_document(&doc_id);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let final_eng = engine.read().unwrap();
    // 10 initial - 5 deleted + (2 * 25) = 55 documents
    assert_eq!(final_eng.doc_count(), 55);
}

#[test]
fn test_rag_full_crash_recovery_with_compaction_and_overwrites() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(512)
        .build();

    let config = RagConfigBuilder::new()
        .embedding_dim(3)
        .similarity_metric(SimilarityMetric::Cosine)
        .build();

    // Session 1: Create, write documents, update some, delete one
    {
        let store = Arc::new(FlashStore::open(options.clone()).unwrap());
        let mut engine = RagEngine::with_store(config.clone(), store.clone());

        for i in 0..5 {
            let doc = Document::builder(
                format!("doc_{}", i),
                format!("Original version of document number {}", i),
            )
            .embedding(vec![0.1 * (i as f32), 0.5, 0.5])
            .metadata_field("version", 1i64)
            .metadata_field("author", "alice")
            .build();
            engine.add_document_model(doc).unwrap();
        }

        // Overwrite doc_1 and doc_2 with version 2 and different text/embeddings
        let doc1_v2 = Document::builder(
            "doc_1",
            "Updated version of document 1 with high performance rust storage keywords",
        )
        .embedding(vec![0.9, 0.1, 0.0])
        .metadata_field("version", 2i64)
        .metadata_field("author", "bob")
        .build();
        engine.add_document_model(doc1_v2).unwrap();

        let doc2_v2 = Document::builder(
            "doc_2",
            "Updated version of document 2 covering distributed consensus algorithms",
        )
        .embedding(vec![0.0, 0.9, 0.1])
        .metadata_field("version", 2i64)
        .metadata_field("author", "charlie")
        .build();
        engine.add_document_model(doc2_v2).unwrap();

        // Delete doc_4
        let deleted = engine.delete_document("doc_4").unwrap();
        assert!(deleted);

        // Force flush and compaction on the underlying LSM store
        store.flush().unwrap();
        store.compact().unwrap();
        assert_eq!(engine.doc_count(), 4);
    }

    // Session 2: Crash recovery into a fresh engine instance
    {
        let reopened_store = Arc::new(FlashStore::open(options).unwrap());
        let mut recovered_engine = RagEngine::recover(config, reopened_store).unwrap();

        assert_eq!(recovered_engine.doc_count(), 4);

        // Verify doc_4 is gone
        assert!(recovered_engine.get_document("doc_4").is_none());

        // Verify doc_1 was updated
        let d1 = recovered_engine.get_document("doc_1").unwrap();
        assert!(d1.text.contains("Updated version of document 1"));
        assert_eq!(d1.metadata.get_i64("version"), Some(2));
        assert_eq!(d1.metadata.get_string("author"), Some("bob"));

        // Verify doc_2 was updated
        let d2 = recovered_engine.get_document("doc_2").unwrap();
        assert!(d2.text.contains("distributed consensus"));
        assert_eq!(d2.metadata.get_i64("version"), Some(2));
        assert_eq!(d2.metadata.get_string("author"), Some("charlie"));

        // Verify untouched doc_0
        let d0 = recovered_engine.get_document("doc_0").unwrap();
        assert!(d0.text.contains("Original version of document number 0"));
        assert_eq!(d0.metadata.get_i64("version"), Some(1));

        // Verify search finds updated document keywords
        let hits = recovered_engine.search("consensus", None, 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc_2");

        // Verify search for deleted document terms does not retrieve doc_4
        let hits4 = recovered_engine.search("document number 4", None, 5).unwrap();
        assert!(hits4.iter().all(|h| h.doc_id != "doc_4"));
    }
}

#[test]
fn test_rag_complex_nested_metadata_filtering() {
    let config = RagConfigBuilder::new().embedding_dim(2).build();
    let mut engine = RagEngine::new(config);

    let doc1 = Document::builder("doc1", "Enterprise Rust database internals storage")
        .metadata_field("tier", "enterprise")
        .metadata_field("score", 95.5)
        .metadata_field("nodes", 16i64)
        .metadata_field("active", true)
        .metadata_field("tags", vec!["rust", "storage", "enterprise"])
        .build();

    let doc2 = Document::builder("doc2", "Community edition key value cache storage database")
        .metadata_field("tier", "community")
        .metadata_field("score", 82.0)
        .metadata_field("nodes", 4i64)
        .metadata_field("active", true)
        .metadata_field("tags", vec!["cache", "fast"])
        .build();

    let doc3 = Document::builder("doc3", "Deprecated storage engine database prototype")
        .metadata_field("tier", "deprecated")
        .metadata_field("score", 45.0)
        .metadata_field("nodes", 1i64)
        .metadata_field("active", false)
        .metadata_field("tags", vec!["prototype"])
        .build();

    let doc4 = Document::builder("doc4", "Enterprise distributed vector index storage database")
        .metadata_field("tier", "enterprise")
        .metadata_field("score", 91.0)
        .metadata_field("nodes", 32i64)
        .metadata_field("active", false)
        .metadata_field("tags", vec!["vector", "enterprise"])
        .build();

    engine.batch_add_documents(&[doc1, doc2, doc3, doc4]).unwrap();

    // 1. Gte & Lte numeric filters
    let q_score = RagQuery::builder("storage database")
        .filter(MetadataFilter::all(vec![
            MetadataFilter::condition(FilterCondition::Gte("score".into(), 80.0)),
            MetadataFilter::condition(FilterCondition::Lte("score".into(), 93.0)),
        ]))
        .top_k(10)
        .build();
    let hits = engine.search_query(&q_score).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"doc2"));
    assert!(ids.contains(&"doc4"));
    assert!(!ids.contains(&"doc1")); // score 95.5 > 93.0
    assert!(!ids.contains(&"doc3")); // score 45.0 < 80.0

    // 2. In & NotIn filter
    let q_tier = RagQuery::builder("storage")
        .filter(MetadataFilter::condition(FilterCondition::In(
            "tier".into(),
            vec!["community".into(), "deprecated".into()],
        )))
        .top_k(10)
        .build();
    let hits = engine.search_query(&q_tier).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"doc2"));
    assert!(ids.contains(&"doc3"));
    assert!(!ids.contains(&"doc1"));
    assert!(!ids.contains(&"doc4"));

    // 3. String Contains filter
    let q_contains = RagQuery::builder("storage")
        .filter(MetadataFilter::condition(FilterCondition::Contains(
            "tier".into(),
            "enter".into(),
        )))
        .top_k(10)
        .build();
    let hits = engine.search_query(&q_contains).unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"doc1"));
    assert!(ids.contains(&"doc4"));

    // 4. Complex Nested boolean: (tier == "enterprise" OR nodes > 10) AND NOT (active == false)
    let complex_filter = MetadataFilter::all(vec![
        MetadataFilter::any(vec![
            MetadataFilter::condition(FilterCondition::Eq("tier".into(), "enterprise".into())),
            MetadataFilter::condition(FilterCondition::Gt("nodes".into(), 10.0)),
        ]),
        MetadataFilter::negate(MetadataFilter::condition(FilterCondition::Eq(
            "active".into(),
            false.into(),
        ))),
    ]);

    let q_complex = RagQuery::builder("storage database")
        .filter(complex_filter)
        .top_k(10)
        .build();
    let hits = engine.search_query(&q_complex).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "doc1");
}

#[test]
fn test_rag_all_fusion_algorithms_calibration() {
    let config = RagConfigBuilder::new()
        .embedding_dim(3)
        .similarity_metric(SimilarityMetric::Cosine)
        .build();
    let mut engine = RagEngine::new(config);

    // Document A: strong dense match, no keyword match
    let doc_a = Document::builder("doc_dense", "Apple banana cherry fruit salad")
        .embedding(vec![1.0, 0.0, 0.0])
        .build();

    // Document B: strong sparse keyword match, orthogonal vector
    let doc_b = Document::builder("doc_sparse", "Distributed key value storage engine in Rust")
        .embedding(vec![0.0, 1.0, 0.0])
        .build();

    // Document C: moderate on both
    let doc_c = Document::builder("doc_hybrid", "Storage engine key value apple fruit")
        .embedding(vec![0.707, 0.707, 0.0])
        .build();

    engine.batch_add_documents(&[doc_a, doc_b, doc_c]).unwrap();

    let query_text = "storage engine Rust";
    let query_vector = vec![1.0, 0.0, 0.0]; // vector matches doc_dense

    // 1. DenseOnly
    let q_dense = RagQuery::builder(query_text)
        .embedding(query_vector.clone())
        .fusion_strategy(FusionStrategy::DenseOnly)
        .top_k(3)
        .build();
    let hits = engine.search_query(&q_dense).unwrap();
    assert_eq!(hits[0].id.as_str(), "doc_dense");

    // 2. SparseOnly
    let q_sparse = RagQuery::builder(query_text)
        .fusion_strategy(FusionStrategy::SparseOnly)
        .top_k(3)
        .build();
    let hits = engine.search_query(&q_sparse).unwrap();
    assert_eq!(hits[0].id.as_str(), "doc_sparse");

    // 3. WeightedLinear favoring dense
    engine.clear_cache();
    let q_weighted = RagQuery::builder(query_text)
        .embedding(query_vector.clone())
        .fusion_strategy(FusionStrategy::WeightedLinear {
            dense_weight: 0.9,
        })
        .top_k(3)
        .build();
    let hits = engine.search_query(&q_weighted).unwrap();
    assert_eq!(hits[0].id.as_str(), "doc_dense");

    // 4. ReciprocalRankFusion
    let q_rrf = RagQuery::builder(query_text)
        .embedding(query_vector.clone())
        .fusion_strategy(FusionStrategy::Rrf { k: 60 })
        .top_k(3)
        .build();
    let hits = engine.search_query(&q_rrf).unwrap();
    assert_eq!(hits.len(), 3);
    for h in &hits {
        assert!(h.score > 0.0);
    }

    // 5. BordaCount
    let hits_dense = vec![
        ("doc_dense".into(), 1.0),
        ("doc_hybrid".into(), 0.7),
        ("doc_sparse".into(), 0.0),
    ];
    let hits_sparse = vec![
        ("doc_sparse".into(), 1.0),
        ("doc_hybrid".into(), 0.8),
        ("doc_dense".into(), 0.1),
    ];
    let hits_borda = borda_count_fusion(&hits_dense, &hits_sparse, 3);
    assert_eq!(hits_borda.len(), 3);

    // 6. Min score cutoff
    let q_cutoff = RagQuery::builder(query_text)
        .embedding(query_vector)
        .fusion_strategy(FusionStrategy::DenseOnly)
        .min_score(0.8) // Only doc_dense has cosine >= 0.8
        .top_k(3)
        .build();
    let hits = engine.search_query(&q_cutoff).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "doc_dense");
}

#[test]
fn test_rag_mmr_diversity_reranking_edge_cases() {
    let candidates = vec![
        ("doc_a".to_string(), vec![1.0, 0.0], 0.99),
        ("doc_b".to_string(), vec![0.999, 0.001], 0.98),
        ("doc_c".to_string(), vec![0.0, 1.0], 0.70),
    ];

    let query_vec = vec![1.0, 0.0];

    // High diversity (lambda = 0.1): doc_c should be picked over doc_b as 2nd item
    let reranked_diverse = maximal_marginal_relevance(
        &query_vec,
        &candidates,
        0.1,
        2,
    );
    assert_eq!(reranked_diverse.len(), 2);
    assert_eq!(reranked_diverse[0].doc_id, "doc_a");
    assert_eq!(reranked_diverse[1].doc_id, "doc_c");

    // Pure relevance (lambda = 1.0): doc_b should be picked 2nd
    let reranked_pure = maximal_marginal_relevance(
        &query_vec,
        &candidates,
        1.0,
        2,
    );
    assert_eq!(reranked_pure.len(), 2);
    assert_eq!(reranked_pure[0].doc_id, "doc_a");
    assert_eq!(reranked_pure[1].doc_id, "doc_b");

    // Edge case 1: top_k > candidates
    let reranked_all = maximal_marginal_relevance(
        &query_vec,
        &candidates,
        0.5,
        10,
    );
    assert_eq!(reranked_all.len(), 3);

    // Edge case 2: empty candidates
    let empty_res = maximal_marginal_relevance(
        &query_vec,
        &[],
        0.5,
        5,
    );
    assert!(empty_res.is_empty());
}

#[test]
fn test_rag_cross_encoder_scoring_and_explanations() {
    let scorer = LexicalSemanticCrossEncoder::default();

    let query = "FlashStore LSM storage";
    let doc_exact = "FlashStore LSM storage engine";
    let doc_partial = "FlashStore is an embedded database";
    let doc_irrelevant = "Cooking Italian pasta with tomatoes";

    let exp_exact = scorer.explain(query, doc_exact);
    let exp_partial = scorer.explain(query, doc_partial);
    let exp_irrel = scorer.explain(query, doc_irrelevant);

    assert!(exp_exact.combined_score > exp_partial.combined_score);
    assert!(exp_partial.combined_score > exp_irrel.combined_score);
    assert!(exp_exact.phrase_match > 0.0);
    assert_eq!(exp_irrel.token_coverage, 0.0);
    assert_eq!(exp_irrel.combined_score, 0.0);

    let candidates = vec![
        ("doc1".to_string(), doc_exact.to_string(), 0.5),
        ("doc2".to_string(), doc_irrelevant.to_string(), 0.6),
    ];

    let reranked = rerank_candidates_with_explanation(&scorer, query, &candidates, 2);
    assert_eq!(reranked.len(), 2);
    assert_eq!(reranked[0].doc_id, "doc1");
    assert!(reranked[0].explanation.is_some());
    assert!(reranked[0].reranked_score > reranked[1].reranked_score);
}

#[test]
fn test_rag_semantic_cache_lru_invalidation_and_clear() {
    let mut cache = SemanticCache::new(2, 0.90, 3600);
    assert_eq!(cache.capacity(), 2);
    assert_eq!(cache.len(), 0);

    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![0.0, 1.0];
    let vec3 = vec![-1.0, 0.0];

    let hits1 = vec![SearchResult {
        doc_id: "doc1".into(),
        score: 0.99,
        dense_score: Some(0.99),
        sparse_score: None,
    }];
    let hits2 = vec![SearchResult {
        doc_id: "doc2".into(),
        score: 0.95,
        dense_score: Some(0.95),
        sparse_score: None,
    }];
    let hits3 = vec![SearchResult {
        doc_id: "doc3".into(),
        score: 0.90,
        dense_score: Some(0.90),
        sparse_score: None,
    }];

    cache.insert("q1", vec1.clone(), hits1).unwrap();
    cache.insert("q2", vec2.clone(), hits2).unwrap();
    assert_eq!(cache.len(), 2);

    // Lookup vec1 -> Cache hit!
    let res = cache.lookup(&vec1);
    assert!(res.is_some());
    assert_eq!(res.unwrap()[0].doc_id, "doc1");
    assert_eq!(cache.total_hits(), 1);

    // Insert 3rd entry -> evicts an entry to maintain capacity limit 2
    cache.insert("q3", vec3.clone(), hits3).unwrap();
    assert_eq!(cache.len(), 2);

    // Newly inserted vec3 should hit
    assert!(cache.lookup(&vec3).is_some());

    // Invalidate doc3
    let removed = cache.invalidate_for_doc("doc3");
    assert_eq!(removed, 1);
    assert!(cache.lookup(&vec3).is_none());

    // Clear cache
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_rag_chunking_unicode_and_boundary_cases() {
    let unicode_text = "🦀 Rust is awesome! 🚀 データベース Fast KV store. 日本語テキスト. Café résumé.";
    let config_chars = ChunkingConfig {
        strategy: ChunkingStrategy::FixedChars { size: 15, overlap: 5 },
        min_chunk_size: 5,
    };
    let chunks = chunk_text(unicode_text, &config_chars);
    assert!(!chunks.is_empty());
    for (start, end, text) in &chunks {
        assert!(!text.is_empty());
        assert!(*end > *start);
        let expected = &unicode_text[*start..*end];
        assert_eq!(text.as_str(), expected);
    }

    let paragraph_text = "Paragraph 1 line A.\nParagraph 1 line B.\n\n\nParagraph 2 single line.\n\nParagraph 3 final line.";
    let config_para = ChunkingConfig {
        strategy: ChunkingStrategy::Paragraph,
        min_chunk_size: 5,
    };
    let p_chunks = chunk_text(paragraph_text, &config_para);
    assert_eq!(p_chunks.len(), 3);
    assert!(p_chunks[0].2.contains("Paragraph 1"));
    assert!(p_chunks[1].2.contains("Paragraph 2"));
    assert!(p_chunks[2].2.contains("Paragraph 3"));

    let sentence_text = "FlashStore v1.0 is released! Is it fast? Yes, absolutely... Dr. Smith confirmed 99.9% reliability. It works well.";
    let config_sent = ChunkingConfig {
        strategy: ChunkingStrategy::Sentence,
        min_chunk_size: 5,
    };
    let s_chunks = chunk_text(sentence_text, &config_sent);
    assert!(s_chunks.len() >= 3);

    let empty_chunks = chunk_text("", &config_chars);
    assert!(empty_chunks.is_empty());

    let tiny_chunks = chunk_text("Hi", &config_chars);
    assert_eq!(tiny_chunks.len(), 1);
    assert_eq!(tiny_chunks[0].2, "Hi");
}

#[test]
fn test_rag_context_assembler_all_formats_and_truncation() {
    let mut meta = DocumentMetadata::new();
    meta.insert("author", "Zein");
    meta.insert("source", "https://flashstore.dev/docs");

    let docs = vec![
        ScoredDocument {
            id: DocumentId::new("doc_xml"),
            score: 0.98,
            dense_score: Some(0.98),
            sparse_score: Some(0.90),
            text: Some("Storage engine supports <atomic> writes & 'consistent' reads.".into()),
            metadata: Some(meta.clone()),
            explanation: None,
        },
        ScoredDocument {
            id: DocumentId::new("doc_secondary"),
            score: 0.75,
            dense_score: Some(0.75),
            sparse_score: Some(0.70),
            text: Some("Secondary indexing and background compaction jobs.".into()),
            metadata: Some(meta),
            explanation: None,
        },
    ];

    // 1. XML format
    let xml_config = ContextConfig::new()
        .format(ContextFormat::Xml)
        .include_scores(true)
        .include_metadata(true);
    let xml_context = ContextAssembler::assemble(&docs, &xml_config);
    assert!(xml_context.text.contains("<context>"));
    assert!(xml_context.text.contains("</context>"));
    assert!(xml_context.text.contains("<document id=\"doc_xml\""));
    assert!(xml_context.text.contains("Storage engine supports <atomic>"));

    // 2. Compact format
    let compact_config = ContextConfig::new().format(ContextFormat::Compact);
    let compact_context = ContextAssembler::assemble(&docs, &compact_config);
    assert!(compact_context.text.contains("Storage engine supports <atomic>"));

    // 3. Truncation with TruncationStrategy::DropOversized
    let drop_config = ContextConfig::new()
        .format(ContextFormat::Markdown)
        .max_chars(150)
        .truncation_strategy(TruncationStrategy::DropOversized);
    let drop_context = ContextAssembler::assemble(&docs, &drop_config);
    assert_eq!(drop_context.doc_count, 1);
    assert_eq!(drop_context.citations.len(), 1);
    assert_eq!(drop_context.citations[0].doc_id, "doc_xml");
    assert!(drop_context.was_truncated);

    // 4. Truncation with TruncationStrategy::TruncateLast
    let trunc_config = ContextConfig::new()
        .format(ContextFormat::Numbered)
        .max_chars(100)
        .truncation_strategy(TruncationStrategy::TruncateLast);
    let trunc_context = ContextAssembler::assemble(&docs, &trunc_config);
    assert!(trunc_context.text.len() <= 120);
    assert!(trunc_context.was_truncated);
}

#[test]
fn test_rag_ir_evaluator_comprehensive_metrics() {
    let samples = vec![
        QueryEvaluationSample {
            query: "query 1".into(),
            ground_truth_ids: vec!["target_1".into()],
            retrieved_ids: vec!["target_1".into(), "other_a".into(), "other_b".into()],
        },
        QueryEvaluationSample {
            query: "query 2".into(),
            ground_truth_ids: vec!["target_2".into()],
            retrieved_ids: vec!["other_c".into(), "target_2".into(), "other_d".into()],
        },
        QueryEvaluationSample {
            query: "query 3".into(),
            ground_truth_ids: vec!["target_3".into()],
            retrieved_ids: vec!["other_e".into(), "other_f".into(), "target_3".into()],
        },
        QueryEvaluationSample {
            query: "query 4".into(),
            ground_truth_ids: vec!["target_4".into()],
            retrieved_ids: vec!["other_g".into(), "other_h".into(), "other_i".into()],
        },
    ];

    let eval_k2 = RetrievalEvaluator::evaluate(&samples, 2);
    assert_eq!(eval_k2.sample_count, 4);
    assert_eq!(eval_k2.k, 2);
    assert!((eval_k2.mrr - 0.375).abs() < 1e-4);
    assert!((eval_k2.hit_rate_at_k - 0.5).abs() < 1e-4);
    assert!((eval_k2.precision_at_k - 0.25).abs() < 1e-4);
    assert!((eval_k2.recall_at_k - 0.5).abs() < 1e-4);
    assert!(eval_k2.ndcg_at_k > 0.0);

    let empty_summary = RetrievalEvaluator::evaluate(&[], 5);
    assert_eq!(empty_summary.sample_count, 0);
    assert_eq!(empty_summary.mrr, 0.0);
    assert_eq!(empty_summary.hit_rate_at_k, 0.0);
}

#[test]
fn test_rag_pipeline_end_to_end_lifecycle_and_deletion() {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options).unwrap());

    let config = RagConfigBuilder::new().embedding_dim(8).build();
    let engine = RagEngine::with_store(config, store);
    let embedder = Arc::new(MockEmbeddingProvider::new(8));

    let chunk_cfg = ChunkingConfig {
        strategy: ChunkingStrategy::FixedChars { size: 50, overlap: 10 },
        min_chunk_size: 10,
    };

    let mut pipeline = RagPipeline::new(engine)
        .with_embedder(embedder)
        .with_chunking(chunk_cfg);

    let report = pipeline
        .batch_ingest(vec![
            (
                "doc_pip_1".into(),
                "FlashStore implements high throughput LSM trees with zero allocation writes.".into(),
                None,
            ),
            (
                "doc_pip_2".into(),
                "Semantic search combines dense vectors with BM25 plus sparse keyword indexes.".into(),
                None,
            ),
            (
                "doc_pip_3".into(),
                "Reranking models prioritize documents based on maximal marginal relevance.".into(),
                None,
            ),
        ])
        .unwrap();

    assert_eq!(report.docs_ingested, 3);
    assert_eq!(pipeline.list_documents().len(), 3);

    let q_res = pipeline.query("LSM trees throughput writes", 2).unwrap();
    assert!(!q_res.documents.is_empty());
    assert_eq!(q_res.documents[0].id.as_str(), "doc_pip_1");
    assert!(q_res.prompt.contains("FlashStore implements"));

    let deleted = pipeline.delete_document("doc_pip_1").unwrap();
    assert!(deleted);
    assert_eq!(pipeline.list_documents().len(), 2);
    assert!(pipeline.get_document("doc_pip_1").is_none());

    let q_res2 = pipeline.query("LSM trees throughput writes", 2).unwrap();
    assert!(q_res2.documents.iter().all(|d| d.id.as_str() != "doc_pip_1"));
}

#[test]
fn test_rag_store_adapter_data_integrity_and_recovery() {
    let dir = tempdir().unwrap();
    let store = Arc::new(FlashStore::open(OptionsBuilder::new().dir(dir.path()).build()).unwrap());
    let adapter = RagStoreAdapter::new(store.clone());

    store
        .put(
            bytes::Bytes::from_static(b"user:zein"),
            bytes::Bytes::from_static(b"admin"),
        )
        .unwrap();
    store
        .put(
            bytes::Bytes::from_static(b"orders:1001"),
            bytes::Bytes::from_static(b"paid"),
        )
        .unwrap();

    let mut meta = DocumentMetadata::new();
    meta.insert("env", "production");
    meta.insert("replicas", 3i64);

    let doc = Document {
        id: DocumentId::new("doc_store_1"),
        text: "Full document body for storage adapter testing.".into(),
        embedding: Some(Embedding::new(vec![0.1, 0.2, 0.3])),
        metadata: meta,
        chunks: vec![
            DocumentChunk {
                chunk_id: "doc_store_1_c0".into(),
                doc_id: DocumentId::new("doc_store_1"),
                chunk_index: 0,
                start_char: 0,
                end_char: 20,
                text: "Full document body f".into(),
                embedding: None,
                metadata: DocumentMetadata::default(),
            },
            DocumentChunk {
                chunk_id: "doc_store_1_c1".into(),
                doc_id: DocumentId::new("doc_store_1"),
                chunk_index: 1,
                start_char: 20,
                end_char: 48,
                text: "or storage adapter testing.".into(),
                embedding: None,
                metadata: DocumentMetadata::default(),
            },
        ],
        created_at: 1000,
        updated_at: 2000,
    };

    adapter.persist_document(&doc).unwrap();

    let loaded = adapter
        .load_document("doc_store_1")
        .unwrap()
        .expect("document not found");
    assert_eq!(loaded.id.as_str(), "doc_store_1");
    assert_eq!(loaded.text, "Full document body for storage adapter testing.");
    assert_eq!(loaded.embedding.unwrap().as_slice(), &[0.1, 0.2, 0.3]);
    assert_eq!(loaded.metadata.get_string("env"), Some("production"));
    assert_eq!(loaded.metadata.get_i64("replicas"), Some(3));
    assert_eq!(loaded.chunks.len(), 2);
    assert_eq!(loaded.chunks[0].chunk_index, 0);
    assert_eq!(loaded.chunks[1].chunk_index, 1);

    let batch_docs: Vec<Document> = (2..=10)
        .map(|i| {
            Document::builder(
                format!("doc_store_{}", i),
                format!("Document payload number {}", i),
            )
            .embedding(vec![0.1 * i as f32])
            .build()
        })
        .collect();

    adapter.persist_documents_batch(&batch_docs).unwrap();

    let all = adapter.recover_all().unwrap();
    assert_eq!(all.len(), 10);
    assert!(all.iter().all(|d| d.id.as_str().starts_with("doc_store_")));

    let deleted = adapter.delete_document("doc_store_1").unwrap();
    assert!(deleted);
    assert!(adapter.load_document("doc_store_1").unwrap().is_none());

    let chunk_key = RagStoreAdapter::chunk_key("doc_store_1", 0);
    assert!(store.get(&chunk_key).unwrap().is_none());

    assert_eq!(
        store
            .get(bytes::Bytes::from_static(b"user:zein"))
            .unwrap(),
        Some(bytes::Bytes::from_static(b"admin"))
    );
}

