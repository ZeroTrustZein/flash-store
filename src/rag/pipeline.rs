use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{FlashStoreError, Result};
use crate::rag::context::{AssembledContext, ContextAssembler, ContextConfig, RagPromptTemplate};
use crate::rag::engine::RagEngine;
use crate::rag::types::{
    ChunkingConfig, Document, DocumentId, DocumentMetadata, Embedding, RagQuery, ScoredDocument,
};

/// Interface for generating vector embeddings from text passages or queries.
pub trait EmbeddingProvider: Send + Sync {
    /// Generates an embedding vector for a single query string.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;

    /// Generates embedding vectors for a batch of text passages.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Expected dimensionality of vectors produced by this provider.
    fn dimension(&self) -> usize;
}

/// Deterministic, offline pseudo-embedding provider that converts text into
/// unit-normalized vectors using feature hashing and token frequencies.
///
/// Ensures semantic overlap: sentences with shared terms produce high cosine similarity.
pub struct MockEmbeddingProvider {
    dim: usize,
}

impl MockEmbeddingProvider {
    /// Creates a `MockEmbeddingProvider` with the specified vector dimensionality.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "Dimension must be greater than zero");
        Self { dim }
    }

    /// Internal token hash using FNV-1a.
    fn hash_token(token: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in token.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
        hash
    }

    /// Computes a normalized vector for text.
    fn text_to_vector(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        if words.is_empty() {
            vec[0] = 1.0;
            return vec;
        }

        for (pos, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let h = Self::hash_token(&lower);
            let bucket = (h as usize) % self.dim;
            let sign = if (h >> 32) % 2 == 0 { 1.0f32 } else { -1.0f32 };
            let pos_decay = 1.0 / ((pos as f32) * 0.05 + 1.0);
            vec[bucket] += sign * (1.0 + (word.len() as f32).min(5.0) * 0.2) * pos_decay;
        }

        // L2 Normalize
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in &mut vec {
                *v /= norm;
            }
        } else {
            vec[0] = 1.0;
        }

        vec
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        Ok(self.text_to_vector(query))
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.text_to_vector(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Static map-backed embedding provider useful for testing predetermined vectors.
pub struct StaticEmbeddingProvider {
    dim: usize,
    lookup: HashMap<String, Vec<f32>>,
}

impl StaticEmbeddingProvider {
    /// Creates a `StaticEmbeddingProvider` with a fixed dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            lookup: HashMap::new(),
        }
    }

    /// Registers a fixed vector for a text string.
    pub fn register(&mut self, text: impl Into<String>, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");
        self.lookup.insert(text.into(), vector);
    }
}

impl EmbeddingProvider for StaticEmbeddingProvider {
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.lookup.get(query) {
            Ok(v.clone())
        } else {
            Err(FlashStoreError::InvalidArgument(format!(
                "No registered embedding for query: '{}'",
                query
            )))
        }
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for t in texts {
            if let Some(v) = self.lookup.get(*t) {
                results.push(v.clone());
            } else {
                return Err(FlashStoreError::InvalidArgument(format!(
                    "No registered embedding for text: '{}'",
                    t
                )));
            }
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Ingestion telemetry report summarizing results of a batch ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestReport {
    /// Total number of documents successfully processed.
    pub docs_ingested: usize,
    /// Total number of chunks generated and indexed.
    pub chunks_created: usize,
    /// Total raw character count indexed.
    pub total_chars: usize,
    /// Ingestion duration in milliseconds.
    pub elapsed_ms: u64,
}

/// Pipeline query execution result containing retrieved documents, assembled context, and ready prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineQueryResult {
    /// The user query executed.
    pub query: String,
    /// Ranked retrieved documents.
    pub documents: Vec<ScoredDocument>,
    /// Formatted context block with source citations.
    pub context: AssembledContext,
    /// Final assembled LLM prompt.
    pub prompt: String,
}

/// End-to-end RAG pipeline coordinating document ingestion, chunking,
/// automatic embedding generation, hybrid search, and context assembly.
pub struct RagPipeline {
    engine: RagEngine,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    chunking_config: Option<ChunkingConfig>,
    context_config: ContextConfig,
    prompt_template: RagPromptTemplate,
}

impl RagPipeline {
    /// Creates a `RagPipeline` wrapping a `RagEngine`.
    pub fn new(engine: RagEngine) -> Self {
        Self {
            engine,
            embedder: None,
            chunking_config: None,
            context_config: ContextConfig::default(),
            prompt_template: RagPromptTemplate::default(),
        }
    }

    /// Attaches an embedding provider for automatic query and document vectorization.
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Sets an automatic text chunking configuration.
    pub fn with_chunking(mut self, chunking: ChunkingConfig) -> Self {
        self.chunking_config = Some(chunking);
        self
    }

    /// Sets the context assembly configuration.
    pub fn with_context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = config;
        self
    }

    /// Sets a custom prompt template.
    pub fn with_prompt_template(mut self, template: RagPromptTemplate) -> Self {
        self.prompt_template = template;
        self
    }

    /// Reference to underlying `RagEngine`.
    pub fn engine(&self) -> &RagEngine {
        &self.engine
    }

    /// Mutable reference to underlying `RagEngine`.
    pub fn engine_mut(&mut self) -> &mut RagEngine {
        &mut self.engine
    }

    /// Returns all indexed document IDs in sorted order.
    pub fn document_ids(&self) -> Vec<String> {
        self.engine.document_ids()
    }

    /// Retrieves a document by ID.
    pub fn get_document(&self, doc_id: &str) -> Option<Document> {
        self.engine.get_document(doc_id)
    }

    /// Deletes a document by ID.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<bool> {
        self.engine.delete_document(doc_id)
    }

    /// Lists all indexed documents.
    pub fn list_documents(&self) -> Vec<Document> {
        let mut docs = Vec::new();
        for id in self.engine.document_ids() {
            if let Some(doc) = self.engine.get_document(&id) {
                docs.push(doc);
            }
        }
        docs
    }

    /// Clears the semantic vector query cache.
    pub fn clear_cache(&mut self) {
        self.engine.clear_cache();
    }

    /// Ingests a raw text document, automatically chunking and embedding if configured.
    pub fn ingest_text(
        &mut self,
        doc_id: impl Into<String>,
        text: &str,
        metadata: Option<DocumentMetadata>,
    ) -> Result<()> {
        let id_str = doc_id.into();

        // 1. Generate document-level embedding if provider configured
        let doc_emb = if let Some(ref emb_provider) = self.embedder {
            Some(Embedding::new(emb_provider.embed_query(text)?))
        } else {
            None
        };

        // 2. Build Document
        let mut doc = Document {
            id: DocumentId::new(&id_str),
            text: text.to_string(),
            embedding: doc_emb,
            metadata: metadata.unwrap_or_default(),
            chunks: Vec::new(),
            created_at: 0,
            updated_at: 0,
        };

        // 3. Chunk if configured
        if let Some(ref chunk_cfg) = self.chunking_config {
            doc = doc.chunk(chunk_cfg);

            // Generate chunk embeddings if provider available
            if let Some(ref emb_provider) = self.embedder {
                for chunk in &mut doc.chunks {
                    let c_emb = emb_provider.embed_query(&chunk.text)?;
                    chunk.embedding = Some(Embedding::new(c_emb));
                }
            }
        }

        self.engine.add_document_model(doc)
    }

    /// Ingests a batch of documents, reporting timing and counts.
    pub fn batch_ingest(
        &mut self,
        items: Vec<(String, String, Option<DocumentMetadata>)>,
    ) -> Result<IngestReport> {
        let start = Instant::now();
        let mut total_chars = 0;
        let mut chunks_created = 0;
        let docs_ingested = items.len();

        for (id, text, meta) in items {
            total_chars += text.len();
            if let Some(ref chunk_cfg) = self.chunking_config {
                let temp_doc = Document::new(id.as_str(), &text);
                chunks_created += temp_doc.chunk(chunk_cfg).chunks.len();
            }
            self.ingest_text(id, &text, meta)?;
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(IngestReport {
            docs_ingested,
            chunks_created,
            total_chars,
            elapsed_ms,
        })
    }

    /// Executes an end-to-end RAG query: embedding -> retrieval -> rerank -> context assembly -> prompt.
    pub fn query(&mut self, query_text: &str, top_k: usize) -> Result<PipelineQueryResult> {
        // 1. Generate query embedding if embedder configured
        let query_emb = if let Some(ref emb) = self.embedder {
            Some(Embedding::new(emb.embed_query(query_text)?))
        } else {
            None
        };

        // 2. Build structured query
        let mut q_builder = RagQuery::builder(query_text)
            .top_k(top_k)
            .rerank(true, Some(top_k));

        if let Some(emb) = query_emb {
            q_builder = q_builder.embedding(emb.as_slice().to_vec());
        }

        let rag_query = q_builder.build();

        // 3. Execute search
        let documents = self.engine.search_query(&rag_query)?;

        // 4. Assemble context block
        let context = ContextAssembler::assemble(&documents, &self.context_config);

        // 5. Format prompt
        let prompt = self.prompt_template.format_prompt(query_text, &context);

        Ok(PipelineQueryResult {
            query: query_text.to_string(),
            documents,
            context,
            prompt,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::config::RagConfigBuilder;
    use crate::rag::dense::cosine_similarity;

    #[test]
    fn test_mock_embedding_provider_cosine_similarity() {
        let provider = MockEmbeddingProvider::new(32);
        let v1 = provider.embed_query("Rust LSM storage engine").unwrap();
        let v2 = provider.embed_query("Rust LSM tree database").unwrap();
        let v3 = provider.embed_query("Banana apple pineapple fruit").unwrap();

        assert_eq!(v1.len(), 32);
        let sim_similar = cosine_similarity(&v1, &v2);
        let sim_unrelated = cosine_similarity(&v1, &v3);

        // Semantically similar sentences should score higher than unrelated ones
        assert!(
            sim_similar > sim_unrelated,
            "sim_similar: {}, sim_unrelated: {}",
            sim_similar,
            sim_unrelated
        );
    }

    #[test]
    fn test_rag_pipeline_end_to_end() {
        let config = RagConfigBuilder::new().embedding_dim(16).build();
        let engine = RagEngine::new(config);
        let embedder = Arc::new(MockEmbeddingProvider::new(16));

        let mut pipeline = RagPipeline::new(engine)
            .with_embedder(embedder)
            .with_chunking(ChunkingConfig::default());

        let report = pipeline
            .batch_ingest(vec![
                (
                    "doc1".into(),
                    "FlashStore is a high performance LSM engine in Rust.".into(),
                    None,
                ),
                (
                    "doc2".into(),
                    "Relational databases use B-Trees and write-ahead logs.".into(),
                    None,
                ),
            ])
            .unwrap();

        assert_eq!(report.docs_ingested, 2);

        let result = pipeline.query("FlashStore LSM engine", 1).unwrap();
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].id.as_str(), "doc1");
        assert!(result.prompt.contains("FlashStore"));
        assert!(result.context.doc_count >= 1);
    }
}
