use std::collections::HashMap;
use std::sync::Arc;

use crate::batch::WriteBatch;
use crate::engine::FlashStore;
use crate::error::Result;
use crate::rag::types::{Document, DocumentChunk, DocumentId, DocumentMetadata, Embedding};

/// Prefix for document raw text in FlashStore KV engine.
pub const PREFIX_DOC: &[u8] = b"rag:doc:";
/// Prefix for document vector embeddings in FlashStore KV engine.
pub const PREFIX_VEC: &[u8] = b"rag:vec:";
/// Prefix for document metadata in FlashStore KV engine.
pub const PREFIX_META: &[u8] = b"rag:meta:";
/// Prefix for document chunk data in FlashStore KV engine.
pub const PREFIX_CHUNK: &[u8] = b"rag:chunk:";
/// Prefix for system / index state in FlashStore KV engine.
pub const PREFIX_SYS: &[u8] = b"rag:sys:";

/// Storage adapter providing atomic persistence, recovery, and hydration
/// between the in-memory RAG subsystem and the persistent FlashStore LSM-tree engine.
#[derive(Clone)]
pub struct RagStoreAdapter {
    store: Arc<FlashStore>,
}

impl RagStoreAdapter {
    /// Creates a new `RagStoreAdapter` wrapping the given `FlashStore` engine.
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying `FlashStore` engine.
    pub fn store(&self) -> &Arc<FlashStore> {
        &self.store
    }

    /// Formats a key for document raw text.
    pub fn doc_key(doc_id: &str) -> Vec<u8> {
        [PREFIX_DOC, doc_id.as_bytes()].concat()
    }

    /// Formats a key for document embedding vector.
    pub fn vec_key(doc_id: &str) -> Vec<u8> {
        [PREFIX_VEC, doc_id.as_bytes()].concat()
    }

    /// Formats a key for document metadata JSON.
    pub fn meta_key(doc_id: &str) -> Vec<u8> {
        [PREFIX_META, doc_id.as_bytes()].concat()
    }

    /// Formats a key for a specific document chunk.
    pub fn chunk_key(doc_id: &str, chunk_index: usize) -> Vec<u8> {
        format!("rag:chunk:{}:{}", doc_id, chunk_index).into_bytes()
    }

    /// Atomically persists a single document (text, embedding, metadata, and chunks)
    /// into FlashStore via a single `WriteBatch`.
    pub fn persist_document(&self, doc: &Document) -> Result<()> {
        let mut batch = WriteBatch::new();
        self.stage_document_put(&mut batch, doc)?;
        self.store.write_batch(batch)
    }

    /// Atomically persists multiple documents in a single atomic `WriteBatch`.
    pub fn persist_documents_batch(&self, docs: &[Document]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new();
        for doc in docs {
            self.stage_document_put(&mut batch, doc)?;
        }
        self.store.write_batch(batch)
    }

    /// Atomically deletes a document and all its associated vectors, metadata, and chunks.
    pub fn delete_document(&self, doc_id: &str) -> Result<bool> {
        let dkey = Self::doc_key(doc_id);
        let exists = self.store.get(&dkey)?.is_some();
        if !exists {
            return Ok(false);
        }

        let mut batch = WriteBatch::new();
        batch.delete(dkey);
        batch.delete(Self::vec_key(doc_id));
        batch.delete(Self::meta_key(doc_id));

        // Delete any chunks associated with this doc_id
        let chunk_prefix = format!("rag:chunk:{}:", doc_id);
        let prefix_bytes = chunk_prefix.as_bytes();
        let entries = self
            .store
            .scan(Some(bytes::Bytes::copy_from_slice(prefix_bytes)), None)?;
        for (k, _) in entries {
            if !k.starts_with(prefix_bytes) {
                break;
            }
            batch.delete(k);
        }

        self.store.write_batch(batch)?;
        Ok(true)
    }

    /// Loads a single `Document` by `doc_id` from FlashStore, reconstructing
    /// its text, embedding, metadata, and chunks.
    pub fn load_document(&self, doc_id: &str) -> Result<Option<Document>> {
        let text_bytes = match self.store.get(Self::doc_key(doc_id))? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let text = String::from_utf8(text_bytes.to_vec())
            .map_err(|e| crate::error::FlashStoreError::Corruption(e.to_string()))?;

        let embedding = if let Some(vbytes) = self.store.get(Self::vec_key(doc_id))? {
            let vec: Vec<f32> = bincode::deserialize(&vbytes)?;
            Some(Embedding::new(vec))
        } else {
            None
        };

        let metadata = if let Some(mbytes) = self.store.get(Self::meta_key(doc_id))? {
            serde_json::from_slice(&mbytes)?
        } else {
            DocumentMetadata::default()
        };

        // Load chunks
        let chunk_prefix = format!("rag:chunk:{}:", doc_id);
        let prefix_bytes = chunk_prefix.as_bytes();
        let mut chunks = Vec::new();
        let entries = self
            .store
            .scan(Some(bytes::Bytes::copy_from_slice(prefix_bytes)), None)?;
        for (k, v) in entries {
            if !k.starts_with(prefix_bytes) {
                break;
            }
            let chunk: DocumentChunk = serde_json::from_slice(&v)?;
            chunks.push(chunk);
        }
        chunks.sort_by_key(|c| c.chunk_index);

        Ok(Some(Document {
            id: DocumentId::new(doc_id),
            text,
            embedding,
            metadata,
            chunks,
            created_at: 0,
            updated_at: 0,
        }))
    }

    /// Recovers all documents stored under the `rag:` key range in FlashStore.
    /// Reconstructs texts, embeddings, metadata, and chunks into full `Document` objects.
    pub fn recover_all(&self) -> Result<Vec<Document>> {
        let entries = self
            .store
            .scan(Some(bytes::Bytes::from_static(b"rag:")), None)?;

        let mut docs_map: HashMap<String, String> = HashMap::new();
        let mut vecs_map: HashMap<String, Embedding> = HashMap::new();
        let mut meta_map: HashMap<String, DocumentMetadata> = HashMap::new();
        let mut chunks_map: HashMap<String, Vec<DocumentChunk>> = HashMap::new();

        for (key, val) in entries {
            if !key.starts_with(b"rag:") {
                break;
            }

            if key.starts_with(PREFIX_DOC) {
                let id_bytes = &key[PREFIX_DOC.len()..];
                if let Ok(id) = std::str::from_utf8(id_bytes) {
                    if let Ok(text) = std::str::from_utf8(&val) {
                        docs_map.insert(id.to_string(), text.to_string());
                    }
                }
            } else if key.starts_with(PREFIX_VEC) {
                let id_bytes = &key[PREFIX_VEC.len()..];
                if let Ok(id) = std::str::from_utf8(id_bytes) {
                    if let Ok(vec) = bincode::deserialize::<Vec<f32>>(&val) {
                        vecs_map.insert(id.to_string(), Embedding::new(vec));
                    }
                }
            } else if key.starts_with(PREFIX_META) {
                let id_bytes = &key[PREFIX_META.len()..];
                if let Ok(id) = std::str::from_utf8(id_bytes) {
                    if let Ok(meta) = serde_json::from_slice::<DocumentMetadata>(&val) {
                        meta_map.insert(id.to_string(), meta);
                    }
                }
            } else if key.starts_with(PREFIX_CHUNK) {
                if let Ok(chunk) = serde_json::from_slice::<DocumentChunk>(&val) {
                    chunks_map
                        .entry(chunk.doc_id.to_string())
                        .or_default()
                        .push(chunk);
                }
            }
        }

        let mut documents = Vec::with_capacity(docs_map.len());
        for (id, text) in docs_map {
            let embedding = vecs_map.remove(&id);
            let metadata = meta_map.remove(&id).unwrap_or_default();
            let mut chunks = chunks_map.remove(&id).unwrap_or_default();
            chunks.sort_by_key(|c| c.chunk_index);

            documents.push(Document {
                id: DocumentId::new(&id),
                text,
                embedding,
                metadata,
                chunks,
                created_at: 0,
                updated_at: 0,
            });
        }

        documents.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        Ok(documents)
    }

    /// Internal helper to stage puts for a single document into a WriteBatch.
    fn stage_document_put(&self, batch: &mut WriteBatch, doc: &Document) -> Result<()> {
        let id_str = doc.id.as_str();

        // 1. Raw text
        batch.put(Self::doc_key(id_str), doc.text.as_bytes());

        // 2. Vector embedding
        if let Some(ref emb) = doc.embedding {
            let encoded_vec = bincode::serialize(emb.as_slice())?;
            batch.put(Self::vec_key(id_str), encoded_vec);
        }

        // 3. Metadata
        if !doc.metadata.is_empty() {
            let encoded_meta = serde_json::to_vec(&doc.metadata)?;
            batch.put(Self::meta_key(id_str), encoded_meta);
        }

        // 4. Chunks
        for chunk in &doc.chunks {
            let chunk_key = Self::chunk_key(id_str, chunk.chunk_index);
            let encoded_chunk = serde_json::to_vec(chunk)?;
            batch.put(chunk_key, encoded_chunk);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_rag_store_adapter_atomic_lifecycle() {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let store = Arc::new(FlashStore::open(options).unwrap());
        let adapter = RagStoreAdapter::new(store.clone());

        let doc1 = Document::builder("doc_atom_1", "Atomic storage test document 1")
            .embedding(vec![0.1, 0.2, 0.3])
            .metadata_field("tag", "alpha")
            .build();

        let doc2 = Document::builder("doc_atom_2", "Atomic storage test document 2")
            .embedding(vec![0.4, 0.5, 0.6])
            .metadata_field("tag", "beta")
            .build();

        // Atomic batch persist
        adapter
            .persist_documents_batch(&[doc1.clone(), doc2.clone()])
            .unwrap();

        // Load single
        let loaded1 = adapter.load_document("doc_atom_1").unwrap().unwrap();
        assert_eq!(loaded1.id.as_str(), "doc_atom_1");
        assert_eq!(loaded1.text, "Atomic storage test document 1");
        assert_eq!(loaded1.embedding.unwrap().as_slice(), &[0.1, 0.2, 0.3]);
        assert_eq!(loaded1.metadata.get_string("tag"), Some("alpha"));

        // Recover all
        let recovered = adapter.recover_all().unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].id.as_str(), "doc_atom_1");
        assert_eq!(recovered[1].id.as_str(), "doc_atom_2");

        // Delete doc1
        assert!(adapter.delete_document("doc_atom_1").unwrap());
        assert!(adapter.load_document("doc_atom_1").unwrap().is_none());

        let remaining = adapter.recover_all().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id.as_str(), "doc_atom_2");
    }
}
