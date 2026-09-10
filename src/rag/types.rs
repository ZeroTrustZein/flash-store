//! # Domain Models and Type Annotations for RAG Subsystem
//!
//! Provides rich, type-safe domain models for documents, embeddings,
//! metadata filtering, query specifications, search fusion, and reranking explanations.

use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::error::{FlashStoreError, Result};
use crate::rag::dense::{cosine_similarity, dot_product, euclidean_distance, l2_norm};
use crate::rag::hybrid::SearchResult;

/// Strongly typed identifier for documents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocumentId(String);

impl DocumentId {
    /// Creates a new `DocumentId`.
    #[inline]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrows document id as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for DocumentId {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for DocumentId {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for DocumentId {
    #[inline]
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DocumentId {
    #[inline]
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<DocumentId> for String {
    #[inline]
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

/// Primitive metadata value type supported in document attributes and filters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    /// Textual string value.
    String(String),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point number.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
    /// Nested list of metadata values.
    List(Vec<MetadataValue>),
}

impl MetadataValue {
    /// Returns the string value if this is a string variant.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the integer value if this is an int variant.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MetadataValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the floating point value if this is a float or integer variant.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Returns the boolean value if this is a bool variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns a slice of metadata values if this is a list variant.
    pub fn as_list(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::List(l) => Some(l.as_slice()),
            _ => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataValue::String(s) => write!(f, "\"{}\"", s),
            MetadataValue::Int(i) => write!(f, "{}", i),
            MetadataValue::Float(v) => write!(f, "{}", v),
            MetadataValue::Bool(b) => write!(f, "{}", b),
            MetadataValue::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl From<&str> for MetadataValue {
    #[inline]
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    #[inline]
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<i64> for MetadataValue {
    #[inline]
    fn from(i: i64) -> Self {
        MetadataValue::Int(i)
    }
}

impl From<i32> for MetadataValue {
    #[inline]
    fn from(i: i32) -> Self {
        MetadataValue::Int(i as i64)
    }
}

impl From<f64> for MetadataValue {
    #[inline]
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<f32> for MetadataValue {
    #[inline]
    fn from(f: f32) -> Self {
        MetadataValue::Float(f as f64)
    }
}

impl From<bool> for MetadataValue {
    #[inline]
    fn from(b: bool) -> Self {
        MetadataValue::Bool(b)
    }
}

impl<T: Into<MetadataValue>> From<Vec<T>> for MetadataValue {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        MetadataValue::List(vec.into_iter().map(Into::into).collect())
    }
}

/// Key-value metadata store associated with a document or chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    fields: HashMap<String, MetadataValue>,
}

impl DocumentMetadata {
    /// Creates an empty metadata dictionary.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Sets or updates a metadata field.
    pub fn insert(&mut self, key: impl Into<String>, val: impl Into<MetadataValue>) -> &mut Self {
        self.fields.insert(key.into(), val.into());
        self
    }

    /// Retrieves a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.fields.get(key)
    }

    /// Helper to get a string property.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(MetadataValue::as_string)
    }

    /// Helper to get an integer property.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(MetadataValue::as_i64)
    }

    /// Helper to get a float property.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.fields.get(key).and_then(MetadataValue::as_f64)
    }

    /// Helper to get a boolean property.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).and_then(MetadataValue::as_bool)
    }

    /// Checks if a metadata key is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    /// Returns the number of metadata fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns true if no metadata fields are present.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Consumes the wrapper returning the underlying `HashMap`.
    pub fn into_inner(self) -> HashMap<String, MetadataValue> {
        self.fields
    }
}

impl Deref for DocumentMetadata {
    type Target = HashMap<String, MetadataValue>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

/// Atomic predicate condition for filtering document metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterCondition {
    /// Field equals value.
    Eq(String, MetadataValue),
    /// Field does not equal value.
    Ne(String, MetadataValue),
    /// Field matches any of the given values.
    In(String, Vec<MetadataValue>),
    /// Field does not match any of the given values.
    NotIn(String, Vec<MetadataValue>),
    /// Numeric field strictly greater than value.
    Gt(String, f64),
    /// Numeric field greater than or equal to value.
    Gte(String, f64),
    /// Numeric field strictly less than value.
    Lt(String, f64),
    /// Numeric field less than or equal to value.
    Lte(String, f64),
    /// String field contains substring.
    Contains(String, String),
}

impl FilterCondition {
    /// Evaluates this filter condition against the given document metadata.
    pub fn matches(&self, metadata: &DocumentMetadata) -> bool {
        match self {
            FilterCondition::Eq(key, expected) => metadata.get(key) == Some(expected),
            FilterCondition::Ne(key, not_expected) => metadata.get(key) != Some(not_expected),
            FilterCondition::In(key, allowed) => metadata
                .get(key)
                .is_some_and(|v| allowed.iter().any(|item| item == v)),
            FilterCondition::NotIn(key, disallowed) => metadata
                .get(key)
                .is_none_or(|v| !disallowed.iter().any(|item| item == v)),
            FilterCondition::Gt(key, target) => metadata
                .get(key)
                .and_then(MetadataValue::as_f64)
                .is_some_and(|v| v > *target),
            FilterCondition::Gte(key, target) => metadata
                .get(key)
                .and_then(MetadataValue::as_f64)
                .is_some_and(|v| v >= *target),
            FilterCondition::Lt(key, target) => metadata
                .get(key)
                .and_then(MetadataValue::as_f64)
                .is_some_and(|v| v < *target),
            FilterCondition::Lte(key, target) => metadata
                .get(key)
                .and_then(MetadataValue::as_f64)
                .is_some_and(|v| v <= *target),
            FilterCondition::Contains(key, needle) => metadata
                .get_string(key)
                .is_some_and(|s| s.contains(needle.as_str())),
        }
    }
}

/// Composite boolean expression filter for metadata queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// Single condition leaf.
    Condition(FilterCondition),
    /// All sub-filters must match (AND).
    And(Vec<MetadataFilter>),
    /// At least one sub-filter must match (OR).
    Or(Vec<MetadataFilter>),
    /// Sub-filter must not match (NOT).
    Not(Box<MetadataFilter>),
}

impl MetadataFilter {
    /// Creates a leaf condition filter.
    pub fn condition(condition: FilterCondition) -> Self {
        MetadataFilter::Condition(condition)
    }

    /// Combines multiple filters with AND.
    pub fn all(filters: Vec<MetadataFilter>) -> Self {
        MetadataFilter::And(filters)
    }

    /// Combines multiple filters with OR.
    pub fn any(filters: Vec<MetadataFilter>) -> Self {
        MetadataFilter::Or(filters)
    }

    /// Negates a filter.
    pub fn negate(filter: MetadataFilter) -> Self {
        MetadataFilter::Not(Box::new(filter))
    }

    /// Evaluates the complete filter tree against document metadata.
    pub fn matches(&self, metadata: &DocumentMetadata) -> bool {
        match self {
            MetadataFilter::Condition(cond) => cond.matches(metadata),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            MetadataFilter::Not(filter) => !filter.matches(metadata),
        }
    }
}

/// Dense vector embedding representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Creates a new embedding from a float vector.
    pub fn new(vector: Vec<f32>) -> Self {
        Self(vector)
    }

    /// Creates an embedding from a slice copy.
    pub fn from_slice(slice: &[f32]) -> Self {
        Self(slice.to_vec())
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Borrows the underlying float slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Consumes the embedding into the inner `Vec<f32>`.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Calculates Euclidean L2 norm of the embedding.
    pub fn l2_norm(&self) -> f32 {
        l2_norm(&self.0)
    }

    /// In-place unit L2 normalization.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 0.0 {
            for x in &mut self.0 {
                *x /= norm;
            }
        }
    }

    /// Returns a new unit-normalized embedding.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Verifies dimension equality.
    fn check_dim(&self, other: &Embedding) -> Result<()> {
        if self.dim() != other.dim() {
            Err(FlashStoreError::VectorDimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            })
        } else {
            Ok(())
        }
    }

    /// Computes dot product with another embedding.
    pub fn dot_product(&self, other: &Embedding) -> Result<f32> {
        self.check_dim(other)?;
        Ok(dot_product(&self.0, &other.0))
    }

    /// Computes cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f32> {
        self.check_dim(other)?;
        Ok(cosine_similarity(&self.0, &other.0))
    }

    /// Computes Euclidean distance to another embedding.
    pub fn euclidean_distance(&self, other: &Embedding) -> Result<f32> {
        self.check_dim(other)?;
        Ok(euclidean_distance(&self.0, &other.0))
    }
}

impl Deref for Embedding {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<f32>> for Embedding {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        Self::new(v)
    }
}

impl From<&[f32]> for Embedding {
    #[inline]
    fn from(s: &[f32]) -> Self {
        Self::from_slice(s)
    }
}

/// Represents a passage or chunk from a larger document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique identifier for this chunk (e.g., "doc1_chunk_0").
    pub chunk_id: String,
    /// Parent document identifier.
    pub doc_id: DocumentId,
    /// Zero-based chunk index within parent document.
    pub chunk_index: usize,
    /// Textual content of this chunk.
    pub text: String,
    /// Dense vector embedding for this chunk.
    pub embedding: Option<Embedding>,
    /// Start character offset in original document.
    pub start_char: usize,
    /// End character offset in original document.
    pub end_char: usize,
    /// Additional chunk-specific metadata.
    pub metadata: DocumentMetadata,
}

/// Complete document domain model in the RAG subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier.
    pub id: DocumentId,
    /// Full text content of the document.
    pub text: String,
    /// Optional document-level dense embedding.
    pub embedding: Option<Embedding>,
    /// Structured key-value metadata.
    pub metadata: DocumentMetadata,
    /// Optional decomposed chunks.
    pub chunks: Vec<DocumentChunk>,
    /// Creation Unix timestamp in seconds.
    pub created_at: u64,
    /// Last update Unix timestamp in seconds.
    pub updated_at: u64,
}

impl Document {
    /// Creates a new document with minimal required fields.
    pub fn new(id: impl Into<DocumentId>, text: impl Into<String>) -> Self {
        let now = crate::rag::cache::current_timestamp_secs();
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: DocumentMetadata::new(),
            chunks: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Sets the document embedding.
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Sets the document metadata.
    pub fn with_metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the document chunks.
    pub fn with_chunks(mut self, chunks: Vec<DocumentChunk>) -> Self {
        self.chunks = chunks;
        self
    }

    /// Decomposes this document into chunks according to `config`.
    pub fn chunk(mut self, config: &ChunkingConfig) -> Self {
        self.chunks = crate::rag::chunker::chunk_document(&self, config);
        self
    }

    /// Creates a builder for `Document`.
    pub fn builder(id: impl Into<DocumentId>, text: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder::new(id, text)
    }
}

/// Fluent builder for [`Document`].
#[derive(Debug)]
pub struct DocumentBuilder {
    id: DocumentId,
    text: String,
    embedding: Option<Embedding>,
    metadata: DocumentMetadata,
    chunks: Vec<DocumentChunk>,
    created_at: Option<u64>,
    updated_at: Option<u64>,
}

impl DocumentBuilder {
    /// Creates a new document builder.
    pub fn new(id: impl Into<DocumentId>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: DocumentMetadata::new(),
            chunks: Vec::new(),
            created_at: None,
            updated_at: None,
        }
    }

    /// Sets the dense vector embedding.
    pub fn embedding(mut self, embedding: impl Into<Embedding>) -> Self {
        self.embedding = Some(embedding.into());
        self
    }

    /// Adds a single metadata field.
    pub fn metadata_field(mut self, key: impl Into<String>, val: impl Into<MetadataValue>) -> Self {
        self.metadata.insert(key, val);
        self
    }

    /// Sets entire metadata.
    pub fn metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets pre-chunked passages.
    pub fn chunks(mut self, chunks: Vec<DocumentChunk>) -> Self {
        self.chunks = chunks;
        self
    }

    /// Automatically splits document text into chunks using `config`.
    pub fn chunk_with(mut self, config: &ChunkingConfig) -> Self {
        let temp_doc = Document {
            id: self.id.clone(),
            text: self.text.clone(),
            embedding: self.embedding.clone(),
            metadata: self.metadata.clone(),
            chunks: Vec::new(),
            created_at: 0,
            updated_at: 0,
        };
        self.chunks = crate::rag::chunker::chunk_document(&temp_doc, config);
        self
    }

    /// Overrides timestamps.
    pub fn timestamps(mut self, created_at: u64, updated_at: u64) -> Self {
        self.created_at = Some(created_at);
        self.updated_at = Some(updated_at);
        self
    }

    /// Builds the `Document`.
    pub fn build(self) -> Document {
        let now = crate::rag::cache::current_timestamp_secs();
        Document {
            id: self.id,
            text: self.text,
            embedding: self.embedding,
            metadata: self.metadata,
            chunks: self.chunks,
            created_at: self.created_at.unwrap_or(now),
            updated_at: self.updated_at.unwrap_or(now),
        }
    }
}

/// Hybrid retrieval fusion strategy selector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion with smoothing parameter k.
    Rrf { k: usize },
    /// Normalized linear combination: `alpha * dense + (1 - alpha) * sparse`.
    WeightedLinear { dense_weight: f32 },
    /// Dense vector retrieval only.
    DenseOnly,
    /// Sparse BM25 retrieval only.
    SparseOnly,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::Rrf { k: 60 }
    }
}

/// Strategy for splitting long documents into passages/chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed character window with overlap.
    FixedChars { size: usize, overlap: usize },
    /// Fixed token window with overlap.
    FixedTokens { size: usize, overlap: usize },
    /// Split by double newlines into paragraphs.
    Paragraph,
    /// Split by punctuation into sentences.
    Sentence,
}

/// Configuration for document chunking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Primary chunking strategy.
    pub strategy: ChunkingStrategy,
    /// Minimum chunk length to avoid tiny fragments.
    pub min_chunk_size: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::FixedTokens {
                size: 256,
                overlap: 32,
            },
            min_chunk_size: 20,
        }
    }
}

/// Detailed explanation of cross-encoder reranking score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Lexical token coverage fraction in [0.0, 1.0].
    pub token_coverage: f32,
    /// Contiguous phrase match score in [0.0, 1.0].
    pub phrase_match: f32,
    /// Term span proximity density in [0.0, 1.0].
    pub proximity: f32,
    /// Final combined score.
    pub combined_score: f32,
}

/// Comprehensive search result domain model including metadata and score breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredDocument {
    /// Document identifier.
    pub id: DocumentId,
    /// Aggregated relevance score.
    pub score: f32,
    /// Dense retrieval score if available.
    pub dense_score: Option<f32>,
    /// Sparse BM25 retrieval score if available.
    pub sparse_score: Option<f32>,
    /// Document text content if loaded.
    pub text: Option<String>,
    /// Associated document metadata if loaded.
    pub metadata: Option<DocumentMetadata>,
    /// Cross-encoder score explanation if reranked.
    pub explanation: Option<ScoreExplanation>,
}

impl From<SearchResult> for ScoredDocument {
    fn from(sr: SearchResult) -> Self {
        Self {
            id: DocumentId::new(sr.doc_id),
            score: sr.score,
            dense_score: sr.dense_score,
            sparse_score: sr.sparse_score,
            text: None,
            metadata: None,
            explanation: None,
        }
    }
}

impl From<ScoredDocument> for SearchResult {
    fn from(sd: ScoredDocument) -> Self {
        Self {
            doc_id: sd.id.into(),
            score: sd.score,
            dense_score: sd.dense_score,
            sparse_score: sd.sparse_score,
        }
    }
}

/// Structured query specification for RAG retrieval and reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagQuery {
    /// Query string for BM25 and cross-encoder reranking.
    pub text: String,
    /// Dense query embedding for vector similarity search.
    pub embedding: Option<Embedding>,
    /// Number of final search candidates to retrieve.
    pub top_k: usize,
    /// Optional metadata filtering predicate.
    pub filter: Option<MetadataFilter>,
    /// Strategy for fusing sparse and dense candidate lists.
    pub fusion_strategy: FusionStrategy,
    /// Whether to run cross-encoder reranking on candidate hits.
    pub rerank: bool,
    /// Number of candidates to rerank if cross-encoder is active.
    pub rerank_top_k: Option<usize>,
    /// Minimum score threshold to qualify as a hit.
    pub min_score: Option<f32>,
    /// Optional diversity reranking using Maximal Marginal Relevance (MMR) with balance factor lambda.
    #[serde(default)]
    pub mmr_lambda: Option<f32>,
}

impl RagQuery {
    /// Creates a minimal query for the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            top_k: 10,
            filter: None,
            fusion_strategy: FusionStrategy::default(),
            rerank: false,
            rerank_top_k: None,
            min_score: None,
            mmr_lambda: None,
        }
    }

    /// Returns a builder for fluent `RagQuery` construction.
    pub fn builder(text: impl Into<String>) -> RagQueryBuilder {
        RagQueryBuilder::new(text)
    }
}

/// Builder for [`RagQuery`].
#[derive(Debug)]
pub struct RagQueryBuilder {
    query: RagQuery,
}

impl RagQueryBuilder {
    /// Creates a builder with the search text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            query: RagQuery::new(text),
        }
    }

    /// Sets query vector embedding.
    pub fn embedding(mut self, embedding: impl Into<Embedding>) -> Self {
        self.query.embedding = Some(embedding.into());
        self
    }

    /// Sets top-k documents to return.
    pub fn top_k(mut self, k: usize) -> Self {
        self.query.top_k = k;
        self
    }

    /// Sets metadata filtering predicate.
    pub fn filter(mut self, filter: MetadataFilter) -> Self {
        self.query.filter = Some(filter);
        self
    }

    /// Sets fusion strategy.
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.query.fusion_strategy = strategy;
        self
    }

    /// Enables cross-encoder reranking with optional top-k cutoff.
    pub fn rerank(mut self, enabled: bool, top_k: Option<usize>) -> Self {
        self.query.rerank = enabled;
        self.query.rerank_top_k = top_k;
        self
    }

    /// Sets minimum score cutoff.
    pub fn min_score(mut self, min: f32) -> Self {
        self.query.min_score = Some(min);
        self
    }

    /// Enables diversity reranking using Maximal Marginal Relevance (MMR) with balance parameter lambda.
    pub fn mmr(mut self, lambda: f32) -> Self {
        self.query.mmr_lambda = Some(lambda.clamp(0.0, 1.0));
        self
    }

    /// Builds the `RagQuery`.
    pub fn build(self) -> RagQuery {
        self.query
    }
}

/// Semantic cache telemetry statistics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SemanticCacheStats {
    /// Maximum cache entries.
    pub capacity: usize,
    /// Currently stored entries.
    pub len: usize,
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Hit rate fraction in [0.0, 1.0].
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_id_basics() {
        let id = DocumentId::new("doc_123");
        assert_eq!(id.as_str(), "doc_123");
        assert_eq!(&*id, "doc_123");
        assert_eq!(id.to_string(), "doc_123");

        let s: String = id.clone().into();
        assert_eq!(s, "doc_123");

        let from_str: DocumentId = "doc_456".into();
        assert_eq!(from_str.as_ref(), "doc_456");
    }

    #[test]
    fn test_metadata_value_conversions_and_display() {
        let s: MetadataValue = "hello".into();
        assert_eq!(s.as_string(), Some("hello"));
        assert_eq!(s.to_string(), "\"hello\"");

        let i: MetadataValue = 42i64.into();
        assert_eq!(i.as_i64(), Some(42));
        assert_eq!(i.as_f64(), Some(42.0));
        assert_eq!(i.to_string(), "42");

        let f: MetadataValue = 42.5f64.into();
        assert!((f.as_f64().unwrap() - 42.5).abs() < 1e-6);

        let b: MetadataValue = true.into();
        assert_eq!(b.as_bool(), Some(true));

        let l: MetadataValue = vec![MetadataValue::Int(1), MetadataValue::Int(2)].into();
        assert_eq!(l.as_list().unwrap().len(), 2);
        assert_eq!(l.to_string(), "[1, 2]");
    }

    #[test]
    fn test_document_metadata_operations() {
        let mut meta = DocumentMetadata::new();
        assert!(meta.is_empty());
        assert_eq!(meta.len(), 0);

        meta.insert("author", "Zein");
        meta.insert("version", 2i64);
        meta.insert("rating", 4.8f64);
        meta.insert("published", true);

        assert_eq!(meta.len(), 4);
        assert!(!meta.is_empty());
        assert!(meta.contains_key("author"));
        assert_eq!(meta.get_string("author"), Some("Zein"));
        assert_eq!(meta.get_i64("version"), Some(2));
        assert!((meta.get_f64("rating").unwrap() - 4.8).abs() < 1e-6);
        assert_eq!(meta.get_bool("published"), Some(true));
    }

    #[test]
    fn test_metadata_filtering() {
        let mut meta = DocumentMetadata::new();
        meta.insert("category", "tech");
        meta.insert("year", 2026i64);
        meta.insert("score", 95.5f64);

        // Simple condition
        let cond_eq = FilterCondition::Eq("category".into(), "tech".into());
        assert!(cond_eq.matches(&meta));

        let cond_gt = FilterCondition::Gt("score".into(), 90.0);
        assert!(cond_gt.matches(&meta));

        let cond_in = FilterCondition::In("category".into(), vec!["news".into(), "tech".into()]);
        assert!(cond_in.matches(&meta));

        let cond_contains = FilterCondition::Contains("category".into(), "ec".into());
        assert!(cond_contains.matches(&meta));

        // Composite filters: AND, OR, NOT
        let filter_and = MetadataFilter::all(vec![
            MetadataFilter::condition(cond_eq),
            MetadataFilter::condition(cond_gt),
        ]);
        assert!(filter_and.matches(&meta));

        let filter_not = MetadataFilter::negate(MetadataFilter::condition(FilterCondition::Eq(
            "category".into(),
            "sports".into(),
        )));
        assert!(filter_not.matches(&meta));

        let filter_or = MetadataFilter::any(vec![
            MetadataFilter::condition(FilterCondition::Eq("category".into(), "sports".into())),
            MetadataFilter::condition(FilterCondition::Gte("year".into(), 2020.0)),
        ]);
        assert!(filter_or.matches(&meta));
    }

    #[test]
    fn test_embedding_vector_operations() {
        let mut emb1 = Embedding::new(vec![3.0, 4.0]);
        assert_eq!(emb1.dim(), 2);
        assert_eq!(emb1.l2_norm(), 5.0);

        emb1.normalize();
        assert!((emb1.l2_norm() - 1.0).abs() < 1e-6);
        assert!((emb1[0] - 0.6).abs() < 1e-6);
        assert!((emb1[1] - 0.8).abs() < 1e-6);

        let emb2 = Embedding::from_slice(&[1.0, 0.0]);
        let emb3 = Embedding::from_slice(&[0.0, 1.0]);

        assert_eq!(emb2.dot_product(&emb3).unwrap(), 0.0);
        assert_eq!(emb2.cosine_similarity(&emb3).unwrap(), 0.0);
        assert!((emb2.euclidean_distance(&emb3).unwrap() - 2.0f32.sqrt()).abs() < 1e-6);

        // Dimension mismatch
        let mismatch = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert!(emb2.dot_product(&mismatch).is_err());
    }

    #[test]
    fn test_document_builder_and_domain_model() {
        let doc = Document::builder("doc_01", "High performance LSM tree")
            .embedding(vec![1.0, 0.0, 0.0])
            .metadata_field("tag", "database")
            .metadata_field("stars", 100i64)
            .build();

        assert_eq!(doc.id.as_str(), "doc_01");
        assert_eq!(doc.text, "High performance LSM tree");
        assert!(doc.embedding.is_some());
        assert_eq!(doc.embedding.as_ref().unwrap().dim(), 3);
        assert_eq!(doc.metadata.get_string("tag"), Some("database"));
        assert_eq!(doc.metadata.get_i64("stars"), Some(100));
    }

    #[test]
    fn test_rag_query_builder() {
        let query = RagQuery::builder("flash store lsm")
            .embedding(vec![0.5, 0.5])
            .top_k(5)
            .fusion_strategy(FusionStrategy::WeightedLinear { dense_weight: 0.7 })
            .rerank(true, Some(3))
            .min_score(0.1)
            .build();

        assert_eq!(query.text, "flash store lsm");
        assert_eq!(query.top_k, 5);
        assert_eq!(
            query.fusion_strategy,
            FusionStrategy::WeightedLinear { dense_weight: 0.7 }
        );
        assert!(query.rerank);
        assert_eq!(query.rerank_top_k, Some(3));
        assert_eq!(query.min_score, Some(0.1));
    }

    #[test]
    fn test_scored_document_conversions() {
        let sr = SearchResult {
            doc_id: "docA".to_string(),
            score: 0.88,
            dense_score: Some(0.9),
            sparse_score: Some(0.85),
        };

        let scored_doc: ScoredDocument = sr.clone().into();
        assert_eq!(scored_doc.id.as_str(), "docA");
        assert_eq!(scored_doc.score, 0.88);

        let back_sr: SearchResult = scored_doc.into();
        assert_eq!(back_sr, sr);
    }
}
