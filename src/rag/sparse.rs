use std::collections::HashMap;

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Basic whitespace & punctuation tokenizer that normalizes terms to lowercase alphanumeric tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Common English stopwords filtered out during sparse indexing when configured.
pub const DEFAULT_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
];

/// Tokenizer that normalizes text and filters out common English stopwords.
pub fn tokenize_filtered(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| !DEFAULT_STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Computes Okapi BM25 Inverse Document Frequency (IDF) with smoothing.
pub fn compute_idf(doc_freq: usize, total_docs: usize) -> f32 {
    let n = doc_freq as f32;
    let total = total_docs as f32;
    ((total - n + 0.5) / (n + 0.5) + 1.0).ln().max(0.0)
}

/// In-memory inverted index for sparse lexical retrieval using Okapi BM25 / BM25+ scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndex {
    k1: f32,
    b: f32,
    #[serde(default)]
    delta: f32,
    /// Term -> (doc_id -> term frequency)
    postings: HashMap<String, HashMap<String, usize>>,
    /// doc_id -> document token length
    doc_lengths: HashMap<String, usize>,
    /// Cached total tokens across all documents
    total_tokens: usize,
}

impl SparseIndex {
    /// Creates a new sparse index with standard BM25 hyperparameters.
    pub fn new(k1: f32, b: f32) -> Self {
        Self::with_delta(k1, b, 0.0)
    }

    /// Creates a new sparse index with BM25+ hyperparameters, including lower-bound offset `delta`.
    pub fn with_delta(k1: f32, b: f32, delta: f32) -> Self {
        Self {
            k1,
            b,
            delta: delta.max(0.0),
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_tokens: 0,
        }
    }

    /// Returns the BM25+ delta parameter.
    pub fn delta(&self) -> f32 {
        self.delta
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Checks if the index contains any documents.
    pub fn is_empty(&self) -> bool {
        self.doc_lengths.is_empty()
    }

    /// Number of distinct vocabulary terms indexed.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Average document length across the collection.
    pub fn avg_dl(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            0.0
        } else {
            self.total_tokens as f32 / self.doc_lengths.len() as f32
        }
    }

    /// Indexes a document's text content.
    pub fn add_document(&mut self, doc_id: impl Into<String>, text: &str) {
        let doc_id = doc_id.into();
        self.remove_document(&doc_id);

        let tokens = tokenize(text);
        let doc_len = tokens.len();
        self.total_tokens += doc_len;
        self.doc_lengths.insert(doc_id.clone(), doc_len);

        for token in tokens {
            let entry = self.postings.entry(token).or_default();
            *entry.entry(doc_id.clone()).or_insert(0) += 1;
        }
    }

    /// Removes a document from the index.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        if let Some(len) = self.doc_lengths.remove(doc_id) {
            self.total_tokens = self.total_tokens.saturating_sub(len);
            for map in self.postings.values_mut() {
                map.remove(doc_id);
            }
            // Retain terms that still have postings
            self.postings.retain(|_, map| !map.is_empty());
            true
        } else {
            false
        }
    }

    /// Searches for documents matching the query string using BM25 ranking.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let total_docs = self.len();
        let avg_dl = self.avg_dl().max(1.0);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings_map) = self.postings.get(token) {
                let idf = compute_idf(postings_map.len(), total_docs);
                for (doc_id, &tf) in postings_map {
                    let doc_len = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                    let tf_f32 = tf as f32;
                    let numerator = tf_f32 * (self.k1 + 1.0);
                    let denominator =
                        tf_f32 + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl));
                    let score = idf * (numerator / denominator + self.delta);
                    *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Clears the entire sparse index.
    pub fn clear(&mut self) {
        self.postings.clear();
        self.doc_lengths.clear();
        self.total_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "FlashStore: High-Performance, Embedded LSM-Tree!";
        let tokens = tokenize(text);
        assert_eq!(
            tokens,
            vec![
                "flashstore",
                "high",
                "performance",
                "embedded",
                "lsm",
                "tree"
            ]
        );
    }

    #[test]
    fn test_sparse_bm25_search() {
        let mut index = SparseIndex::new(1.2, 0.75);
        index.add_document("doc1", "rust storage engine lsm tree");
        index.add_document("doc2", "relational database sql engine");
        index.add_document("doc3", "rust embedded key value store");

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());

        let results = index.search("rust engine", 5).unwrap();
        assert!(!results.is_empty());
        // doc1 matches both "rust" and "engine"
        assert_eq!(results[0].0, "doc1");

        let single = index.search("sql", 1).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].0, "doc2");

        // Document removal
        assert!(index.remove_document("doc2"));
        assert_eq!(index.len(), 2);
        let empty_search = index.search("sql", 5).unwrap();
        assert!(empty_search.is_empty());
    }

    #[test]
    fn test_bm25_plus_with_delta() {
        let mut index = SparseIndex::with_delta(1.2, 0.75, 0.5);
        assert_eq!(index.delta(), 0.5);
        index.add_document("doc1", "performance database engine");
        index.add_document("doc2", "performance key value storage");

        let res = index.search("performance", 2).unwrap();
        assert_eq!(res.len(), 2);
        assert!(res[0].1 > 0.0);
    }

    #[test]
    fn test_tokenize_filtered() {
        let text = "This is a book on the table with an apple";
        let filtered = tokenize_filtered(text);
        assert!(!filtered.contains(&"is".to_string()));
        assert!(!filtered.contains(&"a".to_string()));
        assert!(!filtered.contains(&"the".to_string()));
        assert!(!filtered.contains(&"with".to_string()));
        assert!(filtered.contains(&"book".to_string()));
        assert!(filtered.contains(&"apple".to_string()));
    }
}
