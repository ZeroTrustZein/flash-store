use crate::rag::sparse::tokenize;
use serde::{Deserialize, Serialize};

/// Result of cross-encoder reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    /// Document identifier.
    pub doc_id: String,
    /// Initial candidate retrieval score before reranking.
    pub original_score: f32,
    /// Reranked relevance score.
    pub reranked_score: f32,
    /// Shift in rank position (e.g. +2 means moved up 2 spots, -1 moved down 1 spot).
    pub rank_delta: i32,
}

/// Trait defining a cross-encoder pair scoring model.
pub trait CrossEncoderScorer: Send + Sync {
    /// Computes relevance score for query and document text.
    fn score(&self, query: &str, doc_text: &str) -> f32;
}

/// Lexical-semantic cross-encoder scorer.
///
/// Evaluates query-document pairs using token overlap, term proximity,
/// contiguous phrase matching, and document length normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalSemanticCrossEncoder {
    /// Weight given to term overlap ratio (default: 0.5).
    pub token_match_weight: f32,
    /// Weight given to contiguous phrase / bigram matching (default: 0.3).
    pub phrase_match_weight: f32,
    /// Weight given to term proximity / density (default: 0.2).
    pub proximity_weight: f32,
}

impl Default for LexicalSemanticCrossEncoder {
    fn default() -> Self {
        Self {
            token_match_weight: 0.5,
            phrase_match_weight: 0.3,
            proximity_weight: 0.2,
        }
    }
}

impl CrossEncoderScorer for LexicalSemanticCrossEncoder {
    fn score(&self, query: &str, doc_text: &str) -> f32 {
        let q_tokens = tokenize(query);
        let d_tokens = tokenize(doc_text);

        if q_tokens.is_empty() || d_tokens.is_empty() {
            return 0.0;
        }

        // 1. Token match coverage
        let mut matched_tokens = 0usize;
        for qt in &q_tokens {
            if d_tokens.contains(qt) {
                matched_tokens += 1;
            }
        }
        let token_coverage = matched_tokens as f32 / q_tokens.len() as f32;

        // 2. Phrase / bigram match
        let mut phrase_matches = 0usize;
        if q_tokens.len() >= 2 && d_tokens.len() >= 2 {
            for i in 0..q_tokens.len() - 1 {
                let pair = (&q_tokens[i], &q_tokens[i + 1]);
                for j in 0..d_tokens.len() - 1 {
                    if pair == (&d_tokens[j], &d_tokens[j + 1]) {
                        phrase_matches += 1;
                        break;
                    }
                }
            }
        }
        let phrase_score = if q_tokens.len() >= 2 {
            phrase_matches as f32 / (q_tokens.len() - 1) as f32
        } else {
            token_coverage
        };

        // 3. Proximity / span density
        let mut first_idx = None;
        let mut last_idx = None;
        for (idx, dt) in d_tokens.iter().enumerate() {
            if q_tokens.contains(dt) {
                if first_idx.is_none() {
                    first_idx = Some(idx);
                }
                last_idx = Some(idx);
            }
        }

        let proximity_score = match (first_idx, last_idx) {
            (Some(f), Some(l)) if matched_tokens > 1 => {
                let span = (l - f + 1) as f32;
                (matched_tokens as f32 / span).clamp(0.0, 1.0)
            }
            (Some(_), Some(_)) => 1.0,
            _ => 0.0,
        };

        // Total weighted score
        self.token_match_weight * token_coverage
            + self.phrase_match_weight * phrase_score
            + self.proximity_weight * proximity_score
    }
}

/// Reranks candidate documents for a given query.
///
/// `candidates`: slice of tuples `(doc_id, document_text, initial_score)`.
pub fn rerank_candidates<S: CrossEncoderScorer>(
    scorer: &S,
    query: &str,
    candidates: &[(String, String, f32)],
    top_k: usize,
) -> Vec<RerankResult> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(usize, String, f32, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(initial_rank, (id, text, orig_score))| {
            let cross_score = scorer.score(query, text);
            // Blend cross-encoder score (70%) with original candidate score (30%)
            let final_score = 0.7 * cross_score + 0.3 * orig_score;
            (initial_rank, id.clone(), *orig_score, final_score)
        })
        .collect();

    scored.sort_by(|a, b| b.3.total_cmp(&a.3));
    scored.truncate(top_k);

    scored
        .into_iter()
        .enumerate()
        .map(
            |(new_rank, (initial_rank, doc_id, orig_score, final_score))| {
                let rank_delta = initial_rank as i32 - new_rank as i32;
                RerankResult {
                    doc_id,
                    original_score: orig_score,
                    reranked_score: final_score,
                    rank_delta,
                }
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_encoder_scoring() {
        let scorer = LexicalSemanticCrossEncoder::default();
        let query = "embedded key value store";
        let doc_relevant = "FlashStore is an embedded key value store in Rust";
        let doc_partial = "Rust is a fast programming language with value types";
        let doc_irrelevant = "Cooking recipes with pasta and tomato sauce";

        let score_rel = scorer.score(query, doc_relevant);
        let score_part = scorer.score(query, doc_partial);
        let score_irrel = scorer.score(query, doc_irrelevant);

        assert!(score_rel > score_part);
        assert!(score_part > score_irrel);
        assert_eq!(score_irrel, 0.0);
    }

    #[test]
    fn test_rerank_candidates() {
        let scorer = LexicalSemanticCrossEncoder::default();
        let query = "rust lsm tree";
        let candidates = vec![
            (
                "doc1".to_string(),
                "Python web framework with django".to_string(),
                0.9,
            ),
            (
                "doc2".to_string(),
                "High performance rust lsm tree storage engine".to_string(),
                0.3,
            ),
        ];

        let reranked = rerank_candidates(&scorer, query, &candidates, 2);
        assert_eq!(reranked.len(), 2);
        // doc2 had lower initial score (0.3), but cross-encoder reranks it to top
        assert_eq!(reranked[0].doc_id, "doc2");
        assert!(reranked[0].rank_delta > 0); // moved up
    }
}
