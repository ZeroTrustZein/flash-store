use std::collections::HashSet;

use crate::rag::dense::cosine_similarity;
use crate::rag::sparse::tokenize;
use crate::rag::types::ScoreExplanation;
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
    /// Optional detailed scoring breakdown.
    pub explanation: Option<ScoreExplanation>,
}

/// Trait defining a cross-encoder pair scoring model.
pub trait CrossEncoderScorer: Send + Sync {
    /// Computes relevance score for query and document text.
    fn score(&self, query: &str, doc_text: &str) -> f32;

    /// Computes detailed score explanation.
    fn explain(&self, query: &str, doc_text: &str) -> ScoreExplanation {
        ScoreExplanation {
            token_coverage: 0.0,
            phrase_match: 0.0,
            proximity: 0.0,
            combined_score: self.score(query, doc_text),
        }
    }
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

impl LexicalSemanticCrossEncoder {
    /// Computes a detailed score explanation breaking down token match, phrase match, and term proximity.
    pub fn explain(&self, query: &str, doc_text: &str) -> ScoreExplanation {
        let q_tokens = tokenize(query);
        let d_tokens = tokenize(doc_text);

        if q_tokens.is_empty() || d_tokens.is_empty() {
            return ScoreExplanation {
                token_coverage: 0.0,
                phrase_match: 0.0,
                proximity: 0.0,
                combined_score: 0.0,
            };
        }

        // 1. Token match coverage with O(1) set lookup
        let d_set: HashSet<&str> = d_tokens.iter().map(|s| s.as_str()).collect();
        let matched_tokens = q_tokens
            .iter()
            .filter(|qt| d_set.contains(qt.as_str()))
            .count();
        let token_coverage = matched_tokens as f32 / q_tokens.len() as f32;

        // 2. Phrase / bigram match
        let phrase_score = if q_tokens.len() >= 2 && d_tokens.len() >= 2 {
            let d_bigrams: HashSet<(&str, &str)> = d_tokens
                .windows(2)
                .map(|w| (w[0].as_str(), w[1].as_str()))
                .collect();
            let phrase_matches = q_tokens
                .windows(2)
                .filter(|w| d_bigrams.contains(&(w[0].as_str(), w[1].as_str())))
                .count();
            phrase_matches as f32 / (q_tokens.len() - 1) as f32
        } else if q_tokens.len() >= 2 {
            0.0
        } else {
            token_coverage
        };

        // 3. Proximity / span density with O(1) query set lookup
        let q_set: HashSet<&str> = q_tokens.iter().map(|s| s.as_str()).collect();
        let mut first_idx = None;
        let mut last_idx = None;
        for (idx, dt) in d_tokens.iter().enumerate() {
            if q_set.contains(dt.as_str()) {
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

        let combined_score = self.token_match_weight * token_coverage
            + self.phrase_match_weight * phrase_score
            + self.proximity_weight * proximity_score;

        ScoreExplanation {
            token_coverage,
            phrase_match: phrase_score,
            proximity: proximity_score,
            combined_score,
        }
    }
}

impl CrossEncoderScorer for LexicalSemanticCrossEncoder {
    fn score(&self, query: &str, doc_text: &str) -> f32 {
        self.explain(query, doc_text).combined_score
    }

    fn explain(&self, query: &str, doc_text: &str) -> ScoreExplanation {
        self.explain(query, doc_text)
    }
}

/// Reranks candidates using a cross-encoder model with customizable blend weight and optional explanation.
pub fn rerank_candidates_weighted<S: CrossEncoderScorer>(
    scorer: &S,
    query: &str,
    candidates: &[(String, String, f32)],
    cross_weight: f32,
    top_k: usize,
    with_explanation: bool,
) -> Vec<RerankResult> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let alpha = cross_weight.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;

    let mut scored: Vec<(usize, String, f32, f32, Option<ScoreExplanation>)> = candidates
        .iter()
        .enumerate()
        .map(|(initial_rank, (id, text, orig_score))| {
            let explanation = if with_explanation {
                Some(scorer.explain(query, text))
            } else {
                None
            };
            let cross_score = explanation
                .as_ref()
                .map(|e| e.combined_score)
                .unwrap_or_else(|| scorer.score(query, text));
            let final_score = alpha * cross_score + beta * orig_score;
            (
                initial_rank,
                id.clone(),
                *orig_score,
                final_score,
                explanation,
            )
        })
        .collect();

    scored.sort_by(|a, b| b.3.total_cmp(&a.3));
    scored.truncate(top_k);

    scored
        .into_iter()
        .enumerate()
        .map(
            |(new_rank, (initial_rank, doc_id, orig_score, final_score, explanation))| {
                let rank_delta = initial_rank as i32 - new_rank as i32;
                RerankResult {
                    doc_id,
                    original_score: orig_score,
                    reranked_score: final_score,
                    rank_delta,
                    explanation,
                }
            },
        )
        .collect()
}

/// Reranks candidate documents for a given query, attaching detailed scoring explanations.
pub fn rerank_candidates_with_explanation<S: CrossEncoderScorer>(
    scorer: &S,
    query: &str,
    candidates: &[(String, String, f32)],
    top_k: usize,
) -> Vec<RerankResult> {
    rerank_candidates_weighted(scorer, query, candidates, 0.7, top_k, true)
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
    rerank_candidates_weighted(scorer, query, candidates, 0.7, top_k, false)
}

/// Computes Maximal Marginal Relevance (MMR) selection of documents to balance relevance and diversity.
///
/// Iteratively selects candidates that maximize:
/// `MMR(d) = lambda * Sim(d, Query) - (1 - lambda) * max_{s in Selected} Sim(d, s)`
///
/// - `query_vec`: query embedding vector.
/// - `candidates`: slice of tuples `(doc_id, embedding, initial_score)`.
/// - `lambda`: balance factor in `[0.0, 1.0]`. 1.0 focuses entirely on query relevance, 0.0 maximizes diversity.
/// - `top_k`: maximum number of diverse documents to select.
pub fn maximal_marginal_relevance(
    query_vec: &[f32],
    candidates: &[(String, Vec<f32>, f32)],
    lambda: f32,
    top_k: usize,
) -> Vec<RerankResult> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(top_k.min(candidates.len()));

    // Precompute query relevance score for each candidate once
    let query_scores: Vec<f32> = candidates
        .iter()
        .map(|(_, cand_vec, init_score)| {
            if query_vec.is_empty() || cand_vec.is_empty() {
                *init_score
            } else {
                let cos = cosine_similarity(query_vec, cand_vec);
                0.7 * cos + 0.3 * init_score
            }
        })
        .collect();

    while selected.len() < top_k && !remaining.is_empty() {
        let mut best_cand_idx = remaining[0];
        let mut best_rem_pos = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (rem_pos, &cand_idx) in remaining.iter().enumerate() {
            let (_, cand_vec, _) = &candidates[cand_idx];
            let sim_query = query_scores[cand_idx];

            let max_sim_selected = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|&(sel_idx, _)| cosine_similarity(cand_vec, &candidates[sel_idx].1))
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            let mmr_score = lambda * sim_query - (1.0 - lambda) * max_sim_selected;
            if mmr_score > best_mmr {
                best_mmr = mmr_score;
                best_cand_idx = cand_idx;
                best_rem_pos = rem_pos;
            }
        }

        remaining.swap_remove(best_rem_pos);
        selected.push((best_cand_idx, best_mmr));
    }

    selected
        .into_iter()
        .enumerate()
        .map(|(new_rank, (initial_rank, mmr_score))| {
            let (doc_id, _, orig_score) = &candidates[initial_rank];
            let rank_delta = initial_rank as i32 - new_rank as i32;
            RerankResult {
                doc_id: doc_id.clone(),
                original_score: *orig_score,
                reranked_score: mmr_score,
                rank_delta,
                explanation: None,
            }
        })
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

    #[test]
    fn test_rerank_candidates_with_explanation() {
        let scorer = LexicalSemanticCrossEncoder::default();
        let query = "rust lsm";
        let candidates = vec![(
            "doc1".to_string(),
            "rust lsm tree storage engine".to_string(),
            0.5,
        )];

        let reranked = rerank_candidates_with_explanation(&scorer, query, &candidates, 1);
        assert_eq!(reranked.len(), 1);
        assert!(reranked[0].explanation.is_some());
        let exp = reranked[0].explanation.as_ref().unwrap();
        assert!(exp.token_coverage > 0.0);
        assert!(exp.combined_score > 0.0);
    }

    #[test]
    fn test_maximal_marginal_relevance_diversity() {
        let query_vec = vec![1.0, 0.0, 0.0];
        // doc1 is identical to doc2 in vector space, both highly relevant
        // doc3 is orthogonal (diverse) but moderate relevance
        let candidates = vec![
            ("doc1".to_string(), vec![1.0, 0.0, 0.0], 0.95),
            ("doc2".to_string(), vec![0.99, 0.01, 0.0], 0.94),
            ("doc3".to_string(), vec![0.0, 1.0, 0.0], 0.60),
        ];

        // With lambda = 0.5, after doc1 is selected, doc2 is heavily penalized for redundancy,
        // and doc3 is promoted for diversity!
        let mmr_results = maximal_marginal_relevance(&query_vec, &candidates, 0.3, 2);
        assert_eq!(mmr_results.len(), 2);
        assert_eq!(mmr_results[0].doc_id, "doc1");
        assert_eq!(mmr_results[1].doc_id, "doc3");
    }
}
