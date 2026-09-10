use crate::cli::OutputFormat;
use crate::engine::Stats;
use crate::rag::{Document, EvaluationSummary, PipelineQueryResult, RagEngine, ScoredDocument};
use crate::types::{Key, Value};
use serde_json::json;

/// Truncates string to `max_bytes`, safely snapping down to the nearest UTF-8 character boundary.
pub fn safe_truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        let mut end = max_bytes;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// Format and display a slice of key-value pairs according to the chosen format.
pub fn format_kv_pairs(pairs: &[(Key, Value)], format: OutputFormat, quiet: bool) {
    match format {
        OutputFormat::Json => {
            let json_array: Vec<serde_json::Value> = pairs
                .iter()
                .map(|(k, v)| {
                    json!({
                        "key": String::from_utf8_lossy(k).into_owned(),
                        "value": String::from_utf8_lossy(v).into_owned(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json_array).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            for (k, v) in pairs {
                println!(
                    "{}\t{}",
                    String::from_utf8_lossy(k),
                    String::from_utf8_lossy(v)
                );
            }
        }
        OutputFormat::Text => {
            if pairs.is_empty() {
                if !quiet {
                    println!("(empty scan result)");
                }
                return;
            }
            for (k, v) in pairs {
                println!(
                    "{}: {}",
                    String::from_utf8_lossy(k),
                    String::from_utf8_lossy(v)
                );
            }
            if !quiet {
                println!("Total entries: {}", pairs.len());
            }
        }
    }
}

/// Format and print database statistics.
pub fn format_stats(stats: &Stats, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let json_obj = json!({
                "active_memtable_size_bytes": stats.active_memtable_size,
                "immutable_memtables_count": stats.immutable_memtables_count,
                "levels_file_count": stats.levels_file_count,
                "total_sstable_files": stats.levels_file_count.iter().sum::<usize>(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!("metric\tvalue");
            println!("active_memtable_size_bytes\t{}", stats.active_memtable_size);
            println!(
                "immutable_memtables_count\t{}",
                stats.immutable_memtables_count
            );
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                println!("level_{}_sstable_count\t{}", lvl, count);
            }
        }
        OutputFormat::Text => {
            println!("=== FlashStore Statistics ===");
            println!(
                "Active Memtable Size:     {} bytes",
                stats.active_memtable_size
            );
            println!(
                "Immutable Memtables:      {}",
                stats.immutable_memtables_count
            );
            let total_sst: usize = stats.levels_file_count.iter().sum();
            println!("Total SSTable Files:      {}", total_sst);
            println!("Levels Distribution:");
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                println!("  Level {}: {} SSTable(s)", lvl, count);
            }
        }
    }
}

/// Format and display retrieved RAG search hits.
pub fn format_rag_results(hits: &[ScoredDocument], format: OutputFormat, show_explanation: bool) {
    match format {
        OutputFormat::Json => {
            let json_arr: Vec<serde_json::Value> = hits
                .iter()
                .enumerate()
                .map(|(rank, doc)| {
                    let mut obj = json!({
                        "rank": rank + 1,
                        "doc_id": doc.id.as_str(),
                        "score": doc.score,
                        "dense_score": doc.dense_score,
                        "sparse_score": doc.sparse_score,
                        "text": doc.text,
                        "metadata": doc.metadata,
                    });
                    if show_explanation {
                        if let Some(ref exp) = doc.explanation {
                            obj["explanation"] = json!({
                                "token_coverage": exp.token_coverage,
                                "phrase_match": exp.phrase_match,
                                "proximity": exp.proximity,
                                "combined_score": exp.combined_score,
                            });
                        }
                    }
                    obj
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json_arr).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("rank\tdoc_id\tscore\tdense_score\tsparse_score\ttext");
            for (i, d) in hits.iter().enumerate() {
                let text_preview = d
                    .text
                    .as_deref()
                    .unwrap_or("")
                    .replace(['\n', '\r', '\t'], " ");
                let text_preview = if text_preview.len() > 60 {
                    format!("{}...", safe_truncate_str(&text_preview, 60))
                } else {
                    text_preview
                };
                println!(
                    "{}\t{}\t{:.4}\t{}\t{}\t{}",
                    i + 1,
                    d.id.as_str(),
                    d.score,
                    d.dense_score
                        .map(|s| format!("{:.4}", s))
                        .unwrap_or_else(|| "-".into()),
                    d.sparse_score
                        .map(|s| format!("{:.4}", s))
                        .unwrap_or_else(|| "-".into()),
                    text_preview
                );
            }
        }
        OutputFormat::Text => {
            if hits.is_empty() {
                println!("(no documents found)");
                return;
            }
            println!("Found {} relevant document(s):", hits.len());
            for (i, d) in hits.iter().enumerate() {
                let scores_detail = match (d.dense_score, d.sparse_score) {
                    (Some(dense), Some(sparse)) => {
                        format!(" (dense: {:.4}, sparse: {:.4})", dense, sparse)
                    }
                    (Some(dense), None) => format!(" (dense: {:.4})", dense),
                    (None, Some(sparse)) => format!(" (sparse: {:.4})", sparse),
                    (None, None) => String::new(),
                };
                println!(
                    "\n{}) [{}] score: {:.4}{}",
                    i + 1,
                    d.id.as_str(),
                    d.score,
                    scores_detail
                );
                if let Some(ref text) = d.text {
                    let snippet = if text.len() > 200 {
                        format!("{}...", safe_truncate_str(text, 200))
                    } else {
                        text.clone()
                    };
                    println!("   Content: {}", snippet);
                }
                if let Some(ref meta) = d.metadata {
                    if !meta.is_empty() {
                        let meta_str: Vec<String> =
                            meta.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                        println!("   Metadata: {{{}}}", meta_str.join(", "));
                    }
                }
                if show_explanation {
                    if let Some(ref exp) = d.explanation {
                        println!(
                            "   Explanation: coverage={:.2}, phrase={:.2}, proximity={:.2}, combined={:.4}",
                            exp.token_coverage,
                            exp.phrase_match,
                            exp.proximity,
                            exp.combined_score
                        );
                    }
                }
            }
        }
    }
}

/// Format and display end-to-end pipeline query result with prompt and citations.
pub fn format_pipeline_query_result(res: &PipelineQueryResult, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(res).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!("query\tdocuments_retrieved\tcontext_chars\twas_truncated");
            println!(
                "{}\t{}\t{}\t{}",
                res.query,
                res.documents.len(),
                res.context.total_chars,
                res.context.was_truncated
            );
        }
        OutputFormat::Text => {
            println!("=== Pipeline Query: \"{}\" ===", res.query);
            println!(
                "Retrieved: {} document(s), Context Size: {} chars (truncated: {})",
                res.documents.len(),
                res.context.total_chars,
                res.context.was_truncated
            );
            if !res.context.citations.is_empty() {
                println!("Citations:");
                for c in &res.context.citations {
                    let title = c
                        .source
                        .as_deref()
                        .or_else(|| c.metadata.get_string("title"))
                        .unwrap_or("Untitled");
                    println!("  - [{}] {} ({})", c.index, c.doc_id, title);
                }
            }
            println!("\n=== Generated Prompt ===");
            println!("{}", res.prompt);
        }
    }
}

/// Format and display RAG engine statistics and telemetry.
pub fn format_rag_stats(engine: &RagEngine, format: OutputFormat) {
    let cache = engine.semantic_cache_stats();
    let metrics = engine.metrics();
    match format {
        OutputFormat::Json => {
            let json_stats = json!({
                "documents_count": engine.doc_count(),
                "dense_vectors_count": engine.dense_vector_count(),
                "sparse_terms_count": engine.sparse_term_count(),
                "semantic_cache": {
                    "capacity": cache.capacity,
                    "len": cache.len,
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "hit_rate": cache.hit_rate,
                },
                "telemetry": {
                    "total_searches": metrics.total_searches,
                    "cache_hits": metrics.cache_hits,
                    "cache_misses": metrics.cache_misses,
                    "search_latency_micros": {
                        "avg": metrics.search_latency.avg_micros,
                        "min": metrics.search_latency.min_micros,
                        "max": metrics.search_latency.max_micros,
                    }
                }
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json_stats).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!("metric\tvalue");
            println!("documents_count\t{}", engine.doc_count());
            println!("dense_vectors_count\t{}", engine.dense_vector_count());
            println!("sparse_terms_count\t{}", engine.sparse_term_count());
            println!("cache_capacity\t{}", cache.capacity);
            println!("cache_entries\t{}", cache.len);
            println!("cache_hits\t{}", cache.hits);
            println!("cache_misses\t{}", cache.misses);
            println!("cache_hit_rate\t{:.4}", cache.hit_rate);
        }
        OutputFormat::Text => {
            println!("=== FlashStore RAG Telemetry & Statistics ===");
            println!("Indexed Documents:      {}", engine.doc_count());
            println!("Dense Vectors:          {}", engine.dense_vector_count());
            println!("Sparse Terms:           {}", engine.sparse_term_count());
            println!("Semantic Query Cache:");
            println!("  Capacity:             {}", cache.capacity);
            println!("  Entries:              {}", cache.len);
            println!("  Hits:                 {}", cache.hits);
            println!("  Misses:               {}", cache.misses);
            println!("  Hit Rate:             {:.2}%", cache.hit_rate * 100.0);
            println!("Search Telemetry:");
            println!("  Total Searches:       {}", metrics.total_searches);
            println!(
                "  Latency (avg / max):  {:.1} µs / {} µs",
                metrics.search_latency.avg_micros, metrics.search_latency.max_micros
            );
        }
    }
}

/// Format and display Information Retrieval benchmark evaluation summary.
pub fn format_evaluation_summary(summary: &EvaluationSummary, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(summary).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!("metric\tvalue");
            println!("sample_count\t{}", summary.sample_count);
            println!("cutoff_k\t{}", summary.k);
            println!("mrr\t{:.4}", summary.mrr);
            println!("ndcg_at_k\t{:.4}", summary.ndcg_at_k);
            println!("precision_at_k\t{:.4}", summary.precision_at_k);
            println!("recall_at_k\t{:.4}", summary.recall_at_k);
            println!("hit_rate_at_k\t{:.4}", summary.hit_rate_at_k);
        }
        OutputFormat::Text => {
            println!("=== RAG IR Retrieval Evaluation (K = {}) ===", summary.k);
            println!("Evaluated Queries:      {}", summary.sample_count);
            println!("Mean Reciprocal Rank:   {:.4}", summary.mrr);
            println!(
                "NDCG@{}:                 {:.4}",
                summary.k, summary.ndcg_at_k
            );
            println!(
                "Precision@{}:            {:.4}",
                summary.k, summary.precision_at_k
            );
            println!(
                "Recall@{}:               {:.4}",
                summary.k, summary.recall_at_k
            );
            println!(
                "Hit Rate@{}:             {:.2}%",
                summary.k,
                summary.hit_rate_at_k * 100.0
            );
        }
    }
}

/// Format and display a single indexed document.
pub fn format_rag_doc(doc: &Document, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(doc).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!("doc_id\ttext_len\tchunks_count\thas_embedding");
            println!(
                "{}\t{}\t{}\t{}",
                doc.id.as_str(),
                doc.text.len(),
                doc.chunks.len(),
                doc.embedding.is_some()
            );
        }
        OutputFormat::Text => {
            println!("Document: [{}]", doc.id.as_str());
            println!("Length:   {} chars", doc.text.len());
            println!("Chunks:   {}", doc.chunks.len());
            println!(
                "Vector:   {}",
                if doc.embedding.is_some() {
                    "Indexed"
                } else {
                    "None"
                }
            );
            if !doc.metadata.is_empty() {
                println!("Metadata:");
                for (k, v) in doc.metadata.iter() {
                    println!("  {}: {}", k, v);
                }
            }
            println!("Content:\n{}", doc.text);
        }
    }
}

/// Format and display a list of documents.
pub fn format_rag_doc_list(docs: &[Document], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let list: Vec<serde_json::Value> = docs
                .iter()
                .map(|d| {
                    json!({
                        "id": d.id.as_str(),
                        "text_length": d.text.len(),
                        "chunks_count": d.chunks.len(),
                        "has_embedding": d.embedding.is_some(),
                        "metadata": d.metadata,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&list).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("id\tlength\tchunks\thas_embedding\ttitle");
            for d in docs {
                let title = d.metadata.get_string("title").unwrap_or("-");
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    d.id.as_str(),
                    d.text.len(),
                    d.chunks.len(),
                    d.embedding.is_some(),
                    title
                );
            }
        }
        OutputFormat::Text => {
            if docs.is_empty() {
                println!("(no documents indexed)");
                return;
            }
            println!("=== Indexed Documents (Total: {}) ===", docs.len());
            for (i, d) in docs.iter().enumerate() {
                let title = d
                    .metadata
                    .get_string("title")
                    .map(|t| format!(" \"{}\"", t))
                    .unwrap_or_default();
                println!(
                    "{}) [{}] ({} chars, {} chunks){}",
                    i + 1,
                    d.id.as_str(),
                    d.text.len(),
                    d.chunks.len(),
                    title
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_format_stats_structure() {
        let stats = Stats {
            active_memtable_size: 1024,
            immutable_memtables_count: 2,
            levels_file_count: vec![3, 1, 0, 0],
        };
        format_stats(&stats, OutputFormat::Json);
        format_stats(&stats, OutputFormat::Tsv);
        format_stats(&stats, OutputFormat::Text);
    }

    #[test]
    fn test_format_kv_pairs_output() {
        let pairs = vec![
            (Bytes::from_static(b"k1"), Bytes::from_static(b"v1")),
            (Bytes::from_static(b"k2"), Bytes::from_static(b"v2")),
        ];
        format_kv_pairs(&pairs, OutputFormat::Json, false);
        format_kv_pairs(&pairs, OutputFormat::Tsv, false);
        format_kv_pairs(&pairs, OutputFormat::Text, false);
    }

    #[test]
    fn test_format_rag_results_and_documents() {
        use crate::rag::{DocumentId, DocumentMetadata, RagConfig};

        let mut meta = DocumentMetadata::new();
        meta.insert("title", "Test Title");
        meta.insert("author", "Zein");

        let doc = Document {
            id: DocumentId::new("doc_100"),
            text: "This is test document text for CLI formatting tests.".into(),
            embedding: None,
            metadata: meta.clone(),
            chunks: Vec::new(),
            created_at: 0,
            updated_at: 0,
        };

        format_rag_doc(&doc, OutputFormat::Json);
        format_rag_doc(&doc, OutputFormat::Tsv);
        format_rag_doc(&doc, OutputFormat::Text);

        format_rag_doc_list(std::slice::from_ref(&doc), OutputFormat::Json);
        format_rag_doc_list(std::slice::from_ref(&doc), OutputFormat::Tsv);
        format_rag_doc_list(std::slice::from_ref(&doc), OutputFormat::Text);
        format_rag_doc_list(&[], OutputFormat::Text);

        let hit = ScoredDocument {
            id: DocumentId::new("doc_100"),
            score: 0.95,
            dense_score: Some(0.92),
            sparse_score: Some(0.88),
            text: Some("This is test document text.".into()),
            metadata: Some(meta),
            explanation: None,
        };

        format_rag_results(std::slice::from_ref(&hit), OutputFormat::Json, true);
        format_rag_results(std::slice::from_ref(&hit), OutputFormat::Tsv, true);
        format_rag_results(std::slice::from_ref(&hit), OutputFormat::Text, true);
        format_rag_results(&[], OutputFormat::Text, false);

        let engine = RagEngine::new(RagConfig::default());
        format_rag_stats(&engine, OutputFormat::Json);
        format_rag_stats(&engine, OutputFormat::Tsv);
        format_rag_stats(&engine, OutputFormat::Text);

        let summary = EvaluationSummary {
            sample_count: 5,
            k: 5,
            mrr: 0.85,
            precision_at_k: 0.8,
            recall_at_k: 0.9,
            ndcg_at_k: 0.87,
            hit_rate_at_k: 1.0,
        };
        format_evaluation_summary(&summary, OutputFormat::Json);
        format_evaluation_summary(&summary, OutputFormat::Tsv);
        format_evaluation_summary(&summary, OutputFormat::Text);
    }
}
