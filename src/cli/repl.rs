use crate::cli::GlobalOpts;
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use crate::rag::{
    MockEmbeddingProvider, QueryEvaluationSample, RagConfig, RagEngine, RagPipeline,
    RetrievalEvaluator,
};
use bytes::Bytes;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

/// Tokenize an input line respecting single and double quotes and escaped spaces.
pub fn tokenize_line(line: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escaped = false;

    for ch in line.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }

        match ch {
            '\\' => {
                escaped = true;
            }
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
            }
            ' ' | '\t' | '\r' | '\n' if !in_single_quote && !in_double_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if in_single_quote || in_double_quote {
        return Err(FlashStoreError::InvalidArgument(
            "Unclosed quote in input line".into(),
        ));
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

/// Execute a single REPL command against a FlashStore instance and format the response.
pub fn execute_repl_command(db: &FlashStore, tokens: &[String]) -> Result<Option<String>> {
    if tokens.is_empty() {
        return Ok(None);
    }

    let cmd = tokens[0].to_uppercase();
    let args = &tokens[1..];
    let start_time = Instant::now();

    match cmd.as_str() {
        "PUT" | "SET" => {
            if args.len() < 2 {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: PUT <key> <value>".into(),
                ));
            }
            let key = &args[0];
            let value = &args[1];
            db.put(Bytes::from(key.clone()), Bytes::from(value.clone()))?;
            let elapsed = start_time.elapsed();
            Ok(Some(format!("OK ({:?})", elapsed)))
        }
        "GET" => {
            if args.is_empty() {
                return Err(FlashStoreError::InvalidArgument("Usage: GET <key>".into()));
            }
            let key = &args[0];
            let res = db.get(Bytes::from(key.clone()))?;
            let elapsed = start_time.elapsed();
            match res {
                Some(val) => {
                    let s = String::from_utf8_lossy(&val);
                    Ok(Some(format!("\"{}\" ({:?})", s, elapsed)))
                }
                None => Ok(Some(format!("(nil) ({:?})", elapsed))),
            }
        }
        "DEL" | "DELETE" => {
            if args.is_empty() {
                return Err(FlashStoreError::InvalidArgument("Usage: DEL <key>".into()));
            }
            let key = &args[0];
            db.delete(Bytes::from(key.clone()))?;
            let elapsed = start_time.elapsed();
            Ok(Some(format!("OK ({:?})", elapsed)))
        }
        "SCAN" => {
            let start = args.first().cloned().map(Bytes::from);
            let end = args.get(1).cloned().map(Bytes::from);
            let limit = args.get(2).and_then(|s| s.parse::<usize>().ok());

            let mut results = db.scan(start, end)?;
            if let Some(lim) = limit {
                results.truncate(lim);
            }
            let elapsed = start_time.elapsed();

            let mut out = String::new();
            if results.is_empty() {
                out.push_str("(empty)\n");
            } else {
                for (idx, (k, v)) in results.iter().enumerate() {
                    out.push_str(&format!(
                        "{}) {}: {}\n",
                        idx + 1,
                        String::from_utf8_lossy(k),
                        String::from_utf8_lossy(v)
                    ));
                }
            }
            out.push_str(&format!("({} entries, took {:?})", results.len(), elapsed));
            Ok(Some(out))
        }
        "FLUSH" => {
            db.flush()?;
            let elapsed = start_time.elapsed();
            Ok(Some(format!("Flushed memtable to SSTable ({:?})", elapsed)))
        }
        "COMPACT" => {
            db.compact()?;
            let elapsed = start_time.elapsed();
            Ok(Some(format!("Compaction completed ({:?})", elapsed)))
        }
        "STATS" => {
            let stats = db.stats();
            let mut out = String::new();
            out.push_str(&format!(
                "Active memtable: {} bytes\n",
                stats.active_memtable_size
            ));
            out.push_str(&format!(
                "Immutable memtables: {}\n",
                stats.immutable_memtables_count
            ));
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                out.push_str(&format!("Level {}: {} SSTables\n", lvl, count));
            }
            Ok(Some(out.trim_end().to_string()))
        }
        "RAG" => execute_rag_repl(db, args, start_time),
        s if s.starts_with("RAG-") || s.starts_with("RAG_") => {
            let sub = &s[4..];
            let mut combined = vec![sub.to_string()];
            combined.extend_from_slice(args);
            execute_rag_repl(db, &combined, start_time)
        }
        "HELP" | "?" => {
            let help = r#"Available REPL commands:
  PUT <key> <value>            Insert or update key-value pair
  GET <key>                    Retrieve value by key
  DEL <key>                    Delete key
  SCAN [start] [end] [limit]   Range scan entries
  FLUSH                        Flush active memtable to L0 SSTable
  COMPACT                      Trigger manual background compaction
  STATS                        Display engine statistics
  RAG INGEST <id> <text>       Ingest document into RAG engine
  RAG GET <id>                 Retrieve document content and metadata
  RAG DEL <id>                 Delete document from RAG engine
  RAG QUERY <query> [top_k]    Hybrid retrieval + reranker search
  RAG PROMPT <query> [top_k]   Assemble LLM prompt with citations
  RAG STATS                    Display RAG & semantic cache stats
  RAG LIST [limit]             List indexed documents
  RAG CACHE CLEAR              Clear semantic query cache
  RAG EVAL                     Run IR evaluation metrics benchmark
  CLEAR                        Clear terminal screen
  HELP / ?                     Display this help text
  EXIT / QUIT / Q              Exit interactive shell"#;
            Ok(Some(help.to_string()))
        }
        "CLEAR" => {
            // ANSI clear screen
            print!("\x1B[2J\x1B[1;1H");
            let _ = io::stdout().flush();
            Ok(None)
        }
        "EXIT" | "QUIT" | "Q" => Ok(Some("BYE".to_string())),
        other => Err(FlashStoreError::InvalidArgument(format!(
            "Unknown command '{}'. Type 'HELP' for commands.",
            other
        ))),
    }
}

/// Execute RAG subsystem commands within the REPL.
pub fn execute_rag_repl(
    db: &FlashStore,
    args: &[String],
    start_time: Instant,
) -> Result<Option<String>> {
    if args.is_empty() {
        return Err(FlashStoreError::InvalidArgument(
            "Usage: RAG <INGEST|GET|DEL|QUERY|PROMPT|STATS|LIST|CLEAR-CACHE|EVAL> ...".into(),
        ));
    }

    let subcmd = args[0].to_uppercase();
    let sub_args = &args[1..];

    let arc_db = Arc::new(db.clone());
    let config = RagConfig::default();
    let engine = RagEngine::recover(config.clone(), arc_db)?;
    let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim));
    let mut pipeline = RagPipeline::new(engine).with_embedder(embedder);

    match subcmd.as_str() {
        "INGEST" => {
            if sub_args.len() < 2 {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG INGEST <id> <text>".into(),
                ));
            }
            let id = &sub_args[0];
            let text = sub_args[1..].join(" ");
            pipeline.ingest_text(id.clone(), &text, None)?;
            let elapsed = start_time.elapsed();
            Ok(Some(format!(
                "OK (Ingested document '{}' in {:?})",
                id, elapsed
            )))
        }
        "GET" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG GET <id>".into(),
                ));
            }
            let id = &sub_args[0];
            match pipeline.get_document(id) {
                Some(doc) => {
                    let mut out = format!(
                        "Document: [{}]\nLength: {} chars\nChunks: {}\n",
                        doc.id.as_str(),
                        doc.text.len(),
                        doc.chunks.len()
                    );
                    if !doc.metadata.is_empty() {
                        out.push_str("Metadata:\n");
                        for (k, v) in doc.metadata.iter() {
                            out.push_str(&format!("  {}: {}\n", k, v));
                        }
                    }
                    out.push_str(&format!(
                        "Content:\n{}\n({:?})",
                        doc.text,
                        start_time.elapsed()
                    ));
                    Ok(Some(out))
                }
                None => Ok(Some(format!("(nil) ({:?})", start_time.elapsed()))),
            }
        }
        "DEL" | "DELETE" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG DEL <id>".into(),
                ));
            }
            let id = &sub_args[0];
            let deleted = pipeline.delete_document(id)?;
            let elapsed = start_time.elapsed();
            if deleted {
                Ok(Some(format!(
                    "OK (Deleted document '{}' in {:?})",
                    id, elapsed
                )))
            } else {
                Ok(Some(format!("(not found: '{}' in {:?})", id, elapsed)))
            }
        }
        "QUERY" | "SEARCH" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG QUERY <query> [top_k]".into(),
                ));
            }
            let (query_text, top_k) = if sub_args.len() > 1 {
                if let Ok(k) = sub_args[sub_args.len() - 1].parse::<usize>() {
                    (sub_args[..sub_args.len() - 1].join(" "), k)
                } else {
                    (sub_args.join(" "), 5)
                }
            } else {
                (sub_args[0].clone(), 5)
            };

            let res = pipeline.query(&query_text, top_k)?;
            let elapsed = start_time.elapsed();

            let mut out = String::new();
            if res.documents.is_empty() {
                out.push_str("(no documents found)\n");
            } else {
                for (i, d) in res.documents.iter().enumerate() {
                    let snippet = d.text.as_deref().unwrap_or("");
                    let snippet = if snippet.len() > 120 {
                        format!("{}...", &snippet[..120])
                    } else {
                        snippet.to_string()
                    };
                    out.push_str(&format!(
                        "{}) [{}] score: {:.4}\n   {}\n",
                        i + 1,
                        d.id.as_str(),
                        d.score,
                        snippet
                    ));
                }
            }
            out.push_str(&format!(
                "({} hits, took {:?})",
                res.documents.len(),
                elapsed
            ));
            Ok(Some(out))
        }
        "PROMPT" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG PROMPT <query> [top_k]".into(),
                ));
            }
            let (query_text, top_k) = if sub_args.len() > 1 {
                if let Ok(k) = sub_args[sub_args.len() - 1].parse::<usize>() {
                    (sub_args[..sub_args.len() - 1].join(" "), k)
                } else {
                    (sub_args.join(" "), 5)
                }
            } else {
                (sub_args[0].clone(), 5)
            };

            let res = pipeline.query(&query_text, top_k)?;
            let elapsed = start_time.elapsed();

            let out = format!(
                "=== Generated RAG Prompt ({:?}) ===\n{}\n",
                elapsed, res.prompt
            );
            Ok(Some(out))
        }
        "STATS" => {
            let engine = pipeline.engine();
            let cache = engine.semantic_cache_stats();
            let metrics = engine.metrics();
            let mut out = String::new();
            out.push_str(&format!("Documents:     {}\n", engine.doc_count()));
            out.push_str(&format!("Dense Vectors: {}\n", engine.dense_vector_count()));
            out.push_str(&format!("Sparse Terms:  {}\n", engine.sparse_term_count()));
            out.push_str(&format!(
                "Cache:         {}/{} entries, {} hits, {} misses ({:.2}% hit rate)\n",
                cache.len,
                cache.capacity,
                cache.hits,
                cache.misses,
                cache.hit_rate * 100.0
            ));
            out.push_str(&format!(
                "Searches:      {} (avg latency: {:.1} µs)",
                metrics.total_searches, metrics.search_latency.avg_micros
            ));
            Ok(Some(out))
        }
        "LIST" => {
            let limit = sub_args
                .first()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(20);
            let mut docs = pipeline.list_documents();
            if docs.len() > limit {
                docs.truncate(limit);
            }
            let mut out = String::new();
            if docs.is_empty() {
                out.push_str("(no documents indexed)\n");
            } else {
                for (i, d) in docs.iter().enumerate() {
                    let title = d
                        .metadata
                        .get_string("title")
                        .map(|t| format!(" \"{}\"", t))
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "{}) [{}] ({} chars, {} chunks){}\n",
                        i + 1,
                        d.id.as_str(),
                        d.text.len(),
                        d.chunks.len(),
                        title
                    ));
                }
            }
            out.push_str(&format!(
                "({} documents listed, took {:?})",
                docs.len(),
                start_time.elapsed()
            ));
            Ok(Some(out))
        }
        "CLEAR-CACHE" => {
            pipeline.clear_cache();
            Ok(Some(format!(
                "Semantic query cache cleared ({:?})",
                start_time.elapsed()
            )))
        }
        "CACHE" => {
            if sub_args
                .first()
                .map(|s| s.to_uppercase())
                .as_deref()
                == Some("CLEAR")
            {
                pipeline.clear_cache();
                Ok(Some(format!(
                    "Semantic query cache cleared ({:?})",
                    start_time.elapsed()
                )))
            } else {
                Err(FlashStoreError::InvalidArgument(
                    "Usage: RAG CACHE CLEAR".into(),
                ))
            }
        }
        "EVAL" => {
            let docs = pipeline.list_documents();
            if docs.is_empty() {
                return Ok(Some(
                    "No documents indexed. Ingest documents before running eval.".into(),
                ));
            }
            let mut eval_samples = Vec::new();
            for d in docs.iter().take(10) {
                let words: Vec<&str> = d.text.split_whitespace().collect();
                let query = if words.len() >= 3 {
                    words[..3].join(" ")
                } else {
                    d.text.clone()
                };
                let search_res = pipeline.engine_mut().search(&query, None, 5)?;
                let retrieved_ids = search_res.into_iter().map(|r| r.doc_id).collect();
                eval_samples.push(QueryEvaluationSample {
                    query,
                    ground_truth_ids: vec![d.id.to_string()],
                    retrieved_ids,
                });
            }
            let summary = RetrievalEvaluator::evaluate(&eval_samples, 5);
            let out = format!(
                "=== RAG IR Evaluation (K = 5) ===\nSamples:   {}\nMRR:       {:.4}\nNDCG@5:    {:.4}\nPrecision: {:.4}\nRecall:    {:.4}\nHit Rate:  {:.2}%\n({:?})",
                summary.sample_count,
                summary.mrr,
                summary.ndcg_at_k,
                summary.precision_at_k,
                summary.recall_at_k,
                summary.hit_rate_at_k * 100.0,
                start_time.elapsed()
            );
            Ok(Some(out))
        }
        other => Err(FlashStoreError::InvalidArgument(format!(
            "Unknown RAG command '{}'. Available: INGEST, GET, DEL, QUERY, PROMPT, STATS, LIST, CACHE CLEAR, EVAL",
            other
        ))),
    }
}

/// Run interactive REPL loop using stdin and stdout.
pub fn run_repl(db: FlashStore, opts: &GlobalOpts) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let db_name = opts.path.display().to_string();

    println!("===========================================================");
    println!("  FlashStore Interactive LSM-Tree Storage Engine REPL");
    println!("  Database: {}", db_name);
    println!("  Type 'HELP' for a list of commands, 'EXIT' to quit.");
    println!("===========================================================");

    let mut line_buffer = String::new();

    loop {
        print!("flash-store [{}]> ", db_name);
        io::stdout().flush().map_err(FlashStoreError::Io)?;

        line_buffer.clear();
        let bytes_read = reader
            .read_line(&mut line_buffer)
            .map_err(FlashStoreError::Io)?;
        if bytes_read == 0 {
            // EOF reached
            println!("\nGoodbye.");
            break;
        }

        let trimmed = line_buffer.trim();
        if trimmed.is_empty() {
            continue;
        }

        match tokenize_line(trimmed) {
            Ok(tokens) => match execute_repl_command(&db, &tokens) {
                Ok(Some(output)) => {
                    if output == "BYE" {
                        println!("Goodbye.");
                        break;
                    }
                    println!("{}", output);
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Parse error: {}", e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_tokenize_line() -> Result<()> {
        let tokens = tokenize_line("PUT \"my key\" 'my value' 123")?;
        assert_eq!(tokens, vec!["PUT", "my key", "my value", "123"]);

        let tokens2 = tokenize_line("SCAN a b 10")?;
        assert_eq!(tokens2, vec!["SCAN", "a", "b", "10"]);

        let unclosed = tokenize_line("PUT \"unclosed key val");
        assert!(unclosed.is_err());

        Ok(())
    }

    #[test]
    fn test_repl_commands_execution() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let db = FlashStore::open(options)?;

        // PUT
        let res = execute_repl_command(&db, &["PUT".into(), "k1".into(), "v1".into()])?;
        assert!(res.unwrap().contains("OK"));

        // GET
        let res = execute_repl_command(&db, &["GET".into(), "k1".into()])?;
        assert!(res.unwrap().contains("v1"));

        // GET non-existent
        let res = execute_repl_command(&db, &["GET".into(), "k_missing".into()])?;
        assert!(res.unwrap().contains("(nil)"));

        // SCAN
        let res = execute_repl_command(&db, &["SCAN".into()])?;
        assert!(res.unwrap().contains("k1: v1"));

        // DEL
        let res = execute_repl_command(&db, &["DEL".into(), "k1".into()])?;
        assert!(res.unwrap().contains("OK"));

        // GET after DEL
        let res = execute_repl_command(&db, &["GET".into(), "k1".into()])?;
        assert!(res.unwrap().contains("(nil)"));

        // FLUSH & STATS & COMPACT & HELP & EXIT
        assert!(execute_repl_command(&db, &["FLUSH".into()])?.is_some());
        assert!(execute_repl_command(&db, &["STATS".into()])?.is_some());
        assert!(execute_repl_command(&db, &["COMPACT".into()])?.is_some());
        assert!(execute_repl_command(&db, &["HELP".into()])?.is_some());
        assert_eq!(
            execute_repl_command(&db, &["EXIT".into()])?,
            Some("BYE".into())
        );

        // RAG commands
        let ingest_res = execute_repl_command(
            &db,
            &[
                "RAG".into(),
                "INGEST".into(),
                "repl_doc1".into(),
                "FlashStore supports interactive hybrid search in REPL".into(),
            ],
        )?;
        assert!(ingest_res.unwrap().contains("OK"));

        let get_res = execute_repl_command(
            &db,
            &["RAG".into(), "GET".into(), "repl_doc1".into()],
        )?;
        assert!(get_res.unwrap().contains("repl_doc1"));

        let query_res = execute_repl_command(
            &db,
            &[
                "RAG".into(),
                "QUERY".into(),
                "interactive search".into(),
                "2".into(),
            ],
        )?;
        assert!(query_res.unwrap().contains("repl_doc1"));

        let prompt_res = execute_repl_command(
            &db,
            &["RAG".into(), "PROMPT".into(), "hybrid search".into()],
        )?;
        assert!(prompt_res.unwrap().contains("Prompt"));

        let stats_res = execute_repl_command(&db, &["RAG".into(), "STATS".into()])?;
        assert!(stats_res.unwrap().contains("Documents:"));

        let list_res = execute_repl_command(&db, &["RAG".into(), "LIST".into()])?;
        assert!(list_res.unwrap().contains("repl_doc1"));

        let cache_res = execute_repl_command(
            &db,
            &["RAG".into(), "CACHE".into(), "CLEAR".into()],
        )?;
        assert!(cache_res.unwrap().contains("cleared"));

        let eval_res = execute_repl_command(&db, &["RAG".into(), "EVAL".into()])?;
        assert!(eval_res.unwrap().contains("Evaluation"));

        let del_res = execute_repl_command(
            &db,
            &["RAG".into(), "DEL".into(), "repl_doc1".into()],
        )?;
        assert!(del_res.unwrap().contains("OK"));

        Ok(())
    }
}
