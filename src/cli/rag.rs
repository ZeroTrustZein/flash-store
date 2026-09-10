use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Subcommand;
use serde::Deserialize;

use crate::cli::formatter;
use crate::cli::{open_db, GlobalOpts, OutputFormat};
use crate::error::{FlashStoreError, Result};
use crate::rag::{
    ChunkingConfig, ChunkingStrategy, DocumentMetadata, FilterCondition, FusionStrategy,
    MetadataFilter, MetadataValue, MockEmbeddingProvider, QueryEvaluationSample, RagConfig,
    RagEngine, RagPipeline, RagQuery, RetrievalEvaluator,
};

/// All RAG retrieval, vector search, and reranker subcommands.
#[derive(Debug, Subcommand)]
pub enum RagCmd {
    /// Ingest a document or batch into the RAG engine
    Ingest {
        /// Document identifier (generated if omitted)
        #[arg(short, long)]
        id: Option<String>,

        /// Text content to ingest
        #[arg(short, long)]
        text: Option<String>,

        /// Path to a text file to ingest
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Path to a batch JSON file containing an array of documents
        #[arg(long)]
        batch_file: Option<PathBuf>,

        /// Document title metadata
        #[arg(long)]
        title: Option<String>,

        /// Document author metadata
        #[arg(long)]
        author: Option<String>,

        /// Comma-separated list of tags
        #[arg(long)]
        tags: Option<String>,

        /// Additional metadata in key=value format (repeatable: --meta source=wiki)
        #[arg(long = "meta")]
        metadata: Vec<String>,

        /// Chunk size in characters or tokens
        #[arg(long)]
        chunk_size: Option<usize>,

        /// Chunk overlap
        #[arg(long)]
        chunk_overlap: Option<usize>,

        /// Chunking strategy: paragraphs, sentences, fixed-tokens, fixed-chars
        #[arg(long, default_value = "paragraphs")]
        chunk_strategy: String,
    },

    /// Query using hybrid dense-sparse retrieval and cross-encoder reranking
    Query {
        /// Query text (positional)
        #[arg(value_name = "QUERY")]
        query: Option<String>,

        /// Query text via flag
        #[arg(short, long = "query")]
        query_flag: Option<String>,

        /// Number of top documents to retrieve
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,

        /// Weight for dense vector search (0.0 to 1.0, default 0.5)
        #[arg(long)]
        dense_weight: Option<f32>,

        /// Weight for sparse lexical search (0.0 to 1.0, default 0.5)
        #[arg(long)]
        sparse_weight: Option<f32>,

        /// Fusion strategy: linear, rrf, borda
        #[arg(long, default_value = "linear")]
        fusion: String,

        /// Disable cross-encoder reranking
        #[arg(long)]
        no_rerank: bool,

        /// Enable Maximal Marginal Relevance (MMR) diversity reranking
        #[arg(long)]
        mmr: bool,

        /// MMR diversity lambda parameter (0.0 to 1.0, default 0.7)
        #[arg(long, default_value_t = 0.7)]
        mmr_lambda: f32,

        /// Minimum relevance score threshold
        #[arg(long)]
        min_score: Option<f32>,

        /// Metadata equality filters in key=value format (repeatable)
        #[arg(long = "filter")]
        filters: Vec<String>,

        /// Output format: text, json, tsv, prompt
        #[arg(long, default_value = "text")]
        format: String,

        /// Emit ready LLM prompt with citations
        #[arg(long)]
        prompt: bool,

        /// Include reranker score explanations
        #[arg(long)]
        explain: bool,
    },

    /// Retrieve an indexed document by ID
    Get {
        /// Document identifier (positional)
        #[arg(value_name = "ID")]
        id: Option<String>,

        /// Document identifier via flag
        #[arg(short, long = "id")]
        doc_id: Option<String>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Delete an indexed document by ID
    Delete {
        /// Document identifier (positional)
        #[arg(value_name = "ID")]
        id: Option<String>,

        /// Document identifier via flag
        #[arg(short, long = "id")]
        doc_id: Option<String>,
    },

    /// List indexed documents
    List {
        /// Maximum number of documents to list
        #[arg(short, long, default_value_t = 20)]
        limit: usize,

        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Display RAG telemetry, cache metrics, and index statistics
    Stats {
        /// Output format: text, json, tsv
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Clear the semantic vector query cache
    ClearCache,

    /// Run Information Retrieval (IR) benchmark evaluation
    Eval {
        /// Path to JSON file containing evaluation samples
        #[arg(long)]
        samples_file: Option<PathBuf>,

        /// Evaluation rank cutoff K
        #[arg(short, long, default_value_t = 5)]
        k: usize,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },
}

/// Structure for parsing batch document files.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchDocumentInput {
    pub id: Option<String>,
    pub text: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Open a persistent `RagPipeline` configured with a deterministic embedder and FlashStore store.
pub fn open_rag_pipeline(opts: &GlobalOpts) -> Result<RagPipeline> {
    let db = Arc::new(open_db(opts)?);
    let config = RagConfig::default();
    let engine = RagEngine::recover(config.clone(), db)?;
    let embedder = Arc::new(MockEmbeddingProvider::new(config.embedding_dim));
    let pipeline = RagPipeline::new(engine).with_embedder(embedder);
    Ok(pipeline)
}

/// Parse format string into `OutputFormat`.
pub fn parse_output_format(s: &str, json_flag: bool) -> OutputFormat {
    if json_flag || s.eq_ignore_ascii_case("json") {
        OutputFormat::Json
    } else if s.eq_ignore_ascii_case("tsv") {
        OutputFormat::Tsv
    } else {
        OutputFormat::Text
    }
}

/// Parse metadata CLI flags into a `DocumentMetadata` structure.
pub fn parse_metadata_flags(
    title: Option<String>,
    author: Option<String>,
    tags: Option<String>,
    metadata_flags: &[String],
) -> DocumentMetadata {
    let mut meta = DocumentMetadata::new();

    if let Some(t) = title {
        meta.insert("title", t);
    }
    if let Some(a) = author {
        meta.insert("author", a);
    }
    if let Some(tag_str) = tags {
        let tag_list: Vec<MetadataValue> = tag_str
            .split(',')
            .map(|t| MetadataValue::String(t.trim().to_string()))
            .collect();
        meta.insert("tags", MetadataValue::List(tag_list));
    }

    for item in metadata_flags {
        if let Some(idx) = item.find('=') {
            let key = item[..idx].trim();
            let val = item[idx + 1..].trim();
            let parsed_val = if let Ok(i) = val.parse::<i64>() {
                MetadataValue::Int(i)
            } else if let Ok(f) = val.parse::<f64>() {
                MetadataValue::Float(f)
            } else if val.eq_ignore_ascii_case("true") {
                MetadataValue::Bool(true)
            } else if val.eq_ignore_ascii_case("false") {
                MetadataValue::Bool(false)
            } else {
                MetadataValue::String(val.to_string())
            };
            meta.insert(key, parsed_val);
        }
    }

    meta
}

/// Parse filter flags (repeatable `--filter key=val`) into a composite `MetadataFilter`.
pub fn parse_filter_flags(filters: &[String]) -> Option<MetadataFilter> {
    if filters.is_empty() {
        return None;
    }

    let mut list = Vec::new();
    for f in filters {
        if let Some(idx) = f.find('=') {
            let key = f[..idx].trim().to_string();
            let val = f[idx + 1..].trim();
            let meta_val = if let Ok(i) = val.parse::<i64>() {
                MetadataValue::Int(i)
            } else if let Ok(fl) = val.parse::<f64>() {
                MetadataValue::Float(fl)
            } else if val.eq_ignore_ascii_case("true") {
                MetadataValue::Bool(true)
            } else if val.eq_ignore_ascii_case("false") {
                MetadataValue::Bool(false)
            } else {
                MetadataValue::String(val.to_string())
            };
            list.push(MetadataFilter::condition(FilterCondition::Eq(
                key, meta_val,
            )));
        }
    }

    if list.is_empty() {
        None
    } else if list.len() == 1 {
        Some(list.remove(0))
    } else {
        Some(MetadataFilter::all(list))
    }
}

/// Construct a `ChunkingConfig` from CLI strategy and size parameters.
pub fn parse_chunk_config(
    strategy: &str,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
) -> ChunkingConfig {
    match strategy.to_lowercase().as_str() {
        "fixed-chars" | "chars" => ChunkingConfig {
            strategy: ChunkingStrategy::FixedChars {
                size: chunk_size.unwrap_or(500),
                overlap: chunk_overlap.unwrap_or(50),
            },
            min_chunk_size: 10,
        },
        "fixed-tokens" | "tokens" => ChunkingConfig {
            strategy: ChunkingStrategy::FixedTokens {
                size: chunk_size.unwrap_or(200),
                overlap: chunk_overlap.unwrap_or(20),
            },
            min_chunk_size: 10,
        },
        "sentences" | "sentence" => ChunkingConfig {
            strategy: ChunkingStrategy::Sentence,
            min_chunk_size: 10,
        },
        _ => ChunkingConfig {
            strategy: ChunkingStrategy::Paragraph,
            min_chunk_size: 10,
        },
    }
}

/// Execute a RAG subcommand with the given global options.
pub fn run_rag_command(cmd: RagCmd, opts: &GlobalOpts) -> Result<()> {
    match cmd {
        RagCmd::Ingest {
            id,
            text,
            file,
            batch_file,
            title,
            author,
            tags,
            metadata,
            chunk_size,
            chunk_overlap,
            chunk_strategy,
        } => {
            let mut pipeline = open_rag_pipeline(opts)?;
            let chunk_cfg = parse_chunk_config(&chunk_strategy, chunk_size, chunk_overlap);
            pipeline = pipeline.with_chunking(chunk_cfg);

            if let Some(batch_path) = batch_file {
                let content = fs::read_to_string(&batch_path).map_err(FlashStoreError::Io)?;
                let items: Vec<BatchDocumentInput> = serde_json::from_str(&content)
                    .or_else(|_| {
                        // Fallback: JSON Lines
                        let mut lines = Vec::new();
                        for line in content.lines() {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                let item: BatchDocumentInput = serde_json::from_str(trimmed)?;
                                lines.push(item);
                            }
                        }
                        Ok::<Vec<BatchDocumentInput>, serde_json::Error>(lines)
                    })
                    .map_err(|e| {
                        FlashStoreError::InvalidArgument(format!("Invalid batch JSON: {}", e))
                    })?;

                let mut to_ingest = Vec::with_capacity(items.len());
                for (idx, item) in items.into_iter().enumerate() {
                    let doc_id = item.id.unwrap_or_else(|| format!("batch_doc_{}", idx + 1));
                    let mut meta = DocumentMetadata::new();
                    if let Some(t) = item.title {
                        meta.insert("title", t);
                    }
                    if let Some(a) = item.author {
                        meta.insert("author", a);
                    }
                    if let Some(serde_json::Value::Object(map)) = item.metadata {
                        for (k, v) in map {
                            if let Some(s) = v.as_str() {
                                meta.insert(k, s.to_string());
                            } else if let Some(i) = v.as_i64() {
                                meta.insert(k, i);
                            } else if let Some(f) = v.as_f64() {
                                meta.insert(k, f);
                            } else if let Some(b) = v.as_bool() {
                                meta.insert(k, b);
                            }
                        }
                    }
                    to_ingest.push((doc_id, item.text, Some(meta)));
                }

                let report = pipeline.batch_ingest(to_ingest)?;
                if !opts.quiet {
                    println!(
                        "Ingested {} documents ({} chunks, {} characters) in {}ms",
                        report.docs_ingested,
                        report.chunks_created,
                        report.total_chars,
                        report.elapsed_ms
                    );
                }
            } else {
                let content = if let Some(t) = text {
                    t
                } else if let Some(f) = file {
                    fs::read_to_string(&f).map_err(FlashStoreError::Io)?
                } else {
                    return Err(FlashStoreError::InvalidArgument(
                        "Must provide --text, --file, or --batch-file".into(),
                    ));
                };

                let doc_id = id.unwrap_or_else(|| {
                    format!(
                        "doc_{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_micros()
                    )
                });

                let meta = parse_metadata_flags(title, author, tags, &metadata);
                pipeline.ingest_text(doc_id.clone(), &content, Some(meta))?;

                if !opts.quiet {
                    println!("OK: Ingested document '{}'", doc_id);
                }
            }
        }

        RagCmd::Query {
            query,
            query_flag,
            top_k,
            dense_weight,
            sparse_weight,
            fusion,
            no_rerank,
            mmr,
            mmr_lambda,
            min_score,
            filters,
            format,
            prompt,
            explain,
        } => {
            let mut pipeline = open_rag_pipeline(opts)?;
            let query_text = query.or(query_flag).ok_or_else(|| {
                FlashStoreError::InvalidArgument("Query text must be provided".into())
            })?;

            let fmt = parse_output_format(&format, opts.json);
            let emit_prompt = prompt || format.eq_ignore_ascii_case("prompt");

            if emit_prompt {
                let res = pipeline.query(&query_text, top_k)?;
                formatter::format_pipeline_query_result(&res, fmt);
            } else {
                let mut q_builder = RagQuery::builder(&query_text).top_k(top_k);

                // Embed query if embedder configured
                let query_emb = pipeline.engine().config().embedding_dim;
                let mock_embedder = MockEmbeddingProvider::new(query_emb);
                use crate::rag::EmbeddingProvider;
                if let Ok(vec) = mock_embedder.embed_query(&query_text) {
                    q_builder = q_builder.embedding(vec);
                }

                // Fusion strategy
                match fusion.to_lowercase().as_str() {
                    "rrf" => {
                        q_builder = q_builder.fusion_strategy(FusionStrategy::Rrf { k: 60 });
                    }
                    "dense" | "dense-only" => {
                        q_builder = q_builder.fusion_strategy(FusionStrategy::DenseOnly);
                    }
                    "sparse" | "sparse-only" => {
                        q_builder = q_builder.fusion_strategy(FusionStrategy::SparseOnly);
                    }
                    _ => {
                        let dw = dense_weight.unwrap_or_else(|| {
                            if let Some(sw) = sparse_weight {
                                (1.0 - sw).clamp(0.0, 1.0)
                            } else {
                                0.5
                            }
                        });
                        q_builder = q_builder
                            .fusion_strategy(FusionStrategy::WeightedLinear { dense_weight: dw });
                    }
                }

                if mmr {
                    q_builder = q_builder.mmr(mmr_lambda);
                } else {
                    q_builder = q_builder.rerank(!no_rerank, Some(top_k));
                }

                if let Some(min_s) = min_score {
                    q_builder = q_builder.min_score(min_s);
                }

                if let Some(filter) = parse_filter_flags(&filters) {
                    q_builder = q_builder.filter(filter);
                }

                let rag_query = q_builder.build();
                let results = pipeline.engine_mut().search_query(&rag_query)?;
                formatter::format_rag_results(&results, fmt, explain);
            }
        }

        RagCmd::Get { id, doc_id, format } => {
            let pipeline = open_rag_pipeline(opts)?;
            let target_id = id.or(doc_id).ok_or_else(|| {
                FlashStoreError::InvalidArgument("Document ID must be provided".into())
            })?;
            let fmt = parse_output_format(&format, opts.json);

            match pipeline.get_document(&target_id) {
                Some(doc) => formatter::format_rag_doc(&doc, fmt),
                None => {
                    if fmt == OutputFormat::Json {
                        println!("null");
                    } else {
                        println!("Document not found: {}", target_id);
                    }
                }
            }
        }

        RagCmd::Delete { id, doc_id } => {
            let mut pipeline = open_rag_pipeline(opts)?;
            let target_id = id.or(doc_id).ok_or_else(|| {
                FlashStoreError::InvalidArgument("Document ID must be provided".into())
            })?;

            let deleted = pipeline.delete_document(&target_id)?;
            if !opts.quiet {
                if deleted {
                    println!("OK: Deleted document '{}'", target_id);
                } else {
                    println!("Document not found: '{}'", target_id);
                }
            }
        }

        RagCmd::List { limit, format } => {
            let pipeline = open_rag_pipeline(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            let mut docs = pipeline.list_documents();
            if docs.len() > limit {
                docs.truncate(limit);
            }
            formatter::format_rag_doc_list(&docs, fmt);
        }

        RagCmd::Stats { format } => {
            let pipeline = open_rag_pipeline(opts)?;
            let fmt = parse_output_format(&format, opts.json);
            formatter::format_rag_stats(pipeline.engine(), fmt);
        }

        RagCmd::ClearCache => {
            let mut pipeline = open_rag_pipeline(opts)?;
            pipeline.clear_cache();
            if !opts.quiet {
                println!("Semantic query cache cleared.");
            }
        }

        RagCmd::Eval {
            samples_file,
            k,
            format,
        } => {
            let mut pipeline = open_rag_pipeline(opts)?;
            let fmt = parse_output_format(&format, opts.json);

            let samples = if let Some(path) = samples_file {
                let content = fs::read_to_string(&path).map_err(FlashStoreError::Io)?;
                serde_json::from_str::<Vec<QueryEvaluationSample>>(&content).map_err(|e| {
                    FlashStoreError::InvalidArgument(format!("Invalid eval JSON: {}", e))
                })?
            } else {
                // Built-in synthetic evaluation queries against indexed documents
                let docs = pipeline.list_documents();
                if docs.is_empty() {
                    if !opts.quiet {
                        println!("No documents indexed. Ingest documents before running eval.");
                    }
                    return Ok(());
                }

                let mut eval_samples = Vec::new();
                for d in docs.iter().take(10) {
                    let words: Vec<&str> = d.text.split_whitespace().collect();
                    let query = if words.len() >= 3 {
                        words[..3].join(" ")
                    } else {
                        d.text.clone()
                    };

                    let search_res = pipeline.engine_mut().search(&query, None, k)?;
                    let retrieved_ids = search_res.into_iter().map(|r| r.doc_id).collect();

                    eval_samples.push(QueryEvaluationSample {
                        query,
                        ground_truth_ids: vec![d.id.to_string()],
                        retrieved_ids,
                    });
                }
                eval_samples
            };

            let summary = RetrievalEvaluator::evaluate(&samples, k);
            formatter::format_evaluation_summary(&summary, fmt);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_metadata_and_filters() {
        let meta = parse_metadata_flags(
            Some("My Title".into()),
            Some("Alice".into()),
            Some("rust,storage,lsm".into()),
            &[
                "views=100".into(),
                "ratio=0.75".into(),
                "active=true".into(),
            ],
        );

        assert_eq!(meta.get_string("title"), Some("My Title"));
        assert_eq!(meta.get_string("author"), Some("Alice"));
        assert_eq!(meta.get_i64("views"), Some(100));
        assert_eq!(meta.get_f64("ratio"), Some(0.75));
        assert_eq!(meta.get_bool("active"), Some(true));

        let filter = parse_filter_flags(&["author=Alice".into(), "views=100".into()]);
        assert!(filter.is_some());
        assert!(filter.unwrap().matches(&meta));
    }

    #[test]
    fn test_parse_chunk_config() {
        let cfg1 = parse_chunk_config("chars", Some(300), Some(30));
        match cfg1.strategy {
            ChunkingStrategy::FixedChars { size, overlap } => {
                assert_eq!(size, 300);
                assert_eq!(overlap, 30);
            }
            _ => panic!("Expected FixedChars"),
        }

        let cfg2 = parse_chunk_config("tokens", Some(150), Some(15));
        match cfg2.strategy {
            ChunkingStrategy::FixedTokens { size, overlap } => {
                assert_eq!(size, 150);
                assert_eq!(overlap, 15);
            }
            _ => panic!("Expected FixedTokens"),
        }
    }

    #[test]
    fn test_rag_cli_end_to_end() -> Result<()> {
        let dir = tempdir().unwrap();
        let opts = GlobalOpts {
            path: dir.path().to_path_buf(),
            quiet: true,
            ..Default::default()
        };

        // Ingest
        run_rag_command(
            RagCmd::Ingest {
                id: Some("doc_alpha".into()),
                text: Some("FlashStore LSM-tree engine with hybrid retrieval".into()),
                file: None,
                batch_file: None,
                title: Some("FlashStore Guide".into()),
                author: Some("Zein".into()),
                tags: Some("storage,rag".into()),
                metadata: vec!["category=engine".into()],
                chunk_size: Some(100),
                chunk_overlap: Some(10),
                chunk_strategy: "paragraphs".into(),
            },
            &opts,
        )?;

        // Get
        run_rag_command(
            RagCmd::Get {
                id: Some("doc_alpha".into()),
                doc_id: None,
                format: "text".into(),
            },
            &opts,
        )?;

        // Query
        run_rag_command(
            RagCmd::Query {
                query: Some("hybrid retrieval".into()),
                query_flag: None,
                top_k: 3,
                dense_weight: Some(0.6),
                sparse_weight: Some(0.4),
                fusion: "linear".into(),
                no_rerank: false,
                mmr: false,
                mmr_lambda: 0.7,
                min_score: None,
                filters: vec!["author=Zein".into()],
                format: "json".into(),
                prompt: false,
                explain: true,
            },
            &opts,
        )?;

        // Query prompt
        run_rag_command(
            RagCmd::Query {
                query: Some("LSM-tree".into()),
                query_flag: None,
                top_k: 2,
                dense_weight: None,
                sparse_weight: None,
                fusion: "linear".into(),
                no_rerank: false,
                mmr: false,
                mmr_lambda: 0.7,
                min_score: None,
                filters: Vec::new(),
                format: "prompt".into(),
                prompt: true,
                explain: false,
            },
            &opts,
        )?;

        // List
        run_rag_command(
            RagCmd::List {
                limit: 10,
                format: "text".into(),
            },
            &opts,
        )?;

        // Stats
        run_rag_command(
            RagCmd::Stats {
                format: "json".into(),
            },
            &opts,
        )?;

        // ClearCache
        run_rag_command(RagCmd::ClearCache, &opts)?;

        // Eval
        run_rag_command(
            RagCmd::Eval {
                samples_file: None,
                k: 3,
                format: "text".into(),
            },
            &opts,
        )?;

        // Delete
        run_rag_command(
            RagCmd::Delete {
                id: Some("doc_alpha".into()),
                doc_id: None,
            },
            &opts,
        )?;

        Ok(())
    }
}
