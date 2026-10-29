use crate::agent::{
    create_standard_tool_registry, AgentId, AgentPipeline, AgentTask, EpisodicMemoryEntry, TaskId,
    TaskPriority,
};
use crate::cli::agent::seed_standard_agents;
use crate::cli::formatter::safe_truncate_str;
use crate::cli::GlobalOpts;
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use crate::mcp::{
    JsonRpcRequest, McpContent, McpServerConfigBuilder, McpSubsystem, DEFAULT_SERVER_NAME,
    DEFAULT_SERVER_VERSION, MCP_PROTOCOL_VERSION,
};
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
        "AGENT" => execute_agent_repl(db, args, start_time),
        s if s.starts_with("AGENT-") || s.starts_with("AGENT_") => {
            let sub = &s[6..];
            let mut combined = vec![sub.to_string()];
            combined.extend_from_slice(args);
            execute_agent_repl(db, &combined, start_time)
        }
        "MCP" => execute_mcp_repl(db, args, start_time),
        s if s.starts_with("MCP-") || s.starts_with("MCP_") => {
            let sub = &s[4..];
            let mut combined = vec![sub.to_string()];
            combined.extend_from_slice(args);
            execute_mcp_repl(db, &combined, start_time)
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
  AGENT RUN <prompt>           Execute autonomous agent task
  AGENT AGENTS                 List active swarm agents
  AGENT TASKS                  List workflow tasks
  AGENT GET <task_id>          Inspect task details
  AGENT TOOLS                  List available tools and schemas
  AGENT CALL <tool> [args]     Invoke tool with JSON arguments
  AGENT MEMORY <query|RECORD>  Query or record episodic memory
  AGENT STATS                  Display agent swarm metrics and latency
  AGENT CHECKPOINT             Save workflow state to FlashStore
  MCP TOOLS                    List registered MCP tools
  MCP CALL <tool> [args]       Invoke MCP tool with arguments
  MCP RESOURCES                List available MCP resources
  MCP READ <uri>               Inspect MCP resource by URI
  MCP PROMPTS                  List MCP prompt templates
  MCP GET-PROMPT <name> [args] Render MCP prompt template
  MCP INFO / STATUS / PING     Inspect MCP server status
  MCP REQ <raw_json_rpc>       Execute raw JSON-RPC 2.0 request
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
                        format!("{}...", safe_truncate_str(snippet, 120))
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

/// Execute Agent subsystem commands within the REPL.
pub fn execute_agent_repl(
    db: &FlashStore,
    args: &[String],
    start_time: Instant,
) -> Result<Option<String>> {
    if args.is_empty() {
        return Err(FlashStoreError::InvalidArgument(
            "Usage: AGENT <RUN|AGENTS|TASKS|GET|TOOLS|CALL|MEMORY|STATS|CHECKPOINT> ...".into(),
        ));
    }

    let subcmd = args[0].to_uppercase();
    let sub_args = &args[1..];

    let arc_db = Arc::new(db.clone());
    let workflow_id = "repl_agent_session";
    let mut pipeline = match AgentPipeline::restore_from_store(arc_db.clone(), workflow_id) {
        Ok(p) => p,
        Err(_) => AgentPipeline::new(workflow_id).with_store(arc_db.clone()),
    };
    seed_standard_agents(&mut pipeline)?;

    match subcmd.as_str() {
        "RUN" | "TASK" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: AGENT RUN <prompt>".into(),
                ));
            }
            let prompt = sub_args.join(" ");
            let task = AgentTask::builder(TaskId::generate(), prompt.as_str())
                .description(&prompt)
                .priority(TaskPriority::Medium)
                .build();
            let report = pipeline.run_single_task(task)?;
            let _ = pipeline.checkpoint();
            let elapsed = start_time.elapsed();
            let out = format!(
                "=== Task [{}] Completed ({:?}) ===\nAssigned: {}\nStatus:   {:?}\nRouting:  {}\nTool Calls: {}\nReflections: {}\nOutput:   {}\n",
                report.task_id.as_str(),
                elapsed,
                report.assigned_agent.as_str(),
                report.status,
                report.routing_decision.rationale,
                report.tool_calls_executed,
                report.reflection_iterations,
                report.output.map(|o| o.to_string()).unwrap_or_else(|| "(none)".into())
            );
            Ok(Some(out))
        }
        "AGENTS" | "LIST-AGENTS" => {
            let agents: Vec<_> = pipeline.engine().state().active_agents.values().collect();
            if agents.is_empty() {
                Ok(Some(format!(
                    "(no active agents registered) ({:?})",
                    start_time.elapsed()
                )))
            } else {
                let mut out = format!("Active Swarm Agents ({}):\n", agents.len());
                for (i, a) in agents.iter().enumerate() {
                    let caps: Vec<String> = a.capabilities.iter().map(|c| c.to_string()).collect();
                    out.push_str(&format!(
                        "{}) [{}] {} (Role: {}, Caps: [{}])\n",
                        i + 1,
                        a.id.as_str(),
                        a.name,
                        a.role,
                        caps.join(", ")
                    ));
                }
                out.push_str(&format!("({:?})", start_time.elapsed()));
                Ok(Some(out))
            }
        }
        "TASKS" | "LIST-TASKS" => {
            let tasks: Vec<_> = pipeline.engine().state().tasks.values().collect();
            if tasks.is_empty() {
                Ok(Some(format!(
                    "(no workflow tasks found) ({:?})",
                    start_time.elapsed()
                )))
            } else {
                let mut out = format!("Workflow Tasks ({}):\n", tasks.len());
                for (i, t) in tasks.iter().enumerate() {
                    let assigned = t
                        .assigned_agent
                        .as_ref()
                        .map(|a| a.as_str())
                        .unwrap_or("-");
                    out.push_str(&format!(
                        "{}) [{}] \"{}\" | {:?} | agent: {}\n",
                        i + 1,
                        t.id.as_str(),
                        t.title,
                        t.status,
                        assigned
                    ));
                }
                out.push_str(&format!("({:?})", start_time.elapsed()));
                Ok(Some(out))
            }
        }
        "GET" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: AGENT GET <task_id>".into(),
                ));
            }
            let tid = TaskId::new(&sub_args[0]);
            match pipeline.engine().state().tasks.get(&tid) {
                Some(t) => {
                    let assigned = t
                        .assigned_agent
                        .as_ref()
                        .map(|a| a.as_str())
                        .unwrap_or("(none)");
                    let out = format!(
                        "Task:        [{}]\nTitle:       {}\nDescription: {}\nStatus:      {:?}\nPriority:    {}\nAssigned:    {}\nCreated:     {}\n({:?})",
                        t.id.as_str(),
                        t.title,
                        t.description,
                        t.status,
                        t.priority.as_str(),
                        assigned,
                        t.created_at,
                        start_time.elapsed()
                    );
                    Ok(Some(out))
                }
                None => Ok(Some(format!(
                    "(task not found: '{}' in {:?})",
                    sub_args[0],
                    start_time.elapsed()
                ))),
            }
        }
        "TOOLS" => {
            let registry = create_standard_tool_registry(Some(arc_db));
            let defs = registry.list_definitions();
            let mut out = format!("Registered Tools ({}):\n", defs.len());
            for (i, d) in defs.iter().enumerate() {
                out.push_str(&format!("{}) '{}' - {}\n", i + 1, d.name, d.description));
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "CALL" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: AGENT CALL <tool_name> [json_args]".into(),
                ));
            }
            let tool_name = &sub_args[0];
            let raw_args = if sub_args.len() > 1 {
                sub_args[1..].join(" ")
            } else {
                "{}".into()
            };
            let registry = create_standard_tool_registry(Some(arc_db));
            let tool = registry.get(tool_name).ok_or_else(|| {
                FlashStoreError::Agent(format!("Tool '{}' not found", tool_name))
            })?;
            let json_args: serde_json::Value =
                serde_json::from_str(&raw_args).map_err(|e| {
                    FlashStoreError::InvalidArgument(format!("Invalid tool args: {}", e))
                })?;
            let start = Instant::now();
            let res = tool.execute(&json_args)?;
            let elapsed = start.elapsed();
            Ok(Some(format!(
                "Tool '{}' Result (took {:?}):\n{}\n",
                tool_name,
                elapsed,
                serde_json::to_string_pretty(&res).unwrap_or_default()
            )))
        }
        "MEMORY" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: AGENT MEMORY <query|RECORD <text>>".into(),
                ));
            }
            if sub_args[0].eq_ignore_ascii_case("RECORD") {
                if sub_args.len() < 2 {
                    return Err(FlashStoreError::InvalidArgument(
                        "Usage: AGENT MEMORY RECORD <text>".into(),
                    ));
                }
                let content = sub_args[1..].join(" ");
                let entry = EpisodicMemoryEntry::new(AgentId::new("repl_agent"), content, 0.8);
                pipeline.memory().record_episode(entry)?;
                let _ = pipeline.checkpoint();
                Ok(Some(format!(
                    "OK (Recorded memory in {:?})",
                    start_time.elapsed()
                )))
            } else {
                let query = sub_args.join(" ");
                let recalled = pipeline.memory().recall(&query, 5)?;
                if recalled.is_empty() {
                    Ok(Some(format!(
                        "(no memories matched '{}' in {:?})",
                        query,
                        start_time.elapsed()
                    )))
                } else {
                    let mut out =
                        format!("Recalled Memories ({}) for '{}':\n", recalled.len(), query);
                    for (i, m) in recalled.iter().enumerate() {
                        out.push_str(&format!(
                            "{}) [score: {:.4}, imp: {:.2}] {}\n",
                            i + 1,
                            m.score,
                            m.memory.importance,
                            m.memory.content
                        ));
                    }
                    out.push_str(&format!("({:?})", start_time.elapsed()));
                    Ok(Some(out))
                }
            }
        }
        "STATS" => {
            let snap = pipeline.metrics_snapshot();
            let out = format!(
                "=== Swarm Telemetry & Metrics ===\nTasks Submitted:     {}\nTasks Completed:     {}\nTasks Failed:        {}\nSuccess Rate:        {:.2}%\nTotal Tool Calls:    {}\nTool Success Rate:   {:.2}%\nTotal Reflections:   {}\nEpisodic Records:    {}\nEpisodic Recalls:    {}\n({:?})",
                snap.tasks_submitted,
                snap.tasks_completed,
                snap.tasks_failed,
                snap.task_success_rate * 100.0,
                snap.total_tool_calls,
                snap.tool_success_rate * 100.0,
                snap.total_reflection_passes,
                snap.episodic_records,
                snap.episodic_recalls,
                start_time.elapsed()
            );
            Ok(Some(out))
        }
        "CHECKPOINT" => {
            pipeline.checkpoint()?;
            Ok(Some(format!(
                "OK (Workflow checkpoint saved in {:?})",
                start_time.elapsed()
            )))
        }
        other => Err(FlashStoreError::InvalidArgument(format!(
            "Unknown AGENT command '{}'. Available: RUN, AGENTS, TASKS, GET, TOOLS, CALL, MEMORY, STATS, CHECKPOINT",
            other
        ))),
    }
}

/// Execute MCP subsystem commands within the REPL.
pub fn execute_mcp_repl(
    db: &FlashStore,
    args: &[String],
    start_time: Instant,
) -> Result<Option<String>> {
    if args.is_empty() {
        return Err(FlashStoreError::InvalidArgument(
            "Usage: MCP <TOOLS|CALL|RESOURCES|READ|PROMPTS|GET-PROMPT|INFO|REQ> ...".into(),
        ));
    }

    let subcmd = args[0].to_uppercase();
    let sub_args = &args[1..];

    let arc_db = Arc::new(db.clone());
    let rag_engine = {
        let config = RagConfig::default();
        RagEngine::recover(config, arc_db.clone())
            .ok()
            .map(|e| Arc::new(parking_lot::RwLock::new(e)))
    };
    let agent_engine = {
        let mut engine = crate::agent::AgentEngine::new("repl_agent_session").with_store(arc_db.clone());
        crate::cli::mcp::seed_standard_agents_in_engine(&mut engine);
        Some(Arc::new(parking_lot::RwLock::new(engine)))
    };

    let config = McpServerConfigBuilder::new()
        .server_name(DEFAULT_SERVER_NAME)
        .build();

    let subsystem = McpSubsystem::build(config, arc_db, rag_engine, agent_engine);

    match subcmd.as_str() {
        "TOOLS" | "LIST-TOOLS" => {
            let tools = subsystem.server().tools().list();
            let mut out = format!("=== MCP Tools ({}) ===\n", tools.len());
            for (i, t) in tools.iter().enumerate() {
                out.push_str(&format!(
                    "{}. {}\n   {}\n",
                    i + 1,
                    t.name,
                    t.description.as_deref().unwrap_or("-")
                ));
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "CALL" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: MCP CALL <tool_name> [json_args]".into(),
                ));
            }
            let tool_name = &sub_args[0];
            let json_str = if sub_args.len() > 1 {
                sub_args[1..].join(" ")
            } else {
                "{}".to_string()
            };
            let parsed_args: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Invalid JSON arguments: {}", e))
            })?;
            let res = subsystem
                .server()
                .tools()
                .call(tool_name, Some(parsed_args))?;
            let status = if res.is_error.unwrap_or(false) {
                "ERROR"
            } else {
                "SUCCESS"
            };
            let mut out = format!("=== MCP Tool Result [{}] ===\n", status);
            for c in &res.content {
                match c {
                    McpContent::Text { text } => out.push_str(&format!("{}\n", text)),
                    McpContent::Image { mime_type, .. } => {
                        out.push_str(&format!("[Image: {}]\n", mime_type));
                    }
                    McpContent::Resource { resource } => {
                        out.push_str(&format!("[Resource: {}]\n", resource));
                    }
                }
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "RESOURCES" | "LIST-RESOURCES" => {
            let resources = subsystem.server().resources().list();
            let mut out = format!("=== MCP Resources ({}) ===\n", resources.len());
            for (i, r) in resources.iter().enumerate() {
                out.push_str(&format!(
                    "{}. [{}] {}\n   URI: {}\n",
                    i + 1,
                    r.name,
                    r.mime_type.as_deref().unwrap_or("-"),
                    r.uri
                ));
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "READ" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: MCP READ <uri>".into(),
                ));
            }
            let uri = &sub_args[0];
            let contents = subsystem.server().resources().read(uri)?;
            let mut out = format!("=== MCP Resource [{}] ===\n", uri);
            for c in contents {
                if let Some(txt) = c.text {
                    out.push_str(&format!("{}\n", txt));
                } else if let Some(b) = c.blob {
                    out.push_str(&format!("[Blob: {} bytes]\n", b.len()));
                }
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "PROMPTS" | "LIST-PROMPTS" => {
            let prompts = subsystem.server().prompts().list();
            let mut out = format!("=== MCP Prompt Templates ({}) ===\n", prompts.len());
            for (i, p) in prompts.iter().enumerate() {
                out.push_str(&format!("{}. {}\n", i + 1, p.name));
                if let Some(desc) = &p.description {
                    out.push_str(&format!("   {}\n", desc));
                }
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "GET-PROMPT" | "GETPROMPT" | "PROMPT" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: MCP GET-PROMPT <prompt_name> [json_args]".into(),
                ));
            }
            let prompt_name = &sub_args[0];
            let json_str = if sub_args.len() > 1 {
                sub_args[1..].join(" ")
            } else {
                "{}".to_string()
            };
            let parsed_args: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Invalid JSON arguments: {}", e))
            })?;
            let map = crate::cli::mcp::value_to_string_map(parsed_args);
            let res = subsystem
                .server()
                .prompts()
                .render(prompt_name, map.as_ref())?;
            let mut out = format!("=== MCP Prompt [{}] ===\n", prompt_name);
            for m in res.messages {
                match m.content {
                    McpContent::Text { text } => {
                        out.push_str(&format!("[{}]\n{}\n\n", m.role, text));
                    }
                    _ => out.push_str(&format!("[{}] (non-text content)\n\n", m.role)),
                }
            }
            out.push_str(&format!("({:?})", start_time.elapsed()));
            Ok(Some(out))
        }
        "INFO" | "STATUS" | "PING" => {
            let tools_count = subsystem.server().tools().list().len();
            let resources_count = subsystem.server().resources().list().len();
            let prompts_count = subsystem.server().prompts().list().len();

            let out = format!(
                "=== FlashStore MCP Server ===\nServer:    {} (v{})\nProtocol:  {}\nTools:     {}\nResources: {}\nPrompts:   {}\n({:?})",
                DEFAULT_SERVER_NAME,
                DEFAULT_SERVER_VERSION,
                MCP_PROTOCOL_VERSION,
                tools_count,
                resources_count,
                prompts_count,
                start_time.elapsed()
            );
            Ok(Some(out))
        }
        "REQ" | "REQUEST" | "RPC" => {
            if sub_args.is_empty() {
                return Err(FlashStoreError::InvalidArgument(
                    "Usage: MCP REQ <raw_json_rpc>".into(),
                ));
            }
            let raw_json = sub_args.join(" ");
            let req: JsonRpcRequest = serde_json::from_str(&raw_json).map_err(|e| {
                FlashStoreError::InvalidArgument(format!("Failed parsing JSON-RPC request: {}", e))
            })?;
            let resp = subsystem.server().handle_request(req);
            let out = format!(
                "{}\n({:?})",
                serde_json::to_string_pretty(&resp).unwrap_or_default(),
                start_time.elapsed()
            );
            Ok(Some(out))
        }
        _ => Err(FlashStoreError::InvalidArgument(format!(
            "Unknown MCP command: '{}'. Valid commands: TOOLS, CALL, RESOURCES, READ, PROMPTS, GET-PROMPT, INFO, REQ",
            subcmd
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

        let get_res = execute_repl_command(&db, &["RAG".into(), "GET".into(), "repl_doc1".into()])?;
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

        let cache_res = execute_repl_command(&db, &["RAG".into(), "CACHE".into(), "CLEAR".into()])?;
        assert!(cache_res.unwrap().contains("cleared"));

        let eval_res = execute_repl_command(&db, &["RAG".into(), "EVAL".into()])?;
        assert!(eval_res.unwrap().contains("Evaluation"));

        let del_res = execute_repl_command(&db, &["RAG".into(), "DEL".into(), "repl_doc1".into()])?;
        assert!(del_res.unwrap().contains("OK"));

        // AGENT commands
        let run_res = execute_repl_command(
            &db,
            &[
                "AGENT".into(),
                "RUN".into(),
                "Process telemetry batch".into(),
            ],
        )?;
        assert!(run_res.unwrap().contains("Completed"));

        let agents_res = execute_repl_command(&db, &["AGENT".into(), "AGENTS".into()])?;
        assert!(agents_res.unwrap().contains("Active Swarm Agents"));

        let tasks_res = execute_repl_command(&db, &["AGENT".into(), "TASKS".into()])?;
        assert!(tasks_res.unwrap().contains("Workflow Tasks"));

        let tools_res = execute_repl_command(&db, &["AGENT".into(), "TOOLS".into()])?;
        assert!(tools_res.unwrap().contains("Registered Tools"));

        let call_res = execute_repl_command(
            &db,
            &[
                "AGENT".into(),
                "CALL".into(),
                "calculator".into(),
                "{\"operation\": \"mul\", \"a\": 100, \"b\": 2}".into(),
            ],
        )?;
        assert!(call_res.unwrap().contains("200"));

        let mem_rec = execute_repl_command(
            &db,
            &[
                "AGENT".into(),
                "MEMORY".into(),
                "RECORD".into(),
                "Completed telemetry data processing batch successfully".into(),
            ],
        )?;
        assert!(mem_rec.unwrap().contains("OK"));

        let mem_query =
            execute_repl_command(&db, &["AGENT".into(), "MEMORY".into(), "telemetry".into()])?;
        assert!(mem_query.unwrap().contains("Recalled Memories"));

        let agent_stats = execute_repl_command(&db, &["AGENT".into(), "STATS".into()])?;
        assert!(agent_stats.unwrap().contains("Swarm Telemetry"));

        let agent_cp = execute_repl_command(&db, &["AGENT".into(), "CHECKPOINT".into()])?;
        assert!(agent_cp.unwrap().contains("OK"));

        // MCP commands
        let mcp_tools = execute_repl_command(&db, &["MCP".into(), "TOOLS".into()])?;
        assert!(mcp_tools.unwrap().contains("MCP Tools"));

        let mcp_call = execute_repl_command(
            &db,
            &[
                "MCP".into(),
                "CALL".into(),
                "flash_put".into(),
                "{\"key\": \"mcp_repl_key\", \"value\": \"mcp_repl_val\"}".into(),
            ],
        )?;
        assert!(mcp_call.unwrap().contains("SUCCESS"));

        let mcp_resources = execute_repl_command(&db, &["MCP".into(), "RESOURCES".into()])?;
        assert!(mcp_resources.unwrap().contains("MCP Resources"));

        let mcp_read = execute_repl_command(&db, &["MCP".into(), "READ".into(), "flash://stats".into()])?;
        assert!(mcp_read.unwrap().contains("stats"));

        let mcp_prompts = execute_repl_command(&db, &["MCP".into(), "PROMPTS".into()])?;
        assert!(mcp_prompts.unwrap().contains("flash_troubleshoot"));

        let mcp_get_prompt = execute_repl_command(
            &db,
            &[
                "MCP".into(),
                "GET-PROMPT".into(),
                "flash_troubleshoot".into(),
                "{\"issue\": \"bad crc\"}".into(),
            ],
        )?;
        assert!(mcp_get_prompt.unwrap().contains("bad crc"));

        let mcp_info = execute_repl_command(&db, &["MCP".into(), "INFO".into()])?;
        assert!(mcp_info.unwrap().contains("FlashStore MCP Server"));

        let mcp_req = execute_repl_command(
            &db,
            &[
                "MCP".into(),
                "REQ".into(),
                "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"ping\"}".into(),
            ],
        )?;
        assert!(mcp_req.unwrap().contains("result"));

        Ok(())
    }
}
