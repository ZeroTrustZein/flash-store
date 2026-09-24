use crate::agent::{
    AgentDefinition, AgentMetricsSnapshot, AgentTask, ScoredMemory, TaskExecutionReport,
    ToolDefinition, ToolOutput, WorkflowSummary,
};
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

/// Format and print a list of registered agents.
pub fn format_agent_list(agents: &[AgentDefinition], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(agents).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("id\tname\trole\tcapabilities\titerations");
            for a in agents {
                let caps: Vec<String> = a.capabilities.iter().map(|c| c.to_string()).collect();
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    a.id.as_str(),
                    a.name,
                    a.role,
                    caps.join(","),
                    a.max_iterations
                );
            }
        }
        OutputFormat::Text => {
            if agents.is_empty() {
                println!("(no active agents registered)");
                return;
            }
            println!("=== Active Swarm Agents ({}) ===", agents.len());
            for (i, a) in agents.iter().enumerate() {
                let caps: Vec<String> = a.capabilities.iter().map(|c| c.to_string()).collect();
                println!(
                    "{}) [{}] {} (Role: {})",
                    i + 1,
                    a.id.as_str(),
                    a.name,
                    a.role
                );
                println!(
                    "   Capabilities:   {}",
                    if caps.is_empty() {
                        "none".into()
                    } else {
                        caps.join(", ")
                    }
                );
                println!("   Max Iterations: {}", a.max_iterations);
                if !a.system_prompt.is_empty() {
                    let prompt_snippet = safe_truncate_str(&a.system_prompt, 80);
                    let suffix = if a.system_prompt.len() > 80 {
                        "..."
                    } else {
                        ""
                    };
                    println!("   Prompt:         {}{}", prompt_snippet, suffix);
                }
            }
        }
    }
}

/// Format and print a list of workflow tasks.
pub fn format_task_list(tasks: &[AgentTask], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(tasks).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("id\ttitle\tpriority\tstatus\tassigned");
            for t in tasks {
                let assigned = t.assigned_agent.as_ref().map(|a| a.as_str()).unwrap_or("-");
                println!(
                    "{}\t{}\t{}\t{:?}\t{}",
                    t.id.as_str(),
                    t.title,
                    t.priority.as_str(),
                    t.status,
                    assigned
                );
            }
        }
        OutputFormat::Text => {
            if tasks.is_empty() {
                println!("(no tasks found)");
                return;
            }
            println!("=== Swarm Workflow Tasks ({}) ===", tasks.len());
            for (i, t) in tasks.iter().enumerate() {
                let assigned = t
                    .assigned_agent
                    .as_ref()
                    .map(|a| a.as_str())
                    .unwrap_or("unassigned");
                println!(
                    "{}) [{}] \"{}\" | status: {:?} | priority: {} | agent: {}",
                    i + 1,
                    t.id.as_str(),
                    t.title,
                    t.status,
                    t.priority.as_str(),
                    assigned
                );
                let desc_truncated = safe_truncate_str(&t.description, 100);
                let desc_suffix = if t.description.len() > 100 { "..." } else { "" };
                println!("   Desc: {}{}", desc_truncated, desc_suffix);
                if let Some(ref out) = t.output_artifacts {
                    let out_str = out.to_string();
                    let out_truncated = safe_truncate_str(&out_str, 80);
                    let out_suffix = if out_str.len() > 80 { "..." } else { "" };
                    println!("   Out:  {}{}", out_truncated, out_suffix);
                }
            }
        }
    }
}

/// Format and print detailed task metadata and execution history.
pub fn format_task_detail(task: &AgentTask, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(task).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            let assigned = task
                .assigned_agent
                .as_ref()
                .map(|a| a.as_str())
                .unwrap_or("-");
            println!(
                "{}\t{}\t{}\t{:?}\t{}",
                task.id.as_str(),
                task.title,
                task.priority.as_str(),
                task.status,
                assigned
            );
        }
        OutputFormat::Text => {
            println!("=== Task Detail: [{}] ===", task.id.as_str());
            println!("Title:       {}", task.title);
            println!("Status:      {:?}", task.status);
            println!("Priority:    {}", task.priority.as_str());
            println!(
                "Assigned:    {}",
                task.assigned_agent
                    .as_ref()
                    .map(|a| a.as_str())
                    .unwrap_or("(none)")
            );
            println!("Description: {}", task.description);
            if !task.input_context.is_null() {
                println!("Input:       {}", task.input_context);
            }
            if let Some(ref output) = task.output_artifacts {
                println!("Output:      {}", output);
            }
            if let Some(ref err) = task.error_message {
                println!("Error:       {}", err);
            }
            println!("Created At:  {}", task.created_at);
            println!("Updated At:  {}", task.updated_at);
        }
    }
}

/// Format and print execution report of a single task.
pub fn format_task_report(report: &TaskExecutionReport, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!(
                "{}\t{}\t{:?}\t{}\t{}",
                report.task_id.as_str(),
                report.assigned_agent.as_str(),
                report.status,
                report.execution_time_ms,
                report.tool_calls_executed
            );
        }
        OutputFormat::Text => {
            println!("=== Task Execution Report ===");
            println!("Task ID:             {}", report.task_id.as_str());
            println!("Assigned Agent:      {}", report.assigned_agent.as_str());
            println!("Status:              {:?}", report.status);
            println!("Execution Time:      {} ms", report.execution_time_ms);
            println!("Routing:             {}", report.routing_decision.rationale);
            println!("Tool Calls:          {}", report.tool_calls_executed);
            println!("Reflection Passes:   {}", report.reflection_iterations);
            if let Some(ref verdict) = report.reflection_verdict {
                println!("Reflection Verdict:  {:?}", verdict);
            }
            if let Some(ref out) = report.output {
                println!(
                    "Output:\n{}",
                    serde_json::to_string_pretty(out).unwrap_or_default()
                );
            }
            if let Some(ref err) = report.error {
                println!("Error:               {}", err);
            }
        }
    }
}

/// Format and print summary of full workflow execution.
pub fn format_workflow_summary(summary: &WorkflowSummary, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(summary).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!(
                "{}\t{:?}\t{}\t{}\t{}\t{}",
                summary.workflow_id,
                summary.status,
                summary.total_tasks,
                summary.completed_tasks,
                summary.failed_tasks,
                summary.total_duration_ms
            );
        }
        OutputFormat::Text => {
            println!("=== Workflow Execution Summary ===");
            println!("Workflow ID:     {}", summary.workflow_id);
            println!("Status:          {:?}", summary.status);
            println!("Total Tasks:     {}", summary.total_tasks);
            println!("Completed Tasks: {}", summary.completed_tasks);
            println!("Failed Tasks:    {}", summary.failed_tasks);
            println!("Tool Calls:      {}", summary.total_tool_calls);
            println!("Messages:        {}", summary.total_messages);
            println!("Duration:        {} ms", summary.total_duration_ms);
            println!("\nTask Breakdown:");
            for r in &summary.reports {
                println!(
                    "  - [{}] agent: {}, status: {:?}, time: {}ms",
                    r.task_id.as_str(),
                    r.assigned_agent.as_str(),
                    r.status,
                    r.execution_time_ms
                );
            }
        }
    }
}

/// Format and print list of registered tools.
pub fn format_tools_list(tools: &[ToolDefinition], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("name\tdescription\tparameters_count\tdeterministic\ttimeout_ms");
            for t in tools {
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    t.name,
                    t.description,
                    t.parameters.len(),
                    t.is_deterministic,
                    t.timeout_ms
                );
            }
        }
        OutputFormat::Text => {
            if tools.is_empty() {
                println!("(no registered tools)");
                return;
            }
            println!("=== Registered Swarm Tools ({}) ===", tools.len());
            for (i, t) in tools.iter().enumerate() {
                println!("{}) Tool: '{}'", i + 1, t.name);
                println!("   Description: {}", t.description);
                if !t.parameters.is_empty() {
                    let param_strs: Vec<String> = t
                        .parameters
                        .iter()
                        .map(|p| {
                            let req = if p.required { "*" } else { "" };
                            format!("{}{}: {:?}", p.name, req, p.param_type)
                        })
                        .collect();
                    println!("   Parameters:  {}", param_strs.join(", "));
                }
            }
        }
    }
}

/// Format and print output of a tool execution.
pub fn format_tool_output(output: &ToolOutput, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(output).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!(
                "{}\t{}\t{}\t{}\t{}",
                output.call_id,
                output.tool_name,
                output.success,
                output.execution_time_ms,
                output.result
            );
        }
        OutputFormat::Text => {
            println!("=== Tool Output: '{}' ===", output.tool_name);
            println!("Call ID:        {}", output.call_id);
            println!("Success:        {}", output.success);
            println!("Execution Time: {} ms", output.execution_time_ms);
            if let Some(ref err) = output.error {
                println!("Error:          {}", err);
            }
            println!(
                "Result:\n{}",
                serde_json::to_string_pretty(&output.result).unwrap_or_default()
            );
        }
    }
}

/// Format and print episodic memory entries.
pub fn format_agent_memories(memories: &[ScoredMemory], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(memories).unwrap_or_else(|_| "[]".into())
            );
        }
        OutputFormat::Tsv => {
            println!("id\tagent\tscore\timportance\tcontent");
            for m in memories {
                println!(
                    "{}\t{}\t{:.4}\t{:.2}\t{}",
                    m.memory.id,
                    m.memory.agent_id.as_str(),
                    m.score,
                    m.memory.importance,
                    m.memory.content
                );
            }
        }
        OutputFormat::Text => {
            if memories.is_empty() {
                println!("(no memories found)");
                return;
            }
            println!("=== Recalled Swarm Memories ({}) ===", memories.len());
            for (i, m) in memories.iter().enumerate() {
                println!(
                    "{}) [{}] Agent: '{}' (Score: {:.4}, Imp: {:.2})",
                    i + 1,
                    m.memory.id,
                    m.memory.agent_id.as_str(),
                    m.score,
                    m.memory.importance
                );
                if !m.memory.tags.is_empty() {
                    println!("   Tags:    {}", m.memory.tags.join(", "));
                }
                println!("   Content: {}", m.memory.content);
            }
        }
    }
}

/// Format and print agent telemetry metrics snapshot.
pub fn format_agent_metrics(metrics: &AgentMetricsSnapshot, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(metrics).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv => {
            println!(
                "{}\t{}\t{}\t{:.2}\t{}\t{}\t{}",
                metrics.tasks_submitted,
                metrics.tasks_completed,
                metrics.tasks_failed,
                metrics.task_success_rate * 100.0,
                metrics.total_tool_calls,
                metrics.successful_tool_calls,
                metrics.total_reflection_passes
            );
        }
        OutputFormat::Text => {
            println!("=== Swarm Telemetry & Metrics ===");
            println!("Tasks Submitted:     {}", metrics.tasks_submitted);
            println!("Tasks Completed:     {}", metrics.tasks_completed);
            println!("Tasks Failed:        {}", metrics.tasks_failed);
            println!("Tasks Preempted:     {}", metrics.tasks_preempted);
            println!(
                "Success Rate:        {:.2}%",
                metrics.task_success_rate * 100.0
            );
            println!(
                "Task Latency (avg):  {:.2} ms",
                metrics.task_execution_latency.avg_ms
            );
            println!(
                "Task Latency (max):  {:.2} ms",
                metrics.task_execution_latency.max_ms
            );
            println!("Total Tool Calls:    {}", metrics.total_tool_calls);
            println!(
                "Tool Success Rate:   {:.2}%",
                metrics.tool_success_rate * 100.0
            );
            println!(
                "Tool Latency (avg):  {:.2} ms",
                metrics.tool_execution_latency.avg_ms
            );
            println!("Total Reflections:   {}", metrics.total_reflection_passes);
            println!(
                "Reflection Approval: {:.2}%",
                metrics.reflection_approval_rate * 100.0
            );
            println!("Episodic Records:    {}", metrics.episodic_records);
            println!("Episodic Recalls:    {}", metrics.episodic_recalls);
            println!(
                "Blackboard Ops:      {} writes, {} reads",
                metrics.blackboard_writes, metrics.blackboard_reads
            );
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

    #[test]
    fn test_format_agent_structures() {
        use crate::agent::{
            AgentCapability, AgentDefinition, AgentId, AgentRole, AgentTask, EpisodicMemoryEntry,
            LatencySummary, ReflectionVerdict, RoutingDecision, RoutingStrategy, TaskPriority,
            TaskStatus, ToolParamType, ToolParameter, WorkflowStatus,
        };

        let agent = AgentDefinition::builder("coder_1", AgentRole::Coder)
            .name("Coder Alpha")
            .capability(AgentCapability::CodeExecution)
            .capability(AgentCapability::ToolCalling)
            .system_prompt("You are a professional software engineer.")
            .max_iterations(5)
            .build();

        format_agent_list(std::slice::from_ref(&agent), OutputFormat::Json);
        format_agent_list(std::slice::from_ref(&agent), OutputFormat::Tsv);
        format_agent_list(&[agent], OutputFormat::Text);
        format_agent_list(&[], OutputFormat::Text);

        let mut task = AgentTask::builder("task_001", "Quicksort")
            .description("Implement quicksort in Rust")
            .priority(TaskPriority::High)
            .build();
        task.assign(AgentId::new("coder_1"));

        format_task_list(std::slice::from_ref(&task), OutputFormat::Json);
        format_task_list(std::slice::from_ref(&task), OutputFormat::Tsv);
        format_task_list(std::slice::from_ref(&task), OutputFormat::Text);
        format_task_list(&[], OutputFormat::Text);

        format_task_detail(&task, OutputFormat::Json);
        format_task_detail(&task, OutputFormat::Tsv);
        format_task_detail(&task, OutputFormat::Text);

        let report = TaskExecutionReport {
            task_id: task.id.clone(),
            assigned_agent: AgentId::new("coder_1"),
            status: TaskStatus::Completed,
            routing_decision: RoutingDecision::new(
                task.id.clone(),
                AgentId::new("coder_1"),
                RoutingStrategy::Adaptive,
                0.95,
                "Best match for code",
            ),
            tool_calls_executed: 2,
            reflection_iterations: 1,
            reflection_verdict: Some(ReflectionVerdict::Approved),
            output: Some(json!({"result_text": "Fn implemented"})),
            execution_time_ms: 120,
            error: None,
        };

        format_task_report(&report, OutputFormat::Json);
        format_task_report(&report, OutputFormat::Tsv);
        format_task_report(&report, OutputFormat::Text);

        let summary = WorkflowSummary {
            workflow_id: "wf_1".into(),
            status: WorkflowStatus::Succeeded,
            total_tasks: 1,
            completed_tasks: 1,
            failed_tasks: 0,
            total_tool_calls: 2,
            total_messages: 5,
            total_duration_ms: 150,
            reports: vec![report],
        };

        format_workflow_summary(&summary, OutputFormat::Json);
        format_workflow_summary(&summary, OutputFormat::Tsv);
        format_workflow_summary(&summary, OutputFormat::Text);

        let tool = ToolDefinition::new("calculator", "Performs math calculations").with_parameter(
            ToolParameter {
                name: "expression".into(),
                description: "Math string".into(),
                param_type: ToolParamType::String,
                required: true,
                default_value: None,
            },
        );

        format_tools_list(&[tool], OutputFormat::Json);
        format_tools_list(&[], OutputFormat::Text);

        let tool_out = ToolOutput::success("call_1", "calculator", json!({"value": 42}), 5);
        format_tool_output(&tool_out, OutputFormat::Json);
        format_tool_output(&tool_out, OutputFormat::Tsv);
        format_tool_output(&tool_out, OutputFormat::Text);

        let entry = EpisodicMemoryEntry::new(
            AgentId::new("coder_1"),
            "Recalled solution for quicksort",
            0.85,
        )
        .with_task_id(task.id)
        .with_tags(vec!["algo".into()]);

        let scored = ScoredMemory {
            memory: entry,
            score: 0.92,
            recency_score: 0.95,
            importance_score: 0.85,
            relevance_score: 0.90,
        };

        format_agent_memories(&[scored], OutputFormat::Json);
        format_agent_memories(&[], OutputFormat::Text);

        let metrics = AgentMetricsSnapshot {
            tasks_submitted: 10,
            tasks_completed: 9,
            tasks_failed: 1,
            tasks_preempted: 0,
            task_success_rate: 0.9,
            task_execution_latency: LatencySummary::default(),
            total_routing_decisions: 10,
            preemptive_routings: 0,
            fallback_routings: 1,
            avg_routing_confidence: 0.95,
            total_tool_calls: 15,
            successful_tool_calls: 14,
            failed_tool_calls: 1,
            tool_success_rate: 0.933,
            tool_execution_latency: LatencySummary::default(),
            tool_usage_by_name: std::collections::HashMap::new(),
            total_reflection_passes: 8,
            approved_reflections: 7,
            revision_reflections: 1,
            rejected_reflections: 0,
            escalated_reflections: 0,
            reflection_approval_rate: 0.875,
            avg_reflection_iterations: 1.1,
            blackboard_reads: 20,
            blackboard_writes: 15,
            episodic_records: 10,
            episodic_recalls: 5,
        };

        format_agent_metrics(&metrics, OutputFormat::Json);
        format_agent_metrics(&metrics, OutputFormat::Tsv);
        format_agent_metrics(&metrics, OutputFormat::Text);
    }
}
