use bytes::Bytes;
use flash_store::cli::batch_parser::{parse_json_ops, parse_plain_lines};
use flash_store::cli::formatter::{format_kv_pairs, format_stats};
use flash_store::cli::repl::{execute_repl_command, tokenize_line};
use flash_store::cli::{
    open_agent_pipeline, open_db, open_rag_pipeline, run, AgentCmd, Cmd, GlobalOpts, OutputFormat,
    RagCmd,
};
use flash_store::config::OptionsBuilder;
use flash_store::engine::{FlashStore, Stats};
use flash_store::error::Result;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_cli_put_get_delete_lifecycle() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // 1. Put
    run(
        Cmd::Put {
            key: "cli_key_1".into(),
            value: "cli_val_1".into(),
        },
        &opts,
    )?;

    // 2. Get via direct DB check
    let db = open_db(&opts)?;
    assert_eq!(
        db.get(Bytes::from_static(b"cli_key_1"))?,
        Some(Bytes::from_static(b"cli_val_1"))
    );

    // 3. Get command in Text, JSON, and TSV
    run(
        Cmd::Get {
            key: "cli_key_1".into(),
            format: OutputFormat::Text,
        },
        &opts,
    )?;
    run(
        Cmd::Get {
            key: "cli_key_1".into(),
            format: OutputFormat::Json,
        },
        &opts,
    )?;
    run(
        Cmd::Get {
            key: "cli_key_1".into(),
            format: OutputFormat::Tsv,
        },
        &opts,
    )?;

    // 4. Delete
    run(
        Cmd::Delete {
            key: "cli_key_1".into(),
        },
        &opts,
    )?;

    // 5. Verify deleted
    let db2 = open_db(&opts)?;
    assert_eq!(db2.get(Bytes::from_static(b"cli_key_1"))?, None);

    // 6. Get missing key in JSON & text
    run(
        Cmd::Get {
            key: "missing_key".into(),
            format: OutputFormat::Json,
        },
        &opts,
    )?;
    run(
        Cmd::Get {
            key: "missing_key".into(),
            format: OutputFormat::Text,
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_cli_scan_options_and_formatting() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // Seed data
    let db = open_db(&opts)?;
    db.put(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"))?;
    db.put(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"))?;
    db.put(Bytes::from_static(b"k3"), Bytes::from_static(b"v3"))?;
    db.put(Bytes::from_static(b"k4"), Bytes::from_static(b"v4"))?;

    // Unbounded scan
    run(
        Cmd::Scan {
            start: None,
            end: None,
            limit: None,
            reverse: false,
            format: OutputFormat::Text,
        },
        &opts,
    )?;

    // Bounded scan with limit and reverse
    run(
        Cmd::Scan {
            start: Some("k2".into()),
            end: Some("k4".into()),
            limit: Some(1),
            reverse: true,
            format: OutputFormat::Json,
        },
        &opts,
    )?;

    // TSV scan
    run(
        Cmd::Scan {
            start: None,
            end: None,
            limit: Some(2),
            reverse: false,
            format: OutputFormat::Tsv,
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_cli_batch_execution_from_json_string() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    let json_batch = r#"[
        {"op": "put", "key": "batch_1", "value": "val_1"},
        {"type": "set", "key": "batch_2", "value": "val_2"},
        {"action": "insert", "key": "batch_3", "value": "val_3"},
        {"op": "delete", "key": "batch_2"}
    ]"#;

    run(
        Cmd::Batch {
            ops: Some(json_batch.into()),
            file: None,
        },
        &opts,
    )?;

    let db = open_db(&opts)?;
    assert_eq!(
        db.get(Bytes::from_static(b"batch_1"))?,
        Some(Bytes::from_static(b"val_1"))
    );
    assert_eq!(db.get(Bytes::from_static(b"batch_2"))?, None);
    assert_eq!(
        db.get(Bytes::from_static(b"batch_3"))?,
        Some(Bytes::from_static(b"val_3"))
    );

    Ok(())
}

#[test]
fn test_cli_batch_execution_from_file() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    let file_path = dir.path().join("batch_ops.txt");
    let batch_content = "PUT user:1 Alice\nPUT user:2 Bob\nDEL user:1\n";
    fs::write(&file_path, batch_content)?;

    run(
        Cmd::Batch {
            ops: None,
            file: Some(file_path),
        },
        &opts,
    )?;

    let db = open_db(&opts)?;
    assert_eq!(db.get(Bytes::from_static(b"user:1"))?, None);
    assert_eq!(
        db.get(Bytes::from_static(b"user:2"))?,
        Some(Bytes::from_static(b"Bob"))
    );

    Ok(())
}

#[test]
fn test_cli_flush_compact_stats_commands() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // Insert keys and flush
    let db = open_db(&opts)?;
    for i in 0..10 {
        db.put(
            format!("k_{}", i).into_bytes(),
            format!("v_{}", i).into_bytes(),
        )?;
    }

    run(Cmd::Flush, &opts)?;

    // Compact
    run(Cmd::Compact, &opts)?;

    // Stats in all formats
    run(
        Cmd::Stats {
            format: OutputFormat::Text,
        },
        &opts,
    )?;
    run(
        Cmd::Stats {
            format: OutputFormat::Json,
        },
        &opts,
    )?;
    run(
        Cmd::Stats {
            format: OutputFormat::Tsv,
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_cli_inspect_sstable() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    let db = open_db(&opts)?;
    db.put(Bytes::from_static(b"inspect_a"), Bytes::from_static(b"100"))?;
    db.put(Bytes::from_static(b"inspect_b"), Bytes::from_static(b"200"))?;
    db.flush()?;

    let sst_path = flash_store::sstable::table_path(dir.path(), 1);
    assert!(sst_path.exists());

    // Inspect by direct path
    run(
        Cmd::Inspect {
            path: sst_path.to_string_lossy().into_owned(),
            format: OutputFormat::Text,
        },
        &opts,
    )?;

    // Inspect by number resolution
    run(
        Cmd::Inspect {
            path: "1".into(),
            format: OutputFormat::Json,
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_cli_bench_command() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    run(
        Cmd::Bench {
            ops: 100,
            value_size: 32,
            read_ratio: 0.5,
        },
        &opts,
    )?;

    // Edge case: 0 ops
    run(
        Cmd::Bench {
            ops: 0,
            value_size: 16,
            read_ratio: 0.5,
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_repl_tokenizer_edge_cases() -> Result<()> {
    // Normal words
    assert_eq!(tokenize_line("SET key value")?, vec!["SET", "key", "value"]);

    // Multiple spaces
    assert_eq!(tokenize_line("  PUT    k    v   ")?, vec!["PUT", "k", "v"]);

    // Double quotes with spaces
    assert_eq!(
        tokenize_line(r#"PUT "first key" "first value""#)?,
        vec!["PUT", "first key", "first value"]
    );

    // Single quotes with spaces
    assert_eq!(
        tokenize_line("PUT 'my key' 'my value'")?,
        vec!["PUT", "my key", "my value"]
    );

    // Escaped spaces
    assert_eq!(
        tokenize_line(r"PUT key\ name val\ val")?,
        vec!["PUT", "key name", "val val"]
    );

    // Unclosed double quote
    assert!(tokenize_line(r#"PUT "unclosed key value"#).is_err());

    // Unclosed single quote
    assert!(tokenize_line("PUT 'unclosed key value").is_err());

    // Empty line
    assert_eq!(tokenize_line("")?, Vec::<String>::new());
    assert_eq!(tokenize_line("   \t  ")?, Vec::<String>::new());

    Ok(())
}

#[test]
fn test_repl_command_dispatch() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    // Empty tokens
    assert_eq!(execute_repl_command(&db, &[])?, None);

    // Missing arguments
    assert!(execute_repl_command(&db, &["PUT".into()]).is_err());
    assert!(execute_repl_command(&db, &["PUT".into(), "only_key".into()]).is_err());
    assert!(execute_repl_command(&db, &["GET".into()]).is_err());
    assert!(execute_repl_command(&db, &["DEL".into()]).is_err());

    // Successful operations
    let put_res = execute_repl_command(&db, &["PUT".into(), "user:100".into(), "Alice".into()])?;
    assert!(put_res.unwrap().contains("OK"));

    let get_res = execute_repl_command(&db, &["GET".into(), "user:100".into()])?;
    assert!(get_res.unwrap().contains("Alice"));

    let scan_res = execute_repl_command(&db, &["SCAN".into()])?;
    assert!(scan_res.unwrap().contains("user:100: Alice"));

    let del_res = execute_repl_command(&db, &["DEL".into(), "user:100".into()])?;
    assert!(del_res.unwrap().contains("OK"));

    let get_nil = execute_repl_command(&db, &["GET".into(), "user:100".into()])?;
    assert!(get_nil.unwrap().contains("(nil)"));

    // Maintenance commands
    assert!(execute_repl_command(&db, &["FLUSH".into()])?.is_some());
    assert!(execute_repl_command(&db, &["COMPACT".into()])?.is_some());
    assert!(execute_repl_command(&db, &["STATS".into()])?.is_some());
    assert!(execute_repl_command(&db, &["HELP".into()])?.is_some());
    assert_eq!(
        execute_repl_command(&db, &["EXIT".into()])?,
        Some("BYE".into())
    );
    assert_eq!(
        execute_repl_command(&db, &["QUIT".into()])?,
        Some("BYE".into())
    );
    assert_eq!(
        execute_repl_command(&db, &["Q".into()])?,
        Some("BYE".into())
    );

    // Unknown command
    assert!(execute_repl_command(&db, &["FOOBAR".into()]).is_err());

    Ok(())
}

#[test]
fn test_batch_parser_nested_and_plain() -> Result<()> {
    let json_nested = r#"[
        {"put": {"key": "alpha", "value": "1"}},
        {"delete": {"key": "beta"}},
        {"del": {"key": "gamma"}}
    ]"#;
    let b = parse_json_ops(json_nested)?;
    assert_eq!(b.len(), 3);

    let empty = parse_json_ops("")?;
    assert_eq!(empty.len(), 0);

    let plain_with_comments = r#"
        # Header comment
        // Another comment
        PUT k1 v1
        SET k2 v2
        DEL k1
        DELETE k2
    "#;
    let b_plain = parse_plain_lines(plain_with_comments)?;
    assert_eq!(b_plain.len(), 4);

    Ok(())
}

#[test]
fn test_formatter_functions() {
    let stats = Stats {
        active_memtable_size: 2048,
        immutable_memtables_count: 1,
        levels_file_count: vec![2, 1, 0],
    };
    format_stats(&stats, OutputFormat::Json);
    format_stats(&stats, OutputFormat::Tsv);
    format_stats(&stats, OutputFormat::Text);

    let pairs = vec![
        (Bytes::from_static(b"k1"), Bytes::from_static(b"v1")),
        (Bytes::from_static(b"k2"), Bytes::from_static(b"v2")),
    ];
    format_kv_pairs(&pairs, OutputFormat::Json, false);
    format_kv_pairs(&pairs, OutputFormat::Tsv, false);
    format_kv_pairs(&pairs, OutputFormat::Text, false);
    format_kv_pairs(&[], OutputFormat::Text, false);
}

#[test]
fn test_global_opts_and_custom_options() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        memtable_size: Some(1024),
        block_size: Some(128),
        sync_wal: true,
        block_cache_size: Some(512),
        json: true,
        quiet: false,
    };

    assert_eq!(opts.effective_format(), OutputFormat::Json);

    let db = open_db(&opts)?;
    db.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"))?;
    assert_eq!(
        db.get(Bytes::from_static(b"a"))?,
        Some(Bytes::from_static(b"1"))
    );

    Ok(())
}

#[test]
fn test_cli_rag_ingest_and_query_lifecycle() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // 1. Ingest document via CLI
    run(
        Cmd::Rag {
            command: RagCmd::Ingest {
                id: Some("rag_doc_1".into()),
                text: Some(
                    "FlashStore is a thread-safe embedded LSM-Tree key-value database in Rust."
                        .into(),
                ),
                file: None,
                batch_file: None,
                title: Some("FlashStore Architecture".into()),
                author: Some("Zein".into()),
                tags: Some("database,rust,storage".into()),
                metadata: vec!["tier=engine".into(), "year=2026".into()],
                chunk_size: Some(100),
                chunk_overlap: Some(20),
                chunk_strategy: "paragraphs".into(),
            },
        },
        &opts,
    )?;

    // 2. Verify document in pipeline
    let pipeline = open_rag_pipeline(&opts)?;
    let doc = pipeline.get_document("rag_doc_1");
    assert!(doc.is_some());
    assert_eq!(
        doc.as_ref().unwrap().metadata.get_string("author"),
        Some("Zein")
    );

    // 3. Get document via CLI Get in text and json
    run(
        Cmd::Rag {
            command: RagCmd::Get {
                id: Some("rag_doc_1".into()),
                doc_id: None,
                format: "text".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::Get {
                id: None,
                doc_id: Some("rag_doc_1".into()),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // 4. Query in text, JSON, and TSV
    run(
        Cmd::Rag {
            command: RagCmd::Query {
                query: Some("embedded database LSM".into()),
                query_flag: None,
                top_k: 5,
                dense_weight: Some(0.5),
                sparse_weight: Some(0.5),
                fusion: "linear".into(),
                no_rerank: false,
                mmr: false,
                mmr_lambda: 0.7,
                min_score: None,
                filters: vec!["author=Zein".into()],
                format: "text".into(),
                prompt: false,
                explain: true,
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::Query {
                query: None,
                query_flag: Some("database".into()),
                top_k: 2,
                dense_weight: None,
                sparse_weight: None,
                fusion: "rrf".into(),
                no_rerank: true,
                mmr: false,
                mmr_lambda: 0.7,
                min_score: None,
                filters: Vec::new(),
                format: "json".into(),
                prompt: false,
                explain: false,
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::Query {
                query: Some("thread-safe storage".into()),
                query_flag: None,
                top_k: 2,
                dense_weight: None,
                sparse_weight: None,
                fusion: "dense".into(),
                no_rerank: false,
                mmr: true,
                mmr_lambda: 0.8,
                min_score: None,
                filters: Vec::new(),
                format: "tsv".into(),
                prompt: false,
                explain: false,
            },
        },
        &opts,
    )?;

    // 5. Query with prompt generation
    run(
        Cmd::Rag {
            command: RagCmd::Query {
                query: Some("How does FlashStore work?".into()),
                query_flag: None,
                top_k: 3,
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
        },
        &opts,
    )?;

    // 6. Delete document via CLI
    run(
        Cmd::Rag {
            command: RagCmd::Delete {
                id: Some("rag_doc_1".into()),
                doc_id: None,
            },
        },
        &opts,
    )?;

    let pipeline_after = open_rag_pipeline(&opts)?;
    assert!(pipeline_after.get_document("rag_doc_1").is_none());

    Ok(())
}

#[test]
fn test_cli_rag_batch_ingest_and_list() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // Create a batch JSON file
    let batch_file_path = dir.path().join("batch_documents.json");
    let batch_json = r#"[
        {
            "id": "doc_b1",
            "text": "First batch document about distributed file systems.",
            "title": "DFS Overview",
            "author": "Alice"
        },
        {
            "id": "doc_b2",
            "text": "Second batch document about vector databases and HNSW indexing.",
            "title": "Vector Search",
            "author": "Bob"
        }
    ]"#;
    fs::write(&batch_file_path, batch_json)?;

    // Ingest batch file via CLI
    run(
        Cmd::Rag {
            command: RagCmd::Ingest {
                id: None,
                text: None,
                file: None,
                batch_file: Some(batch_file_path),
                title: None,
                author: None,
                tags: None,
                metadata: Vec::new(),
                chunk_size: Some(100),
                chunk_overlap: Some(10),
                chunk_strategy: "chars".into(),
            },
        },
        &opts,
    )?;

    // List documents in all formats
    run(
        Cmd::Rag {
            command: RagCmd::List {
                limit: 10,
                format: "text".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::List {
                limit: 10,
                format: "json".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::List {
                limit: 10,
                format: "tsv".into(),
            },
        },
        &opts,
    )?;

    // Stats in all formats
    run(
        Cmd::Rag {
            command: RagCmd::Stats {
                format: "text".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Rag {
            command: RagCmd::Stats {
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // Clear Cache
    run(
        Cmd::Rag {
            command: RagCmd::ClearCache,
        },
        &opts,
    )?;

    // Eval
    run(
        Cmd::Rag {
            command: RagCmd::Eval {
                samples_file: None,
                k: 2,
                format: "json".into(),
            },
        },
        &opts,
    )?;

    Ok(())
}

#[test]
fn test_repl_rag_full_session() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    // RAG INGEST
    let res = execute_repl_command(
        &db,
        &[
            "RAG".into(),
            "INGEST".into(),
            "r_doc".into(),
            "FlashStore embedded storage in Rust".into(),
        ],
    )?;
    assert!(res.unwrap().contains("OK"));

    // RAG GET
    let res = execute_repl_command(&db, &["RAG".into(), "GET".into(), "r_doc".into()])?;
    assert!(res.unwrap().contains("r_doc"));

    // RAG QUERY
    let res = execute_repl_command(
        &db,
        &[
            "RAG".into(),
            "QUERY".into(),
            "embedded storage".into(),
            "1".into(),
        ],
    )?;
    assert!(res.unwrap().contains("r_doc"));

    // RAG PROMPT
    let res = execute_repl_command(
        &db,
        &["RAG".into(), "PROMPT".into(), "storage in Rust".into()],
    )?;
    assert!(res.unwrap().contains("=== Generated RAG Prompt"));

    // RAG STATS
    let res = execute_repl_command(&db, &["RAG".into(), "STATS".into()])?;
    assert!(res.unwrap().contains("Documents:"));

    // RAG LIST
    let res = execute_repl_command(&db, &["RAG".into(), "LIST".into()])?;
    assert!(res.unwrap().contains("r_doc"));

    // RAG CACHE CLEAR
    let res = execute_repl_command(&db, &["RAG".into(), "CACHE".into(), "CLEAR".into()])?;
    assert!(res.unwrap().contains("cleared"));

    // RAG EVAL
    let res = execute_repl_command(&db, &["RAG".into(), "EVAL".into()])?;
    assert!(res.unwrap().contains("RAG IR Evaluation"));

    // RAG-DEL alias
    let res = execute_repl_command(&db, &["RAG-DEL".into(), "r_doc".into()])?;
    assert!(res.unwrap().contains("OK"));

    // Missing key check
    let res = execute_repl_command(&db, &["RAG".into(), "GET".into(), "r_doc".into()])?;
    assert!(res.unwrap().contains("(nil)"));

    Ok(())
}

#[test]
fn test_cli_rag_error_handling_and_validation() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // 1. Ingest without text, file, or batch_file -> InvalidArgument error
    let res = run(
        Cmd::Rag {
            command: RagCmd::Ingest {
                id: Some("doc_fail".into()),
                text: None,
                file: None,
                batch_file: None,
                title: None,
                author: None,
                tags: None,
                metadata: Vec::new(),
                chunk_size: None,
                chunk_overlap: None,
                chunk_strategy: "paragraphs".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 2. Query without query string or query_flag -> InvalidArgument error
    let res = run(
        Cmd::Rag {
            command: RagCmd::Query {
                query: None,
                query_flag: None,
                top_k: 5,
                dense_weight: None,
                sparse_weight: None,
                fusion: "linear".into(),
                no_rerank: false,
                mmr: false,
                mmr_lambda: 0.7,
                min_score: None,
                filters: Vec::new(),
                format: "text".into(),
                prompt: false,
                explain: false,
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 3. Get without id or doc_id -> InvalidArgument error
    let res = run(
        Cmd::Rag {
            command: RagCmd::Get {
                id: None,
                doc_id: None,
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 4. Delete without id or doc_id -> InvalidArgument error
    let res = run(
        Cmd::Rag {
            command: RagCmd::Delete {
                id: None,
                doc_id: None,
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 5. REPL invalid RAG command
    let db = open_db(&opts)?;
    let res = execute_repl_command(&db, &["RAG".into(), "INVALID_SUBCOMMAND".into()]);
    assert!(res.is_err());

    // 6. REPL missing arguments for RAG INGEST
    let res = execute_repl_command(&db, &["RAG".into(), "INGEST".into()]);
    assert!(res.is_err());

    // 7. REPL missing arguments for RAG GET
    let res = execute_repl_command(&db, &["RAG".into(), "GET".into()]);
    assert!(res.is_err());

    // 8. REPL missing arguments for RAG QUERY
    let res = execute_repl_command(&db, &["RAG".into(), "QUERY".into()]);
    assert!(res.is_err());

    // 9. REPL invalid cache subcommand
    let res = execute_repl_command(&db, &["RAG".into(), "CACHE".into(), "INVALID".into()]);
    assert!(res.is_err());

    Ok(())
}

#[test]
fn test_cli_agent_lifecycle_and_execution() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // 1. Register custom agent
    run(
        Cmd::Agent {
            command: AgentCmd::Register {
                id: "sec_auditor".into(),
                name: Some("Security Auditor".into()),
                role: "critic".into(),
                capabilities: Some("reflection,text_generation".into()),
                prompt: Some("Audit all actions for security vulnerabilities".into()),
                max_iterations: 8,
                workflow: "audit_wf".into(),
            },
        },
        &opts,
    )?;

    // 2. List agents in text, JSON, and TSV
    for fmt in ["text", "json", "tsv"] {
        run(
            Cmd::Agent {
                command: AgentCmd::Agents {
                    workflow: "audit_wf".into(),
                    format: fmt.into(),
                },
            },
            &opts,
        )?;
    }

    // 3. Run autonomous task
    run(
        Cmd::Agent {
            command: AgentCmd::Run {
                prompt: Some("Execute mathematical calculation of compound interest".into()),
                prompt_flag: None,
                id: Some("task_interest".into()),
                priority: "high".into(),
                agent: None,
                capabilities: Some("tool_calling,code_execution".into()),
                input: Some("{\"rate\": 0.05, \"principal\": 1000}".into()),
                workflow: "audit_wf".into(),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // 4. Inspect task in text and json
    run(
        Cmd::Agent {
            command: AgentCmd::GetTask {
                id: "task_interest".into(),
                workflow: "audit_wf".into(),
                format: "text".into(),
            },
        },
        &opts,
    )?;
    run(
        Cmd::Agent {
            command: AgentCmd::GetTask {
                id: "task_interest".into(),
                workflow: "audit_wf".into(),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // 5. List tasks
    run(
        Cmd::Agent {
            command: AgentCmd::Tasks {
                workflow: "audit_wf".into(),
                status: None,
                format: "tsv".into(),
            },
        },
        &opts,
    )?;

    // 6. Tools inspection & direct invocation
    run(
        Cmd::Agent {
            command: AgentCmd::Tools {
                workflow: "audit_wf".into(),
                format: "text".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Agent {
            command: AgentCmd::Call {
                tool: "calculator".into(),
                args: "{\"operation\": \"mul\", \"a\": 12, \"b\": 8}".into(),
                workflow: "audit_wf".into(),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // 7. Memory operations
    run(
        Cmd::Agent {
            command: AgentCmd::Memory {
                query: None,
                record: Some("Audited compound interest task for input validation".into()),
                agent: "sec_auditor".into(),
                importance: 0.85,
                limit: 5,
                workflow: "audit_wf".into(),
                format: "text".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Agent {
            command: AgentCmd::Memory {
                query: Some("compound interest validation".into()),
                record: None,
                agent: "sec_auditor".into(),
                importance: 0.5,
                limit: 5,
                workflow: "audit_wf".into(),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // 8. Stats & Checkpoint
    run(
        Cmd::Agent {
            command: AgentCmd::Stats {
                workflow: "audit_wf".into(),
                format: "json".into(),
            },
        },
        &opts,
    )?;

    run(
        Cmd::Agent {
            command: AgentCmd::Checkpoint {
                workflow: "audit_wf".into(),
            },
        },
        &opts,
    )?;

    // 9. Reopen pipeline and verify persisted state
    let restored_pipeline = open_agent_pipeline(&opts, "audit_wf")?;
    assert!(restored_pipeline
        .engine()
        .state()
        .active_agents
        .contains_key(&flash_store::agent::AgentId::new("sec_auditor")));

    Ok(())
}

#[test]
fn test_cli_agent_workflow_batch_file() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    let workflow_json = r#"{
        "workflow_id": "batch_analysis_dag",
        "tasks": [
            {
                "id": "step_1",
                "title": "Data Ingestion",
                "description": "Ingest sensory metrics into working store",
                "priority": "high",
                "depends_on": []
            },
            {
                "id": "step_2",
                "title": "Data Transformation",
                "description": "Calculate statistical aggregates",
                "priority": "medium",
                "depends_on": ["step_1"]
            },
            {
                "id": "step_3",
                "title": "Result Verification",
                "description": "Validate invariants and generate report",
                "priority": "critical",
                "depends_on": ["step_2"]
            }
        ]
    }"#;

    let file_path = dir.path().join("workflow_dag.json");
    fs::write(&file_path, workflow_json)?;

    // Execute batch DAG workflow
    run(
        Cmd::Agent {
            command: AgentCmd::Workflow {
                file: file_path,
                id: None,
                format: "json".into(),
            },
        },
        &opts,
    )?;

    // Verify all tasks recorded
    let pipeline = open_agent_pipeline(&opts, "batch_analysis_dag")?;
    let state = pipeline.engine().state();
    assert_eq!(state.tasks.len(), 3);
    assert!(state
        .tasks
        .contains_key(&flash_store::agent::TaskId::new("step_1")));
    assert!(state
        .tasks
        .contains_key(&flash_store::agent::TaskId::new("step_2")));
    assert!(state
        .tasks
        .contains_key(&flash_store::agent::TaskId::new("step_3")));

    Ok(())
}

#[test]
fn test_repl_agent_full_session() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    let db = open_db(&opts)?;

    // 1. HELP should list AGENT commands
    let help = execute_repl_command(&db, &["HELP".into()])?;
    assert!(help.as_ref().unwrap().contains("AGENT RUN"));
    assert!(help.as_ref().unwrap().contains("AGENT TOOLS"));

    // 2. AGENT AGENTS
    let agents_res = execute_repl_command(&db, &["AGENT".into(), "AGENTS".into()])?;
    assert!(agents_res.unwrap().contains("Active Swarm Agents"));

    // 3. AGENT TOOLS
    let tools_res = execute_repl_command(&db, &["AGENT".into(), "TOOLS".into()])?;
    assert!(tools_res.unwrap().contains("Registered Tools"));

    // 4. AGENT CALL
    let call_res = execute_repl_command(
        &db,
        &[
            "AGENT".into(),
            "CALL".into(),
            "calculator".into(),
            "{\"operation\": \"add\", \"a\": 40, \"b\": 2}".into(),
        ],
    )?;
    assert!(call_res.unwrap().contains("42"));

    // 5. AGENT RUN
    let run_res = execute_repl_command(
        &db,
        &["AGENT".into(), "RUN".into(), "Summarize memory logs".into()],
    )?;
    assert!(run_res.unwrap().contains("Completed"));

    // 6. AGENT TASKS
    let tasks_res = execute_repl_command(&db, &["AGENT".into(), "TASKS".into()])?;
    assert!(tasks_res.unwrap().contains("Workflow Tasks"));

    // 7. AGENT MEMORY RECORD & RECALL
    let mem_rec = execute_repl_command(
        &db,
        &[
            "AGENT".into(),
            "MEMORY".into(),
            "RECORD".into(),
            "System telemetry check: memory consumption optimal".into(),
        ],
    )?;
    assert!(mem_rec.unwrap().contains("OK"));

    let mem_recall = execute_repl_command(
        &db,
        &[
            "AGENT".into(),
            "MEMORY".into(),
            "consumption optimal".into(),
        ],
    )?;
    assert!(mem_recall.unwrap().contains("Recalled Memories"));

    // 8. AGENT STATS
    let stats_res = execute_repl_command(&db, &["AGENT".into(), "STATS".into()])?;
    assert!(stats_res.unwrap().contains("Swarm Telemetry"));

    // 9. AGENT CHECKPOINT
    let cp_res = execute_repl_command(&db, &["AGENT".into(), "CHECKPOINT".into()])?;
    assert!(cp_res.unwrap().contains("OK"));

    Ok(())
}

#[test]
fn test_cli_agent_error_handling_and_validation() -> Result<()> {
    let dir = tempdir().unwrap();
    let opts = GlobalOpts {
        path: dir.path().to_path_buf(),
        quiet: true,
        ..Default::default()
    };

    // 1. Run without prompt
    let res = run(
        Cmd::Agent {
            command: AgentCmd::Run {
                prompt: None,
                prompt_flag: None,
                id: None,
                priority: "medium".into(),
                agent: None,
                capabilities: None,
                input: None,
                workflow: "default".into(),
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 2. Call missing tool
    let res = run(
        Cmd::Agent {
            command: AgentCmd::Call {
                tool: "non_existent_tool_xyz".into(),
                args: "{}".into(),
                workflow: "default".into(),
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 3. Call tool with invalid JSON arguments
    let res = run(
        Cmd::Agent {
            command: AgentCmd::Call {
                tool: "calculator".into(),
                args: "not-json-string".into(),
                workflow: "default".into(),
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 4. Memory without query or record
    let res = run(
        Cmd::Agent {
            command: AgentCmd::Memory {
                query: None,
                record: None,
                agent: "default".into(),
                importance: 0.5,
                limit: 10,
                workflow: "default".into(),
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 5. Workflow with non-existent file
    let res = run(
        Cmd::Agent {
            command: AgentCmd::Workflow {
                file: PathBuf::from("non_existent_workflow_file_1234.json"),
                id: None,
                format: "text".into(),
            },
        },
        &opts,
    );
    assert!(res.is_err());

    // 6. REPL invalid agent subcommands
    let db = open_db(&opts)?;
    let res = execute_repl_command(&db, &["AGENT".into(), "UNKNOWN_SUBCOMMAND".into()]);
    assert!(res.is_err());

    let res = execute_repl_command(&db, &["AGENT".into(), "RUN".into()]);
    assert!(res.is_err());

    let res = execute_repl_command(&db, &["AGENT".into(), "GET".into()]);
    assert!(res.is_err());

    let res = execute_repl_command(&db, &["AGENT".into(), "CALL".into()]);
    assert!(res.is_err());

    let res = execute_repl_command(&db, &["AGENT".into(), "MEMORY".into()]);
    assert!(res.is_err());

    Ok(())
}
