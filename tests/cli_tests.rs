use bytes::Bytes;
use flash_store::cli::batch_parser::{parse_json_ops, parse_plain_lines};
use flash_store::cli::formatter::{format_kv_pairs, format_stats};
use flash_store::cli::inspect::inspect_sst;
use flash_store::cli::repl::{execute_repl_command, tokenize_line};
use flash_store::cli::{open_db, run, Cmd, GlobalOpts, OutputFormat};
use flash_store::config::OptionsBuilder;
use flash_store::engine::{FlashStore, Stats};
use flash_store::error::Result;
use std::fs;
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
    assert_eq!(db.get(Bytes::from_static(b"cli_key_1"))?, None);

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
        db.put(format!("k_{}", i).into_bytes(), format!("v_{}", i).into_bytes())?;
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

    let sst_path = dir.path().join("000001.sst");
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
    assert_eq!(
        tokenize_line("SET key value")?,
        vec!["SET", "key", "value"]
    );

    // Multiple spaces
    assert_eq!(
        tokenize_line("  PUT    k    v   ")?,
        vec!["PUT", "k", "v"]
    );

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
    assert_eq!(execute_repl_command(&db, &["EXIT".into()])?, Some("BYE".into()));
    assert_eq!(execute_repl_command(&db, &["QUIT".into()])?, Some("BYE".into()));
    assert_eq!(execute_repl_command(&db, &["Q".into()])?, Some("BYE".into()));

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
