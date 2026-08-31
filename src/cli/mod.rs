pub mod batch_parser;
pub mod formatter;
pub mod inspect;
pub mod repl;

use crate::config::OptionsBuilder;
use crate::engine::FlashStore;
use crate::error::Result;
use bytes::Bytes;
use clap::Subcommand;
use std::path::PathBuf;

/// Output format for commands that emit rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Tsv,
}

/// Global CLI options (shared by all subcommands via the binary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalOpts {
    pub path: PathBuf,
    pub memtable_size: Option<usize>,
    pub block_size: Option<usize>,
    pub sync_wal: bool,
    pub block_cache_size: Option<usize>,
    pub json: bool,
    pub quiet: bool,
}

impl Default for GlobalOpts {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./data"),
            memtable_size: None,
            block_size: None,
            sync_wal: false,
            block_cache_size: None,
            json: false,
            quiet: false,
        }
    }
}

impl GlobalOpts {
    pub fn effective_format(&self) -> OutputFormat {
        if self.json {
            OutputFormat::Json
        } else {
            OutputFormat::Text
        }
    }
}

/// All subcommands.
#[derive(Debug, Subcommand)]
pub enum Cmd {
    /// Insert a key-value pair.
    Put { key: String, value: String },
    /// Query a key.
    Get {
        key: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Delete a key.
    Delete { key: String },
    /// Range scan.
    Scan {
        /// Start key (inclusive).
        #[arg(long)]
        start: Option<String>,
        /// End key (exclusive).
        #[arg(long)]
        end: Option<String>,
        /// Max results.
        #[arg(long)]
        limit: Option<usize>,
        /// Reverse order.
        #[arg(long)]
        reverse: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Atomic batch of PUT/DEL ops from args, JSON string, or file.
    Batch {
        /// JSON array of ops: [{"op":"put","key":"k","value":"v"}, ...].
        #[arg(long, conflicts_with = "file")]
        ops: Option<String>,
        /// Path to JSON file containing ops.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Flush active memtable to SSTable.
    Flush,
    /// Trigger manual compaction.
    Compact,
    /// Print engine statistics.
    Stats {
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Inspect an SSTable file.
    Inspect {
        /// Path to .sst file (or file number, zero-padded to 6 digits).
        path: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Built-in lightweight benchmark.
    Bench {
        /// Number of operations.
        #[arg(long, default_value_t = 10_000)]
        ops: usize,
        /// Value size in bytes.
        #[arg(long, default_value_t = 64)]
        value_size: usize,
        /// Read ratio (0.0–1.0).
        #[arg(long, default_value_t = 0.5)]
        read_ratio: f64,
    },
    /// Start interactive REPL.
    Repl,
}

/// Open FlashStore from global opts.
pub fn open_db(opts: &GlobalOpts) -> Result<FlashStore> {
    let mut builder = OptionsBuilder::new().dir(opts.path.clone());
    if let Some(ms) = opts.memtable_size {
        builder = builder.memtable_size(ms);
    }
    if let Some(bs) = opts.block_size {
        builder = builder.block_size(bs);
    }
    if opts.sync_wal {
        builder = builder.sync_wal(true);
    }
    if let Some(cs) = opts.block_cache_size {
        builder = builder.block_cache_size(cs);
    }
    FlashStore::open(builder.build())
}

/// Run a subcommand with the given global opts.
pub fn run(cmd: Cmd, opts: &GlobalOpts) -> Result<()> {
    match cmd {
        Cmd::Put { key, value } => {
            let db = open_db(opts)?;
            db.put(Bytes::from(key), Bytes::from(value))?;
            if !opts.quiet {
                println!("OK");
            }
        }
        Cmd::Get { key, format } => {
            let db = open_db(opts)?;
            let fmt = if opts.json {
                OutputFormat::Json
            } else {
                format
            };
            match db.get(Bytes::from(key.clone()))? {
                Some(val) => {
                    let s = String::from_utf8_lossy(&val).into_owned();
                    match fmt {
                        OutputFormat::Json => {
                            println!(
                                "{}",
                                serde_json::to_string(&serde_json::json!({
                                    "key": key,
                                    "value": s,
                                }))
                                .unwrap()
                            );
                        }
                        OutputFormat::Tsv => println!("{}\t{}", key, s),
                        OutputFormat::Text => println!("{}", s),
                    }
                }
                None => {
                    if fmt == OutputFormat::Json {
                        println!(
                            "{}",
                            serde_json::to_string(&serde_json::json!({
                                "key": key,
                                "value": null,
                            }))
                            .unwrap()
                        );
                    } else {
                        println!("(nil)");
                    }
                }
            }
        }
        Cmd::Delete { key } => {
            let db = open_db(opts)?;
            db.delete(Bytes::from(key))?;
            if !opts.quiet {
                println!("OK");
            }
        }
        Cmd::Scan {
            start,
            end,
            limit,
            reverse,
            format,
        } => {
            let db = open_db(opts)?;
            let fmt = if opts.json {
                OutputFormat::Json
            } else {
                format
            };
            let start_key = start.map(Bytes::from);
            let end_key = end.map(Bytes::from);
            let mut results = db.scan(start_key, end_key)?;
            if reverse {
                results.reverse();
            }
            if let Some(lim) = limit {
                results.truncate(lim);
            }
            formatter::format_kv_pairs(&results, fmt, opts.quiet);
        }
        Cmd::Batch { ops, file } => {
            let db = open_db(opts)?;
            let batch = if let Some(ops_json) = ops {
                batch_parser::parse_json_ops(&ops_json)?
            } else if let Some(path) = file {
                let content = std::fs::read_to_string(&path)?;
                batch_parser::parse_json_ops(&content)?
            } else {
                // Read from stdin
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                batch_parser::parse_json_ops(&buf)?
            };
            let count = batch.len();
            db.write_batch(batch)?;
            if !opts.quiet {
                println!("Applied {} ops", count);
            }
        }
        Cmd::Flush => {
            let db = open_db(opts)?;
            db.flush()?;
            if !opts.quiet {
                println!("Flushed.");
            }
        }
        Cmd::Compact => {
            let db = open_db(opts)?;
            db.compact()?;
            if !opts.quiet {
                println!("Compaction done.");
            }
        }
        Cmd::Stats { format } => {
            let db = open_db(opts)?;
            let fmt = if opts.json {
                OutputFormat::Json
            } else {
                format
            };
            let stats = db.stats();
            formatter::format_stats(&stats, fmt);
        }
        Cmd::Inspect { path, format } => {
            let fmt = if opts.json {
                OutputFormat::Json
            } else {
                format
            };
            // Resolve path: if it looks like a number, look for it in opts.path dir
            let sst_path = if let Ok(num) = path.parse::<u64>() {
                crate::sstable::table_path(&opts.path, num)
            } else {
                PathBuf::from(&path)
            };
            inspect::inspect_sst(&sst_path, fmt)?;
        }
        Cmd::Bench {
            ops,
            value_size,
            read_ratio,
        } => {
            let db = open_db(opts)?;
            run_bench(&db, ops, value_size, read_ratio, opts.quiet)?;
        }
        Cmd::Repl => {
            let db = open_db(opts)?;
            repl::run_repl(db, opts)?;
        }
    }
    Ok(())
}

fn run_bench(
    db: &FlashStore,
    ops: usize,
    value_size: usize,
    read_ratio: f64,
    quiet: bool,
) -> Result<()> {
    use std::time::Instant;

    if ops == 0 {
        if !quiet {
            println!("Bench: 0 ops requested, nothing to run.");
        }
        return Ok(());
    }

    let value: Bytes = Bytes::from(vec![b'x'; value_size]);
    let reads = (ops as f64 * read_ratio.clamp(0.0, 1.0)) as usize;
    let writes = ops - reads;

    // Seed some data first
    let seed = writes.clamp(1, 1000);
    for i in 0..seed {
        let k = Bytes::from(format!("bench_key_{:08}", i));
        db.put(k, value.clone())?;
    }

    let start = Instant::now();
    let mut hits = 0usize;

    // Writes
    for i in 0..writes {
        let k = Bytes::from(format!("bench_key_{:08}", i));
        db.put(k, value.clone())?;
    }

    // Reads
    for i in 0..reads {
        let k = Bytes::from(format!("bench_key_{:08}", i % seed));
        if db.get(k)?.is_some() {
            hits += 1;
        }
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(0.000_001);
    let throughput = ops as f64 / secs;

    if !quiet {
        println!(
            "Bench: {} ops in {:.3}s = {:.0} ops/s",
            ops,
            elapsed.as_secs_f64(),
            throughput
        );
        println!("  Writes: {}, Reads: {} ({} hits)", writes, reads, hits);
        println!(
            "  Avg latency: {:.2} µs/op",
            elapsed.as_micros() as f64 / ops as f64
        );
    }
    Ok(())
}
