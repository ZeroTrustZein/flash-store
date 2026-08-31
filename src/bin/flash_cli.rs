use clap::Parser;
use flash_store::cli::{Cmd, GlobalOpts};
use flash_store::error::Result;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "flash-cli",
    about = "Production-grade CLI & REPL for FlashStore LSM-Tree Key-Value Engine",
    version
)]
struct Cli {
    /// Path to the FlashStore data directory
    #[arg(short, long, default_value = "./data")]
    path: PathBuf,

    /// Override memtable size threshold in bytes
    #[arg(long)]
    memtable_size: Option<usize>,

    /// Override SSTable block size in bytes
    #[arg(long)]
    block_size: Option<usize>,

    /// Enable synchronous WAL writes
    #[arg(long)]
    sync_wal: bool,

    /// Override block cache size in bytes
    #[arg(long)]
    block_cache_size: Option<usize>,

    /// Emit JSON output where applicable
    #[arg(long)]
    json: bool,

    /// Suppress verbose / informational output
    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Cmd,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let opts = GlobalOpts {
        path: cli.path,
        memtable_size: cli.memtable_size,
        block_size: cli.block_size,
        sync_wal: cli.sync_wal,
        block_cache_size: cli.block_cache_size,
        json: cli.json,
        quiet: cli.quiet,
    };

    flash_store::cli::run(cli.command, &opts)
}
