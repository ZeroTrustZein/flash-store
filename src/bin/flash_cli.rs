use clap::{Parser, Subcommand};
use flash_store::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "flash-cli", about = "CLI tool for FlashStore LSM-Tree KV engine")]
struct Cli {
    #[arg(short, long, default_value = "./data")]
    path: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Put { key: String, value: String },
    Get { key: String },
    Delete { key: String },
    Flush,
    Compact,
    Stats,
    Repl,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let options = OptionsBuilder::new().dir(cli.path).build();
    let db = FlashStore::open(options)?;

    match cli.command {
        Commands::Put { key, value } => {
            db.put(key.into_bytes(), value.into_bytes())?;
            println!("OK");
        }
        Commands::Get { key } => {
            match db.get(key.into_bytes())? {
                Some(val) => {
                    let s = String::from_utf8_lossy(&val);
                    println!("{}", s);
                }
                None => println!("(nil)"),
            }
        }
        Commands::Delete { key } => {
            db.delete(key.into_bytes())?;
            println!("OK");
        }
        Commands::Flush => {
            db.flush()?;
            println!("Flushed successfully.");
        }
        Commands::Compact => {
            db.compact()?;
            println!("Compaction triggered.");
        }
        Commands::Stats => {
            let stats = db.stats();
            println!("Active memtable bytes: {}", stats.active_memtable_size);
            println!("Immutable memtables: {}", stats.immutable_memtables_count);
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                println!("Level {}: {} SSTables", lvl, count);
            }
        }
        Commands::Repl => {
            println!("FlashStore REPL - type 'exit' to quit");
            // Placeholder REPL
        }
    }

    Ok(())
}
