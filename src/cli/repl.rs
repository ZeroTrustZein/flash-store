use crate::cli::GlobalOpts;
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};
use bytes::Bytes;
use std::io::{self, BufRead, Write};
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
        "HELP" | "?" => {
            let help = r#"Available REPL commands:
  PUT <key> <value>            Insert or update key-value pair
  GET <key>                    Retrieve value by key
  DEL <key>                    Delete key
  SCAN [start] [end] [limit]   Range scan entries
  FLUSH                        Flush active memtable to L0 SSTable
  COMPACT                      Trigger manual background compaction
  STATS                        Display engine statistics
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

        Ok(())
    }
}
