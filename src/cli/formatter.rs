use crate::cli::OutputFormat;
use crate::engine::Stats;
use crate::types::{Key, Value};
use serde_json::json;

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
            println!("{}", serde_json::to_string_pretty(&json_array).unwrap_or_else(|_| "[]".into()));
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
                println!("{}: {}", String::from_utf8_lossy(k), String::from_utf8_lossy(v));
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
            println!("{}", serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "{}".into()));
        }
        OutputFormat::Tsv => {
            println!("metric\tvalue");
            println!("active_memtable_size_bytes\t{}", stats.active_memtable_size);
            println!("immutable_memtables_count\t{}", stats.immutable_memtables_count);
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                println!("level_{}_sstable_count\t{}", lvl, count);
            }
        }
        OutputFormat::Text => {
            println!("=== FlashStore Statistics ===");
            println!("Active Memtable Size:     {} bytes", stats.active_memtable_size);
            println!("Immutable Memtables:      {}", stats.immutable_memtables_count);
            let total_sst: usize = stats.levels_file_count.iter().sum();
            println!("Total SSTable Files:      {}", total_sst);
            println!("Levels Distribution:");
            for (lvl, count) in stats.levels_file_count.iter().enumerate() {
                println!("  Level {}: {} SSTable(s)", lvl, count);
            }
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
}
