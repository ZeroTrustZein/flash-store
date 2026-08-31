use crate::cli::OutputFormat;
use crate::error::{FlashStoreError, Result};
use crate::sstable::reader::TableReader;
use serde_json::json;
use std::fs::File;
use std::path::Path;

/// Inspect an SSTable file and display its internal structure and metadata.
pub fn inspect_sst<P: AsRef<Path>>(path: P, format: OutputFormat) -> Result<()> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        return Err(FlashStoreError::InvalidArgument(format!(
            "SSTable file not found: {}",
            path_ref.display()
        )));
    }

    let file = File::open(path_ref)?;
    let file_size = file.metadata()?.len();

    let mut reader = TableReader::open(path_ref)?;
    let entries = reader.read_all_entries()?;

    let total_entries = entries.len();
    let value_count = entries.iter().filter(|e| e.is_value()).count();
    let tombstone_count = entries.iter().filter(|e| e.is_tombstone()).count();

    let smallest_key = entries
        .first()
        .map(|e| String::from_utf8_lossy(&e.key).into_owned());
    let largest_key = entries
        .last()
        .map(|e| String::from_utf8_lossy(&e.key).into_owned());

    match format {
        OutputFormat::Json => {
            let json_out = json!({
                "file_path": path_ref.to_string_lossy(),
                "file_size_bytes": file_size,
                "total_entries": total_entries,
                "values": value_count,
                "tombstones": tombstone_count,
                "smallest_key": smallest_key,
                "largest_key": largest_key,
                "entries_preview": entries.iter().take(10).map(|e| {
                    json!({
                        "key": String::from_utf8_lossy(&e.key),
                        "type": format!("{}", e.value_type),
                        "seq_no": e.seq_no,
                        "value_len": e.value.len(),
                    })
                }).collect::<Vec<_>>(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json_out).unwrap_or_else(|_| "{}".into())
            );
        }
        OutputFormat::Tsv | OutputFormat::Text => {
            println!("=== SSTable Inspection: {} ===", path_ref.display());
            println!("File Size:        {} bytes", file_size);
            println!("Total Entries:    {}", total_entries);
            println!("  Values:         {}", value_count);
            println!("  Tombstones:     {}", tombstone_count);
            println!(
                "Smallest Key:     {}",
                smallest_key.unwrap_or_else(|| "(none)".into())
            );
            println!(
                "Largest Key:      {}",
                largest_key.unwrap_or_else(|| "(none)".into())
            );
            println!("\n--- Entries Preview (first 10) ---");
            for (idx, entry) in entries.iter().take(10).enumerate() {
                println!(
                    "  [{:03}] [{}] key='{}', seq={}, val_len={}",
                    idx,
                    entry.value_type,
                    String::from_utf8_lossy(&entry.key),
                    entry.seq_no,
                    entry.value.len()
                );
            }
            if total_entries > 10 {
                println!("  ... and {} more entries", total_entries - 10);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use crate::sstable::TableBuilder;
    use crate::types::Entry;
    use bytes::Bytes;
    use tempfile::tempdir;

    #[test]
    fn test_inspect_sst_execution() -> Result<()> {
        let dir = tempdir().unwrap();
        let sst_path = crate::sstable::table_path(dir.path(), 1);
        let options = OptionsBuilder::new().block_size(64).build();

        let mut builder = TableBuilder::new(&sst_path, options)?;
        builder.add(Entry::new_value(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"1"),
            1,
        ))?;
        builder.add(Entry::new_tombstone(Bytes::from_static(b"b"), 2))?;
        builder.finish()?;

        inspect_sst(&sst_path, OutputFormat::Text)?;
        inspect_sst(&sst_path, OutputFormat::Json)?;

        let non_existent = dir.path().join("999999.sst");
        assert!(inspect_sst(&non_existent, OutputFormat::Text).is_err());

        Ok(())
    }
}
