use crate::config::Options;
use crate::error::Result;
use crate::manifest::version::{FileMetaData, VersionEdit};
use crate::sstable::{TableBuilder, TableReader};
use crate::types::{Entry, Key, ValueType};
use bytes::Bytes;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CompactionTask {
    pub level: usize,
    pub input_files: Vec<FileMetaData>,
    pub target_level: usize,
    pub target_input_files: Vec<FileMetaData>,
}

pub struct Compactor {
    max_levels: usize,
    base_level_size_bytes: usize,
    l0_compaction_trigger: usize,
}

impl Compactor {
    pub fn new(max_levels: usize, base_level_size_bytes: usize) -> Self {
        Self {
            max_levels,
            base_level_size_bytes,
            l0_compaction_trigger: 4,
        }
    }

    pub fn with_l0_trigger(mut self, trigger: usize) -> Self {
        self.l0_compaction_trigger = trigger;
        self
    }

    /// Check if two key ranges [min1, max1] and [min2, max2] overlap.
    pub fn ranges_overlap(
        min1: &Bytes,
        max1: &Bytes,
        min2: &Bytes,
        max2: &Bytes,
    ) -> bool {
        !(max1 < min2 || max2 < min1)
    }

    /// Find all files in target level that overlap with the given key range.
    pub fn find_overlapping_files(
        level_files: &[FileMetaData],
        smallest: &Bytes,
        largest: &Bytes,
    ) -> Vec<FileMetaData> {
        level_files
            .iter()
            .filter(|f| Self::ranges_overlap(smallest, largest, &f.smallest_key, &f.largest_key))
            .cloned()
            .collect()
    }

    /// Pick a compaction task from the current LSM levels.
    pub fn pick_compaction(&self, levels: &[Vec<FileMetaData>]) -> Option<CompactionTask> {
        if levels.is_empty() || self.max_levels <= 1 {
            return None;
        }

        // 1. Check Level 0 file count trigger
        if let Some(l0) = levels.get(0) {
            if l0.len() >= self.l0_compaction_trigger {
                let mut min_key = &l0[0].smallest_key;
                let mut max_key = &l0[0].largest_key;
                for f in l0.iter().skip(1) {
                    if &f.smallest_key < min_key {
                        min_key = &f.smallest_key;
                    }
                    if &f.largest_key > max_key {
                        max_key = &f.largest_key;
                    }
                }

                let target_input_files = if levels.len() > 1 {
                    Self::find_overlapping_files(&levels[1], min_key, max_key)
                } else {
                    Vec::new()
                };

                return Some(CompactionTask {
                    level: 0,
                    input_files: l0.clone(),
                    target_level: 1,
                    target_input_files,
                });
            }
        }

        // 2. Check Level 1..N size triggers
        for level_idx in 1..levels.len() {
            if level_idx + 1 >= self.max_levels {
                continue;
            }
            let files = &levels[level_idx];
            if files.is_empty() {
                continue;
            }

            let level_size: u64 = files.iter().map(|f| f.file_size).sum();
            let limit = (self.base_level_size_bytes * (10_usize.pow((level_idx - 1) as u32))) as u64;

            if level_size > limit {
                let chosen_file = files[0].clone();
                let next_level = level_idx + 1;
                let target_input_files = if next_level < levels.len() {
                    Self::find_overlapping_files(
                        &levels[next_level],
                        &chosen_file.smallest_key,
                        &chosen_file.largest_key,
                    )
                } else {
                    Vec::new()
                };

                return Some(CompactionTask {
                    level: level_idx,
                    input_files: vec![chosen_file],
                    target_level: next_level,
                    target_input_files,
                });
            }
        }

        None
    }

    /// Execute the compaction task, merge-sort all input SSTables, resolve duplicate keys,
    /// write new SSTables, and produce a VersionEdit.
    pub fn run_compaction<F>(
        &self,
        task: &CompactionTask,
        dir: &Path,
        options: &Options,
        mut next_file_number: F,
    ) -> Result<VersionEdit>
    where
        F: FnMut() -> u64,
    {
        // 1. Gather all entries from all source and target input files
        let mut all_files = Vec::new();
        for f in &task.input_files {
            all_files.push(dir.join(format!("{:06}.sst", f.file_number)));
        }
        for f in &task.target_input_files {
            all_files.push(dir.join(format!("{:06}.sst", f.file_number)));
        }

        // Map key -> (highest seq_no, Entry)
        let mut key_map: BTreeMap<Key, Entry> = BTreeMap::new();

        for path in all_files {
            if path.exists() {
                let mut reader = TableReader::open(&path)?;
                let entries = reader.read_all_entries()?;
                for entry in entries {
                    match key_map.get(&entry.key) {
                        Some(existing) => {
                            if entry.seq_no >= existing.seq_no {
                                key_map.insert(entry.key.clone(), entry);
                            }
                        }
                        None => {
                            key_map.insert(entry.key.clone(), entry);
                        }
                    }
                }
            }
        }

        let is_bottom_level = task.target_level >= self.max_levels - 1;

        // Filter out tombstones if this is the bottom-most level
        let mut final_entries = Vec::new();
        for (_, entry) in key_map {
            if is_bottom_level && entry.value_type == ValueType::Tombstone {
                continue;
            }
            final_entries.push(entry);
        }

        let mut edit = VersionEdit::new();

        // Register deleted files
        for f in &task.input_files {
            edit.delete_file(task.level, f.file_number);
        }
        for f in &task.target_input_files {
            edit.delete_file(task.target_level, f.file_number);
        }

        // Write new SSTable(s) if there are surviving entries
        if !final_entries.is_empty() {
            let file_num = next_file_number();
            let sst_path = dir.join(format!("{:06}.sst", file_num));
            let mut builder = TableBuilder::new(&sst_path, options.clone())?;

            let smallest_key = final_entries.first().unwrap().key.clone();
            let largest_key = final_entries.last().unwrap().key.clone();

            for entry in final_entries {
                builder.add(entry)?;
            }

            let file_size = builder.finish()?;
            let meta = FileMetaData {
                file_number: file_num,
                file_size,
                smallest_key,
                largest_key,
            };

            edit.add_file(task.target_level, meta);
        }

        Ok(edit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_ranges_overlap() {
        let a = Bytes::from_static(b"a");
        let c = Bytes::from_static(b"c");
        let d = Bytes::from_static(b"d");
        let f = Bytes::from_static(b"f");

        // [a, d] and [c, f] overlap
        assert!(Compactor::ranges_overlap(&a, &d, &c, &f));
        // [a, c] and [d, f] do not overlap
        assert!(!Compactor::ranges_overlap(&a, &c, &d, &f));
    }

    #[test]
    fn test_pick_compaction_l0_trigger() {
        let compactor = Compactor::new(7, 10 * 1024 * 1024).with_l0_trigger(4);
        let mut levels = vec![Vec::new(); 7];

        for i in 1..=4 {
            levels[0].push(FileMetaData {
                file_number: i,
                file_size: 1000,
                smallest_key: Bytes::from(format!("k{}", i)),
                largest_key: Bytes::from(format!("k{}", i + 10)),
            });
        }

        let task = compactor.pick_compaction(&levels);
        assert!(task.is_some());
        let task = task.unwrap();
        assert_eq!(task.level, 0);
        assert_eq!(task.target_level, 1);
        assert_eq!(task.input_files.len(), 4);
    }

    #[test]
    fn test_run_compaction_merge_and_dedup() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).block_size(64).build();

        // Create L0 SSTable with key "k1" (seq 1, value "v1_old") and "k2" (seq 2, value "v2")
        let sst1_path = dir.path().join("000001.sst");
        let mut b1 = TableBuilder::new(&sst1_path, options.clone())?;
        b1.add(Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1_old"), 1))?;
        b1.add(Entry::new_value(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"), 2))?;
        let size1 = b1.finish()?;

        // Create L1 SSTable with key "k1" (seq 5, value "v1_new") and "k3" (seq 3, value "v3")
        let sst2_path = dir.path().join("000002.sst");
        let mut b2 = TableBuilder::new(&sst2_path, options.clone())?;
        b2.add(Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1_new"), 5))?;
        b2.add(Entry::new_value(Bytes::from_static(b"k3"), Bytes::from_static(b"v3"), 3))?;
        let size2 = b2.finish()?;

        let task = CompactionTask {
            level: 0,
            input_files: vec![FileMetaData {
                file_number: 1,
                file_size: size1,
                smallest_key: Bytes::from_static(b"k1"),
                largest_key: Bytes::from_static(b"k2"),
            }],
            target_level: 1,
            target_input_files: vec![FileMetaData {
                file_number: 2,
                file_size: size2,
                smallest_key: Bytes::from_static(b"k1"),
                largest_key: Bytes::from_static(b"k3"),
            }],
        };

        let compactor = Compactor::new(7, 10 * 1024 * 1024);
        let mut next_num = 10;
        let edit = compactor.run_compaction(&task, dir.path(), &options, || {
            let n = next_num;
            next_num += 1;
            n
        })?;

        assert_eq!(edit.deleted_files.len(), 2);
        assert_eq!(edit.new_files.len(), 1);
        assert_eq!(edit.new_files[0].0, 1);
        assert_eq!(edit.new_files[0].1.file_number, 10);

        // Verify output SSTable contents
        let out_path = dir.path().join("000010.sst");
        let mut reader = TableReader::open(&out_path)?;
        let entries = reader.read_all_entries()?;
        assert_eq!(entries.len(), 3);
        // k1 should have value "v1_new" (seq 5)
        assert_eq!(entries[0].key, Bytes::from_static(b"k1"));
        assert_eq!(entries[0].value, Bytes::from_static(b"v1_new"));
        assert_eq!(entries[0].seq_no, 5);

        assert_eq!(entries[1].key, Bytes::from_static(b"k2"));
        assert_eq!(entries[2].key, Bytes::from_static(b"k3"));

        Ok(())
    }

    #[test]
    fn test_compactor_tombstone_purged_at_bottom_level() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).block_size(64).build();

        let sst1_path = dir.path().join("000001.sst");
        let mut b1 = TableBuilder::new(&sst1_path, options.clone())?;
        b1.add(Entry::new_tombstone(Bytes::from_static(b"dead_key"), 10))?;
        let size1 = b1.finish()?;

        let task = CompactionTask {
            level: 5,
            input_files: vec![FileMetaData {
                file_number: 1,
                file_size: size1,
                smallest_key: Bytes::from_static(b"dead_key"),
                largest_key: Bytes::from_static(b"dead_key"),
            }],
            target_level: 6, // bottom level (max_levels = 7)
            target_input_files: Vec::new(),
        };

        let compactor = Compactor::new(7, 10 * 1024 * 1024);
        let edit = compactor.run_compaction(&task, dir.path(), &options, || 2)?;

        // Deleted file should be registered, but no new file generated because tombstone was purged!
        assert_eq!(edit.deleted_files.len(), 1);
        assert_eq!(edit.new_files.len(), 0);

        Ok(())
    }
}
