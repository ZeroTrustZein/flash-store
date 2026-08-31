use crate::batch::{BatchOp, WriteBatch};
use crate::cache::{BlockCache, LruBlockCache};
use crate::compaction::Compactor;
use crate::config::Options;
use crate::error::Result;
use crate::manifest::version::FileMetaData;
use crate::manifest::{Manifest, VersionSet};
use crate::memtable::MemTable;
use crate::sstable::{TableBuilder, TableReader};
use crate::types::{Entry, Key, Value, ValueType};
use crate::wal::record::WalRecord;
use crate::wal::{WalReader, WalWriter};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Stats {
    pub active_memtable_size: usize,
    pub immutable_memtables_count: usize,
    pub levels_file_count: Vec<usize>,
}

struct EngineInner {
    options: Options,
    memtable: RwLock<Arc<MemTable>>,
    imm_memtables: RwLock<Vec<Arc<MemTable>>>,
    wal: RwLock<WalWriter>,
    manifest: Manifest,
    version_set: VersionSet,
    block_cache: Arc<dyn BlockCache>,
    memtable_id_seq: AtomicUsize,
}

#[derive(Clone)]
pub struct FlashStore {
    inner: Arc<EngineInner>,
}

impl FlashStore {
    pub fn open(options: Options) -> Result<Self> {
        if !options.dir.exists() {
            if options.create_if_missing {
                fs::create_dir_all(&options.dir)?;
            } else {
                return Err(crate::error::FlashStoreError::InvalidArgument(
                    "Database directory does not exist".into(),
                ));
            }
        }

        let manifest_path = options.dir.join("MANIFEST");
        let edits = Manifest::read_all_edits(&manifest_path)?;
        let manifest = Manifest::open(&manifest_path)?;
        let version_set = VersionSet::new(options.max_levels);

        for edit in edits {
            version_set.log_and_apply(edit);
        }

        let wal_path = options.dir.join("current.wal");
        let initial_memtable = Arc::new(MemTable::new(0));
        let mut max_seq = version_set.last_sequence();

        if wal_path.exists() {
            if let Ok(mut wal_reader) = WalReader::open(&wal_path) {
                if let Ok(records) = wal_reader.read_all() {
                    for record in records {
                        if record.seq_no > max_seq {
                            max_seq = record.seq_no;
                        }
                        if record.is_delete {
                            initial_memtable.delete(record.key, record.seq_no);
                        } else {
                            initial_memtable.put(record.key, record.value, record.seq_no);
                        }
                    }
                }
            }
        }
        version_set.set_last_sequence(max_seq);

        let wal = WalWriter::open(&wal_path, options.sync_wal)?;
        let cache_capacity = (options.block_cache_size / options.block_size.max(1)).max(16);
        let block_cache = Arc::new(LruBlockCache::new(cache_capacity));

        let inner = Arc::new(EngineInner {
            options,
            memtable: RwLock::new(initial_memtable),
            imm_memtables: RwLock::new(Vec::new()),
            wal: RwLock::new(wal),
            manifest,
            version_set,
            block_cache,
            memtable_id_seq: AtomicUsize::new(1),
        });

        Ok(Self { inner })
    }

    pub fn put(&self, key: impl Into<Key>, value: impl Into<Value>) -> Result<()> {
        let key = key.into();
        let value = value.into();
        let seq = self.inner.version_set.next_sequence();

        let record = WalRecord {
            key: key.clone(),
            value: value.clone(),
            is_delete: false,
            seq_no: seq,
        };
        self.inner.wal.read().append(&record)?;
        self.inner.memtable.read().put(key, value, seq);

        self.maybe_schedule_flush()?;
        Ok(())
    }

    pub fn get(&self, key: impl Into<Key>) -> Result<Option<Value>> {
        let key = key.into();

        // 1. Search active memtable
        if let Some(res) = self.inner.memtable.read().get(&key) {
            return Ok(res);
        }

        // 2. Search immutable memtables (newest to oldest)
        {
            let imm = self.inner.imm_memtables.read();
            for mem in imm.iter().rev() {
                if let Some(res) = mem.get(&key) {
                    return Ok(res);
                }
            }
        }

        // 3. Search SSTable levels
        let version = self.inner.version_set.current();

        // Level 0: search files in reverse order (newest to oldest)
        if let Some(l0) = version.levels.get(0) {
            for file_meta in l0.iter().rev() {
                let file_path = self
                    .inner
                    .options
                    .dir
                    .join(format!("{:06}.sst", file_meta.file_number));
                if file_path.exists() {
                    let mut reader = TableReader::open_with_cache(
                        &file_path,
                        file_meta.file_number,
                        Some(self.inner.block_cache.clone()),
                    )?;
                    if let Some(res) = reader.get(&key)? {
                        return Ok(res);
                    }
                }
            }
        }

        // Level 1..N: files are non-overlapping and sorted by key range
        for level in version.levels.iter().skip(1) {
            for file_meta in level {
                if &key < &file_meta.smallest_key || &key > &file_meta.largest_key {
                    continue;
                }
                let file_path = self
                    .inner
                    .options
                    .dir
                    .join(format!("{:06}.sst", file_meta.file_number));
                if file_path.exists() {
                    let mut reader = TableReader::open_with_cache(
                        &file_path,
                        file_meta.file_number,
                        Some(self.inner.block_cache.clone()),
                    )?;
                    if let Some(res) = reader.get(&key)? {
                        return Ok(res);
                    }
                }
            }
        }

        Ok(None)
    }

    pub fn delete(&self, key: impl Into<Key>) -> Result<()> {
        let key = key.into();
        let seq = self.inner.version_set.next_sequence();

        let record = WalRecord {
            key: key.clone(),
            value: bytes::Bytes::new(),
            is_delete: true,
            seq_no: seq,
        };
        self.inner.wal.read().append(&record)?;
        self.inner.memtable.read().delete(key, seq);

        self.maybe_schedule_flush()?;
        Ok(())
    }

    pub fn write_batch(&self, batch: WriteBatch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut records = Vec::with_capacity(batch.len());
        let mut ops = Vec::with_capacity(batch.len());

        for op in batch.ops {
            let seq = self.inner.version_set.next_sequence();
            match op {
                BatchOp::Put(k, v) => {
                    records.push(WalRecord {
                        key: k.clone(),
                        value: v.clone(),
                        is_delete: false,
                        seq_no: seq,
                    });
                    ops.push((k, Some(v), seq));
                }
                BatchOp::Delete(k) => {
                    records.push(WalRecord {
                        key: k.clone(),
                        value: bytes::Bytes::new(),
                        is_delete: true,
                        seq_no: seq,
                    });
                    ops.push((k, None, seq));
                }
            }
        }

        self.inner.wal.read().append_batch(&records)?;

        let memtable = self.inner.memtable.read();
        for (k, v_opt, seq) in ops {
            match v_opt {
                Some(v) => memtable.put(k, v, seq),
                None => memtable.delete(k, seq),
            }
        }

        self.maybe_schedule_flush()?;
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        let old_mem = {
            let mut mem_guard = self.inner.memtable.write();
            if mem_guard.approximate_size() == 0 && mem_guard.is_empty() {
                return Ok(());
            }
            let old = mem_guard.clone();
            let new_id = self.inner.memtable_id_seq.fetch_add(1, Ordering::SeqCst);
            *mem_guard = Arc::new(MemTable::new(new_id));
            self.inner.imm_memtables.write().push(old.clone());
            old
        };

        // Reset WAL for new active memtable
        self.inner.wal.write().reset()?;

        let file_num = self.inner.version_set.next_file_number();
        let sst_path = self
            .inner
            .options
            .dir
            .join(format!("{:06}.sst", file_num));
        let mut builder = TableBuilder::new(&sst_path, self.inner.options.clone())?;

        let mut iter = old_mem.iter();
        let mut smallest_key = None;
        let mut largest_key = None;
        let mut count = 0;

        while iter.valid() {
            if let Some(entry) = iter.item() {
                if smallest_key.is_none() {
                    smallest_key = Some(entry.key.clone());
                }
                largest_key = Some(entry.key.clone());
                builder.add(entry.clone())?;
                count += 1;
            }
            iter.next();
        }

        if count > 0 {
            let file_size = builder.finish()?;
            if let (Some(smallest_key), Some(largest_key)) = (smallest_key, largest_key) {
                let meta = FileMetaData {
                    file_number: file_num,
                    file_size,
                    smallest_key,
                    largest_key,
                };

                let mut edit = crate::manifest::version::VersionEdit::new();
                edit.add_file(0, meta);
                edit.last_sequence = Some(self.inner.version_set.last_sequence());
                edit.next_file_number = Some(self.inner.version_set.next_file_number());
                self.inner.manifest.log_edit(&edit)?;
                self.inner.version_set.log_and_apply(edit);
            }
        }

        self.inner
            .imm_memtables
            .write()
            .retain(|m| m.id() != old_mem.id());

        Ok(())
    }

    pub fn compact(&self) -> Result<()> {
        let compactor = Compactor::new(
            self.inner.options.max_levels,
            self.inner.options.base_level_size_bytes,
        );

        // Run compaction passes as long as pick_compaction returns a task
        for _ in 0..100 {
            let version = self.inner.version_set.current();
            let task = match compactor.pick_compaction(&version.levels) {
                Some(t) => t,
                None => break,
            };

            let edit = compactor.run_compaction(
                &task,
                &self.inner.options.dir,
                &self.inner.options,
                || self.inner.version_set.next_file_number(),
            )?;

            self.inner.manifest.log_edit(&edit)?;
            self.inner.version_set.log_and_apply(edit.clone());

            // Remove obsolete SSTable files from disk
            for (_, file_number) in edit.deleted_files {
                let path = self
                    .inner
                    .options
                    .dir
                    .join(format!("{:06}.sst", file_number));
                if path.exists() {
                    let _ = fs::remove_file(path);
                }
            }
        }

        Ok(())
    }

    pub fn scan(&self, start_key: Option<Key>, end_key: Option<Key>) -> Result<Vec<(Key, Value)>> {
        let mut merged: BTreeMap<Key, (u64, ValueType, Value)> = BTreeMap::new();

        let mut update_entry = |entry: Entry| {
            if let Some((existing_seq, _, _)) = merged.get(&entry.key) {
                if entry.seq_no > *existing_seq {
                    merged.insert(entry.key, (entry.seq_no, entry.value_type, entry.value));
                }
            } else {
                merged.insert(entry.key, (entry.seq_no, entry.value_type, entry.value));
            }
        };

        // 1. Read SSTables
        let version = self.inner.version_set.current();
        for level in &version.levels {
            for file_meta in level {
                let file_path = self
                    .inner
                    .options
                    .dir
                    .join(format!("{:06}.sst", file_meta.file_number));
                if file_path.exists() {
                    let mut reader = TableReader::open_with_cache(
                        &file_path,
                        file_meta.file_number,
                        Some(self.inner.block_cache.clone()),
                    )?;
                    let entries = reader.read_all_entries()?;
                    for entry in entries {
                        update_entry(entry);
                    }
                }
            }
        }

        // 2. Read Immutable Memtables
        {
            let imm = self.inner.imm_memtables.read();
            for mem in imm.iter() {
                let mut iter = mem.iter();
                while iter.valid() {
                    if let Some(entry) = iter.item() {
                        update_entry(entry.clone());
                    }
                    iter.next();
                }
            }
        }

        // 3. Read Active Memtable
        {
            let mem = self.inner.memtable.read();
            let mut iter = mem.iter();
            while iter.valid() {
                if let Some(entry) = iter.item() {
                    update_entry(entry.clone());
                }
                iter.next();
            }
        }

        let mut results = Vec::new();
        for (key, (_, value_type, value)) in merged {
            if value_type == ValueType::Tombstone {
                continue;
            }
            if let Some(ref start) = start_key {
                if &key < start {
                    continue;
                }
            }
            if let Some(ref end) = end_key {
                if &key >= end {
                    continue;
                }
            }
            results.push((key, value));
        }

        Ok(results)
    }

    pub fn stats(&self) -> Stats {
        let version = self.inner.version_set.current();
        Stats {
            active_memtable_size: self.inner.memtable.read().approximate_size(),
            immutable_memtables_count: self.inner.imm_memtables.read().len(),
            levels_file_count: version.levels.iter().map(|lvl| lvl.len()).collect(),
        }
    }

    pub fn close(&self) -> Result<()> {
        self.inner.wal.write().sync()?;
        Ok(())
    }

    fn maybe_schedule_flush(&self) -> Result<()> {
        let size = self.inner.memtable.read().approximate_size();
        if size >= self.inner.options.memtable_size {
            self.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_engine_empty_flush() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let db = FlashStore::open(options)?;

        // Flushing empty memtable should succeed as a no-op
        db.flush()?;
        let stats = db.stats();
        assert_eq!(stats.levels_file_count[0], 0);

        Ok(())
    }

    #[test]
    fn test_engine_delete_non_existent() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let db = FlashStore::open(options)?;

        db.delete(b"non_existent")?;
        assert_eq!(db.get(b"non_existent")?, None);

        Ok(())
    }

    #[test]
    fn test_engine_multiple_flushes_and_scan() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let db = FlashStore::open(options)?;

        db.put(b"k1", b"v1")?;
        db.flush()?;

        db.put(b"k2", b"v2")?;
        db.flush()?;

        db.put(b"k3", b"v3")?;
        db.flush()?;

        let stats = db.stats();
        assert_eq!(stats.levels_file_count[0], 3);

        let entries = db.scan(None, None)?;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, bytes::Bytes::from_static(b"k1"));
        assert_eq!(entries[1].0, bytes::Bytes::from_static(b"k2"));
        assert_eq!(entries[2].0, bytes::Bytes::from_static(b"k3"));

        Ok(())
    }

    #[test]
    fn test_engine_compaction_and_data_integrity() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new()
            .dir(dir.path())
            .block_size(64)
            .build();
        let db = FlashStore::open(options)?;

        // Produce 4 SSTables in L0
        for i in 1..=4 {
            db.put(format!("key_{}", i), format!("val_{}", i))?;
            db.flush()?;
        }

        let stats_before = db.stats();
        assert_eq!(stats_before.levels_file_count[0], 4);

        // Run compaction: L0 files should be compacted into L1
        db.compact()?;

        let stats_after = db.stats();
        assert_eq!(stats_after.levels_file_count[0], 0);
        assert_eq!(stats_after.levels_file_count[1], 1);

        // Verify all keys remain accessible
        for i in 1..=4 {
            let val = db.get(format!("key_{}", i))?;
            assert_eq!(val, Some(bytes::Bytes::from(format!("val_{}", i))));
        }

        Ok(())
    }

    #[test]
    fn test_engine_wal_recovery_and_persistence() -> Result<()> {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();

        {
            let db = FlashStore::open(options.clone())?;
            db.put(b"persist_k1", b"persist_v1")?;
            db.put(b"persist_k2", b"persist_v2")?;
            db.delete(b"persist_k1")?;
            db.close()?;
        }

        // Reopen from same directory
        let db_recovered = FlashStore::open(options)?;
        assert_eq!(db_recovered.get(b"persist_k1")?, None);
        assert_eq!(
            db_recovered.get(b"persist_k2")?,
            Some(bytes::Bytes::from_static(b"persist_v2"))
        );

        Ok(())
    }
}
