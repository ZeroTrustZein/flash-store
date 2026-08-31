use crate::batch::{BatchOp, WriteBatch};
use crate::config::Options;
use crate::error::Result;
use crate::manifest::version::FileMetaData;
use crate::manifest::{Manifest, VersionSet};
use crate::memtable::MemTable;
use crate::sstable::{TableBuilder, TableReader};
use crate::types::{Entry, Key, Value};
use crate::wal::record::WalRecord;
use crate::wal::WalWriter;
use parking_lot::RwLock;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

        let wal_path = options.dir.join("current.wal");
        let wal = WalWriter::open(&wal_path, options.sync_wal)?;

        let manifest_path = options.dir.join("MANIFEST");
        let manifest = Manifest::open(&manifest_path)?;
        let version_set = VersionSet::new(options.max_levels);

        let initial_memtable = Arc::new(MemTable::new(0));

        let inner = Arc::new(EngineInner {
            options,
            memtable: RwLock::new(initial_memtable),
            imm_memtables: RwLock::new(Vec::new()),
            wal: RwLock::new(wal),
            manifest,
            version_set,
            memtable_id_seq: AtomicUsize::new(1),
        });

        Ok(Self { inner })
    }

    pub fn put(&self, key: impl Into<Key>, value: impl Into<Value>) -> Result<()> {
        let key = key.into();
        let value = value.into();
        let seq = self.inner.version_set.next_file_number();

        let record = WalRecord {
            key: key.clone(),
            value: value.clone(),
            is_delete: false,
            seq_no: seq,
        };
        self.inner.wal.write().append(&record)?;
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

        // 2. Search immutable memtables
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
        for level in &version.levels {
            for file_meta in level.iter().rev() {
                let file_path = self.inner.options.dir.join(format!("{:06}.sst", file_meta.file_number));
                if file_path.exists() {
                    let mut reader = TableReader::open(&file_path)?;
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
        let seq = self.inner.version_set.next_file_number();

        let record = WalRecord {
            key: key.clone(),
            value: bytes::Bytes::new(),
            is_delete: true,
            seq_no: seq,
        };
        self.inner.wal.write().append(&record)?;
        self.inner.memtable.read().delete(key, seq);

        self.maybe_schedule_flush()?;
        Ok(())
    }

    pub fn write_batch(&self, batch: WriteBatch) -> Result<()> {
        for op in batch.ops {
            match op {
                BatchOp::Put(k, v) => self.put(k, v)?,
                BatchOp::Delete(k) => self.delete(k)?,
            }
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        let mem = {
            let mut mem_guard = self.inner.memtable.write();
            let old_mem = mem_guard.clone();
            let new_id = self.inner.memtable_id_seq.fetch_add(1, Ordering::SeqCst);
            *mem_guard = Arc::new(MemTable::new(new_id));
            old_mem
        };

        let file_num = self.inner.version_set.next_file_number();
        let sst_path = self.inner.options.dir.join(format!("{:06}.sst", file_num));
        let mut builder = TableBuilder::new(&sst_path, self.inner.options.clone())?;

        let mut iter = mem.iter();
        let mut smallest_key = None;
        let mut largest_key = None;

        while iter.valid() {
            if let Some(entry) = iter.item() {
                if smallest_key.is_none() {
                    smallest_key = Some(entry.key.clone());
                }
                largest_key = Some(entry.key.clone());
                builder.add(entry.clone())?;
            }
            iter.next();
        }

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
            self.inner.manifest.log_edit(&edit)?;
            self.inner.version_set.log_and_apply(edit);
        }

        Ok(())
    }

    pub fn compact(&self) -> Result<()> {
        // Scaffolding manual compaction trigger
        Ok(())
    }

    pub fn scan(&self, _start_key: Option<Key>, _end_key: Option<Key>) -> Result<Vec<(Key, Value)>> {
        // Scan helper placeholder
        Ok(Vec::new())
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
