pub mod version;

use crate::error::Result;
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use version::{FileMetaData, Version, VersionEdit};

pub struct Manifest {
    file: Arc<parking_lot::Mutex<File>>,
    path: PathBuf,
}

impl Manifest {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            file: Arc::new(parking_lot::Mutex::new(file)),
            path: path.as_ref().to_path_buf(),
        })
    }

    pub fn log_edit(&self, edit: &VersionEdit) -> Result<()> {
        let encoded = bincode::serialize(edit)
            .map_err(|e| crate::error::FlashStoreError::Other(e.to_string()))?;
        let len = (encoded.len() as u32).to_le_bytes();
        let mut file = self.file.lock();
        file.write_all(&len)?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        Ok(())
    }

    pub fn read_all_edits<P: AsRef<Path>>(path: P) -> Result<Vec<VersionEdit>> {
        if !path.as_ref().exists() {
            return Ok(Vec::new());
        }
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut edits = Vec::new();
        let mut offset = 0;
        while offset < buffer.len() {
            if offset + 4 > buffer.len() {
                break;
            }
            let len = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + len > buffer.len() {
                break;
            }
            let edit: VersionEdit = bincode::deserialize(&buffer[offset..offset + len])
                .map_err(|e| crate::error::FlashStoreError::Other(e.to_string()))?;
            edits.push(edit);
            offset += len;
        }

        Ok(edits)
    }
}

pub struct VersionSet {
    current: RwLock<Arc<Version>>,
    next_file_number: AtomicU64,
    last_sequence: AtomicU64,
    max_levels: usize,
}

impl VersionSet {
    pub fn new(max_levels: usize) -> Self {
        Self {
            current: RwLock::new(Arc::new(Version::new(max_levels))),
            next_file_number: AtomicU64::new(1),
            last_sequence: AtomicU64::new(0),
            max_levels,
        }
    }

    pub fn current(&self) -> Arc<Version> {
        self.current.read().clone()
    }

    pub fn next_file_number(&self) -> u64 {
        self.next_file_number.fetch_add(1, Ordering::SeqCst)
    }

    pub fn last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::SeqCst)
    }

    pub fn set_last_sequence(&self, seq: u64) {
        self.last_sequence.store(seq, Ordering::SeqCst);
    }

    pub fn log_and_apply(&self, edit: VersionEdit) {
        let mut current_levels = self.current.read().levels.clone();

        for (level, file_number) in edit.deleted_files {
            if level < current_levels.len() {
                current_levels[level].retain(|f| f.file_number != file_number);
            }
        }

        for (level, meta) in edit.new_files {
            if level < current_levels.len() {
                current_levels[level].push(meta);
            }
        }

        if let Some(next) = edit.next_file_number {
            self.next_file_number.store(next, Ordering::SeqCst);
        }
        if let Some(last_seq) = edit.last_sequence {
            self.last_sequence.store(last_seq, Ordering::SeqCst);
        }

        *self.current.write() = Arc::new(Version {
            levels: current_levels,
        });
    }
}
