pub mod version;

use crate::error::Result;
use parking_lot::{Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use version::{Version, VersionEdit};

pub struct Manifest {
    file: Mutex<File>,
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
            file: Mutex::new(file),
            path: path.as_ref().to_path_buf(),
        })
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
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
        while offset + 4 <= buffer.len() {
            let len = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + len > buffer.len() {
                break;
            }
            match bincode::deserialize::<VersionEdit>(&buffer[offset..offset + len]) {
                Ok(edit) => {
                    edits.push(edit);
                    offset += len;
                }
                Err(_) => {
                    // Stop gracefully at corrupted trailing record
                    break;
                }
            }
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

    #[inline]
    pub fn current(&self) -> Arc<Version> {
        self.current.read().clone()
    }

    #[inline]
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    #[inline]
    pub fn next_file_number(&self) -> u64 {
        self.next_file_number.fetch_add(1, Ordering::SeqCst)
    }

    #[inline]
    pub fn set_next_file_number(&self, file_number: u64) {
        self.next_file_number.store(file_number, Ordering::SeqCst);
    }

    #[inline]
    pub fn next_sequence(&self) -> u64 {
        self.last_sequence.fetch_add(1, Ordering::SeqCst) + 1
    }

    #[inline]
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::SeqCst)
    }

    #[inline]
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
                let current_next = self.next_file_number.load(Ordering::SeqCst);
                if meta.file_number >= current_next {
                    self.next_file_number.store(meta.file_number + 1, Ordering::SeqCst);
                }
                current_levels[level].push(meta);
            }
        }

        if let Some(next) = edit.next_file_number {
            let current_next = self.next_file_number.load(Ordering::SeqCst);
            if next > current_next {
                self.next_file_number.store(next, Ordering::SeqCst);
            }
        }
        if let Some(last_seq) = edit.last_sequence {
            let current_last = self.last_sequence.load(Ordering::SeqCst);
            if last_seq > current_last {
                self.last_sequence.store(last_seq, Ordering::SeqCst);
            }
        }

        *self.current.write() = Arc::new(Version {
            levels: current_levels,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::version::FileMetaData;
    use bytes::Bytes;
    use tempfile::tempdir;

    #[test]
    fn test_manifest_log_and_read_edits() -> Result<()> {
        let dir = tempdir().unwrap();
        let manifest_path = dir.path().join("MANIFEST");

        let manifest = Manifest::open(&manifest_path)?;

        let mut edit1 = VersionEdit::new();
        edit1.add_file(
            0,
            FileMetaData {
                file_number: 1,
                file_size: 1024,
                smallest_key: Bytes::from_static(b"a"),
                largest_key: Bytes::from_static(b"z"),
            },
        );
        edit1.last_sequence = Some(10);
        manifest.log_edit(&edit1)?;

        let mut edit2 = VersionEdit::new();
        edit2.delete_file(0, 1);
        edit2.add_file(
            1,
            FileMetaData {
                file_number: 2,
                file_size: 2048,
                smallest_key: Bytes::from_static(b"a"),
                largest_key: Bytes::from_static(b"z"),
            },
        );
        edit2.last_sequence = Some(20);
        manifest.log_edit(&edit2)?;

        let edits = Manifest::read_all_edits(&manifest_path)?;
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0].new_files.len(), 1);
        assert_eq!(edits[0].last_sequence, Some(10));
        assert_eq!(edits[1].deleted_files.len(), 1);
        assert_eq!(edits[1].new_files.len(), 1);
        assert_eq!(edits[1].last_sequence, Some(20));

        // Test VersionSet replay
        let vs = VersionSet::new(7);
        for edit in edits {
            vs.log_and_apply(edit);
        }

        let ver = vs.current();
        assert_eq!(ver.levels[0].len(), 0);
        assert_eq!(ver.levels[1].len(), 1);
        assert_eq!(ver.levels[1][0].file_number, 2);
        assert_eq!(vs.last_sequence(), 20);
        assert!(vs.next_file_number() > 2);

        Ok(())
    }
}
