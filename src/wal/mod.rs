pub mod record;

use crate::error::Result;
use parking_lot::Mutex;
use record::WalRecord;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct WalWriter {
    file: Arc<Mutex<File>>,
    path: PathBuf,
    sync: bool,
}

impl WalWriter {
    pub fn open<P: AsRef<Path>>(path: P, sync: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            path: path.as_ref().to_path_buf(),
            sync,
        })
    }

    pub fn append(&self, record: &WalRecord) -> Result<()> {
        let encoded = record.encode();
        let mut file = self.file.lock();
        file.write_all(&encoded)?;
        if self.sync {
            file.sync_all()?;
        }
        Ok(())
    }

    pub fn append_batch(&self, records: &[WalRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut file = self.file.lock();
        for record in records {
            let encoded = record.encode();
            file.write_all(&encoded)?;
        }
        if self.sync {
            file.sync_all()?;
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        let mut file = self.file.lock();
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        if self.sync {
            file.sync_all()?;
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        let file = self.file.lock();
        file.sync_all()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub struct WalReader {
    file: File,
}

impl WalReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        Ok(Self { file })
    }

    pub fn read_all(&mut self) -> Result<Vec<WalRecord>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut records = Vec::new();
        let mut buffer = Vec::new();
        self.file.read_to_end(&mut buffer)?;

        let mut offset = 0;
        while offset < buffer.len() {
            if offset + 8 > buffer.len() {
                break;
            }
            let len_bytes = &buffer[offset + 4..offset + 8];
            let payload_len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
            let record_len = 8 + payload_len;
            if offset + record_len > buffer.len() {
                break;
            }

            let slice = bytes::Bytes::copy_from_slice(&buffer[offset..offset + record_len]);
            let record = WalRecord::decode(slice)?;
            records.push(record);
            offset += record_len;
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_write_and_read_all() -> Result<()> {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let writer = WalWriter::open(&wal_path, true)?;
        assert_eq!(writer.path(), wal_path);

        let r1 = WalRecord {
            key: bytes::Bytes::from_static(b"k1"),
            value: bytes::Bytes::from_static(b"v1"),
            is_delete: false,
            seq_no: 1,
        };
        let r2 = WalRecord {
            key: bytes::Bytes::from_static(b"k2"),
            value: bytes::Bytes::from_static(b"v2"),
            is_delete: false,
            seq_no: 2,
        };
        let r3 = WalRecord {
            key: bytes::Bytes::from_static(b"k1"),
            value: bytes::Bytes::new(),
            is_delete: true,
            seq_no: 3,
        };

        writer.append(&r1)?;
        writer.append_batch(&[r2.clone(), r3.clone()])?;

        let mut reader = WalReader::open(&wal_path)?;
        let records = reader.read_all()?;
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].key, r1.key);
        assert_eq!(records[0].seq_no, 1);
        assert_eq!(records[1].key, r2.key);
        assert_eq!(records[1].seq_no, 2);
        assert_eq!(records[2].key, r3.key);
        assert_eq!(records[2].is_delete, true);
        assert_eq!(records[2].seq_no, 3);

        // Reset WAL and verify empty
        writer.reset()?;
        let mut reader_after_reset = WalReader::open(&wal_path)?;
        let records_after_reset = reader_after_reset.read_all()?;
        assert_eq!(records_after_reset.len(), 0);

        Ok(())
    }
}
