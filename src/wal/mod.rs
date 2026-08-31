pub mod record;

use crate::error::Result;
use bytes::{Bytes, BytesMut};
use record::WalRecord;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct WalWriter {
    file: File,
    path: PathBuf,
    sync: bool,
}

impl WalWriter {
    pub fn open<P: AsRef<Path>>(path: P, sync: bool) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            path: path.as_ref().to_path_buf(),
            sync,
        })
    }

    #[inline]
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let encoded = record.encode();
        self.file.write_all(&encoded)?;
        if self.sync {
            self.file.sync_all()?;
        }
        Ok(())
    }

    pub fn append_batch(&mut self, records: &[WalRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut batch_buf = BytesMut::new();
        for record in records {
            let encoded = record.encode();
            batch_buf.extend_from_slice(&encoded);
        }
        self.file.write_all(&batch_buf)?;
        if self.sync {
            self.file.sync_all()?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        if self.sync {
            self.file.sync_all()?;
        }
        Ok(())
    }

    #[inline]
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    #[inline]
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

        let bytes_buf = Bytes::from(buffer);
        let mut offset = 0;
        while offset + 8 <= bytes_buf.len() {
            let len_bytes: [u8; 4] = match bytes_buf[offset + 4..offset + 8].try_into() {
                Ok(b) => b,
                Err(_) => break,
            };
            let payload_len = u32::from_le_bytes(len_bytes) as usize;
            let record_len = 8 + payload_len;
            if offset + record_len > bytes_buf.len() {
                // Incomplete payload at end of WAL (torn write)
                break;
            }

            let slice = bytes_buf.slice(offset..offset + record_len);
            match WalRecord::decode(slice) {
                Ok(record) => {
                    records.push(record);
                    offset += record_len;
                }
                Err(_) => {
                    // Corrupted record encountered — stop recovery here to preserve
                    // all committed preceding records rather than discarding everything
                    break;
                }
            }
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

        let mut writer = WalWriter::open(&wal_path, true)?;
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
        assert!(records[2].is_delete);
        assert_eq!(records[2].seq_no, 3);

        // Reset WAL and verify empty
        writer.reset()?;
        let mut reader_after_reset = WalReader::open(&wal_path)?;
        let records_after_reset = reader_after_reset.read_all()?;
        assert_eq!(records_after_reset.len(), 0);

        // Test recovery with corrupt trailing bytes
        writer.append(&r1)?;
        writer.append(&r2)?;
        // Append partial/garbage bytes to simulate uncommitted torn write
        {
            let mut raw_file = std::fs::OpenOptions::new().append(true).open(&wal_path)?;
            raw_file.write_all(b"\xDE\xAD\xBE\xEF\x00\x00")?;
            raw_file.sync_all()?;
        }
        let mut reader_torn = WalReader::open(&wal_path)?;
        let recovered = reader_torn.read_all()?;
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].key, r1.key);
        assert_eq!(recovered[1].key, r2.key);

        Ok(())
    }
}
