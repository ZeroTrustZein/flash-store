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
