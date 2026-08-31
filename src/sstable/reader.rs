use super::block::Block;
use super::bloom::BloomFilter;
use super::footer::{Footer, FOOTER_SIZE};
use crate::error::{FlashStoreError, Result};
use crate::types::{Entry, Key, ValueType};
use bytes::Bytes;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct TableReader {
    file: File,
    footer: Footer,
    first_keys: Vec<Bytes>,
    block_offsets: Vec<u64>,
    bloom_filter: Bytes,
}

impl TableReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < FOOTER_SIZE as u64 {
            return Err(FlashStoreError::Corruption("file size too small".into()));
        }

        // Read footer
        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
        let mut footer_buf = vec![0u8; FOOTER_SIZE];
        file.read_exact(&mut footer_buf)?;
        let footer = Footer::decode(Bytes::from(footer_buf))?;

        // Read Bloom Filter
        file.seek(SeekFrom::Start(footer.meta_index_offset))?;
        let mut bloom_buf = vec![0u8; footer.meta_index_size as usize];
        file.read_exact(&mut bloom_buf)?;
        let bloom_filter = Bytes::from(bloom_buf);

        // Read Index Block
        file.seek(SeekFrom::Start(footer.index_offset))?;
        let mut index_buf = vec![0u8; footer.index_size as usize];
        file.read_exact(&mut index_buf)?;
        let (first_keys, block_offsets): (Vec<Bytes>, Vec<u64>) =
            bincode::deserialize(&index_buf)
                .map_err(|e| FlashStoreError::Corruption(e.to_string()))?;

        Ok(Self {
            file,
            footer,
            first_keys,
            block_offsets,
            bloom_filter,
        })
    }

    pub fn get(&mut self, key: &Key) -> Result<Option<Option<Bytes>>> {
        if !BloomFilter::may_contain(&self.bloom_filter, key) {
            return Ok(None);
        }

        let block_idx = match self.first_keys.binary_search(key) {
            Ok(idx) => idx,
            Err(0) => 0,
            Err(idx) => idx - 1,
        };

        if block_idx >= self.block_offsets.len() {
            return Ok(None);
        }

        let block = self.read_block(block_idx)?;
        for i in 0..block.entries_len() {
            if let Some(entry) = block.get_entry(i) {
                if &entry.key == key {
                    if entry.value_type == ValueType::Tombstone {
                        return Ok(Some(None));
                    } else {
                        return Ok(Some(Some(entry.value)));
                    }
                }
            }
        }

        Ok(None)
    }

    pub fn read_all_entries(&mut self) -> Result<Vec<Entry>> {
        let mut entries = Vec::new();
        for idx in 0..self.block_offsets.len() {
            let block = self.read_block(idx)?;
            for i in 0..block.entries_len() {
                if let Some(entry) = block.get_entry(i) {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }

    fn read_block(&mut self, index: usize) -> Result<Block> {
        let start_offset = self.block_offsets[index];
        let end_offset = if index + 1 < self.block_offsets.len() {
            self.block_offsets[index + 1]
        } else {
            self.footer.meta_index_offset
        };

        let size = (end_offset - start_offset) as usize;
        self.file.seek(SeekFrom::Start(start_offset))?;
        let mut buf = vec![0u8; size];
        self.file.read_exact(&mut buf)?;
        Block::decode(Bytes::from(buf))
    }
}
