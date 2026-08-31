use super::block::Block;
use super::bloom::BloomFilter;
use super::footer::Footer;
use crate::config::Options;
use crate::error::Result;
use crate::types::Entry;
use bytes::Bytes;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct TableBuilder {
    file: File,
    options: Options,
    current_entries: Vec<Entry>,
    block_offsets: Vec<u64>,
    first_keys: Vec<Bytes>,
    all_keys: Vec<Bytes>,
    offset: u64,
}

impl TableBuilder {
    pub fn new<P: AsRef<Path>>(path: P, options: Options) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            file,
            options,
            current_entries: Vec::new(),
            block_offsets: Vec::new(),
            first_keys: Vec::new(),
            all_keys: Vec::new(),
            offset: 0,
        })
    }

    pub fn add(&mut self, entry: Entry) -> Result<()> {
        if self.current_entries.is_empty() {
            self.first_keys.push(entry.key.clone());
        }
        self.all_keys.push(entry.key.clone());
        self.current_entries.push(entry);

        let estimated_size: usize = self.current_entries.iter().map(|e| e.estimated_size()).sum();
        if estimated_size >= self.options.block_size {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> Result<()> {
        if self.current_entries.is_empty() {
            return Ok(());
        }

        let encoded = Block::encode(&self.current_entries);
        self.file.write_all(&encoded)?;
        self.block_offsets.push(self.offset);
        self.offset += encoded.len() as u64;
        self.current_entries.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.flush_block()?;

        // Write Filter Block
        let bloom = BloomFilter::new(self.options.bloom_bits_per_key);
        let filter_data = bloom.build_from_keys(&self.all_keys);
        let meta_index_offset = self.offset;
        self.file.write_all(&filter_data)?;
        let meta_index_size = filter_data.len() as u64;
        self.offset += meta_index_size;

        // Write Index Block
        let index_data = bincode::serialize(&(self.first_keys, self.block_offsets))
            .map_err(|e| crate::error::FlashStoreError::TableError(e.to_string()))?;
        let index_offset = self.offset;
        self.file.write_all(&index_data)?;
        let index_size = index_data.len() as u64;
        self.offset += index_size;

        // Write Footer
        let footer = Footer::new(
            meta_index_offset,
            meta_index_size,
            index_offset,
            index_size,
        );
        let footer_bytes = footer.encode();
        self.file.write_all(&footer_bytes)?;
        self.offset += footer_bytes.len() as u64;

        self.file.sync_all()?;
        Ok(self.offset)
    }
}
