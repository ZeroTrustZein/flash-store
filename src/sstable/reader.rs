use super::block::Block;
use super::bloom::BloomFilter;
use super::footer::{Footer, FOOTER_SIZE};
use crate::cache::BlockCache;
use crate::error::{FlashStoreError, Result};
use crate::types::{Entry, Key, ValueType};
use bytes::Bytes;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

pub struct TableReader {
    file: File,
    file_number: u64,
    footer: Footer,
    first_keys: Vec<Bytes>,
    block_offsets: Vec<u64>,
    bloom_filter: Bytes,
    cache: Option<Arc<dyn BlockCache>>,
}

impl TableReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_cache(path, 0, None)
    }

    pub fn open_with_cache<P: AsRef<Path>>(
        path: P,
        file_number: u64,
        cache: Option<Arc<dyn BlockCache>>,
    ) -> Result<Self> {
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
            file_number,
            footer,
            first_keys,
            block_offsets,
            bloom_filter,
            cache,
        })
    }

    pub fn file_number(&self) -> u64 {
        self.file_number
    }

    pub fn get(&mut self, key: &Key) -> Result<Option<Option<Bytes>>> {
        if self.block_offsets.is_empty() {
            return Ok(None);
        }

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

    pub fn read_block(&mut self, index: usize) -> Result<Block> {
        let cache_key = (self.file_number, index as u64);
        if let Some(ref cache) = self.cache {
            if let Some(cached_bytes) = cache.get(&cache_key) {
                return Block::decode(cached_bytes);
            }
        }

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
        let block_bytes = Bytes::from(buf);

        if let Some(ref cache) = self.cache {
            cache.insert(cache_key, block_bytes.clone());
        }

        Block::decode(block_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LruBlockCache;
    use crate::config::OptionsBuilder;
    use crate::sstable::TableBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_sstable_build_and_read() -> Result<()> {
        let dir = tempdir().unwrap();
        let sst_path = dir.path().join("000001.sst");
        let options = OptionsBuilder::new().block_size(64).build();

        let mut builder = TableBuilder::new(&sst_path, options)?;

        let e1 = Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"), 1);
        let e2 = Entry::new_value(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"), 2);
        let e3 = Entry::new_tombstone(Bytes::from_static(b"k3"), 3);

        builder.add(e1.clone())?;
        builder.add(e2.clone())?;
        builder.add(e3.clone())?;
        let size = builder.finish()?;
        assert!(size > 0);

        let mut reader = TableReader::open(&sst_path)?;
        assert_eq!(
            reader.get(&Bytes::from_static(b"k1"))?,
            Some(Some(Bytes::from_static(b"v1")))
        );
        assert_eq!(
            reader.get(&Bytes::from_static(b"k2"))?,
            Some(Some(Bytes::from_static(b"v2")))
        );
        assert_eq!(reader.get(&Bytes::from_static(b"k3"))?, Some(None));
        assert_eq!(reader.get(&Bytes::from_static(b"k4"))?, None);

        let all_entries = reader.read_all_entries()?;
        assert_eq!(all_entries.len(), 3);
        assert_eq!(all_entries[0], e1);
        assert_eq!(all_entries[1], e2);
        assert_eq!(all_entries[2], e3);

        Ok(())
    }

    #[test]
    fn test_sstable_reader_with_block_cache() -> Result<()> {
        let dir = tempdir().unwrap();
        let sst_path = dir.path().join("000002.sst");
        let options = OptionsBuilder::new().block_size(64).build();

        let mut builder = TableBuilder::new(&sst_path, options)?;
        let e1 = Entry::new_value(Bytes::from_static(b"apple"), Bytes::from_static(b"red"), 1);
        let e2 = Entry::new_value(Bytes::from_static(b"banana"), Bytes::from_static(b"yellow"), 2);
        builder.add(e1)?;
        builder.add(e2)?;
        builder.finish()?;

        let cache = Arc::new(LruBlockCache::new(10));
        let mut reader = TableReader::open_with_cache(&sst_path, 2, Some(cache.clone()))?;

        assert_eq!(cache.len(), 0);
        let val = reader.get(&Bytes::from_static(b"apple"))?;
        assert_eq!(val, Some(Some(Bytes::from_static(b"red"))));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&(2, 0)));

        // Read again, hit cache
        let val2 = reader.get(&Bytes::from_static(b"banana"))?;
        assert_eq!(val2, Some(Some(Bytes::from_static(b"yellow"))));

        Ok(())
    }
}
