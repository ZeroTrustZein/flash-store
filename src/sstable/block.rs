use crate::types::Entry;
use bytes::{Buf, BufMut, Bytes, BytesMut};

#[derive(Debug, Clone)]
pub struct Block {
    pub data: Bytes,
    pub offsets: Vec<u32>,
}

impl Block {
    pub fn encode(entries: &[Entry]) -> Bytes {
        let mut buf = BytesMut::new();
        let mut offsets = Vec::new();

        for entry in entries {
            offsets.push(buf.len() as u32);
            let encoded_entry = bincode::serialize(entry).expect("serialize entry");
            buf.put_u32_le(encoded_entry.len() as u32);
            buf.put_slice(&encoded_entry);
        }

        let num_offsets = offsets.len() as u32;
        for offset in offsets {
            buf.put_u32_le(offset);
        }
        buf.put_u32_le(num_offsets);

        buf.freeze()
    }

    pub fn decode(data: Bytes) -> crate::error::Result<Self> {
        if data.len() < 4 {
            return Err(crate::error::FlashStoreError::Corruption(
                "block too short".into(),
            ));
        }

        let len = data.len();
        let num_offsets_bytes: [u8; 4] = data[len - 4..len].try_into().map_err(|_| {
            crate::error::FlashStoreError::Corruption("corrupted offsets length".into())
        })?;
        let num_offsets = u32::from_le_bytes(num_offsets_bytes) as usize;
        let total_offsets_bytes = num_offsets.checked_mul(4).ok_or_else(|| {
            crate::error::FlashStoreError::Corruption("corrupted offsets count".into())
        })?;

        if len < 4 + total_offsets_bytes {
            return Err(crate::error::FlashStoreError::Corruption(
                "num_offsets exceeds block length".into(),
            ));
        }

        let offsets_start = len - 4 - total_offsets_bytes;

        let mut offsets = Vec::with_capacity(num_offsets);
        let mut offset_bytes = &data[offsets_start..len - 4];
        for _ in 0..num_offsets {
            offsets.push(offset_bytes.get_u32_le());
        }

        let block_data = data.slice(0..offsets_start);
        Ok(Self {
            data: block_data,
            offsets,
        })
    }

    #[inline]
    pub fn get_entry(&self, index: usize) -> Option<Entry> {
        if index >= self.offsets.len() {
            return None;
        }

        let start = self.offsets[index] as usize;
        if start + 4 > self.data.len() {
            return None;
        }
        let mut slice = &self.data[start..];
        let len = slice.get_u32_le() as usize;
        if slice.len() < len {
            return None;
        }
        let entry_bytes = &slice[..len];
        bincode::deserialize(entry_bytes).ok()
    }

    #[inline]
    pub fn entries_len(&self) -> usize {
        self.offsets.len()
    }

    /// Binary search entries in the block for the first matching user key.
    pub fn get_by_key(&self, key: &[u8]) -> Option<Entry> {
        let mut low = 0;
        let mut high = self.offsets.len();

        // Lower bound binary search on sorted entries
        while low < high {
            let mid = low + (high - low) / 2;
            if let Some(entry) = self.get_entry(mid) {
                if entry.key.as_ref() < key {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            } else {
                break;
            }
        }

        if low < self.offsets.len() {
            if let Some(entry) = self.get_entry(low) {
                if entry.key.as_ref() == key {
                    return Some(entry);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_encode_decode() {
        let entries = vec![
            Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"), 1),
            Entry::new_value(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"), 2),
            Entry::new_tombstone(Bytes::from_static(b"k3"), 3),
        ];

        let encoded = Block::encode(&entries);
        let block = Block::decode(encoded).expect("decode block");
        assert_eq!(block.entries_len(), 3);

        assert_eq!(block.get_entry(0), Some(entries[0].clone()));
        assert_eq!(block.get_entry(1), Some(entries[1].clone()));
        assert_eq!(block.get_entry(2), Some(entries[2].clone()));
        assert_eq!(block.get_entry(3), None);

        // Binary search lookup
        assert_eq!(block.get_by_key(b"k1"), Some(entries[0].clone()));
        assert_eq!(block.get_by_key(b"k2"), Some(entries[1].clone()));
        assert_eq!(block.get_by_key(b"k3"), Some(entries[2].clone()));
        assert_eq!(block.get_by_key(b"k0"), None);
        assert_eq!(block.get_by_key(b"k4"), None);
    }
}
