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

    pub fn decode(mut data: Bytes) -> crate::error::Result<Self> {
        if data.len() < 4 {
            return Err(crate::error::FlashStoreError::Corruption("block too short".into()));
        }

        let len = data.len();
        let num_offsets = u32::from_le_bytes(data[len - 4..len].try_into().unwrap()) as usize;
        let offsets_start = len - 4 - num_offsets * 4;

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

    pub fn get_entry(&self, index: usize) -> Option<Entry> {
        if index >= self.offsets.len() {
            return None;
        }

        let start = self.offsets[index] as usize;
        let mut slice = &self.data[start..];
        let len = slice.get_u32_le() as usize;
        let entry_bytes = &slice[..len];
        bincode::deserialize(entry_bytes).ok()
    }

    pub fn entries_len(&self) -> usize {
        self.offsets.len()
    }
}
