use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub key: Bytes,
    pub value: Bytes,
    pub is_delete: bool,
    pub seq_no: u64,
}

impl WalRecord {
    pub fn encode(&self) -> Bytes {
        let payload = bincode::serialize(self).expect("serialization should not fail");
        let crc = crc32fast::Hasher::new();
        let mut hasher = crc;
        hasher.update(&payload);
        let checksum = hasher.finalize();

        let mut buf = BytesMut::with_capacity(4 + 4 + payload.len());
        buf.put_u32_le(checksum);
        buf.put_u32_le(payload.len() as u32);
        buf.put_slice(&payload);
        buf.freeze()
    }

    pub fn decode(mut data: Bytes) -> crate::error::Result<Self> {
        if data.len() < 8 {
            return Err(crate::error::FlashStoreError::WalError(
                "insufficient data for header".into(),
            ));
        }

        let checksum = data.get_u32_le();
        let length = data.get_u32_le() as usize;

        if data.len() < length {
            return Err(crate::error::FlashStoreError::WalError(
                "payload length mismatch".into(),
            ));
        }

        let payload = &data[..length];
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != checksum {
            return Err(crate::error::FlashStoreError::Corruption(
                "WAL checksum mismatch".into(),
            ));
        }

        let record: WalRecord = bincode::deserialize(payload)
            .map_err(|e| crate::error::FlashStoreError::WalError(e.to_string()))?;
        Ok(record)
    }
}
