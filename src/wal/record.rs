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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_record_encode_decode() {
        let record = WalRecord {
            key: Bytes::from_static(b"test_key"),
            value: Bytes::from_static(b"test_val"),
            is_delete: false,
            seq_no: 42,
        };

        let encoded = record.encode();
        let decoded = WalRecord::decode(encoded).expect("decode WAL record");
        assert_eq!(decoded.key, record.key);
        assert_eq!(decoded.value, record.value);
        assert_eq!(decoded.is_delete, record.is_delete);
        assert_eq!(decoded.seq_no, record.seq_no);
    }

    #[test]
    fn test_wal_record_tombstone() {
        let record = WalRecord {
            key: Bytes::from_static(b"del_key"),
            value: Bytes::new(),
            is_delete: true,
            seq_no: 99,
        };

        let encoded = record.encode();
        let decoded = WalRecord::decode(encoded).expect("decode tombstone");
        assert_eq!(decoded.key, record.key);
        assert!(decoded.value.is_empty());
        assert!(decoded.is_delete);
        assert_eq!(decoded.seq_no, 99);
    }

    #[test]
    fn test_wal_record_corrupted_checksum() {
        let record = WalRecord {
            key: Bytes::from_static(b"key"),
            value: Bytes::from_static(b"val"),
            is_delete: false,
            seq_no: 1,
        };

        let mut encoded = record.encode().to_vec();
        // Flip a byte in the payload
        let last_idx = encoded.len() - 1;
        encoded[last_idx] ^= 0xFF;

        let result = WalRecord::decode(Bytes::from(encoded));
        assert!(result.is_err());
    }
}
