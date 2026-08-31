use bytes::{Buf, BufMut, Bytes, BytesMut};

pub const MAGIC_NUMBER: u64 = 0xF1A5_4570_7265_3031; // "FLASHpre01" style magic
pub const FOOTER_SIZE: usize = 48;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Footer {
    pub meta_index_offset: u64,
    pub meta_index_size: u64,
    pub index_offset: u64,
    pub index_size: u64,
    pub magic: u64,
}

impl Footer {
    #[inline]
    pub fn new(
        meta_index_offset: u64,
        meta_index_size: u64,
        index_offset: u64,
        index_size: u64,
    ) -> Self {
        Self {
            meta_index_offset,
            meta_index_size,
            index_offset,
            index_size,
            magic: MAGIC_NUMBER,
        }
    }

    #[inline]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(FOOTER_SIZE);
        buf.put_u64_le(self.meta_index_offset);
        buf.put_u64_le(self.meta_index_size);
        buf.put_u64_le(self.index_offset);
        buf.put_u64_le(self.index_size);
        buf.put_u64_le(0); // reserved
        buf.put_u64_le(self.magic);
        buf.freeze()
    }

    #[inline]
    pub fn decode(mut data: Bytes) -> crate::error::Result<Self> {
        if data.len() < FOOTER_SIZE {
            return Err(crate::error::FlashStoreError::Corruption(
                "footer size too small".into(),
            ));
        }

        let meta_index_offset = data.get_u64_le();
        let meta_index_size = data.get_u64_le();
        let index_offset = data.get_u64_le();
        let index_size = data.get_u64_le();
        let _reserved = data.get_u64_le();
        let magic = data.get_u64_le();

        if magic != MAGIC_NUMBER {
            return Err(crate::error::FlashStoreError::Corruption(
                "invalid SSTable footer magic number".into(),
            ));
        }

        Ok(Self {
            meta_index_offset,
            meta_index_size,
            index_offset,
            index_size,
            magic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footer_encode_decode() {
        let footer = Footer::new(100, 50, 200, 80);
        let encoded = footer.encode();
        assert_eq!(encoded.len(), FOOTER_SIZE);

        let decoded = Footer::decode(encoded).expect("decode footer");
        assert_eq!(footer, decoded);
    }

    #[test]
    fn test_footer_invalid_magic() {
        let mut bad = Footer::new(100, 50, 200, 80).encode().to_vec();
        let len = bad.len();
        bad[len - 1] ^= 0xFF; // corrupt magic

        let result = Footer::decode(Bytes::from(bad));
        assert!(result.is_err());
    }
}
