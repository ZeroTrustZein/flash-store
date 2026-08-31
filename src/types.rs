use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// Trait for converting various byte-like representations into `Bytes` (`Key`/`Value`).
pub trait IntoBytes {
    fn into_bytes(self) -> Bytes;
}

impl IntoBytes for Bytes {
    #[inline]
    fn into_bytes(self) -> Bytes {
        self
    }
}

impl IntoBytes for &Bytes {
    #[inline]
    fn into_bytes(self) -> Bytes {
        self.clone()
    }
}

impl IntoBytes for Vec<u8> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

impl IntoBytes for &Vec<u8> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_slice())
    }
}

impl IntoBytes for &[u8] {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self)
    }
}

impl<const N: usize> IntoBytes for &[u8; N] {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_slice())
    }
}

impl<const N: usize> IntoBytes for [u8; N] {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(&self[..])
    }
}

impl IntoBytes for &str {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }
}

impl IntoBytes for String {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

impl IntoBytes for &String {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }
}

impl IntoBytes for Box<[u8]> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

/// Monotonic logical sequence number for versioning entries.
pub type SequenceNumber = u64;

/// Maximum sequence number supported.
pub const MAX_SEQUENCE_NUMBER: SequenceNumber = u64::MAX;

/// Minimum sequence number supported.
pub const MIN_SEQUENCE_NUMBER: SequenceNumber = 0;

/// Alias for raw user-provided key.
pub type UserKey = Bytes;

/// Storage key type used across the LSM-tree.
pub type Key = Bytes;

/// Value payload type.
pub type Value = Bytes;

/// Type tag representing the operation on an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueType {
    /// Normal put or update entry.
    Value = 0,
    /// Tombstone marker indicating deletion.
    Tombstone = 1,
}

impl ValueType {
    /// Returns true if this entry represents a normal value.
    #[inline]
    pub fn is_value(&self) -> bool {
        matches!(self, ValueType::Value)
    }

    /// Returns true if this entry is a deletion tombstone.
    #[inline]
    pub fn is_tombstone(&self) -> bool {
        matches!(self, ValueType::Tombstone)
    }

    /// Convert to u8 representation.
    #[inline]
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Parse from a u8 representation.
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(ValueType::Value),
            1 => Some(ValueType::Tombstone),
            _ => None,
        }
    }
}

impl From<ValueType> for u8 {
    #[inline]
    fn from(vt: ValueType) -> Self {
        vt.as_u8()
    }
}

impl TryFrom<u8> for ValueType {
    type Error = crate::error::FlashStoreError;

    #[inline]
    fn try_from(val: u8) -> std::result::Result<Self, Self::Error> {
        ValueType::from_u8(val).ok_or_else(|| {
            crate::error::FlashStoreError::Corruption(format!("Invalid ValueType byte: {}", val))
        })
    }
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueType::Value => write!(f, "Value"),
            ValueType::Tombstone => write!(f, "Tombstone"),
        }
    }
}

/// An entry stored in the MemTable or SSTable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub value: Value,
    pub value_type: ValueType,
    pub seq_no: SequenceNumber,
}

impl Entry {
    /// Create a new generic entry.
    pub fn new(
        key: impl IntoBytes,
        value: impl IntoBytes,
        value_type: ValueType,
        seq_no: SequenceNumber,
    ) -> Self {
        Self {
            key: key.into_bytes(),
            value: value.into_bytes(),
            value_type,
            seq_no,
        }
    }

    /// Create a new value entry.
    pub fn new_value(
        key: impl IntoBytes,
        value: impl IntoBytes,
        seq_no: SequenceNumber,
    ) -> Self {
        Self::new(key, value, ValueType::Value, seq_no)
    }

    /// Create a new tombstone (deletion) entry with empty value payload.
    pub fn new_tombstone(key: impl IntoBytes, seq_no: SequenceNumber) -> Self {
        Self::new(key, Bytes::new(), ValueType::Tombstone, seq_no)
    }

    /// Returns true if this is a tombstone entry.
    #[inline]
    pub fn is_tombstone(&self) -> bool {
        self.value_type.is_tombstone()
    }

    /// Returns true if this is a normal value entry.
    #[inline]
    pub fn is_value(&self) -> bool {
        self.value_type.is_value()
    }

    /// Get reference to key.
    #[inline]
    pub fn key(&self) -> &Key {
        &self.key
    }

    /// Get reference to value.
    #[inline]
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Get sequence number.
    #[inline]
    pub fn seq_no(&self) -> SequenceNumber {
        self.seq_no
    }

    /// Get value type.
    #[inline]
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Estimated memory/storage footprint of this entry in bytes.
    #[inline]
    pub fn estimated_size(&self) -> usize {
        self.key.len() + self.value.len() + std::mem::size_of::<SequenceNumber>() + 1
    }
}

/// Internal key representation for LSM-tree multi-version ordering.
/// Keys with the same `user_key` are sorted in descending order of `seq_no`
/// so that the most recent update is encountered first.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InternalKey {
    pub user_key: UserKey,
    pub seq_no: SequenceNumber,
    pub value_type: ValueType,
}

impl InternalKey {
    /// Create a new InternalKey.
    pub fn new(
        user_key: impl IntoBytes,
        seq_no: SequenceNumber,
        value_type: ValueType,
    ) -> Self {
        Self {
            user_key: user_key.into_bytes(),
            seq_no,
            value_type,
        }
    }

    /// Encode internal key to binary bytes: `[user_key][seq_no (8 bytes BE)][value_type (1 byte)]`.
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.user_key.len() + 8 + 1);
        buf.put_slice(&self.user_key);
        buf.put_u64(self.seq_no);
        buf.put_u8(self.value_type.as_u8());
        buf.freeze()
    }

    /// Decode internal key from binary bytes.
    pub fn decode(data: &Bytes) -> crate::error::Result<Self> {
        if data.len() < 9 {
            return Err(crate::error::FlashStoreError::Corruption(
                "InternalKey too short to decode".into(),
            ));
        }
        let user_key_len = data.len() - 9;
        let user_key = data.slice(0..user_key_len);
        let seq_bytes: [u8; 8] = data[user_key_len..user_key_len + 8]
            .try_into()
            .map_err(|_| crate::error::FlashStoreError::Corruption("invalid seq_no bytes".into()))?;
        let seq_no = u64::from_be_bytes(seq_bytes);
        let val_type_byte = data[user_key_len + 8];
        let value_type = ValueType::try_from(val_type_byte)?;

        Ok(Self {
            user_key,
            seq_no,
            value_type,
        })
    }
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // LSM order:
        // 1. user_key ascending
        // 2. seq_no descending (latest first)
        // 3. value_type descending
        self.user_key
            .cmp(&other.user_key)
            .then_with(|| other.seq_no.cmp(&self.seq_no))
            .then_with(|| (other.value_type as u8).cmp(&(self.value_type as u8)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_conversions_and_predicates() {
        assert!(ValueType::Value.is_value());
        assert!(!ValueType::Value.is_tombstone());
        assert_eq!(ValueType::Value.as_u8(), 0);

        assert!(ValueType::Tombstone.is_tombstone());
        assert!(!ValueType::Tombstone.is_value());
        assert_eq!(ValueType::Tombstone.as_u8(), 1);

        assert_eq!(ValueType::from_u8(0), Some(ValueType::Value));
        assert_eq!(ValueType::from_u8(1), Some(ValueType::Tombstone));
        assert_eq!(ValueType::from_u8(2), None);

        assert_eq!(ValueType::try_from(0).unwrap(), ValueType::Value);
        assert_eq!(ValueType::try_from(1).unwrap(), ValueType::Tombstone);
        assert!(ValueType::try_from(99).is_err());

        assert_eq!(u8::from(ValueType::Value), 0);
        assert_eq!(u8::from(ValueType::Tombstone), 1);

        assert_eq!(format!("{}", ValueType::Value), "Value");
        assert_eq!(format!("{}", ValueType::Tombstone), "Tombstone");
    }

    #[test]
    fn test_entry_constructors_and_methods() {
        let entry = Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"), 42);
        assert_eq!(entry.key(), &Bytes::from_static(b"k1"));
        assert_eq!(entry.value(), &Bytes::from_static(b"v1"));
        assert_eq!(entry.seq_no(), 42);
        assert_eq!(entry.value_type(), ValueType::Value);
        assert!(entry.is_value());
        assert!(!entry.is_tombstone());
        assert_eq!(entry.estimated_size(), 2 + 2 + 8 + 1);

        let tombstone = Entry::new_tombstone(Bytes::from_static(b"k1"), 43);
        assert_eq!(tombstone.key(), &Bytes::from_static(b"k1"));
        assert!(tombstone.value().is_empty());
        assert_eq!(tombstone.seq_no(), 43);
        assert_eq!(tombstone.value_type(), ValueType::Tombstone);
        assert!(tombstone.is_tombstone());
        assert!(!tombstone.is_value());
        assert_eq!(tombstone.estimated_size(), 2 + 0 + 8 + 1);
    }

    #[test]
    fn test_entry_serialization_roundtrip() {
        let original = Entry::new_value(
            Bytes::from_static(b"sample_key"),
            Bytes::from_static(b"sample_val"),
            100,
        );
        let encoded = bincode::serialize(&original).expect("serialize entry");
        let decoded: Entry = bincode::deserialize(&encoded).expect("deserialize entry");
        assert_eq!(original, decoded);

        let json = serde_json::to_string(&original).expect("json serialize");
        let from_json: Entry = serde_json::from_str(&json).expect("json deserialize");
        assert_eq!(original, from_json);
    }

    #[test]
    fn test_internal_key_encode_decode() {
        let user_key = Bytes::from_static(b"user_record_key");
        let internal_key = InternalKey::new(user_key.clone(), 123456789, ValueType::Value);
        let encoded = internal_key.encode();

        let decoded = InternalKey::decode(&encoded).expect("decode internal key");
        assert_eq!(internal_key, decoded);
        assert_eq!(decoded.user_key, user_key);
        assert_eq!(decoded.seq_no, 123456789);
        assert_eq!(decoded.value_type, ValueType::Value);
    }

    #[test]
    fn test_internal_key_corrupted_decode() {
        let short_data = Bytes::from_static(b"short");
        assert!(InternalKey::decode(&short_data).is_err());

        // Corrupted value type
        let mut bad_type = BytesMut::new();
        bad_type.put_slice(b"k");
        bad_type.put_u64(10);
        bad_type.put_u8(5); // invalid type
        assert!(InternalKey::decode(&bad_type.freeze()).is_err());
    }

    #[test]
    fn test_internal_key_lsm_ordering() {
        let k1_v10 = InternalKey::new(Bytes::from_static(b"alpha"), 10, ValueType::Value);
        let k1_v20 = InternalKey::new(Bytes::from_static(b"alpha"), 20, ValueType::Value);
        let k2_v5 = InternalKey::new(Bytes::from_static(b"beta"), 5, ValueType::Value);

        // Alpha with seq_no 20 must come BEFORE alpha with seq_no 10 (descending sequence number)
        assert!(k1_v20 < k1_v10);
        // Alpha comes before Beta regardless of seq_no
        assert!(k1_v10 < k2_v5);
        assert!(k1_v20 < k2_v5);

        let mut keys = vec![k1_v10.clone(), k2_v5.clone(), k1_v20.clone()];
        keys.sort();
        assert_eq!(keys, vec![k1_v20, k1_v10, k2_v5]);
    }

    #[test]
    fn test_sequence_number_constants() {
        assert_eq!(MIN_SEQUENCE_NUMBER, 0);
        assert_eq!(MAX_SEQUENCE_NUMBER, u64::MAX);
    }
}
