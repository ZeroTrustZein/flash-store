use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

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

impl IntoBytes for &Box<[u8]> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_ref())
    }
}

impl IntoBytes for Cow<'_, [u8]> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        match self {
            Cow::Borrowed(b) => Bytes::copy_from_slice(b),
            Cow::Owned(v) => Bytes::from(v),
        }
    }
}

impl IntoBytes for Cow<'_, str> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        match self {
            Cow::Borrowed(s) => Bytes::copy_from_slice(s.as_bytes()),
            Cow::Owned(s) => Bytes::from(s.into_bytes()),
        }
    }
}

impl IntoBytes for Arc<[u8]> {
    #[inline]
    fn into_bytes(self) -> Bytes {
        Bytes::copy_from_slice(&self)
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

impl Default for ValueType {
    #[inline]
    fn default() -> Self {
        ValueType::Value
    }
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

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Value => write!(f, "Value"),
            ValueType::Tombstone => write!(f, "Tombstone"),
        }
    }
}

impl AsRef<str> for ValueType {
    #[inline]
    fn as_ref(&self) -> &str {
        match self {
            ValueType::Value => "Value",
            ValueType::Tombstone => "Tombstone",
        }
    }
}

/// Compression algorithm tag for block data compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CompressionType {
    /// No compression.
    None = 0,
    /// Snappy compression.
    Snappy = 1,
    /// Zstandard compression.
    Zstd = 2,
    /// LZ4 compression.
    Lz4 = 3,
}

impl Default for CompressionType {
    #[inline]
    fn default() -> Self {
        CompressionType::None
    }
}

impl CompressionType {
    #[inline]
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(CompressionType::None),
            1 => Some(CompressionType::Snappy),
            2 => Some(CompressionType::Zstd),
            3 => Some(CompressionType::Lz4),
            _ => None,
        }
    }
}

impl From<CompressionType> for u8 {
    #[inline]
    fn from(ct: CompressionType) -> Self {
        ct.as_u8()
    }
}

impl TryFrom<u8> for CompressionType {
    type Error = crate::error::FlashStoreError;

    #[inline]
    fn try_from(val: u8) -> std::result::Result<Self, Self::Error> {
        CompressionType::from_u8(val).ok_or_else(|| {
            crate::error::FlashStoreError::Corruption(format!(
                "Invalid CompressionType byte: {}",
                val
            ))
        })
    }
}

impl fmt::Display for CompressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionType::None => write!(f, "None"),
            CompressionType::Snappy => write!(f, "Snappy"),
            CompressionType::Zstd => write!(f, "Zstd"),
            CompressionType::Lz4 => write!(f, "Lz4"),
        }
    }
}

/// Checksum algorithm tag for data integrity verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChecksumType {
    /// IEEE CRC32 checksum.
    Crc32 = 0,
    /// XXHash64 checksum.
    XxHash = 1,
}

impl Default for ChecksumType {
    #[inline]
    fn default() -> Self {
        ChecksumType::Crc32
    }
}

impl ChecksumType {
    #[inline]
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(ChecksumType::Crc32),
            1 => Some(ChecksumType::XxHash),
            _ => None,
        }
    }
}

impl From<ChecksumType> for u8 {
    #[inline]
    fn from(ct: ChecksumType) -> Self {
        ct.as_u8()
    }
}

impl TryFrom<u8> for ChecksumType {
    type Error = crate::error::FlashStoreError;

    #[inline]
    fn try_from(val: u8) -> std::result::Result<Self, Self::Error> {
        ChecksumType::from_u8(val).ok_or_else(|| {
            crate::error::FlashStoreError::Corruption(format!("Invalid ChecksumType byte: {}", val))
        })
    }
}

impl fmt::Display for ChecksumType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChecksumType::Crc32 => write!(f, "CRC32"),
            ChecksumType::XxHash => write!(f, "XXHash"),
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
    #[inline]
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
    #[inline]
    pub fn new_value(key: impl IntoBytes, value: impl IntoBytes, seq_no: SequenceNumber) -> Self {
        Self::new(key, value, ValueType::Value, seq_no)
    }

    /// Create a new tombstone (deletion) entry with empty value payload.
    #[inline]
    pub fn new_tombstone(key: impl IntoBytes, seq_no: SequenceNumber) -> Self {
        Self::new(key, Bytes::new(), ValueType::Tombstone, seq_no)
    }

    /// Create an entry directly from raw parts.
    #[inline]
    pub fn from_parts(
        key: Key,
        value: Value,
        value_type: ValueType,
        seq_no: SequenceNumber,
    ) -> Self {
        Self {
            key,
            value,
            value_type,
            seq_no,
        }
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

    /// Consume entry into individual tuple components.
    #[inline]
    pub fn into_parts(self) -> (Key, Value, ValueType, SequenceNumber) {
        (self.key, self.value, self.value_type, self.seq_no)
    }

    /// Construct an `InternalKey` referencing this entry's metadata.
    #[inline]
    pub fn to_internal_key(&self) -> InternalKey {
        InternalKey::new(self.key.clone(), self.seq_no, self.value_type)
    }

    /// Convert this entry into an `InternalKey`.
    #[inline]
    pub fn into_internal_key(self) -> InternalKey {
        InternalKey::new(self.key, self.seq_no, self.value_type)
    }

    /// Returns a new entry with the specified sequence number.
    #[inline]
    pub fn with_seq_no(mut self, seq_no: SequenceNumber) -> Self {
        self.seq_no = seq_no;
        self
    }

    /// Returns a new entry with the specified value.
    #[inline]
    pub fn with_value(mut self, value: impl IntoBytes) -> Self {
        self.value = value.into_bytes();
        self
    }
}

impl PartialOrd for Entry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // LSM order:
        // 1. user_key ascending
        // 2. seq_no descending (latest first)
        // 3. value_type descending
        self.key
            .cmp(&other.key)
            .then_with(|| other.seq_no.cmp(&self.seq_no))
            .then_with(|| (other.value_type as u8).cmp(&(self.value_type as u8)))
    }
}

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Entry(key={:?}, type={}, seq={}, val_len={})",
            self.key,
            self.value_type,
            self.seq_no,
            self.value.len()
        )
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
    #[inline]
    pub fn new(user_key: impl IntoBytes, seq_no: SequenceNumber, value_type: ValueType) -> Self {
        Self {
            user_key: user_key.into_bytes(),
            seq_no,
            value_type,
        }
    }

    /// Create a new value InternalKey.
    #[inline]
    pub fn new_value(user_key: impl IntoBytes, seq_no: SequenceNumber) -> Self {
        Self::new(user_key, seq_no, ValueType::Value)
    }

    /// Create a new tombstone InternalKey.
    #[inline]
    pub fn new_tombstone(user_key: impl IntoBytes, seq_no: SequenceNumber) -> Self {
        Self::new(user_key, seq_no, ValueType::Tombstone)
    }

    /// Encoded byte length of this internal key.
    #[inline]
    pub fn encoded_len(&self) -> usize {
        self.user_key.len() + 8 + 1
    }

    /// Append binary encoding of internal key into a `BytesMut` buffer.
    #[inline]
    pub fn encode_to(&self, buf: &mut BytesMut) {
        buf.reserve(self.encoded_len());
        buf.put_slice(&self.user_key);
        buf.put_u64(self.seq_no);
        buf.put_u8(self.value_type.as_u8());
    }

    /// Encode internal key to binary bytes: `[user_key][seq_no (8 bytes BE)][value_type (1 byte)]`.
    #[inline]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.encoded_len());
        self.encode_to(&mut buf);
        buf.freeze()
    }

    /// Decode internal key from a byte slice.
    pub fn decode_from_slice(data: &[u8]) -> crate::error::Result<Self> {
        if data.len() < 9 {
            return Err(crate::error::FlashStoreError::Corruption(
                "InternalKey too short to decode".into(),
            ));
        }
        let user_key_len = data.len() - 9;
        let user_key = Bytes::copy_from_slice(&data[0..user_key_len]);
        let seq_bytes: [u8; 8] = data[user_key_len..user_key_len + 8]
            .try_into()
            .map_err(|_| {
                crate::error::FlashStoreError::Corruption("invalid seq_no bytes".into())
            })?;
        let seq_no = u64::from_be_bytes(seq_bytes);
        let val_type_byte = data[user_key_len + 8];
        let value_type = ValueType::try_from(val_type_byte)?;

        Ok(Self {
            user_key,
            seq_no,
            value_type,
        })
    }

    /// Decode internal key from binary bytes.
    #[inline]
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
            .map_err(|_| {
                crate::error::FlashStoreError::Corruption("invalid seq_no bytes".into())
            })?;
        let seq_no = u64::from_be_bytes(seq_bytes);
        let val_type_byte = data[user_key_len + 8];
        let value_type = ValueType::try_from(val_type_byte)?;

        Ok(Self {
            user_key,
            seq_no,
            value_type,
        })
    }

    /// Get reference to user key.
    #[inline]
    pub fn user_key(&self) -> &UserKey {
        &self.user_key
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

    /// Returns true if this is a normal value.
    #[inline]
    pub fn is_value(&self) -> bool {
        self.value_type.is_value()
    }

    /// Returns true if this is a tombstone.
    #[inline]
    pub fn is_tombstone(&self) -> bool {
        self.value_type.is_tombstone()
    }

    /// Extract user key.
    #[inline]
    pub fn into_user_key(self) -> UserKey {
        self.user_key
    }

    /// Consume into parts.
    #[inline]
    pub fn into_parts(self) -> (UserKey, SequenceNumber, ValueType) {
        (self.user_key, self.seq_no, self.value_type)
    }
}

impl PartialOrd for InternalKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalKey {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
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

impl fmt::Display for InternalKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InternalKey(user_key={:?}, seq={}, type={})",
            self.user_key, self.seq_no, self.value_type
        )
    }
}

impl From<(UserKey, SequenceNumber, ValueType)> for InternalKey {
    #[inline]
    fn from((user_key, seq_no, value_type): (UserKey, SequenceNumber, ValueType)) -> Self {
        Self::new(user_key, seq_no, value_type)
    }
}

impl From<InternalKey> for (UserKey, SequenceNumber, ValueType) {
    #[inline]
    fn from(ik: InternalKey) -> Self {
        (ik.user_key, ik.seq_no, ik.value_type)
    }
}

impl From<&Entry> for InternalKey {
    #[inline]
    fn from(entry: &Entry) -> Self {
        entry.to_internal_key()
    }
}

impl From<Entry> for InternalKey {
    #[inline]
    fn from(entry: Entry) -> Self {
        entry.into_internal_key()
    }
}

/// Interval range representing key bounds `[start, end)` or unbounded intervals.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct KeyRange {
    /// Inclusive lower bound (`None` represents unbounded start).
    pub start: Option<Key>,
    /// Exclusive upper bound (`None` represents unbounded end).
    pub end: Option<Key>,
}

impl KeyRange {
    /// Create a new key range with optional start and end bounds.
    #[inline]
    pub fn new(start: Option<impl IntoBytes>, end: Option<impl IntoBytes>) -> Self {
        Self {
            start: start.map(|s| s.into_bytes()),
            end: end.map(|e| e.into_bytes()),
        }
    }

    /// Create an unbounded key range covering all keys `(-inf, +inf)`.
    #[inline]
    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    /// Create a key range from already typed `Key` bounds.
    #[inline]
    pub fn from_bounds(start: Option<Key>, end: Option<Key>) -> Self {
        Self { start, end }
    }

    /// Create a single-point key range `[key, key + \0)`.
    pub fn point(key: impl IntoBytes) -> Self {
        let k = key.into_bytes();
        let mut end = k.to_vec();
        end.push(0);
        Self {
            start: Some(k),
            end: Some(Bytes::from(end)),
        }
    }

    /// Returns `true` if the range contains the given key slice `[start, end)`.
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        if let Some(ref start) = self.start {
            if key < start.as_ref() {
                return false;
            }
        }
        if let Some(ref end) = self.end {
            if key >= end.as_ref() {
                return false;
            }
        }
        true
    }

    /// Returns `true` if this key range overlaps with another key range.
    pub fn overlaps(&self, other: &KeyRange) -> bool {
        if let (Some(self_start), Some(other_end)) = (&self.start, &other.end) {
            if self_start >= other_end {
                return false;
            }
        }
        if let (Some(self_end), Some(other_start)) = (&self.end, &other.start) {
            if self_end <= other_start {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the range is invalid (i.e. start >= end when both bounds are present).
    #[inline]
    pub fn is_empty(&self) -> bool {
        if let (Some(ref start), Some(ref end)) = (&self.start, &self.end) {
            start >= end
        } else {
            false
        }
    }

    /// Get start bound reference.
    #[inline]
    pub fn start(&self) -> Option<&Key> {
        self.start.as_ref()
    }

    /// Get end bound reference.
    #[inline]
    pub fn end(&self) -> Option<&Key> {
        self.end.as_ref()
    }
}

impl fmt::Display for KeyRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let start_str = match &self.start {
            Some(k) => format!("{:?}", k),
            None => "-inf".to_string(),
        };
        let end_str = match &self.end {
            Some(k) => format!("{:?}", k),
            None => "+inf".to_string(),
        };
        write!(f, "[{}, {})", start_str, end_str)
    }
}

/// Trait for custom key ordering comparisons.
pub trait UserKeyComparator: Send + Sync {
    /// Compare two user keys lexicographically.
    fn compare(&self, a: &[u8], b: &[u8]) -> Ordering;

    /// Comparator identifier name.
    fn name(&self) -> &'static str;
}

/// Standard lexicographical user key comparator.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultUserKeyComparator;

impl UserKeyComparator for DefaultUserKeyComparator {
    #[inline]
    fn compare(&self, a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "flashstore.DefaultComparator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_bytes_implementations() {
        let b = Bytes::from_static(b"hello");
        assert_eq!((&b).into_bytes(), Bytes::from_static(b"hello"));
        assert_eq!(b.into_bytes(), Bytes::from_static(b"hello"));

        let vec_u8 = vec![1, 2, 3];
        assert_eq!((&vec_u8).into_bytes(), Bytes::from_static(&[1, 2, 3]));
        assert_eq!(vec_u8.into_bytes(), Bytes::from_static(&[1, 2, 3]));

        let slice_u8: &[u8] = &[4, 5, 6];
        assert_eq!(slice_u8.into_bytes(), Bytes::from_static(&[4, 5, 6]));

        let arr_u8: [u8; 3] = [7, 8, 9];
        assert_eq!((&arr_u8).into_bytes(), Bytes::from_static(&[7, 8, 9]));
        assert_eq!(arr_u8.into_bytes(), Bytes::from_static(&[7, 8, 9]));

        let str_ref = "rust_str";
        assert_eq!(str_ref.into_bytes(), Bytes::from_static(b"rust_str"));

        let string = String::from("rust_string");
        assert_eq!((&string).into_bytes(), Bytes::from_static(b"rust_string"));
        assert_eq!(string.into_bytes(), Bytes::from_static(b"rust_string"));

        let boxed_slice: Box<[u8]> = vec![10, 20].into_boxed_slice();
        assert_eq!((&boxed_slice).into_bytes(), Bytes::from_static(&[10, 20]));
        assert_eq!(boxed_slice.into_bytes(), Bytes::from_static(&[10, 20]));

        let cow_slice: Cow<'_, [u8]> = Cow::Borrowed(&[30, 40]);
        assert_eq!(cow_slice.into_bytes(), Bytes::from_static(&[30, 40]));
        let cow_owned: Cow<'_, [u8]> = Cow::Owned(vec![50, 60]);
        assert_eq!(cow_owned.into_bytes(), Bytes::from_static(&[50, 60]));

        let cow_str_borrowed: Cow<'_, str> = Cow::Borrowed("cow_str");
        assert_eq!(
            cow_str_borrowed.into_bytes(),
            Bytes::from_static(b"cow_str")
        );
        let cow_str_owned: Cow<'_, str> = Cow::Owned("cow_str_owned".to_string());
        assert_eq!(
            cow_str_owned.into_bytes(),
            Bytes::from_static(b"cow_str_owned")
        );

        let arc_slice: Arc<[u8]> = Arc::from(vec![70, 80].into_boxed_slice());
        assert_eq!(arc_slice.into_bytes(), Bytes::from_static(&[70, 80]));
    }

    #[test]
    fn test_value_type_conversions_and_predicates() {
        assert_eq!(ValueType::default(), ValueType::Value);
        assert!(ValueType::Value.is_value());
        assert!(!ValueType::Value.is_tombstone());
        assert_eq!(ValueType::Value.as_u8(), 0);
        assert_eq!(ValueType::Value.as_ref(), "Value");

        assert!(ValueType::Tombstone.is_tombstone());
        assert!(!ValueType::Tombstone.is_value());
        assert_eq!(ValueType::Tombstone.as_u8(), 1);
        assert_eq!(ValueType::Tombstone.as_ref(), "Tombstone");

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
    fn test_compression_type_and_checksum_type() {
        assert_eq!(CompressionType::default(), CompressionType::None);
        assert_eq!(CompressionType::None.as_u8(), 0);
        assert_eq!(CompressionType::Snappy.as_u8(), 1);
        assert_eq!(CompressionType::Zstd.as_u8(), 2);
        assert_eq!(CompressionType::Lz4.as_u8(), 3);

        assert_eq!(CompressionType::from_u8(0), Some(CompressionType::None));
        assert_eq!(CompressionType::from_u8(1), Some(CompressionType::Snappy));
        assert_eq!(CompressionType::from_u8(2), Some(CompressionType::Zstd));
        assert_eq!(CompressionType::from_u8(3), Some(CompressionType::Lz4));
        assert_eq!(CompressionType::from_u8(4), None);

        assert_eq!(CompressionType::try_from(2).unwrap(), CompressionType::Zstd);
        assert!(CompressionType::try_from(10).is_err());
        assert_eq!(format!("{}", CompressionType::Zstd), "Zstd");

        assert_eq!(ChecksumType::default(), ChecksumType::Crc32);
        assert_eq!(ChecksumType::Crc32.as_u8(), 0);
        assert_eq!(ChecksumType::XxHash.as_u8(), 1);

        assert_eq!(ChecksumType::from_u8(0), Some(ChecksumType::Crc32));
        assert_eq!(ChecksumType::from_u8(1), Some(ChecksumType::XxHash));
        assert_eq!(ChecksumType::from_u8(2), None);

        assert_eq!(ChecksumType::try_from(1).unwrap(), ChecksumType::XxHash);
        assert!(ChecksumType::try_from(5).is_err());
        assert_eq!(format!("{}", ChecksumType::Crc32), "CRC32");
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
        assert_eq!(tombstone.estimated_size(), 2 + 8 + 1);

        let from_parts = Entry::from_parts(
            Bytes::from_static(b"kp"),
            Bytes::from_static(b"vp"),
            ValueType::Value,
            50,
        );
        assert_eq!(from_parts.key(), &Bytes::from_static(b"kp"));

        let updated_seq = from_parts.clone().with_seq_no(60);
        assert_eq!(updated_seq.seq_no(), 60);

        let updated_val = from_parts.clone().with_value("new_val");
        assert_eq!(updated_val.value(), &Bytes::from_static(b"new_val"));

        let ik = from_parts.to_internal_key();
        assert_eq!(ik.user_key(), &Bytes::from_static(b"kp"));
        assert_eq!(ik.seq_no(), 50);

        let (k, v, vt, s) = from_parts.into_parts();
        assert_eq!(k, Bytes::from_static(b"kp"));
        assert_eq!(v, Bytes::from_static(b"vp"));
        assert_eq!(vt, ValueType::Value);
        assert_eq!(s, 50);
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

        assert!(format!("{}", original).contains("sample_key"));
    }

    #[test]
    fn test_internal_key_encode_decode() {
        let user_key = Bytes::from_static(b"user_record_key");
        let internal_key = InternalKey::new(user_key.clone(), 123456789, ValueType::Value);
        let encoded = internal_key.encode();

        assert_eq!(encoded.len(), internal_key.encoded_len());

        let decoded = InternalKey::decode(&encoded).expect("decode internal key");
        assert_eq!(internal_key, decoded);
        assert_eq!(decoded.user_key, user_key);
        assert_eq!(decoded.seq_no, 123456789);
        assert_eq!(decoded.value_type, ValueType::Value);
        assert!(decoded.is_value());
        assert!(!decoded.is_tombstone());

        let decoded_slice = InternalKey::decode_from_slice(&encoded).expect("decode from slice");
        assert_eq!(internal_key, decoded_slice);

        let tombstone_ik = InternalKey::new_tombstone(user_key.clone(), 999);
        assert!(tombstone_ik.is_tombstone());
        assert!(!tombstone_ik.is_value());

        let value_ik = InternalKey::new_value(user_key.clone(), 999);
        assert!(value_ik.is_value());

        assert_eq!(tombstone_ik.into_user_key(), user_key);
    }

    #[test]
    fn test_internal_key_corrupted_decode() {
        let short_data = Bytes::from_static(b"short");
        assert!(InternalKey::decode(&short_data).is_err());
        assert!(InternalKey::decode_from_slice(b"short").is_err());

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

        let e1 = Entry::new_value(Bytes::from_static(b"alpha"), Bytes::new(), 10);
        let e2 = Entry::new_value(Bytes::from_static(b"alpha"), Bytes::new(), 20);
        assert!(e2 < e1);
    }

    #[test]
    fn test_internal_key_conversions_and_display() {
        let ik = InternalKey::new("key", 10, ValueType::Value);
        let tuple: (UserKey, SequenceNumber, ValueType) = ik.clone().into();
        assert_eq!(tuple.0, Bytes::from_static(b"key"));
        assert_eq!(tuple.1, 10);
        assert_eq!(tuple.2, ValueType::Value);

        let ik2: InternalKey = tuple.into();
        assert_eq!(ik, ik2);

        let entry = Entry::new_value("key", "val", 10);
        let ik_from_ref = InternalKey::from(&entry);
        assert_eq!(ik, ik_from_ref);
        let ik_from_val = InternalKey::from(entry);
        assert_eq!(ik, ik_from_val);

        assert!(format!("{}", ik).contains("key"));
    }

    #[test]
    fn test_key_range_operations_and_containment() {
        let unbounded = KeyRange::unbounded();
        assert_eq!(unbounded.start(), None);
        assert_eq!(unbounded.end(), None);
        assert!(!unbounded.is_empty());
        assert!(unbounded.contains(b"any_key"));

        let bounded = KeyRange::new(Some("alpha"), Some("omega"));
        assert!(!bounded.is_empty());
        assert!(bounded.contains(b"beta"));
        assert!(bounded.contains(b"alpha"));
        assert!(!bounded.contains(b"omega"));
        assert!(!bounded.contains(b"zzz"));
        assert!(!bounded.contains(b"aaa"));

        let empty_range = KeyRange::new(Some("zeta"), Some("beta"));
        assert!(empty_range.is_empty());

        let pt = KeyRange::point("point_key");
        assert!(pt.contains(b"point_key"));
        assert!(!pt.contains(b"point_key_other"));

        assert!(format!("{}", bounded).contains("alpha"));
    }

    #[test]
    fn test_key_range_overlaps() {
        let r1 = KeyRange::new(Some("a"), Some("m"));
        let r2 = KeyRange::new(Some("g"), Some("z"));
        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));

        let r3 = KeyRange::new(Some("m"), Some("z"));
        assert!(!r1.overlaps(&r3));
        assert!(!r3.overlaps(&r1));

        let r_unbounded = KeyRange::unbounded();
        assert!(r1.overlaps(&r_unbounded));
        assert!(r_unbounded.overlaps(&r1));
    }

    #[test]
    fn test_default_user_key_comparator() {
        let cmp = DefaultUserKeyComparator;
        assert_eq!(cmp.name(), "flashstore.DefaultComparator");
        assert_eq!(cmp.compare(b"a", b"b"), Ordering::Less);
        assert_eq!(cmp.compare(b"b", b"a"), Ordering::Greater);
        assert_eq!(cmp.compare(b"abc", b"abc"), Ordering::Equal);
    }

    #[test]
    fn test_sequence_number_constants() {
        assert_eq!(MIN_SEQUENCE_NUMBER, 0);
        assert_eq!(MAX_SEQUENCE_NUMBER, u64::MAX);
    }
}
