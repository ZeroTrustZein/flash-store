use bytes::Bytes;
use serde::{Deserialize, Serialize};

pub type SequenceNumber = u64;
pub type UserKey = Bytes;
pub type Key = Bytes;
pub type Value = Bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueType {
    Value = 0,
    Tombstone = 1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub value: Value,
    pub value_type: ValueType,
    pub seq_no: SequenceNumber,
}

impl Entry {
    pub fn new_value(key: Key, value: Value, seq_no: SequenceNumber) -> Self {
        Self {
            key,
            value,
            value_type: ValueType::Value,
            seq_no,
        }
    }

    pub fn new_tombstone(key: Key, seq_no: SequenceNumber) -> Self {
        Self {
            key,
            value: Bytes::new(),
            value_type: ValueType::Tombstone,
            seq_no,
        }
    }

    pub fn is_tombstone(&self) -> bool {
        self.value_type == ValueType::Tombstone
    }

    pub fn estimated_size(&self) -> usize {
        self.key.len() + self.value.len() + std::mem::size_of::<SequenceNumber>() + 1
    }
}
