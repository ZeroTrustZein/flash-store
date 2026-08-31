pub mod skiplist;

use crate::types::{Entry, Key, SequenceNumber, ValueType};
use bytes::Bytes;
use skiplist::SkipList;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct MemTable {
    id: usize,
    table: Arc<SkipList>,
    size_bytes: AtomicUsize,
}

impl MemTable {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            table: Arc::new(SkipList::new()),
            size_bytes: AtomicUsize::new(0),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn put(&self, key: Key, value: Bytes, seq_no: SequenceNumber) {
        let entry = Entry::new_value(key, value, seq_no);
        self.size_bytes
            .fetch_add(entry.estimated_size(), Ordering::Relaxed);
        self.table.insert(entry);
    }

    pub fn delete(&self, key: Key, seq_no: SequenceNumber) {
        let entry = Entry::new_tombstone(key, seq_no);
        self.size_bytes
            .fetch_add(entry.estimated_size(), Ordering::Relaxed);
        self.table.insert(entry);
    }

    pub fn get(&self, key: &Key) -> Option<Option<Bytes>> {
        self.table.get(key).map(|entry| {
            if entry.value_type == ValueType::Tombstone {
                None
            } else {
                Some(entry.value)
            }
        })
    }

    pub fn approximate_size(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    pub fn iter(&self) -> skiplist::SkipListIter {
        self.table.iter()
    }
}
