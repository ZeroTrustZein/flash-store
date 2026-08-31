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
        let new_size = entry.estimated_size();
        let old_size = self.table.insert(entry).unwrap_or(0);
        if new_size >= old_size {
            self.size_bytes
                .fetch_add(new_size - old_size, Ordering::Relaxed);
        } else {
            self.size_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
        }
    }

    pub fn delete(&self, key: Key, seq_no: SequenceNumber) {
        let entry = Entry::new_tombstone(key, seq_no);
        let new_size = entry.estimated_size();
        let old_size = self.table.insert(entry).unwrap_or(0);
        if new_size >= old_size {
            self.size_bytes
                .fetch_add(new_size - old_size, Ordering::Relaxed);
        } else {
            self.size_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
        }
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_crud_and_size() {
        let mem = MemTable::new(1);
        assert_eq!(mem.id(), 1);
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert_eq!(mem.approximate_size(), 0);

        let k1 = Bytes::from_static(b"k1");
        let v1 = Bytes::from_static(b"v1");
        mem.put(k1.clone(), v1.clone(), 100);

        assert!(!mem.is_empty());
        assert_eq!(mem.len(), 1);
        assert!(mem.approximate_size() > 0);
        assert_eq!(mem.get(&k1), Some(Some(v1)));

        // Non-existent key
        let k_missing = Bytes::from_static(b"k_missing");
        assert_eq!(mem.get(&k_missing), None);

        // Delete key (tombstone)
        mem.delete(k1.clone(), 101);
        assert_eq!(mem.get(&k1), Some(None));

        // Overwrite again
        let v1_new = Bytes::from_static(b"v1_new");
        mem.put(k1.clone(), v1_new.clone(), 102);
        assert_eq!(mem.get(&k1), Some(Some(v1_new)));
    }

    #[test]
    fn test_memtable_iteration() {
        let mem = MemTable::new(2);
        mem.put(Bytes::from_static(b"z"), Bytes::from_static(b"26"), 1);
        mem.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"), 2);
        mem.put(Bytes::from_static(b"m"), Bytes::from_static(b"13"), 3);

        let mut iter = mem.iter();
        let mut keys = Vec::new();
        while iter.valid() {
            if let Some(entry) = iter.item() {
                keys.push(entry.key.clone());
            }
            iter.next();
        }

        // Iteration order must be sorted ascending
        assert_eq!(
            keys,
            vec![
                Bytes::from_static(b"a"),
                Bytes::from_static(b"m"),
                Bytes::from_static(b"z"),
            ]
        );
    }
}
