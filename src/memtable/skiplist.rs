use crate::types::{Entry, Key};
use parking_lot::RwLock;
use std::collections::BTreeMap;

/// Thread-safe sorted in-memory skiplist/map backing the MemTable.
#[derive(Default)]
pub struct SkipList {
    map: RwLock<BTreeMap<Key, Entry>>,
}

impl SkipList {
    /// Creates a new empty `SkipList`.
    pub fn new() -> Self {
        Self {
            map: RwLock::new(BTreeMap::new()),
        }
    }

    /// Inserts an entry into the skip list.
    /// Returns the estimated size of the replaced entry, if one previously existed.
    pub fn insert(&self, entry: Entry) -> Option<usize> {
        let mut map = self.map.write();
        let old = map.insert(entry.key.clone(), entry);
        old.map(|e| e.estimated_size())
    }

    /// Look up an entry by key.
    pub fn get(&self, key: &Key) -> Option<Entry> {
        let map = self.map.read();
        map.get(key).cloned()
    }

    /// Returns the number of elements in the skip list.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Returns `true` if the skip list contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }

    /// Creates an iterator over a snapshot of the current entries.
    pub fn iter(&self) -> SkipListIter {
        let entries: Vec<Entry> = self.map.read().values().cloned().collect();
        SkipListIter { entries, index: 0 }
    }
}

/// Iterator over a frozen snapshot of `SkipList` entries.
pub struct SkipListIter {
    entries: Vec<Entry>,
    index: usize,
}

impl SkipListIter {
    /// Returns `true` if the iterator is currently pointing at a valid entry.
    #[inline]
    pub fn valid(&self) -> bool {
        self.index < self.entries.len()
    }

    /// Advances the iterator to the next entry.
    #[inline]
    pub fn next(&mut self) {
        if self.valid() {
            self.index += 1;
        }
    }

    /// Returns a reference to the current entry, or `None` if the iterator is exhausted.
    #[inline]
    pub fn item(&self) -> Option<&Entry> {
        if self.valid() {
            Some(&self.entries[self.index])
        } else {
            None
        }
    }
}
