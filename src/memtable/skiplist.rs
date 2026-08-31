use crate::types::{Entry, Key};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub struct SkipList {
    map: RwLock<BTreeMap<Key, Entry>>,
}

impl SkipList {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn insert(&self, entry: Entry) {
        let mut map = self.map.write();
        map.insert(entry.key.clone(), entry);
    }

    pub fn get(&self, key: &Key) -> Option<Entry> {
        let map = self.map.read();
        map.get(key).cloned()
    }

    pub fn iter(&self) -> SkipListIter {
        let entries: Vec<Entry> = self.map.read().values().cloned().collect();
        SkipListIter {
            entries: Arc::new(entries),
            index: 0,
        }
    }
}

pub struct SkipListIter {
    entries: Arc<Vec<Entry>>,
    index: usize,
}

impl SkipListIter {
    pub fn valid(&self) -> bool {
        self.index < self.entries.len()
    }

    pub fn next(&mut self) {
        if self.valid() {
            self.index += 1;
        }
    }

    pub fn item(&self) -> Option<&Entry> {
        if self.valid() {
            Some(&self.entries[self.index])
        } else {
            None
        }
    }
}
