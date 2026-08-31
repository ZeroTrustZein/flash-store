use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub trait BlockCache: Send + Sync {
    fn get(&self, key: &(u64, u64)) -> Option<Bytes>;
    fn insert(&self, key: (u64, u64), value: Bytes);
}

pub struct LruBlockCache {
    capacity: usize,
    cache: RwLock<HashMap<(u64, u64), Bytes>>,
}

impl LruBlockCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl BlockCache for LruBlockCache {
    fn get(&self, key: &(u64, u64)) -> Option<Bytes> {
        let cache = self.cache.read();
        cache.get(key).cloned()
    }

    fn insert(&self, key: (u64, u64), value: Bytes) {
        let mut cache = self.cache.write();
        if cache.len() >= self.capacity {
            cache.clear(); // Simple eviction stub
        }
        cache.insert(key, value);
    }
}
