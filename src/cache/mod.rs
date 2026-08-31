use bytes::Bytes;
use parking_lot::Mutex;
use std::collections::HashMap;

pub trait BlockCache: Send + Sync {
    fn get(&self, key: &(u64, u64)) -> Option<Bytes>;
    fn insert(&self, key: (u64, u64), value: Bytes);
    fn remove(&self, key: &(u64, u64)) -> Option<Bytes>;
    fn clear(&self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

#[derive(Clone)]
struct LruNode {
    key: (u64, u64),
    value: Bytes,
    prev: Option<usize>,
    next: Option<usize>,
}

struct LruInner {
    nodes: Vec<Option<LruNode>>,
    free_indices: Vec<usize>,
    map: HashMap<(u64, u64), usize>,
    head: Option<usize>, // Most recently used
    tail: Option<usize>, // Least recently used
    capacity: usize,
}

impl LruInner {
    fn new(capacity: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free_indices: Vec::new(),
            map: HashMap::new(),
            head: None,
            tail: None,
            capacity,
        }
    }

    fn detach(&mut self, idx: usize) {
        let node = match self.nodes[idx].as_ref() {
            Some(n) => n,
            None => return,
        };
        let prev = node.prev;
        let next = node.next;

        if let Some(prev_idx) = prev {
            if let Some(prev_node) = self.nodes[prev_idx].as_mut() {
                prev_node.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(next_idx) = next {
            if let Some(next_node) = self.nodes[next_idx].as_mut() {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        if let Some(node_mut) = self.nodes[idx].as_mut() {
            node_mut.prev = None;
            node_mut.next = None;
        }
    }

    fn attach_head(&mut self, idx: usize) {
        if let Some(node_mut) = self.nodes[idx].as_mut() {
            node_mut.prev = None;
            node_mut.next = self.head;
        }

        if let Some(old_head) = self.head {
            if let Some(old_head_node) = self.nodes[old_head].as_mut() {
                old_head_node.prev = Some(idx);
            }
        } else {
            self.tail = Some(idx);
        }

        self.head = Some(idx);
    }

    fn get(&mut self, key: &(u64, u64)) -> Option<Bytes> {
        if let Some(&idx) = self.map.get(key) {
            self.detach(idx);
            self.attach_head(idx);
            self.nodes[idx].as_ref().map(|n| n.value.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, key: (u64, u64), value: Bytes) {
        if self.capacity == 0 {
            return;
        }

        if let Some(&idx) = self.map.get(&key) {
            if let Some(node) = self.nodes[idx].as_mut() {
                node.value = value;
            }
            self.detach(idx);
            self.attach_head(idx);
            return;
        }

        if self.map.len() >= self.capacity {
            if let Some(lru_idx) = self.tail {
                if let Some(lru_node) = self.nodes[lru_idx].take() {
                    self.map.remove(&lru_node.key);
                    self.detach(lru_idx);
                    self.free_indices.push(lru_idx);
                }
            }
        }

        let new_node = LruNode {
            key,
            value,
            prev: None,
            next: None,
        };

        let slot_idx = if let Some(free_idx) = self.free_indices.pop() {
            self.nodes[free_idx] = Some(new_node);
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(new_node));
            idx
        };

        self.attach_head(slot_idx);
        self.map.insert(key, slot_idx);
    }

    fn remove(&mut self, key: &(u64, u64)) -> Option<Bytes> {
        if let Some(idx) = self.map.remove(key) {
            self.detach(idx);
            let val = self.nodes[idx].take().map(|n| n.value);
            self.free_indices.push(idx);
            val
        } else {
            None
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.free_indices.clear();
        self.map.clear();
        self.head = None;
        self.tail = None;
    }
}

pub struct LruBlockCache {
    inner: Mutex<LruInner>,
}

impl LruBlockCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruInner::new(capacity)),
        }
    }

    pub fn contains_key(&self, key: &(u64, u64)) -> bool {
        self.inner.lock().map.contains_key(key)
    }
}

impl BlockCache for LruBlockCache {
    fn get(&self, key: &(u64, u64)) -> Option<Bytes> {
        self.inner.lock().get(key)
    }

    fn insert(&self, key: (u64, u64), value: Bytes) {
        self.inner.lock().insert(key, value);
    }

    fn remove(&self, key: &(u64, u64)) -> Option<Bytes> {
        self.inner.lock().remove(key)
    }

    fn clear(&self) {
        self.inner.lock().clear();
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.lock().map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic_operations() {
        let cache = LruBlockCache::new(2);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let k1 = (1, 0);
        let k2 = (1, 1);
        let k3 = (2, 0);

        let v1 = Bytes::from_static(b"block_1_0");
        let v2 = Bytes::from_static(b"block_1_1");
        let v3 = Bytes::from_static(b"block_2_0");

        cache.insert(k1, v1.clone());
        cache.insert(k2, v2.clone());
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());

        assert_eq!(cache.get(&k1), Some(v1.clone()));
        assert_eq!(cache.get(&k2), Some(v2.clone()));

        // Inserting k3 must evict the least recently used entry (which is k1 because k2 was accessed most recently)
        // Wait, k1 was accessed, then k2 was accessed, so k1 is LRU!
        cache.insert(k3, v3.clone());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&k1), None);
        assert_eq!(cache.get(&k2), Some(v2));
        assert_eq!(cache.get(&k3), Some(v3.clone()));

        // Remove
        assert_eq!(cache.remove(&k3), Some(v3));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&k3), None);

        // Clear
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_cache_update_existing_key() {
        let cache = LruBlockCache::new(2);
        let k1 = (10, 1);
        let k2 = (10, 2);

        cache.insert(k1, Bytes::from_static(b"initial"));
        cache.insert(k2, Bytes::from_static(b"other"));
        cache.insert(k1, Bytes::from_static(b"updated"));

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&k1), Some(Bytes::from_static(b"updated")));
    }
}
