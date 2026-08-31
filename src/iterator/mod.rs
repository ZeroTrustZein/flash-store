use crate::error::Result;
use crate::memtable::skiplist::SkipListIter;
use crate::types::{Entry, Key, Value};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub trait StorageIterator {
    fn valid(&self) -> bool;
    fn next(&mut self) -> Result<()>;
    fn key(&self) -> &Key;
    fn value(&self) -> &Value;
}

pub struct MemtableIterator {
    iter: SkipListIter,
}

impl MemtableIterator {
    pub fn new(iter: SkipListIter) -> Self {
        Self { iter }
    }
}

impl StorageIterator for MemtableIterator {
    fn valid(&self) -> bool {
        self.iter.valid()
    }

    fn next(&mut self) -> Result<()> {
        self.iter.next();
        Ok(())
    }

    fn key(&self) -> &Key {
        &self.iter.item().expect("iterator not valid").key
    }

    fn value(&self) -> &Value {
        &self.iter.item().expect("iterator not valid").value
    }
}

pub struct SSTableIterator {
    entries: Vec<Entry>,
    index: usize,
}

impl SSTableIterator {
    pub fn new(entries: Vec<Entry>) -> Self {
        Self { entries, index: 0 }
    }
}

impl StorageIterator for SSTableIterator {
    fn valid(&self) -> bool {
        self.index < self.entries.len()
    }

    fn next(&mut self) -> Result<()> {
        if self.valid() {
            self.index += 1;
        }
        Ok(())
    }

    fn key(&self) -> &Key {
        &self.entries[self.index].key
    }

    fn value(&self) -> &Value {
        &self.entries[self.index].value
    }
}

struct HeapNode<I: StorageIterator> {
    iter: I,
    index: usize,
}

impl<I: StorageIterator> PartialEq for HeapNode<I> {
    fn eq(&self, other: &Self) -> bool {
        self.iter.key() == other.iter.key()
    }
}

impl<I: StorageIterator> Eq for HeapNode<I> {}

impl<I: StorageIterator> PartialOrd for HeapNode<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: StorageIterator> Ord for HeapNode<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap; lower index = newer data, wins ties
        other
            .iter
            .key()
            .cmp(self.iter.key())
            .then_with(|| self.index.cmp(&other.index))
    }
}

pub struct MergingIterator<I: StorageIterator> {
    heap: BinaryHeap<HeapNode<I>>,
    current: Option<(Key, Value)>,
}

impl<I: StorageIterator> MergingIterator<I> {
    pub fn new(iters: Vec<I>) -> Self {
        let mut heap = BinaryHeap::new();
        for (index, iter) in iters.into_iter().enumerate() {
            if iter.valid() {
                heap.push(HeapNode { iter, index });
            }
        }

        let mut merge_iter = Self {
            heap,
            current: None,
        };
        let _ = merge_iter.advance();
        merge_iter
    }

    fn advance(&mut self) -> Result<()> {
        if let Some(mut top) = self.heap.pop() {
            let key = top.iter.key().clone();
            let value = top.iter.value().clone();
            self.current = Some((key.clone(), value));

            top.iter.next()?;
            if top.iter.valid() {
                self.heap.push(top);
            }

            // Deduplicate matching keys from other iterators
            while let Some(peek) = self.heap.peek() {
                if peek.iter.key() == &key {
                    // pop matching iterator, advance, reinsert if still valid
                    let mut node = self.heap.pop().unwrap();
                    node.iter.next()?;
                    if node.iter.valid() {
                        self.heap.push(node);
                    }
                } else {
                    break;
                }
            }
        } else {
            self.current = None;
        }
        Ok(())
    }
}

impl<I: StorageIterator> StorageIterator for MergingIterator<I> {
    fn valid(&self) -> bool {
        self.current.is_some()
    }

    fn next(&mut self) -> Result<()> {
        self.advance()
    }

    fn key(&self) -> &Key {
        &self.current.as_ref().expect("iterator not valid").0
    }

    fn value(&self) -> &Value {
        &self.current.as_ref().expect("iterator not valid").1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_sstable_iterator() {
        let entries = vec![
            Entry::new_value(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"), 1),
            Entry::new_value(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"), 2),
        ];

        let mut iter = SSTableIterator::new(entries);
        assert!(iter.valid());
        assert_eq!(iter.key(), &Bytes::from_static(b"k1"));
        assert_eq!(iter.value(), &Bytes::from_static(b"v1"));

        iter.next().unwrap();
        assert!(iter.valid());
        assert_eq!(iter.key(), &Bytes::from_static(b"k2"));

        iter.next().unwrap();
        assert!(!iter.valid());
    }

    #[test]
    fn test_merging_iterator_dedup() {
        let iter1 = SSTableIterator::new(vec![
            Entry::new_value(Bytes::from_static(b"a"), Bytes::from_static(b"v_a_1"), 1),
            Entry::new_value(Bytes::from_static(b"c"), Bytes::from_static(b"v_c_1"), 1),
        ]);

        let iter2 = SSTableIterator::new(vec![
            Entry::new_value(Bytes::from_static(b"a"), Bytes::from_static(b"v_a_2"), 2),
            Entry::new_value(Bytes::from_static(b"b"), Bytes::from_static(b"v_b_2"), 2),
        ]);

        let mut merger = MergingIterator::new(vec![iter1, iter2]);

        let mut results = Vec::new();
        while merger.valid() {
            results.push((merger.key().clone(), merger.value().clone()));
            merger.next().unwrap();
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, Bytes::from_static(b"a"));
        assert_eq!(results[1].0, Bytes::from_static(b"b"));
        assert_eq!(results[2].0, Bytes::from_static(b"c"));
    }
}
