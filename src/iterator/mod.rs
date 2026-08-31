use crate::error::Result;
use crate::memtable::skiplist::SkipListIter;
use crate::types::{Entry, Key, Value};
use bytes::Bytes;
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
        // Reverse for min-heap
        other
            .iter
            .key()
            .cmp(self.iter.key())
            .then_with(|| other.index.cmp(&self.index))
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
