use crate::types::{IntoBytes, Key, Value};

#[derive(Debug, Clone)]
pub enum BatchOp {
    Put(Key, Value),
    Delete(Key),
}

#[derive(Default, Debug, Clone)]
pub struct WriteBatch {
    pub ops: Vec<BatchOp>,
}

impl WriteBatch {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn put<K: IntoBytes, V: IntoBytes>(&mut self, key: K, value: V) {
        self.ops
            .push(BatchOp::Put(key.into_bytes(), value.into_bytes()));
    }

    pub fn delete<K: IntoBytes>(&mut self, key: K) {
        self.ops.push(BatchOp::Delete(key.into_bytes()));
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn clear(&mut self) {
        self.ops.clear();
    }
}
