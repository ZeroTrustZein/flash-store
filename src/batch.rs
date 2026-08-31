use crate::types::{IntoBytes, Key, Value};

/// Represents an individual mutating operation within a [`WriteBatch`].
#[derive(Debug, Clone)]
pub enum BatchOp {
    /// Insert or update a key-value entry.
    Put(Key, Value),
    /// Delete an entry by key (records a tombstone).
    Delete(Key),
}

/// An atomic collection of mutating operations (`Put` and `Delete`).
///
/// All operations within a `WriteBatch` are written to the Write-Ahead Log (WAL)
/// and applied to the active MemTable as a single atomic unit.
///
/// # Examples
///
/// ```rust
/// use flash_store::batch::WriteBatch;
///
/// let mut batch = WriteBatch::new();
/// batch.put("user:101", "Alice");
/// batch.put("user:102", "Bob");
/// batch.delete("user:100");
///
/// assert_eq!(batch.len(), 3);
/// assert!(!batch.is_empty());
/// ```
#[derive(Default, Debug, Clone)]
pub struct WriteBatch {
    /// Ordered list of batch operations to apply.
    pub ops: Vec<BatchOp>,
}

impl WriteBatch {
    /// Creates an empty `WriteBatch`.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Queues a `Put` operation in the batch.
    pub fn put<K: IntoBytes, V: IntoBytes>(&mut self, key: K, value: V) {
        self.ops
            .push(BatchOp::Put(key.into_bytes(), value.into_bytes()));
    }

    /// Queues a `Delete` operation in the batch.
    pub fn delete<K: IntoBytes>(&mut self, key: K) {
        self.ops.push(BatchOp::Delete(key.into_bytes()));
    }

    /// Returns the number of operations in this batch.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if the batch contains no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Clears all operations from this batch.
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}
