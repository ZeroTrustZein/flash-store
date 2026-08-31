pub mod batch;
pub mod cache;
pub mod compaction;
pub mod config;
pub mod engine;
pub mod error;
pub mod iterator;
pub mod manifest;
pub mod memtable;
pub mod sstable;
pub mod types;
pub mod wal;

pub use batch::WriteBatch;
pub use cache::{BlockCache, LruBlockCache};
pub use compaction::{CompactionTask, Compactor};
pub use config::{Options, OptionsBuilder};
pub use engine::{FlashStore, Stats};
pub use error::{FlashStoreError, Result};
pub use iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
pub use types::{Entry, InternalKey, Key, SequenceNumber, UserKey, Value, ValueType};

pub mod prelude {
    pub use crate::batch::WriteBatch;
    pub use crate::cache::{BlockCache, LruBlockCache};
    pub use crate::compaction::{CompactionTask, Compactor};
    pub use crate::config::{Options, OptionsBuilder};
    pub use crate::engine::FlashStore;
    pub use crate::error::{FlashStoreError, Result};
    pub use crate::iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
    pub use crate::types::{Entry, InternalKey, Key, Value, ValueType};
}
