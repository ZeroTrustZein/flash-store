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
pub use config::{Options, OptionsBuilder};
pub use engine::{FlashStore, Stats};
pub use error::{FlashStoreError, Result};
pub use types::{Entry, Key, SequenceNumber, UserKey, Value, ValueType};

pub mod prelude {
    pub use crate::batch::WriteBatch;
    pub use crate::config::{Options, OptionsBuilder};
    pub use crate::engine::FlashStore;
    pub use crate::error::{FlashStoreError, Result};
    pub use crate::types::{Entry, Key, Value, ValueType};
}
