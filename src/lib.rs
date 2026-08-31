//! # FlashStore
//!
//! An embedded, thread-safe, high-performance Log-Structured Merge-Tree (LSM-Tree)
//! key-value storage engine implemented in Rust.
//!
//! ## Architecture Overview
//!
//! - **Write-Ahead Log (WAL)**: Append-only persistent journal with CRC32 checksum framing for durability.
//! - **MemTable**: In-memory sorted index backed by concurrent skiplists for rapid writes.
//! - **SSTable**: Immutable disk-based tables with data blocks, Bloom filters, meta index blocks, and fixed-size footers.
//! - **Block Cache**: Configurable LRU cache for uncompressed data blocks.
//! - **Compaction Engine**: Multi-level tiered compaction that purges overwritten versions and tombstones.
//! - **Manifest & Versioning**: ACID version edits tracking SSTable level assignments across crash recoveries.
//!
//! ## Quick Start
//!
//! ```rust
//! use flash_store::prelude::*;
//!
//! # fn main() -> Result<()> {
//! # let dir = tempfile::tempdir().unwrap();
//! let options = OptionsBuilder::new()
//!     .dir(dir.path())
//!     .memtable_size(4 * 1024 * 1024)
//!     .build();
//!
//! let db = FlashStore::open(options)?;
//!
//! // Basic CRUD
//! db.put("user_1001", "Alice")?;
//! if let Some(value) = db.get("user_1001")? {
//!     println!("Found user: {}", String::from_utf8_lossy(&value));
//! }
//! db.delete("user_1001")?;
//! # Ok(())
//! # }
//! ```

pub mod batch;
pub mod cache;
pub mod cli;
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
pub use cli::{Cmd, GlobalOpts, OutputFormat};
pub use compaction::{CompactionTask, Compactor};
pub use config::{Options, OptionsBuilder};
pub use engine::{FlashStore, Stats};
pub use error::{FlashStoreError, Result};
pub use iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
pub use types::{Entry, InternalKey, IntoBytes, Key, SequenceNumber, UserKey, Value, ValueType};

pub mod prelude {
    pub use crate::batch::WriteBatch;
    pub use crate::cache::{BlockCache, LruBlockCache};
    pub use crate::cli::{Cmd, GlobalOpts, OutputFormat};
    pub use crate::compaction::{CompactionTask, Compactor};
    pub use crate::config::{Options, OptionsBuilder};
    pub use crate::engine::{FlashStore, Stats};
    pub use crate::error::{FlashStoreError, Result};
    pub use crate::iterator::{MemtableIterator, MergingIterator, SSTableIterator, StorageIterator};
    pub use crate::types::{Entry, InternalKey, IntoBytes, Key, Value, ValueType};
}
