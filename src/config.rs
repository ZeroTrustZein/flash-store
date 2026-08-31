use std::path::PathBuf;

/// Configuration options for initializing a [`FlashStore`](crate::engine::FlashStore) instance.
#[derive(Debug, Clone)]
pub struct Options {
    /// Directory path where database files (WAL, SSTables, MANIFEST) are stored.
    pub dir: PathBuf,
    /// Maximum byte size of the active MemTable before triggering a flush (default: 4MB).
    pub memtable_size: usize,
    /// Target byte size of uncompressed SSTable data blocks (default: 4KB).
    pub block_size: usize,
    /// Number of bits per key allocated in Bloom filters (default: 10 bits/key, ~1% false positive rate).
    pub bloom_bits_per_key: usize,
    /// Maximum number of LSM-tree levels (default: 7).
    pub max_levels: usize,
    /// Target size in bytes for Level 1 compaction (default: 10MB).
    pub base_level_size_bytes: usize,
    /// If `true`, executes `fsync` on the WAL file after every write for maximum durability (default: `false`).
    pub sync_wal: bool,
    /// Maximum memory capacity for the LRU block cache in bytes (default: 64MB).
    pub block_cache_size: usize,
    /// If `true`, automatically creates the database directory if it does not exist (default: `true`).
    pub create_if_missing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./data"),
            memtable_size: 4 * 1024 * 1024, // 4 MB
            block_size: 4 * 1024,           // 4 KB
            bloom_bits_per_key: 10,         // ~1% false positive rate
            max_levels: 7,
            base_level_size_bytes: 10 * 1024 * 1024, // 10 MB
            sync_wal: false,
            block_cache_size: 64 * 1024 * 1024, // 64 MB
            create_if_missing: true,
        }
    }
}

/// Fluent builder for constructing [`Options`].
///
/// # Examples
///
/// ```rust
/// use flash_store::config::OptionsBuilder;
///
/// let options = OptionsBuilder::new()
///     .dir("/tmp/flashstore_data")
///     .memtable_size(8 * 1024 * 1024)
///     .block_cache_size(128 * 1024 * 1024)
///     .sync_wal(true)
///     .build();
///
/// assert_eq!(options.memtable_size, 8 * 1024 * 1024);
/// assert!(options.sync_wal);
/// ```
pub struct OptionsBuilder {
    options: Options,
}

impl OptionsBuilder {
    /// Creates a new `OptionsBuilder` initialized with default settings.
    pub fn new() -> Self {
        Self {
            options: Options::default(),
        }
    }

    /// Sets the database root directory.
    pub fn dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.options.dir = dir.into();
        self
    }

    /// Sets the maximum memory size (in bytes) of the active MemTable before flushing.
    pub fn memtable_size(mut self, size: usize) -> Self {
        self.options.memtable_size = size;
        self
    }

    /// Sets the target data block size (in bytes) for SSTable construction.
    pub fn block_size(mut self, size: usize) -> Self {
        self.options.block_size = size;
        self
    }

    /// Sets the number of bits allocated per key in SSTable Bloom filters.
    pub fn bloom_bits_per_key(mut self, bits: usize) -> Self {
        self.options.bloom_bits_per_key = bits;
        self
    }

    /// Sets the maximum number of LSM levels (e.g. L0 through L6).
    pub fn max_levels(mut self, levels: usize) -> Self {
        self.options.max_levels = levels;
        self
    }

    /// Sets the base size (in bytes) for Level 1 compaction calculations.
    pub fn base_level_size_bytes(mut self, bytes: usize) -> Self {
        self.options.base_level_size_bytes = bytes;
        self
    }

    /// Enables or disables synchronous `fsync` for the Write-Ahead Log.
    pub fn sync_wal(mut self, sync: bool) -> Self {
        self.options.sync_wal = sync;
        self
    }

    /// Sets the maximum memory capacity (in bytes) for the in-memory LRU block cache.
    pub fn block_cache_size(mut self, size: usize) -> Self {
        self.options.block_cache_size = size;
        self
    }

    /// Sets whether to automatically create missing database directories.
    pub fn create_if_missing(mut self, create: bool) -> Self {
        self.options.create_if_missing = create;
        self
    }

    /// Consumes the builder and returns the configured [`Options`].
    pub fn build(self) -> Options {
        self.options
    }
}

impl Default for OptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}
