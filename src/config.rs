use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub memtable_size: usize,
    pub block_size: usize,
    pub bloom_bits_per_key: usize,
    pub max_levels: usize,
    pub base_level_size_bytes: usize,
    pub sync_wal: bool,
    pub block_cache_size: usize,
    pub create_if_missing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./data"),
            memtable_size: 4 * 1024 * 1024,      // 4 MB
            block_size: 4 * 1024,                // 4 KB
            bloom_bits_per_key: 10,              // ~1% false positive rate
            max_levels: 7,
            base_level_size_bytes: 10 * 1024 * 1024, // 10 MB
            sync_wal: false,
            block_cache_size: 64 * 1024 * 1024,  // 64 MB
            create_if_missing: true,
        }
    }
}

pub struct OptionsBuilder {
    options: Options,
}

impl OptionsBuilder {
    pub fn new() -> Self {
        Self {
            options: Options::default(),
        }
    }

    pub fn dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.options.dir = dir.into();
        self
    }

    pub fn memtable_size(mut self, size: usize) -> Self {
        self.options.memtable_size = size;
        self
    }

    pub fn block_size(mut self, size: usize) -> Self {
        self.options.block_size = size;
        self
    }

    pub fn bloom_bits_per_key(mut self, bits: usize) -> Self {
        self.options.bloom_bits_per_key = bits;
        self
    }

    pub fn max_levels(mut self, levels: usize) -> Self {
        self.options.max_levels = levels;
        self
    }

    pub fn base_level_size_bytes(mut self, bytes: usize) -> Self {
        self.options.base_level_size_bytes = bytes;
        self
    }

    pub fn sync_wal(mut self, sync: bool) -> Self {
        self.options.sync_wal = sync;
        self
    }

    pub fn block_cache_size(mut self, size: usize) -> Self {
        self.options.block_cache_size = size;
        self
    }

    pub fn create_if_missing(mut self, create: bool) -> Self {
        self.options.create_if_missing = create;
        self
    }

    pub fn build(self) -> Options {
        self.options
    }
}

impl Default for OptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}
