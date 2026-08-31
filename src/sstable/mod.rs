pub mod block;
pub mod bloom;
pub mod builder;
pub mod footer;
pub mod reader;

pub use block::Block;
pub use bloom::BloomFilter;
pub use builder::TableBuilder;
pub use footer::Footer;
pub use reader::TableReader;

use std::path::{Path, PathBuf};

/// Formats an SSTable file number to its canonical 6-digit filename (e.g. "000001.sst").
#[inline]
pub fn table_file_name(file_number: u64) -> String {
    format!("{:06}.sst", file_number)
}

/// Constructs the full path to an SSTable file given the database directory and file number.
#[inline]
pub fn table_path<P: AsRef<Path>>(dir: P, file_number: u64) -> PathBuf {
    dir.as_ref().join(table_file_name(file_number))
}
