use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMetaData {
    pub file_number: u64,
    pub file_size: u64,
    pub smallest_key: Bytes,
    pub largest_key: Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VersionEdit {
    pub next_file_number: Option<u64>,
    pub last_sequence: Option<u64>,
    pub new_files: Vec<(usize, FileMetaData)>, // (level, file)
    pub deleted_files: Vec<(usize, u64)>,      // (level, file_number)
}

impl VersionEdit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, level: usize, meta: FileMetaData) {
        self.new_files.push((level, meta));
    }

    pub fn delete_file(&mut self, level: usize, file_number: u64) {
        self.deleted_files.push((level, file_number));
    }
}

#[derive(Debug, Clone)]
pub struct Version {
    pub levels: Vec<Vec<FileMetaData>>,
}

impl Version {
    pub fn new(max_levels: usize) -> Self {
        Self {
            levels: vec![Vec::new(); max_levels],
        }
    }
}
