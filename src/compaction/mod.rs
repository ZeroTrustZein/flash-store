use crate::error::Result;
use crate::manifest::version::FileMetaData;

#[derive(Debug, Clone)]
pub struct CompactionTask {
    pub level: usize,
    pub input_files: Vec<FileMetaData>,
    pub target_level: usize,
}

pub struct Compactor {
    max_levels: usize,
    base_level_size_bytes: usize,
}

impl Compactor {
    pub fn new(max_levels: usize, base_level_size_bytes: usize) -> Self {
        Self {
            max_levels,
            base_level_size_bytes,
        }
    }

    pub fn pick_compaction(
        &self,
        levels: &[Vec<FileMetaData>],
    ) -> Option<CompactionTask> {
        // Simple leveled compaction strategy trigger stub
        for (level_idx, files) in levels.iter().enumerate() {
            if level_idx + 1 >= self.max_levels {
                continue;
            }
            let level_size: u64 = files.iter().map(|f| f.file_size).sum();
            let limit = (self.base_level_size_bytes * (10_usize.pow(level_idx as u32))) as u64;

            if level_size > limit && !files.is_empty() {
                return Some(CompactionTask {
                    level: level_idx,
                    input_files: files.clone(),
                    target_level: level_idx + 1,
                });
            }
        }
        None
    }

    pub fn run_compaction(&self, _task: &CompactionTask) -> Result<()> {
        // Scaffolding stub for running compaction merge
        Ok(())
    }
}
