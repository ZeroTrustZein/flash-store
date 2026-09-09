use thiserror::Error;

#[derive(Error, Debug)]
pub enum FlashStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Data corruption error: {0}")]
    Corruption(String),

    #[error("Compaction error: {0}")]
    CompactionError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("WAL error: {0}")]
    WalError(String),

    #[error("SSTable error: {0}")]
    TableError(String),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Bincode serialization/deserialization error: {0}")]
    BincodeError(#[from] bincode::Error),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("RAG error: {0}")]
    Rag(String),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, FlashStoreError>;
