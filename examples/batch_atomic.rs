//! Example: Atomic Batch Operations in FlashStore
//!
//! This example demonstrates how to use `WriteBatch` to perform multiple
//! Put and Delete operations atomically in a single write operation.

use flash_store::prelude::*;
use tempfile::tempdir;

fn main() -> Result<()> {
    // 1. Initialize temporary directory and storage options
    let dir = tempdir().map_err(FlashStoreError::Io)?;
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(2 * 1024 * 1024)
        .build();

    let db = FlashStore::open(options)?;
    println!("Opened FlashStore at {:?}", dir.path());

    // 2. Pre-populate some existing data
    db.put("account:alice", "100")?;
    db.put("account:bob", "50")?;
    db.put("account:charlie", "20")?;
    println!("Initial balances:");
    println!(
        "  Alice:   {:?}",
        db.get("account:alice")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );
    println!(
        "  Bob:     {:?}",
        db.get("account:bob")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );
    println!(
        "  Charlie: {:?}",
        db.get("account:charlie")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );

    // 3. Construct an atomic batch (Transfer 30 from Alice to Bob, and delete Charlie's account)
    println!("\nExecuting atomic WriteBatch (Alice -30, Bob +30, Delete Charlie)...");
    let mut batch = WriteBatch::new();
    batch.put("account:alice", "70");
    batch.put("account:bob", "80");
    batch.delete("account:charlie");

    println!("Batch size: {} operations", batch.len());

    // 4. Commit batch atomically to WAL and MemTable
    db.write_batch(batch)?;
    println!("WriteBatch applied successfully!");

    // 5. Verify the atomic changes
    println!("\nUpdated balances after batch:");
    println!(
        "  Alice:   {:?}",
        db.get("account:alice")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );
    println!(
        "  Bob:     {:?}",
        db.get("account:bob")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );
    println!(
        "  Charlie: {:?}",
        db.get("account:charlie")?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    );

    assert_eq!(db.get("account:alice")?.unwrap(), "70".as_bytes());
    assert_eq!(db.get("account:bob")?.unwrap(), "80".as_bytes());
    assert_eq!(db.get("account:charlie")?, None);

    println!("\nBatch atomic operations verified successfully!");
    Ok(())
}
