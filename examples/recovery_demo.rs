//! Example: Crash Simulation and WAL/SSTable Recovery in FlashStore
//!
//! This example demonstrates FlashStore's durability guarantees:
//! 1. Writing data to MemTable and WAL without flushing.
//! 2. Simulating an ungraceful process termination by dropping the instance.
//! 3. Re-opening FlashStore on the same directory and verifying complete WAL replay.
//! 4. Writing more data, flushing some to SSTable, adding more WAL data, restarting, and verifying both.

use flash_store::prelude::*;
use tempfile::tempdir;

fn main() -> Result<()> {
    let dir = tempdir().map_err(FlashStoreError::Io)?;
    let db_path = dir.path().to_path_buf();

    println!("=== Phase 1: Initial Writes to WAL & MemTable ===");
    {
        let options = OptionsBuilder::new()
            .dir(&db_path)
            .sync_wal(true) // Ensure synchronous fsync on writes
            .memtable_size(1024 * 1024)
            .build();

        let db = FlashStore::open(options)?;
        db.put("session:101", "active")?;
        db.put("session:102", "pending")?;
        db.put("session:103", "expired")?;
        db.delete("session:103")?; // Tombstone in WAL

        println!(
            "Wrote 3 sessions (1 deleted) to WAL. Dropping DB handle without calling db.flush()..."
        );
        // db is dropped here without flush
    }

    println!("\n=== Phase 2: Simulating Restart / Recovery from WAL ===");
    {
        let options = OptionsBuilder::new().dir(&db_path).sync_wal(true).build();

        let db = FlashStore::open(options)?;
        println!("FlashStore re-opened successfully. Verifying recovered records...");

        assert_eq!(db.get("session:101")?.unwrap(), "active".as_bytes());
        assert_eq!(db.get("session:102")?.unwrap(), "pending".as_bytes());
        assert_eq!(db.get("session:103")?, None);

        println!(
            "  session:101 => {:?}",
            String::from_utf8_lossy(&db.get("session:101")?.unwrap())
        );
        println!(
            "  session:102 => {:?}",
            String::from_utf8_lossy(&db.get("session:102")?.unwrap())
        );
        println!("  session:103 => None (Tombstone correctly recovered)");

        // Add persistent SSTable data
        println!("\nWriting session:201 and flushing to L0 SSTable...");
        db.put("session:201", "archived")?;
        db.flush()?;

        // Add new WAL data on top of SSTables
        println!("Writing session:301 to active WAL...");
        db.put("session:301", "live")?;
    }

    println!("\n=== Phase 3: Second Restart / Recovery from Manifest + SSTable + WAL ===");
    {
        let options = OptionsBuilder::new().dir(&db_path).build();

        let db = FlashStore::open(options)?;
        println!("Verifying combined SSTable and WAL state across restarts...");

        assert_eq!(db.get("session:101")?.unwrap(), "active".as_bytes());
        assert_eq!(db.get("session:102")?.unwrap(), "pending".as_bytes());
        assert_eq!(db.get("session:201")?.unwrap(), "archived".as_bytes());
        assert_eq!(db.get("session:301")?.unwrap(), "live".as_bytes());

        let stats = db.stats();
        println!("Database Stats after full recovery:");
        println!(
            "  Active MemTable Size:       {} bytes",
            stats.active_memtable_size
        );
        println!(
            "  Immutable MemTables Count:  {}",
            stats.immutable_memtables_count
        );
        println!(
            "  L0 SSTable Files:           {:?}",
            stats.levels_file_count
        );
    }

    println!("\nDurability and recovery demonstration succeeded perfectly!");
    Ok(())
}
