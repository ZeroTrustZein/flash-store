//! Example: Range Queries and Prefix Scanning in FlashStore
//!
//! This example demonstrates range scans across memtables and persisted SSTables,
//! including prefix-based filtering and bounded range queries.

use bytes::Bytes;
use flash_store::prelude::*;
use tempfile::tempdir;

fn main() -> Result<()> {
    let dir = tempdir().map_err(FlashStoreError::Io)?;
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(1024 * 1024)
        .build();

    let db = FlashStore::open(options)?;
    println!("Populating multi-domain data across users, orders, and products...");

    // Insert keys across multiple domains
    db.put("orders:2026-01-01:001", "Order #1 Total: $120")?;
    db.put("orders:2026-01-01:002", "Order #2 Total: $45")?;
    db.put("orders:2026-01-02:003", "Order #3 Total: $210")?;
    db.put("orders:2026-01-03:004", "Order #4 Total: $80")?;
    db.put("products:electronics:p101", "Laptop Pro 16")?;
    db.put("products:electronics:p102", "Wireless Mouse")?;
    db.put("products:furniture:f201", "Ergonomic Desk Chair")?;
    db.put("users:zein", "Engineer")?;
    db.put("users:odysseus", "Swarm Commander")?;

    // Flush some data to disk SSTable to verify cross-layer merging
    println!("Flushing memtable to create SSTable on disk...");
    db.flush()?;

    // Insert new writes into active MemTable
    db.put("orders:2026-01-02:005", "Order #5 Total: $350 (Unflushed)")?;
    db.put("users:athena", "Orchestrator")?;

    // 1. Full database scan
    println!("\n=== Full Scan ===");
    let all_items = db.scan(None, None)?;
    for (k, v) in &all_items {
        println!(
            "  {} => {}",
            String::from_utf8_lossy(k),
            String::from_utf8_lossy(v)
        );
    }
    assert_eq!(all_items.len(), 11);

    // 2. Bounded range query for orders on 2026-01-02
    println!("\n=== Bounded Scan (orders:2026-01-02 to orders:2026-01-03) ===");
    let start = Some(Bytes::from("orders:2026-01-02"));
    let end = Some(Bytes::from("orders:2026-01-03"));
    let jan_2_orders = db.scan(start, end)?;
    for (k, v) in &jan_2_orders {
        println!(
            "  {} => {}",
            String::from_utf8_lossy(k),
            String::from_utf8_lossy(v)
        );
    }
    assert_eq!(jan_2_orders.len(), 2);

    // 3. Prefix scan for all products
    println!("\n=== Prefix Scan (products:) ===");
    let prefix_start = Some(Bytes::from("products:"));
    let prefix_end = Some(Bytes::from("products;\x00")); // ';' is ASCII byte after ':'
    let products = db.scan(prefix_start, prefix_end)?;
    for (k, v) in &products {
        println!(
            "  {} => {}",
            String::from_utf8_lossy(k),
            String::from_utf8_lossy(v)
        );
    }
    assert_eq!(products.len(), 3);

    println!("\nAll range queries and scans executed correctly!");
    Ok(())
}
