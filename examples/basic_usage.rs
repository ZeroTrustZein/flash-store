use flash_store::prelude::*;
use tempfile::tempdir;

fn main() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(1024 * 1024)
        .build();

    let db = FlashStore::open(options)?;

    println!("Putting key-value pairs...");
    db.put("key1", "value1")?;
    db.put("key2", "value2")?;

    if let Some(val) = db.get("key1")? {
        println!("Got key1: {}", String::from_utf8_lossy(&val));
    }

    println!("Flushing memtable to disk...");
    db.flush()?;

    if let Some(val) = db.get("key1")? {
        println!("Got key1 after flush: {}", String::from_utf8_lossy(&val));
    }

    println!("Deleting key1...");
    db.delete("key1")?;

    match db.get("key1")? {
        Some(_) => println!("Error: key1 still exists"),
        None => println!("key1 confirmed deleted!"),
    }

    Ok(())
}
