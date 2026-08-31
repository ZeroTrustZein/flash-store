use flash_store::prelude::*;
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

fn main() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = Arc::new(FlashStore::open(options)?);

    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || -> Result<()> {
            for i in 0..100 {
                let key = format!("thread_{}_key_{}", thread_id, i);
                let value = format!("val_{}", i);
                db_clone.put(key.into_bytes(), value.into_bytes())?;
            }
            Ok(())
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap()?;
    }

    println!("All threads finished inserting data.");
    for thread_id in 0..4 {
        let key = format!("thread_{}_key_0", thread_id);
        if let Some(val) = db.get(key.as_bytes())? {
            println!("Verified {}: {}", key, String::from_utf8_lossy(&val));
        }
    }

    Ok(())
}
