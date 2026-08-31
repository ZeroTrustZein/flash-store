use flash_store::prelude::*;
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

#[test]
fn test_basic_crud() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    // Put
    db.put(b"k1", b"v1")?;
    db.put(b"k2", b"v2")?;

    // Get
    assert_eq!(db.get(b"k1")?, Some(bytes::Bytes::from_static(b"v1")));
    assert_eq!(db.get(b"k2")?, Some(bytes::Bytes::from_static(b"v2")));
    assert_eq!(db.get(b"k3")?, None);

    // Overwrite
    db.put(b"k1", b"v1_updated")?;
    assert_eq!(db.get(b"k1")?, Some(bytes::Bytes::from_static(b"v1_updated")));

    // Delete
    db.delete(b"k1")?;
    assert_eq!(db.get(b"k1")?, None);

    Ok(())
}

#[test]
fn test_flush_and_persistence() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    db.put(b"persist_key", b"persist_val")?;
    db.flush()?;

    assert_eq!(
        db.get(b"persist_key")?,
        Some(bytes::Bytes::from_static(b"persist_val"))
    );

    // Overwrite in memtable after flush
    db.put(b"persist_key", b"new_val")?;
    assert_eq!(
        db.get(b"persist_key")?,
        Some(bytes::Bytes::from_static(b"new_val"))
    );

    // Delete in memtable after flush
    db.delete(b"persist_key")?;
    assert_eq!(db.get(b"persist_key")?, None);

    Ok(())
}

#[test]
fn test_write_batch() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    let mut batch = WriteBatch::new();
    batch.put(b"b1", b"v1");
    batch.put(b"b2", b"v2");
    batch.delete(b"b1");

    db.write_batch(batch)?;

    assert_eq!(db.get(b"b1")?, None);
    assert_eq!(db.get(b"b2")?, Some(bytes::Bytes::from_static(b"v2")));

    Ok(())
}

#[test]
fn test_crash_recovery_wal() -> Result<()> {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();

    // 1. Open DB, write unflushed keys to WAL, close DB
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;
        db.put(b"wal_k1", b"wal_v1")?;
        db.put(b"wal_k2", b"wal_v2")?;
        db.delete(b"wal_k1")?;
        db.put(b"wal_k3", b"wal_v3")?;
        db.close()?;
    }

    // 2. Reopen DB from same directory and verify WAL replay
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;

        assert_eq!(db.get(b"wal_k1")?, None);
        assert_eq!(db.get(b"wal_k2")?, Some(bytes::Bytes::from_static(b"wal_v2")));
        assert_eq!(db.get(b"wal_k3")?, Some(bytes::Bytes::from_static(b"wal_v3")));
    }

    Ok(())
}

#[test]
fn test_crash_recovery_manifest_and_sst() -> Result<()> {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();

    // 1. Write and flush
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;
        db.put(b"sst_k1", b"sst_v1")?;
        db.put(b"sst_k2", b"sst_v2")?;
        db.flush()?;
        db.close()?;
    }

    // 2. Reopen and verify persisted SSTables are indexed via Manifest
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;

        assert_eq!(db.get(b"sst_k1")?, Some(bytes::Bytes::from_static(b"sst_v1")));
        assert_eq!(db.get(b"sst_k2")?, Some(bytes::Bytes::from_static(b"sst_v2")));
        assert_eq!(db.get(b"sst_k3")?, None);

        let stats = db.stats();
        assert!(stats.levels_file_count[0] >= 1);
    }

    Ok(())
}

#[test]
fn test_crash_recovery_mixed_wal_and_sst() -> Result<()> {
    let dir = tempdir().unwrap();
    let db_path = dir.path().to_path_buf();

    // 1. Write SSTables + unflushed WAL records
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;
        db.put(b"k1", b"v1_old")?;
        db.put(b"k2", b"v2")?;
        db.flush()?; // Flushed to SSTable

        db.put(b"k1", b"v1_new")?; // Unflushed in WAL
        db.delete(b"k2")?;         // Unflushed delete in WAL
        db.put(b"k3", b"v3")?;     // Unflushed in WAL
        db.close()?;
    }

    // 2. Reopen DB and verify correct layered state
    {
        let options = OptionsBuilder::new().dir(&db_path).build();
        let db = FlashStore::open(options)?;

        assert_eq!(db.get(b"k1")?, Some(bytes::Bytes::from_static(b"v1_new")));
        assert_eq!(db.get(b"k2")?, None);
        assert_eq!(db.get(b"k3")?, Some(bytes::Bytes::from_static(b"v3")));
    }

    Ok(())
}

#[test]
fn test_auto_flush_on_size_threshold() -> Result<()> {
    let dir = tempdir().unwrap();
    // Set a tiny memtable size to trigger auto-flush
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(256)
        .build();
    let db = FlashStore::open(options)?;

    for i in 0..50 {
        let key = format!("auto_key_{:04}", i);
        let val = format!("auto_value_{:08}", i);
        db.put(key.into_bytes(), val.into_bytes())?;
    }

    // Check that auto-flush created L0 SSTables
    let stats = db.stats();
    assert!(stats.levels_file_count[0] > 0);

    // Verify all 50 keys are intact
    for i in 0..50 {
        let key = format!("auto_key_{:04}", i);
        let expected_val = format!("auto_value_{:08}", i);
        let actual = db.get(key.into_bytes())?;
        assert_eq!(actual, Some(bytes::Bytes::from(expected_val)));
    }

    Ok(())
}

#[test]
fn test_range_scan() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    db.put(b"a", b"1")?;
    db.put(b"b", b"2")?;
    db.put(b"c", b"3")?;
    db.flush()?;

    db.delete(b"b")?;
    db.put(b"d", b"4")?;
    db.put(b"e", b"5")?;

    // Full scan
    let all = db.scan(None, None)?;
    assert_eq!(
        all,
        vec![
            (bytes::Bytes::from_static(b"a"), bytes::Bytes::from_static(b"1")),
            (bytes::Bytes::from_static(b"c"), bytes::Bytes::from_static(b"3")),
            (bytes::Bytes::from_static(b"d"), bytes::Bytes::from_static(b"4")),
            (bytes::Bytes::from_static(b"e"), bytes::Bytes::from_static(b"5")),
        ]
    );

    // Bounded scan [b"c", b"e")
    let bounded = db.scan(
        Some(bytes::Bytes::from_static(b"c")),
        Some(bytes::Bytes::from_static(b"e")),
    )?;
    assert_eq!(
        bounded,
        vec![
            (bytes::Bytes::from_static(b"c"), bytes::Bytes::from_static(b"3")),
            (bytes::Bytes::from_static(b"d"), bytes::Bytes::from_static(b"4")),
        ]
    );

    Ok(())
}

#[test]
fn test_concurrent_reads_writes() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(1024)
        .build();
    let db = Arc::new(FlashStore::open(options)?);

    let mut handles = Vec::new();

    // 4 Writer threads
    for t in 0..4 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || -> Result<()> {
            for i in 0..50 {
                let key = format!("thread_{}_{}", t, i);
                let val = format!("val_{}_{}", t, i);
                db_clone.put(key.into_bytes(), val.into_bytes())?;
            }
            Ok(())
        }));
    }

    // 4 Reader threads
    for _ in 0..4 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || -> Result<()> {
            for _ in 0..50 {
                let _ = db_clone.get(b"thread_0_0")?;
            }
            Ok(())
        }));
    }

    for handle in handles {
        handle.join().unwrap()?;
    }

    // Verify written data
    for t in 0..4 {
        for i in 0..50 {
            let key = format!("thread_{}_{}", t, i);
            let expected_val = format!("val_{}_{}", t, i);
            let actual = db.get(key.into_bytes())?;
            assert_eq!(actual, Some(bytes::Bytes::from(expected_val)));
        }
    }

    Ok(())
}

#[test]
fn test_multi_version_overwrites() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    for v in 0..10 {
        let val = format!("v_{}", v);
        db.put(b"multi_key", val.into_bytes())?;
        if v % 3 == 0 {
            db.flush()?;
        }
    }

    assert_eq!(
        db.get(b"multi_key")?,
        Some(bytes::Bytes::from_static(b"v_9"))
    );

    let scan_res = db.scan(None, None)?;
    assert_eq!(scan_res.len(), 1);
    assert_eq!(scan_res[0].1, bytes::Bytes::from_static(b"v_9"));

    Ok(())
}
