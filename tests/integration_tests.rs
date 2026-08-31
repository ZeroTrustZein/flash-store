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

#[test]
fn test_subsystems_compaction_pipeline() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .block_size(64)
        .build();
    let db = FlashStore::open(options)?;

    // Create 4 L0 files
    for batch_idx in 0..4 {
        for item_idx in 0..10 {
            let key = format!("k_{:02}_{:02}", batch_idx, item_idx);
            let val = format!("val_{:02}_{:02}", batch_idx, item_idx);
            db.put(key.into_bytes(), val.into_bytes())?;
        }
        db.flush()?;
    }

    let stats_before = db.stats();
    assert_eq!(stats_before.levels_file_count[0], 4);
    assert_eq!(stats_before.levels_file_count[1], 0);

    // Compact L0 -> L1
    db.compact()?;

    let stats_after = db.stats();
    assert_eq!(stats_after.levels_file_count[0], 0);
    assert_eq!(stats_after.levels_file_count[1], 1);

    // Verify all 40 keys are intact after compaction
    for batch_idx in 0..4 {
        for item_idx in 0..10 {
            let key = format!("k_{:02}_{:02}", batch_idx, item_idx);
            let expected = format!("val_{:02}_{:02}", batch_idx, item_idx);
            assert_eq!(
                db.get(key.into_bytes())?,
                Some(bytes::Bytes::from(expected))
            );
        }
    }

    Ok(())
}

#[test]
fn test_subsystems_merging_iterator_integration() -> Result<()> {
    let e1 = Entry::new_value(bytes::Bytes::from_static(b"apple"), bytes::Bytes::from_static(b"10"), 1);
    let e2 = Entry::new_value(bytes::Bytes::from_static(b"cherry"), bytes::Bytes::from_static(b"30"), 1);
    let iter1 = SSTableIterator::new(vec![e1, e2]);

    let e3 = Entry::new_value(bytes::Bytes::from_static(b"banana"), bytes::Bytes::from_static(b"20"), 2);
    let e4 = Entry::new_value(bytes::Bytes::from_static(b"date"), bytes::Bytes::from_static(b"40"), 2);
    let iter2 = SSTableIterator::new(vec![e3, e4]);

    let mut merger = MergingIterator::new(vec![iter1, iter2]);
    let mut collected = Vec::new();
    while merger.valid() {
        collected.push((merger.key().clone(), merger.value().clone()));
        merger.next()?;
    }

    assert_eq!(collected.len(), 4);
    assert_eq!(collected[0].0, bytes::Bytes::from_static(b"apple"));
    assert_eq!(collected[1].0, bytes::Bytes::from_static(b"banana"));
    assert_eq!(collected[2].0, bytes::Bytes::from_static(b"cherry"));
    assert_eq!(collected[3].0, bytes::Bytes::from_static(b"date"));

    Ok(())
}

#[test]
fn test_large_keys_and_values() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .block_size(256)
        .build();
    let db = FlashStore::open(options)?;

    // 16KB payload
    let large_value = vec![b'x'; 16 * 1024];
    let key = "large_payload_key";

    db.put(key, &large_value)?;
    assert_eq!(db.get(key)?, Some(bytes::Bytes::copy_from_slice(&large_value)));

    // Flush and verify SSTable reader handles multi-block large values
    db.flush()?;
    assert_eq!(db.get(key)?, Some(bytes::Bytes::copy_from_slice(&large_value)));

    Ok(())
}

#[test]
fn test_empty_values_vs_tombstones() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    // Put empty value
    db.put(b"empty_val_key", b"")?;
    assert_eq!(db.get(b"empty_val_key")?, Some(bytes::Bytes::new()));

    // Flush to SSTable and verify empty value is still Some(Bytes::new())
    db.flush()?;
    assert_eq!(db.get(b"empty_val_key")?, Some(bytes::Bytes::new()));

    // Now delete it
    db.delete(b"empty_val_key")?;
    assert_eq!(db.get(b"empty_val_key")?, None);

    // Flush tombstone to SSTable
    db.flush()?;
    assert_eq!(db.get(b"empty_val_key")?, None);

    Ok(())
}

#[test]
fn test_range_scan_edge_cases() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let db = FlashStore::open(options)?;

    db.put(b"k10", b"v10")?;
    db.put(b"k20", b"v20")?;
    db.put(b"k30", b"v30")?;
    db.flush()?;

    // Inverted range [k30, k10) -> should be empty
    let inverted = db.scan(
        Some(bytes::Bytes::from_static(b"k30")),
        Some(bytes::Bytes::from_static(b"k10")),
    )?;
    assert!(inverted.is_empty());

    // Range before all keys [k00, k05) -> should be empty
    let before = db.scan(
        Some(bytes::Bytes::from_static(b"k00")),
        Some(bytes::Bytes::from_static(b"k05")),
    )?;
    assert!(before.is_empty());

    // Range after all keys [k40, k50) -> should be empty
    let after = db.scan(
        Some(bytes::Bytes::from_static(b"k40")),
        Some(bytes::Bytes::from_static(b"k50")),
    )?;
    assert!(after.is_empty());

    // Range with single key [k20, k21)
    let single = db.scan(
        Some(bytes::Bytes::from_static(b"k20")),
        Some(bytes::Bytes::from_static(b"k21")),
    )?;
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].0, bytes::Bytes::from_static(b"k20"));

    Ok(())
}

#[test]
fn test_interleaved_flushes_deletions_compaction() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .block_size(64)
        .build();
    let db = FlashStore::open(options)?;

    // Cycle 1: Put & Flush
    for i in 0..20 {
        db.put(format!("k_{:03}", i), format!("val_{:03}", i))?;
    }
    db.flush()?;

    // Cycle 2: Overwrite even keys, delete odd keys & Flush
    for i in 0..20 {
        if i % 2 == 0 {
            db.put(format!("k_{:03}", i), format!("val_updated_{:03}", i))?;
        } else {
            db.delete(format!("k_{:03}", i))?;
        }
    }
    db.flush()?;

    // Cycle 3: Add new keys & Compact
    for i in 20..30 {
        db.put(format!("k_{:03}", i), format!("val_{:03}", i))?;
    }
    db.flush()?;
    db.compact()?;

    // Verify state
    for i in 0..20 {
        if i % 2 == 0 {
            assert_eq!(
                db.get(format!("k_{:03}", i))?,
                Some(bytes::Bytes::from(format!("val_updated_{:03}", i)))
            );
        } else {
            assert_eq!(db.get(format!("k_{:03}", i))?, None);
        }
    }
    for i in 20..30 {
        assert_eq!(
            db.get(format!("k_{:03}", i))?,
            Some(bytes::Bytes::from(format!("val_{:03}", i)))
        );
    }

    Ok(())
}

#[test]
fn test_concurrent_batch_writes_and_scans() -> Result<()> {
    let dir = tempdir().unwrap();
    let options = OptionsBuilder::new()
        .dir(dir.path())
        .memtable_size(512)
        .build();
    let db = Arc::new(FlashStore::open(options)?);

    let mut handles = Vec::new();

    // 4 Batch Writer threads
    for t in 0..4 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || -> Result<()> {
            for b in 0..10 {
                let mut batch = WriteBatch::new();
                for i in 0..10 {
                    let k = format!("th_{}_b_{}_k_{}", t, b, i);
                    let v = format!("val_{}_{}_{}", t, b, i);
                    batch.put(k, v);
                }
                db_clone.write_batch(batch)?;
            }
            Ok(())
        }));
    }

    // 2 Scanner threads
    for _ in 0..2 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || -> Result<()> {
            for _ in 0..10 {
                let _ = db_clone.scan(None, None)?;
            }
            Ok(())
        }));
    }

    for handle in handles {
        handle.join().unwrap()?;
    }

    // Verify that all 4 * 10 * 10 = 400 keys were written correctly
    for t in 0..4 {
        for b in 0..10 {
            for i in 0..10 {
                let k = format!("th_{}_b_{}_k_{}", t, b, i);
                let expected = format!("val_{}_{}_{}", t, b, i);
                assert_eq!(
                    db.get(k)?,
                    Some(bytes::Bytes::from(expected))
                );
            }
        }
    }

    Ok(())
}
