use flash_store::prelude::*;
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
