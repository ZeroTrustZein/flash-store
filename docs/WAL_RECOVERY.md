# FlashStore Write-Ahead Log (WAL) & Crash Recovery

## 1. Overview

The Write-Ahead Log (WAL) ensures durability across unexpected process termination, kernel crashes, or hardware power failure. Every mutating operation (`put`, `delete`, `write_batch`) is appended to `current.wal` on disk before any in-memory state in the `MemTable` is modified.

---

## 2. WAL Record Binary Framing

Each record in `current.wal` is framed with a 4-byte CRC32-IEEE checksum and a 4-byte payload length header:

```
+-------------------+-------------------+-----------------------------------+
|  CRC32 Checksum   |  Payload Length   |       Bincode-Encoded Payload     |
|     (4 Bytes)     |     (4 Bytes)     |              (N Bytes)            |
+-------------------+-------------------+-----------------------------------+
```

### 2.1 Header Fields
1. **CRC32 Checksum** (`u32` Little-Endian): Computed over the entire serialized payload slice.
2. **Payload Length** (`u32` Little-Endian): Length of payload $N$ in bytes.

### 2.2 Payload Schema (`WalRecord`)
Serialized via `bincode`:
- `key`: Byte buffer of user key.
- `value`: Byte buffer of value (empty if `is_delete` is true).
- `is_delete`: Boolean flag (`true` indicates a tombstone deletion).
- `seq_no`: Monotonic 64-bit sequence number (`u64`).

---

## 3. Durability Modes: `sync_wal`

FlashStore supports two write durability modes configured via `OptionsBuilder::sync_wal`:

1. **Synchronous Mode (`sync_wal: true`)**:
   - Executes `std::fs::File::sync_data()` after each record or batch append.
   - **Guarantee**: Zero data loss even on total operating system crash or power outage.
   - **Cost**: IOPS bounded by underlying SSD / NVMe fsync latency.

2. **Asynchronous Mode (`sync_wal: false`, Default)**:
   - Appends records to the OS page cache buffers.
   - **Guarantee**: Zero data loss on application process crashes; bounded potential loss (OS sync interval) on abrupt hardware power failure.
   - **Throughput**: Extremely high throughput matching in-memory append speeds.

---

## 4. Crash Recovery Protocol

When `FlashStore::open(options)` is invoked, recovery proceeds through the following sequential stages:

```
[Start Recovery]
       |
       v
1. Load MANIFEST
   Replay all VersionEdits -> Reconstruct VersionSet (L0..LN SSTable files)
       |
       v
2. Determine `last_sequence` from Manifest
       |
       v
3. Inspect `current.wal`
   - If missing: start with clean MemTable.
   - If present: Open WalReader.
       |
       v
4. Replay WalRecords
   - Read 8-byte header ([CRC32] [Length]).
   - Read payload bytes.
   - Verify CRC32 matches payload.
   - If EOF or corrupted trailing write: halt replay cleanly (truncate corrupt tail).
   - If seq_no > version_set.last_sequence():
         Apply Put/Delete to initial MemTable.
   - Update max_seq = max(max_seq, record.seq_no).
       |
       v
5. Update VersionSet last_sequence = max_seq
       |
       v
6. Open WalWriter for new incoming writes
       |
       v
[Recovery Complete]
```

### 4.1 Corrupted Tail Handling
In the event of an unclean system crash in the middle of writing a record, the file may contain an incomplete header or mismatched CRC32. FlashStore's `WalReader` identifies incomplete chunks and stops replaying at the point of corruption without failing the database initialization, preserving all previously acknowledged synchronous transactions.

### 4.2 WAL Truncation on Flush
When an active or immutable MemTable is flushed to an L0 SSTable:
1. The new SSTable is written and registered via a `VersionEdit` in the `MANIFEST`.
2. The remaining unflushed records (from other active/immutable tables) are re-written to a fresh WAL segment.
3. The old log file is reset, preventing unbounded WAL disk growth.
