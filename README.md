# ⚡ FlashStore

[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.0-orange.svg)](https://crates.io/crates/flash-store)
[![Documentation](https://docs.rs/flash-store/badge.svg)](https://docs.rs/flash-store)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![CI Status](https://github.com/zeindergham/flash-store/actions/workflows/ci.yml/badge.svg)](https://github.com/zeindergham/flash-store/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/rustc-1.70.0%2B-brightgreen.svg)](https://blog.rust-lang.org/)

**FlashStore** is an embedded, thread-safe, high-performance Log-Structured Merge-Tree (LSM-Tree) key-value storage engine engineered in 100% safe Rust. Designed for predictable write throughput, robust crash resilience, low-latency point lookups, and fast range scans, FlashStore provides a modular architecture with Write-Ahead Logging (WAL), concurrent in-memory MemTables, immutable SSTables with probabilistic Bloom filters, an LRU Block Cache, multi-level Compaction, and atomic VersionSet metadata management.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [Architecture & Design](#-architecture--design)
  - [High-Level Swarm Architecture](#high-level-swarm-architecture)
  - [Write Path (Insert / Update / Delete / Batch)](#write-path-insert--update--delete--batch)
  - [Read Path (Point Lookup)](#read-path-point-lookup)
  - [SSTable File Layout](#sstable-file-layout)
- [Feature Matrix](#-feature-matrix)
- [Installation & Setup](#-installation--setup)
- [Rust API Reference](#-rust-api-reference)
  - [Opening a Database & Options](#opening-a-database--options)
  - [Point Operations (`put`, `get`, `delete`)](#point-operations-put-get-delete)
  - [Atomic Batch Operations (`WriteBatch`)](#atomic-batch-operations-writebatch)
  - [Range Iteration & Scanning (`scan`, `StorageIterator`)](#range-iteration--scanning-scan-storageiterator)
  - [Thread-Safe Concurrent Access](#thread-safe-concurrent-access)
  - [Engine Statistics & Monitoring](#engine-statistics--monitoring)
  - [Manual & Automated Flush / Compaction](#manual--automated-flush--compaction)
  - [Clean Shutdown & Crash Recovery](#clean-shutdown--crash-recovery)
- [Command Line Interface (`flash-cli`)](#-command-line-interface-flash-cli)
  - [CLI Overview & Global Flags](#cli-overview--global-flags)
  - [One-Shot Subcommands](#one-shot-subcommands)
  - [Interactive REPL Shell](#interactive-repl-shell)
  - [Batch Script File Formats](#batch-script-file-formats)
  - [SSTable Deep Inspection](#sstable-deep-inspection)
- [Engine Internals Deep Dive](#-engine-internals-deep-dive)
  - [1. Write-Ahead Log (WAL)](#1-write-ahead-log-wal)
  - [2. MemTable & SkipList](#2-memtable--skiplist)
  - [3. SSTable Binary Format & Bloom Filters](#3-sstable-binary-format--bloom-filters)
  - [4. LRU Block Cache](#4-lru-block-cache)
  - [5. Leveled Compaction Engine](#5-leveled-compaction-engine)
  - [6. Manifest & VersionSet Architecture](#6-manifest--versionset-architecture)
  - [7. Concurrency & Multi-Version Ordering](#7-concurrency--multi-version-ordering)
- [Configuration Reference](#-configuration-reference)
- [Code Examples](#-code-examples)
  - [Basic CRUD Lifecycle](#basic-crud-lifecycle)
  - [Multi-Threaded Worker Pool](#multi-threaded-worker-pool)
- [Benchmarks & Performance](#-benchmarks--performance)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License & Contributions](#-license--contributions)

---

## 🚀 Key Highlights

- **Pure Rust**: Zero unsafe dependencies, memory safe, and thread safe by construction.
- **Append-Only Write Durability**: High-throughput writes via CRC32-framed append-only WAL with optional synchronous `fsync`.
- **Fast Lock-Free Read Concurrency**: `Arc`-wrapped concurrent SkipList MemTables with Read-Copy-Update (RCU) version switching.
- **Probabilistic Bloom Filtering**: Bit-level Murmur/CRC32 double-hashing Bloom filters per SSTable eliminate disk I/O on missing key lookups.
- **LRU Block Cache**: In-memory LRU caching of decoded SSTable data blocks prevents redundant disk access on repeated or sequential reads.
- **Deterministic Multi-Level Compaction**: Background compactor merges overlapping SSTable key ranges across levels, reclaims disk space, and purges obsolete versions and tombstones.
- **ACID Manifest & Crash Resilience**: Atomic `VersionEdit` delta logging guarantees zero data loss on unexpected power cuts or OS crashes.
- **Rich CLI & Interactive REPL**: Built-in CLI tool (`flash-cli`) supporting text, JSON, and TSV formats, file-based batch execution, and full SSTable binary dissection.

---

## 🏗 Architecture & Design

### High-Level Swarm Architecture

```
                    +--------------------------------------------------------+
                    |                      Client Code / CLI                 |
                    +---------------------------+----------------------------+
                                                |
                        +-----------------------v------------------------+
                        |                 FlashStore (API)               |
                        +-----------+------------------------+-----------+
                                    |                        |
                       (Write Path) |                        | (Read Path)
                                    v                        v
                    +---------------+----+       +-----------+--------------------+
                    |  Write-Ahead Log   |       |        Active MemTable         |
                    |   (current.wal)    |       +-----------+--------------------+
                    +---------------+----+                   |
                                    |                        v
                                    |            +-----------+--------------------+
                                    +----------->|   Immutable MemTables (Freeze) |
                                                 +-----------+--------------------+
                                                             |
                                               [Flush]       v
                                    +---------------------------------------------+
                                    |        Level 0 SSTables (000001.sst)        |
                                    +----------------------+----------------------+
                                                           |
                                            [Compaction]   v
                                    +---------------------------------------------+
                                    |        Level 1 SSTables (Non-overlapping)   |
                                    +----------------------+----------------------+
                                                           |
                                            [Compaction]   v
                                    +---------------------------------------------+
                                    |        Level N SSTables (Capacity x 10^N)   |
                                    +---------------------------------------------+
```

### Write Path (Insert / Update / Delete / Batch)

1. **Sequence Allocation**: Monotonically increments atomic sequence counter (`AtomicU64`).
2. **WAL Journaling**: Encodes a `WalRecord` (`[CRC32: 4B][Length: 4B][Payload]`) and writes to `current.wal`. If `sync_wal = true`, invokes OS `fsync`.
3. **MemTable Insertion**: Writes the `Entry` into the active MemTable sorted by `(user_key ASC, seq_no DESC)`.
4. **Flush Evaluation**: If `active_memtable.approximate_size() >= options.memtable_size`, the active MemTable is frozen into an immutable MemTable, a fresh MemTable is created, and an asynchronous flush writes an L0 SSTable to disk and updates the `MANIFEST`.

```
Client Put/Delete ──> Next Sequence ──> Append to WAL (CRC32) ──> Insert to MemTable ──> Flush Check
```

### Read Path (Point Lookup)

FlashStore evaluates storage tiers in strictly chronological order to ensure the latest sequence update is returned first:

1. **Active MemTable**: Checked first via in-memory lock-free binary search.
2. **Immutable MemTables**: Checked in reverse order (newest frozen table to oldest).
3. **Level 0 SSTables**: Checked in reverse chronological order (newest file number to oldest).
4. **Levels 1 to $N$ SSTables**: Each level has mutually non-overlapping key ranges. A binary search pinpoints candidate SSTables via `[smallest_key, largest_key]`.
5. **Bloom Filter Probe**: Evaluates the in-memory Bloom filter block. If bitmask returns negative, the file is bypassed with zero disk I/O.
6. **Block Cache Lookup**: Consults `LruBlockCache` for cached block data; falls back to disk I/O on cache miss.

```
Read Key ──> Active MemTable ──> Imm MemTables ──> L0 SSTables (Rev) ──> L1..N (Binary Search) ──> Bloom Check ──> Block Cache / Disk
```

### SSTable File Layout

```
+-----------------------------------------------------------------------+
|  Data Block 0: [len][Entry 0] ... [len][Entry K][Offsets...][#Off:4B] |
+-----------------------------------------------------------------------+
|  Data Block 1: [len][Entry 0] ... [len][Entry M][Offsets...][#Off:4B] |
+-----------------------------------------------------------------------+
|  ...                                                                  |
+-----------------------------------------------------------------------+
|  Meta Block (Bloom Filter): [Bit Array Bitmap] [k_hashes: 1B]         |
+-----------------------------------------------------------------------+
|  Index Block: (Vec<first_keys>, Vec<block_offsets>) (Bincode)         |
+-----------------------------------------------------------------------+
|  Footer (Fixed 48 Bytes):                                             |
|    - meta_index_offset (8B LE)                                        |
|    - meta_index_size   (8B LE)                                        |
|    - index_offset      (8B LE)                                        |
|    - index_size       (8B LE)                                        |
|    - reserved          (8B LE)                                        |
|    - magic number      (8B LE = 0xF1A5457072653031)                   |
+-----------------------------------------------------------------------+
```

---

## 📊 Feature Matrix

| Feature | FlashStore Support | Details |
| :--- | :---: | :--- |
| **Point Writes (`put`)** | ✅ | WAL journaled, concurrent MemTable buffered |
| **Point Reads (`get`)** | ✅ | Tiered MemTable $\to$ SSTable traversal with Bloom pruning |
| **Point Deletes (`delete`)** | ✅ | Tombstone markers with bottom-level garbage eviction |
| **Atomic Batches (`write_batch`)** | ✅ | Multi-operation atomic commit to WAL and MemTable |
| **Range Scans (`scan`)** | ✅ | Merged stream across MemTables and all SSTable levels |
| **Multi-Thread Safety** | ✅ | Safe concurrent reads & writes via `Arc<FlashStore>` |
| **Bloom Filters** | ✅ | Dynamic bits-per-key tuning (~1% false positive at 10 bits) |
| **Block Cache** | ✅ | LRU uncompressed block cache with configurable memory limit |
| **Compaction** | ✅ | Size-tiered & Leveled compaction with overlapping range detection |
| **Crash Recovery** | ✅ | Auto-recovery on start from WAL replay + MANIFEST edits |
| **Zero External C Libraries** | ✅ | 100% pure Rust code base with standard cargo toolchain |
| **CLI & REPL** | ✅ | Full-featured command-line interface with interactive mode |

---

## 📦 Installation & Setup

Add FlashStore to your project's `Cargo.toml`:

```toml
[dependencies]
flash-store = { git = "https://github.com/zeindergham/flash-store.git" }
bytes = "1.5"
```

Or install the standalone `flash-cli` tool:

```bash
cargo install --path . --bin flash-cli
```

---

## 📖 Rust API Reference

### Opening a Database & Options

Databases are opened using `FlashStore::open` with an `Options` struct configured directly or via `OptionsBuilder`.

```rust
use flash_store::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let options = OptionsBuilder::new()
        .dir("./flashstore_db")
        .memtable_size(8 * 1024 * 1024)       // 8 MB MemTable
        .block_size(4 * 1024)                 // 4 KB Block Size
        .bloom_bits_per_key(10)               // 10 bits (~1% FPR)
        .block_cache_size(128 * 1024 * 1024)  // 128 MB Block Cache
        .max_levels(7)                        // 7 LSM Levels
        .base_level_size_bytes(10 * 1024 * 1024) // 10 MB Base Level
        .sync_wal(false)                      // Async WAL writes for speed
        .create_if_missing(true)              // Auto-create directory
        .build();

    let db = FlashStore::open(options)?;
    println!("Database initialized successfully.");
    Ok(())
}
```

### Point Operations (`put`, `get`, `delete`)

`FlashStore` accepts any type implementing `IntoBytes` (`&str`, `String`, `&[u8]`, `Vec<u8>`, `Bytes`, `[u8; N]`).

```rust
use flash_store::prelude::*;

fn point_ops(db: &FlashStore) -> Result<()> {
    // Insert or update
    db.put("sensor:temp:room1", "22.5")?;
    db.put(b"sensor:humidity:room1", b"45.2".to_vec())?;

    // Point lookup
    if let Some(val) = db.get("sensor:temp:room1")? {
        println!("Temperature: {}", String::from_utf8_lossy(&val));
    }

    // Deletion
    db.delete("sensor:temp:room1")?;
    assert_eq!(db.get("sensor:temp:room1")?, None);

    Ok(())
}
```

### Atomic Batch Operations (`WriteBatch`)

Group multiple mutations into a single atomic disk write. Batches are logged in a single WAL write and applied atomically to the active MemTable.

```rust
use flash_store::prelude::*;

fn batch_ops(db: &FlashStore) -> Result<()> {
    let mut batch = WriteBatch::new();

    batch.put("account:101:balance", "5000");
    batch.put("account:102:balance", "12500");
    batch.delete("account:pending:99");

    println!("Executing batch with {} ops...", batch.len());
    db.write_batch(batch)?;
    Ok(())
}
```

### Range Iteration & Scanning (`scan`, `StorageIterator`)

`db.scan(start_key, end_key)` performs a range query with boundary filtering and multi-version deduplication.

```rust
use flash_store::prelude::*;
use bytes::Bytes;

fn scan_example(db: &FlashStore) -> Result<()> {
    // Unbounded scan: all entries
    let all_entries = db.scan(None, None)?;
    for (k, v) in all_entries {
        println!("{}: {}", String::from_utf8_lossy(&k), String::from_utf8_lossy(&v));
    }

    // Bounded scan: [start, end)
    let start = Some(Bytes::from("sensor:2026-08-01"));
    let end = Some(Bytes::from("sensor:2026-08-31"));
    let august_readings = db.scan(start, end)?;

    println!("Found {} readings in August.", august_readings.len());
    Ok(())
}
```

### Thread-Safe Concurrent Access

`FlashStore` implements `Clone` internally backed by `Arc<EngineInner>`, making it cheap to share across threads or thread pools.

```rust
use flash_store::prelude::*;
use std::sync::Arc;
use std::thread;

fn concurrent_usage() -> Result<()> {
    let options = OptionsBuilder::new().dir("./concurrent_db").build();
    let db = Arc::new(FlashStore::open(options)?);
    let mut handles = Vec::new();

    for thread_id in 0..8 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || -> Result<()> {
            for i in 0..1_000 {
                let key = format!("worker:{}:item:{}", thread_id, i);
                let val = format!("val_{}", i);
                db_clone.put(key, val)?;
            }
            Ok(())
        }));
    }

    for h in handles {
        h.join().unwrap()?;
    }

    Ok(())
}
```

### Engine Statistics & Monitoring

Inspect internal LSM hierarchy metrics at runtime:

```rust
use flash_store::prelude::*;

fn check_stats(db: &FlashStore) {
    let stats: Stats = db.stats();
    println!("Active MemTable: {} bytes", stats.active_memtable_size);
    println!("Immutable MemTables: {}", stats.immutable_memtables_count);
    for (level, count) in stats.levels_file_count.iter().enumerate() {
        println!("Level {}: {} SSTable file(s)", level, count);
    }
}
```

### Manual & Automated Flush / Compaction

While flushes and compactions trigger automatically based on size thresholds, they can be invoked manually:

```rust
use flash_store::prelude::*;

fn maintenance(db: &FlashStore) -> Result<()> {
    // Force freeze active memtable and flush to L0 SSTable
    db.flush()?;

    // Trigger full background compaction pass
    db.compact()?;
    Ok(())
}
```

### Clean Shutdown & Crash Recovery

```rust
use flash_store::prelude::*;

fn lifecycle() -> Result<()> {
    let options = OptionsBuilder::new().dir("./prod_data").build();

    // Open & Write
    {
        let db = FlashStore::open(options.clone())?;
        db.put("resilient_key", "persisted_value")?;
        db.close()?; // Flushes & syncs WAL buffers
    }

    // Reopen & Recover
    {
        let db = FlashStore::open(options)?;
        assert_eq!(
            db.get("resilient_key")?,
            Some(bytes::Bytes::from_static(b"persisted_value"))
        );
    }
    Ok(())
}
```

---

## 💻 Command Line Interface (`flash-cli`)

FlashStore includes a high-performance CLI binary for administrative tasks, REPL interaction, SSTable debugging, and benchmarking.

```
Production-grade CLI & REPL for FlashStore LSM-Tree Key-Value Engine

Usage: flash-cli [OPTIONS] <COMMAND>

Commands:
  put      Insert a key-value pair
  get      Query a key
  delete   Delete a key
  scan     Range scan
  batch    Atomic batch of PUT/DEL ops from args, JSON string, or file
  flush    Flush active memtable to SSTable
  compact  Trigger manual compaction
  stats    Print engine statistics
  inspect  Inspect an SSTable file
  bench    Built-in lightweight benchmark
  repl     Start interactive REPL
  help     Print this message or the help of the given subcommand(s)

Options:
  -p, --path <PATH>                  Path to FlashStore data directory [default: ./data]
      --memtable-size <MEMTABLE_SIZE> Override memtable size threshold in bytes
      --block-size <BLOCK_SIZE>       Override SSTable block size in bytes
      --sync-wal                     Enable synchronous WAL writes
      --block-cache-size <BYTES>     Override block cache size in bytes
      --json                         Emit JSON output where applicable
  -q, --quiet                        Suppress verbose / informational output
  -h, --help                         Print help
  -V, --version                      Print version
```

### One-Shot Subcommands

#### PUT
```bash
flash-cli --path ./my_db put user:100 '{"name":"Alice","role":"Admin"}'
# Output: OK
```

#### GET
```bash
flash-cli --path ./my_db get user:100
# Output: {"name":"Alice","role":"Admin"}

# With JSON envelope
flash-cli --path ./my_db --json get user:100
# Output: {"key":"user:100","value":"{\"name\":\"Alice\",\"role\":\"Admin\"}"}
```

#### DELETE
```bash
flash-cli --path ./my_db delete user:100
# Output: OK
```

#### SCAN
```bash
# Full scan in tabular text
flash-cli --path ./my_db scan

# Bounded scan with limit, reverse order, and JSON output
flash-cli --path ./my_db scan --start user:100 --end user:200 --limit 50 --reverse --format json
```

#### STATS
```bash
flash-cli --path ./my_db stats --format text
# Output:
# === FlashStore Statistics ===
# Active Memtable Size:     4194304 bytes
# Immutable Memtables:      0
# Total SSTable Files:      3
# Levels Distribution:
#   Level 0: 2 SSTable(s)
#   Level 1: 1 SSTable(s)
```

#### FLUSH & COMPACT
```bash
flash-cli --path ./my_db flush
flash-cli --path ./my_db compact
```

#### BENCH
```bash
flash-cli --path ./bench_db bench --ops 50000 --value-size 128 --read-ratio 0.7
```

---

### Interactive REPL Shell

Launch the interactive REPL with `flash-cli --path <DIR> repl`:

```text
===========================================================
  FlashStore Interactive LSM-Tree Storage Engine REPL
  Database: ./my_db
  Type 'HELP' for a list of commands, 'EXIT' to quit.
===========================================================
flash-store [./my_db]> PUT user:1 "Alice In Chains"
OK (42.1µs)
flash-store [./my_db]> PUT user:2 "Bob Marley"
OK (28.4µs)
flash-store [./my_db]> GET user:1
"Alice In Chains" (12.3µs)
flash-store [./my_db]> SCAN user:1 user:3 10
1) user:1: Alice In Chains
2) user:2: Bob Marley
(2 entries, took 65.2µs)
flash-store [./my_db]> STATS
Active Memtable: 182 bytes
Immutable Memtables: 0
Level 0: 0 SSTables
flash-store [./my_db]> FLUSH
Flushed memtable to SSTable (1.2ms)
flash-store [./my_db]> STATS
Active Memtable: 0 bytes
Immutable Memtables: 0
Level 0: 1 SSTables
flash-store [./my_db]> EXIT
Goodbye.
```

---

### Batch Script File Formats

`flash-cli batch` accepts two file formats:

#### Format 1: JSON Array
```json
[
  { "op": "put", "key": "k1", "value": "val1" },
  { "op": "put", "key": "k2", "value": "val2" },
  { "op": "delete", "key": "k1" }
]
```

Run with:
```bash
flash-cli --path ./my_db batch --file ./ops.json
```

#### Format 2: Line-Delimited Commands
```text
# User data import
PUT user:100 Alice
PUT user:101 Bob Jones
DEL user:99
```

Run with:
```bash
flash-cli --path ./my_db batch --file ./commands.txt
```

---

### SSTable Deep Inspection

Inspect SSTable binary structures directly:

```bash
flash-cli --path ./my_db inspect 000001.sst --format text
```

```text
=== SSTable Inspection: ./my_db/000001.sst ===
File Size:        4280 bytes
Total Entries:    50
  Values:         48
  Tombstones:     2
Smallest Key:     account:001
Largest Key:      account:050

--- Entries Preview (first 10) ---
  [000] [Value] key='account:001', seq=1, val_len=12
  [001] [Value] key='account:002', seq=2, val_len=12
  [002] [Tombstone] key='account:003', seq=3, val_len=0
  ... and 40 more entries
```

---

## 🔬 Engine Internals Deep Dive

### 1. Write-Ahead Log (WAL)

Every mutation is framed and flushed into `current.wal` before touching RAM.

```
+----------------+-------------------+-----------------------------------+
| Checksum (4B)  | Payload Len (4B)  | Bincode Encoded WalRecord Payload |
| (CRC32 IEEE)   | (u32 Little End)  | (key, value, is_delete, seq_no)   |
+----------------+-------------------+-----------------------------------+
```

- **Torn Write Protection**: On startup, `WalReader` streams records sequentially, verifies CRC32 checksums, and cleanly halts on trailing incomplete writes without corrupting previously committed data.
- **Atomic Log Reset**: When a MemTable is successfully flushed to an SSTable and registered in the `MANIFEST`, the active WAL is truncated to zero bytes via `WalWriter::reset()`.

### 2. MemTable & SkipList

- **SkipList Structure**: Thread-safe in-memory ordered map guarded with `parking_lot::RwLock`.
- **LSM Key Ordering**: Entries are ordered lexicographically by user key ascending. For matching keys, latest sequence numbers come first (`seq_no DESC`).
- **Freeze & Flush**: When the MemTable exceeds `memtable_size`, it is swapped with an atomic pointer swap into `imm_memtables`. Reads seamlessly query both active and immutable MemTables.

### 3. SSTable Binary Format & Bloom Filters

An SSTable is an immutable sorted file divided into fixed-size data blocks (default: 4 KB):

1. **Data Blocks**: Each block stores consecutive key-value pairs followed by an offset restart table.
2. **Meta Block (Bloom Filter)**: Uses $k = \text{round}(\text{bits\_per\_key} \cdot \ln 2)$ hash functions. Keys are hashed with Murmur/CRC32 bit-shift variants to produce uniform distribution across the bit array.
3. **Index Block**: Maps the first key of every data block to its physical byte offset in the SSTable file.
4. **Footer**: Fixed 48-byte record at `[file_len - 48]` storing byte offsets, sizes, and the magic constant `0xF1A5457072653031`.

### 4. LRU Block Cache

- High-efficiency in-memory LRU block cache (`LruBlockCache`) keyed by `(file_number: u64, block_index: u64)`.
- Eliminates redundant OS read syscalls for hot data blocks during random point lookups and sequential range iterations.

### 5. Leveled Compaction Engine

FlashStore implements a multi-tier compaction scheduler:

- **Level 0 Trigger**: When L0 file count reaches $\ge 4$, all L0 files are gathered and compacted against overlapping files in Level 1.
- **Level $N$ Size Limits**: Each level $N \ge 1$ has a capacity limit of $\text{base\_level\_size\_bytes} \times 10^{N-1}$.
- **Tombstone Purging**: Deletion tombstones are retained across higher levels to mask older values, but are permanently purged once they reach the bottom-most level ($L_{\text{max}-1}$).

### 6. Manifest & VersionSet Architecture

- Changes to LSM levels are represented as immutable `VersionEdit` records:
  - `new_files: Vec<(level, FileMetaData)>`
  - `deleted_files: Vec<(level, file_number)>`
  - `next_file_number: Option<u64>`
  - `last_sequence: Option<u64>`
- Edits are appended to `MANIFEST`. On database restart, FlashStore replays all edits to reconstruct the exact `VersionSet` without scanning individual SSTable files.

### 7. Concurrency & Multi-Version Ordering

- **Writer Synchronization**: MemTable flushes and compaction runs acquire non-blocking individual mutexes (`flush_lock`, `compact_lock`), allowing client point writes and batch inserts to progress uninterrupted.
- **Reader Isolation**: Readers acquire short read locks on `memtable` and `imm_memtables`, avoiding global lock contention.

---

## ⚙️ Configuration Reference

| Option Field | Default Value | Description & Tuning Advice |
| :--- | :---: | :--- |
| `dir` | `./data` | File system directory where WAL, SSTables, and MANIFEST reside. |
| `memtable_size` | `4 * 1024 * 1024` (4 MB) | Size in bytes before freezing active MemTable. Higher values increase write throughput at the cost of higher RAM usage. |
| `block_size` | `4 * 1024` (4 KB) | Target uncompressed block size in SSTables. Smaller blocks improve point lookup latency; larger blocks improve compression and sequential scans. |
| `bloom_bits_per_key` | `10` | Number of Bloom filter bits per key (~1% false positive rate). Set to `0` to disable Bloom filtering. |
| `max_levels` | `7` | Maximum depth of the LSM-tree hierarchy ($L_0$ to $L_6$). |
| `base_level_size_bytes` | `10 * 1024 * 1024` (10 MB) | Maximum size capacity of Level 1. Levels scale by a factor of 10 ($L_1 = 10\text{MB}, L_2 = 100\text{MB}, \dots$). |
| `sync_wal` | `false` | When `true`, calls `fsync` after every WAL append. Ensures maximum durability at the cost of write IOPS. |
| `block_cache_size` | `64 * 1024 * 1024` (64 MB) | In-memory capacity allocated for the LRU Block Cache. |
| `create_if_missing` | `true` | When `true`, automatically creates parent directories if they do not exist. |

---

## 💻 Code Examples

### Basic CRUD Lifecycle

Full source in [`examples/basic_usage.rs`](examples/basic_usage.rs):

```rust
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
```

Run with:
```bash
cargo run --example basic_usage
```

### Multi-Threaded Worker Pool

Full source in [`examples/multi_threaded.rs`](examples/multi_threaded.rs):

```rust
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
```

Run with:
```bash
cargo run --example multi_threaded
```

---

## 📈 Benchmarks & Performance

FlashStore includes microbenchmarks powered by [Criterion.rs](https://bheisler.github.io/criterion.rs/book/index.html).

### Running Benchmarks

```bash
cargo bench
```

The benchmark suite measures:
- **`flashstore_put_seq`**: Sequential write latency and throughput into WAL and MemTable.
- **`flashstore_get_hit`**: Point lookup latency with 10,000 pre-populated keys across MemTables and SSTables.

---

## 🧪 Testing & Quality Assurance

FlashStore maintains comprehensive unit, integration, and CLI test suites:

```bash
# Run all unit and integration tests
cargo test --all-targets --all-features

# Run formatting check
cargo fmt --all -- --check

# Run Clippy linter with strict warning flags
cargo clippy --all-targets --all-features -- -D warnings

# Validate documentation examples
cargo test --doc
```

### GitHub Actions CI Workflow

The CI matrix builds and tests across:
- **Operating Systems**: Ubuntu (`ubuntu-latest`), Windows (`windows-latest`), macOS (`macos-latest`).
- **Rust Toolchains**: Stable, MSRV (`1.70.0`).
- **Security Audits**: Automated cargo security advisory checks via `cargo-audit`.

---

## 📄 License & Contributions

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
- MIT License ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.

Contributions are welcome! Feel free to open issues or submit pull requests on [GitHub](https://github.com/zeindergham/flash-store).
