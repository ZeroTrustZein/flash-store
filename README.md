# ⚡ FlashStore

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI Status](https://github.com/ZeroTrustZein/flash-store/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeroTrustZein/flash-store/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rustc-1.75%2B-brightgreen.svg)](https://blog.rust-lang.org/)

**FlashStore** is an embedded, thread-safe, high-performance Log-Structured Merge-Tree (LSM-Tree) key-value storage engine engineered in 100% safe Rust, featuring a native **RAG (Retrieval-Augmented Generation), Cross-Encoder Reranking, and Semantic Caching Subsystem**. Designed for predictable write throughput, robust crash resilience, low-latency point lookups, fast range scans, and intelligent vector-lexical retrieval, FlashStore provides a modular architecture with Write-Ahead Logging (WAL), concurrent in-memory MemTables, immutable SSTables with probabilistic Bloom filters, an LRU Block Cache, multi-level Compaction, atomic `WriteBatch` persistence, hybrid BM25/vector search, and cross-encoder reranking.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [Architecture & Design](#-architecture--design)
  - [High-Level Layered Architecture](#high-level-layered-architecture)
  - [Write Path (Insert / Update / Delete / Batch)](#write-path-insert--update--delete--batch)
  - [Read Path (Point Lookup)](#read-path-point-lookup)
  - [SSTable File Layout](#sstable-file-layout)
  - [RAG & Reranker Subsystem Architecture](#rag--reranker-subsystem-architecture)
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
  - [RAG Retrieval & Reranker Pipeline API](#rag-retrieval--reranker-pipeline-api)
- [Command Line Interface (`flash-cli`)](#-command-line-interface-flash-cli)
  - [CLI Overview & Global Flags](#cli-overview--global-flags)
  - [One-Shot Subcommands](#one-shot-subcommands)
  - [Interactive REPL Shell](#interactive-repl-shell)
  - [Batch Script File Formats](#batch-script-file-formats)
  - [SSTable Deep Inspection](#sstable-deep-inspection)
  - [RAG & Vector Search Subcommands (`flash-cli rag`)](#rag--vector-search-subcommands-flash-cli-rag)
- [Engine Internals Deep Dive](#-engine-internals-deep-dive)
  - [1. Write-Ahead Log (WAL)](#1-write-ahead-log-wal)
  - [2. MemTable & SkipList](#2-memtable--skiplist)
  - [3. SSTable Binary Format & Bloom Filters](#3-sstable-binary-format--bloom-filters)
  - [4. LRU Block Cache](#4-lru-block-cache)
  - [5. Leveled Compaction Engine](#5-leveled-compaction-engine)
  - [6. Manifest & VersionSet Architecture](#6-manifest--versionset-architecture)
  - [7. Concurrency & Multi-Version Ordering](#7-concurrency--multi-version-ordering)
  - [8. Hybrid RAG Retrieval, Reranking & Caching](#8-hybrid-rag-retrieval-reranking--caching)
- [Configuration Reference](#-configuration-reference)
- [Code Examples](#-code-examples)
  - [Basic CRUD Lifecycle](#basic-crud-lifecycle)
  - [Atomic Batch Operations](#atomic-batch-operations)
  - [Range Queries & Prefix Scans](#range-queries--prefix-scans)
  - [Crash Recovery & Durability](#crash-recovery--durability)
  - [Multi-Threaded Worker Pool](#multi-threaded-worker-pool)
  - [End-to-End RAG Pipeline & Semantic Caching](#end-to-end-rag-pipeline--semantic-caching)
- [Deep Technical Documentation](#-deep-technical-documentation)
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
- **Native RAG & Cross-Encoder Reranking**: Built-in hybrid retrieval combining dense vector similarity with BM25 sparse search, fused via Reciprocal Rank Fusion (RRF), and scored with cross-encoder lexical-semantic alignment.
- **Semantic Vector Query Cache**: Instant sub-millisecond query responses for semantically equivalent queries using cosine similarity thresholds, TTL expiration, and LRU eviction.
- **Context Assembly & LLM Citations**: Token-budget-aware context packing with configurable truncation (`TruncateLast`, `DropOversized`), structured layouts (Markdown, XML, Numbered, Compact), and 1-based citation references (`[1]`, `[2]`).
- **Rich CLI & Interactive REPL**: Built-in CLI tool (`flash-cli`) supporting KV operations, RAG pipelines, text/JSON/TSV formats, batch files, and SSTable binary dissection.

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

### RAG & Reranker Subsystem Architecture

FlashStore includes a native high-performance RAG and reranker layer on top of the LSM-Tree engine:

```
                                 [ User Query ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │     Semantic Vector Cache         │
                     │  (Exact Match / Cosine Threshold) ├───────────┐
                     └─────────────────┬─────────────────┘           │
                                       │ Cache Miss                  │ Cache Hit
                                       ▼                             │ (Fast Return)
                     ┌───────────────────────────────────┐           │
                     │    Query Vector Embedding         │           │
                     │  (EmbeddingProvider Interface)    │           │
                     └─────────────────┬─────────────────┘           │
                                       │                             │
                     ┌─────────────────┴─────────────────┐           │
                     ▼                                   ▼           │
       ┌───────────────────────────┐       ┌─────────────────────────┴─┐
       │    Sparse Lexical Search  │       │    Dense Vector Search    │
       │  (BM25 Inverted Index)    │       │  (Cosine / Dot / L2)      │
       └─────────────┬─────────────┘       └─────────────┬─────────────┘
                     │ Top-K Sparse                      │ Top-K Dense
                     └─────────────────┬─────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │      Hybrid Score Fusion          │
                     │   (RRF / Weighted Linear / Borda) │
                     └─────────────────┬─────────────────┘
                                       │ Fused Candidates
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Cross-Encoder Reranker         │
                     │ (Lexical-Semantic / MMR Diversity)│
                     └─────────────────┬─────────────────┘
                                       │ Reranked Candidates
                                       ▼
                     ┌───────────────────────────────────┐
                     │     Context Assembler & Prompter  │
                     │ (Token Budget / Citations / XML)  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     [ LLM Prompt / Pipeline Result ] ◄──────────────┘
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
| **Hybrid RAG Retrieval** | ✅ | Parallel dense vector and BM25 sparse inverted index search |
| **Score Fusion Strategies** | ✅ | Reciprocal Rank Fusion (RRF), Weighted Linear, and Borda Count |
| **Cross-Encoder Reranker** | ✅ | Deep lexical-semantic pair evaluation with score explanations |
| **MMR Diversity Reranking**| ✅ | Maximal Marginal Relevance balancing relevance vs redundancy |
| **Semantic Query Cache** | ✅ | Multi-tier LRU cache with cosine similarity threshold and TTL |
| **Context Assembly** | ✅ | Token budgeting, truncation (`TruncateLast`), and citations (`[1]`) |
| **IR Telemetry & Eval** | ✅ | Built-in MRR, Recall@K, Precision@K, NDCG@K, and stage latency |

---

## 📦 Installation & Setup

Add FlashStore to your project's `Cargo.toml`:

```toml
[dependencies]
flash-store = { git = "https://github.com/ZeroTrustZein/flash-store.git" }
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

### RAG Retrieval & Reranker Pipeline API

FlashStore provides `RagPipeline` and `RagEngine` for building retrieval-augmented generation applications directly on top of the LSM-tree storage engine:

```rust
use flash_store::prelude::*;
use std::sync::Arc;

fn rag_example(db: Arc<FlashStore>) -> Result<()> {
    // 1. Configure RAG Engine with hybrid weights and semantic caching
    let config = RagConfigBuilder::new()
        .embedding_dim(384)
        .similarity_metric(SimilarityMetric::Cosine)
        .hybrid_dense_weight(0.5)
        .rrf_k(60)
        .semantic_cache(5000, 0.92, 3600)
        .build();

    let engine = RagEngine::with_store(config, db);

    // 2. Wrap in RagPipeline with an embedding provider and chunking
    let embedder = Arc::new(MockEmbeddingProvider::new(384));
    let mut pipeline = RagPipeline::new(engine)
        .with_embedder(embedder)
        .with_chunking(ChunkingConfig {
            strategy: ChunkingStrategy::Paragraph,
            min_chunk_size: 20,
        })
        .with_context_config(ContextConfig {
            format: ContextFormat::Markdown,
            max_chars: 4096,
            include_scores: true,
            include_metadata: true,
            ..Default::default()
        });

    // 3. Ingest documents
    let mut meta = DocumentMetadata::new();
    meta.insert("author", "Zein");
    meta.insert("category", "database");

    pipeline.ingest_text(
        "doc_wal",
        "The Write-Ahead Log guarantees durability by appending records before applying to MemTable.",
        Some(meta),
    )?;

    // 4. Query with hybrid retrieval, cross-encoder reranking, and prompt generation
    let result = pipeline.query("How does the WAL guarantee durability?", 3)?;

    for doc in &result.documents {
        println!("Hit: {} (Score: {:.4})", doc.id, doc.score);
    }

    // Ready-to-use LLM prompt with structured citations
    println!("LLM Prompt:\n{}", result.prompt);
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

### RAG & Vector Search Subcommands (`flash-cli rag`)

`flash-cli rag` exposes the complete RAG, vector search, reranking, and semantic caching engine via the command line and interactive REPL.

#### 1. Ingest Documents (`ingest`)
```bash
# Ingest raw text with metadata
flash-cli --path ./my_db rag ingest \
  --id doc1 \
  --text "LSM-trees organize writes sequentially into WAL and MemTables before flushing to SSTables." \
  --title "LSM Architecture" \
  --tags storage,lsm \
  --meta category=database_internals

# Ingest markdown file with automatic paragraph chunking
flash-cli --path ./my_db rag ingest \
  --file docs/ARCHITECTURE.md \
  --chunk-strategy paragraphs \
  --chunk-size 512

# Ingest batch JSON file
flash-cli --path ./my_db rag ingest --batch-file data/knowledge_base.json
```

#### 2. Hybrid Retrieval & Cross-Encoder Query (`query`)
```bash
# Basic query with top-5 results
flash-cli --path ./my_db rag query "How are writes handled?" --top-k 5

# Hybrid query with Reciprocal Rank Fusion (RRF) and MMR diversity reranking
flash-cli --path ./my_db rag query "SSTable compression" \
  --fusion rrf \
  --mmr \
  --mmr-lambda 0.7

# JSON output with detailed cross-encoder score explanations
flash-cli --path ./my_db rag query "WAL durability" --explain --format json

# Format as an LLM prompt ready for generation with citations
flash-cli --path ./my_db rag query "Explain the block cache" --prompt --format prompt
```

#### 3. Inspect, List, and Delete Documents
```bash
# Retrieve document by ID
flash-cli --path ./my_db rag get doc1

# List indexed documents
flash-cli --path ./my_db rag list --limit 10

# Delete document and associated chunks/embeddings
flash-cli --path ./my_db rag delete doc1
```

#### 4. Telemetry, Cache Management, and IR Evaluation
```bash
# View RAG query stats, cache hit rates, and latency breakdown
flash-cli --path ./my_db rag stats

# Clear the semantic vector cache
flash-cli --path ./my_db rag clearcache

# Run IR benchmark evaluation (Recall@K, MRR, NDCG@K)
flash-cli --path ./my_db rag eval --samples-file benchmarks/eval_queries.json --k 5
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

### 8. Hybrid RAG Retrieval, Reranking & Caching

- **Unified Key-Space**: Documents, chunks, embeddings, and metadata are persisted with binary prefixes (`rag:doc:`, `rag:vec:`, `rag:meta:`, `rag:chunk:`) using atomic `WriteBatch` operations.
- **BM25 Sparse Search**: Tokenizes queries and passages into inverted index postings with configurable saturation $k_1$ and document length normalization $b$.
- **Dense Vector Search**: Computes Cosine Similarity, Dot Product, or Euclidean distance over normalized continuous vector representations.
- **Reciprocal Rank Fusion (RRF)**: Combines dense and sparse rank candidate lists without requiring score scale calibration.
- **Lexical-Semantic Cross-Encoder**: Scores candidate passages against queries using token coverage, phrase/bigram alignment, and term proximity.
- **Semantic Vector Query Cache**: Intercepts queries using exact text matching and cosine similarity thresholding to return cached hits in sub-millisecond time.

---

## ⚙️ Configuration Reference

### Storage Engine Options (`Options`)

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

### RAG Subsystem Options (`RagConfig`)

| Option Field | Default Value | Description & Tuning Advice |
| :--- | :---: | :--- |
| `embedding_dim` | `384` | Dimension of dense vector embeddings (e.g. 384, 768, 1536). |
| `similarity_metric` | `Cosine` | Vector distance metric: `Cosine`, `DotProduct`, or `Euclidean`. |
| `bm25_k1` | `1.2` | BM25 term frequency saturation parameter. |
| `bm25_b` | `0.75` | BM25 document length normalization parameter. |
| `hybrid_dense_weight` | `0.5` | Weight for dense vector scores in linear fusion ($0.0 \to 1.0$). |
| `rrf_k` | `60` | Reciprocal Rank Fusion smoothing parameter. |
| `semantic_cache_capacity` | `10,000` | Maximum number of cached query embeddings in LRU cache. |
| `semantic_cache_threshold` | `0.92` | Minimum cosine similarity to trigger a semantic cache hit. |
| `semantic_cache_ttl_secs` | `3600` | TTL in seconds for cached query results ($0 = \text{infinite}$). |

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

### Atomic Batch Operations

Demonstrates multi-key atomic transactions via `WriteBatch`:
```bash
cargo run --example batch_atomic
```

### Range Queries & Prefix Scans

Demonstrates bounded scans, prefix queries, and multi-layer merging:
```bash
cargo run --example range_queries
```

### Crash Recovery & Durability

Demonstrates persistent WAL replay, crash simulation, and SSTable recovery:
```bash
cargo run --example recovery_demo
```

### End-to-End RAG Pipeline & Semantic Caching

Full source in [`examples/rag_pipeline.rs`](examples/rag_pipeline.rs):

Demonstrates document ingestion with metadata, text chunking, dense vector embeddings, hybrid retrieval (BM25 + vector), Reciprocal Rank Fusion, cross-encoder reranking with score explanations, LLM context assembly with citations, and semantic vector cache hits:

```bash
cargo run --example rag_pipeline
```

---

## 📚 Deep Technical Documentation

Detailed architecture specifications, binary formats, and algorithms are available in the [`docs/`](docs/) directory:

- 🏛 **[Architecture Guide](docs/ARCHITECTURE.md)**: High-level overview, subsystem components, read/write lifecycles, and concurrency lock hierarchy.
- 🧠 **[RAG Retrieval, Reranking & Caching Guide](docs/RAG_RERANKER.md)**: Hybrid BM25/vector search, Reciprocal Rank Fusion, cross-encoder scoring, MMR diversity, semantic cache, and prompt assembly.
- 💾 **[SSTable Binary Format Specification](docs/SSTABLE_FORMAT.md)**: Byte-level layout of Data Blocks, Bloom Filters, Meta Index, Index Blocks, and 48-byte Footers.
- 📜 **[Write-Ahead Log (WAL) & Recovery Protocol](docs/WAL_RECOVERY.md)**: CRC32-IEEE framing, synchronous vs asynchronous durability, and crash recovery algorithm.
- 🔄 **[Leveled Compaction Architecture](docs/COMPACTION.md)**: Dynamic level scoring, multi-way merge iterators, tombstone purging, and manifest updates.
- ⚡ **[Performance, Benchmarking & Tuning](docs/BENCHMARKS.md)**: RUM trade-off matrix, configuration tuning guide, and Criterion benchmarking workflows.

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
- **Rust Toolchains**: Stable, MSRV (`1.75.0`).
- **Security Audits**: Automated cargo security advisory checks via `cargo-audit`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Developed as an open-source technical portfolio project demonstrating database internals, LSM-Tree storage architectures, concurrent data structures, and systems programming in Rust.
