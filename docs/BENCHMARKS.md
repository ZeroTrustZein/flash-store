# FlashStore Performance, Benchmarking & Tuning Guide

## 1. Overview

FlashStore is designed for high-throughput, low-latency key-value workloads on modern solid-state drives (NVMe/SSD). This document describes benchmarking methodology, workload profiles, tuning parameters, and RUM (Read, Update, Memory) trade-offs.

---

## 2. RUM Trade-Off Matrix

LSM-trees navigate the fundamental RUM conjecture (Read Amplification vs. Update Amplification vs. Memory Overhead):

| Configuration Preset | Update Throughput | Read Latency | Space Amplification | Memory Footprint | Recommended Use Case |
|---|---|---|---|---|---|
| **Write-Heavy / Ingestion** | High (100k+ ops/s) | Medium | Moderate | Low | Time-series, telemetry, append-logs |
| **Point-Lookup Heavy** | Moderate | Low (< 50µs) | Low | High (Large Cache + Bloom) | User sessions, feature stores |
| **Balanced (Default)** | High | Low | Low ($\approx 1.1\times$) | Medium (64MB Cache) | General KV storage |

---

## 3. Configuration Knobs & Tuning Parameters

All settings are configured via `OptionsBuilder`:

```rust
use flash_store::prelude::*;

let options = OptionsBuilder::new()
    .dir("/var/data/flashstore")
    .memtable_size(8 * 1024 * 1024)       // 8 MB active memtable
    .block_size(8 * 1024)                  // 8 KB data blocks
    .block_cache_size(256 * 1024 * 1024)   // 256 MB LRU cache
    .bloom_bits_per_key(12)                // 12 bits/key (< 0.5% FP)
    .sync_wal(false)                       // Async WAL append
    .max_levels(7)                         // 7 LSM levels
    .build();
```

### Key Parameter Guide:
- **`memtable_size`**: Larger memtable buffers more writes in RAM and amortizes disk flush overhead, reducing total SSTable count at the cost of more RAM.
- **`block_size`**: Smaller blocks (4KB) optimize random point lookups by reading less unneeded data. Larger blocks (16KB–64KB) optimize sequential range scans and improve compression.
- **`block_cache_size`**: Sized to hold the hot data working set in memory.
- **`bloom_bits_per_key`**:
  - `10 bits`: $\approx 1.0\%$ false positive rate.
  - `12 bits`: $\approx 0.3\%$ false positive rate.
  - `16 bits`: $\approx 0.03\%$ false positive rate.
- **`sync_wal`**: Set to `true` for financial/critical ACID transactions; `false` for high-throughput caching and ingestion.

---

## 4. Running Benchmarks

FlashStore uses `criterion` for statistical benchmarking.

### Execute All Benchmarks:
```bash
cargo bench
```

### Benchmark Suite Breakdown (`benches/benchmarks.rs`):
- `flashstore_put_seq`: Sequential insertion throughput of 10,000 keys.
- `flashstore_get_hit`: Point lookup latency for keys residing in MemTable and cached SSTables.

---

## 5. CLI Performance Profiling

You can inspect the engine internals and performance stats using the built-in CLI:

```bash
# Display live engine statistics
cargo run --bin flash_cli -- stats --dir /path/to/db

# Inspect SSTable block offsets, bloom filters, and metadata
cargo run --bin flash_cli -- sst-inspect --file /path/to/db/000001.sst
```
