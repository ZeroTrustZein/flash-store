# FlashStore Architecture Guide

## 1. Introduction

`FlashStore` is an embedded, thread-safe, high-performance key-value storage engine engineered in Rust. It utilizes a **Log-Structured Merge-Tree (LSM-Tree)** architecture optimized for low-latency sequential writes, efficient point lookups, and bounded multi-level range scans on modern solid-state storage.

---

## 2. High-Level Architecture Overview

```
+========================================================================+
|                       MCP HOST / CLIENT APPLICATION                    |
|           (Claude Desktop, Cursor, OpenCode, Custom Agents)            |
+========================================================================+
                                    |
                         Standard I/O (stdin / stdout)
                         or In-Memory Channel Transport
                         JSON-RPC 2.0 Protocol Framing
                                    |
                                    v
+========================================================================+
|             LAYER 4: MODEL CONTEXT PROTOCOL (MCP) SUBSYSTEM            |
|                                                                        |
|  +--------------------+  +--------------------+  +------------------+  |
|  |   Tool Registry    |  | Resource Registry  |  | Prompt Registry  |  |
|  | (Storage/RAG/Agent)|  | (Stats/Manifest/Wf)|  | (Troubleshoot/RAG|  |
|  +---------+----------+  +---------+----------+  +--------+---------+  |
|            |                       |                      |            |
|            +-----------------------+----------------------+            |
|                                    |                                   |
|                      +-------------v--------------+                    |
|                      |      McpServer Runtime     |                    |
|                      |  (Handlers & Notifications)|                    |
|                      +-------------+--------------+                    |
+====================================|===================================+
                                     |
       +-----------------------------+-----------------------------+
       v                             v                             v
+========================================================================+
|             LAYER 3: AUTONOMOUS MULTI-AGENT SWARM SUBSYSTEM            |
|                                                                        |
|    DAG Task Graphs   ──>  Adaptive Router      ──>  Reflection Engine  |
|    Dynamic Tools     ──>  Episodic Blackboard  ──>  Pub/Sub Telemetry  |
+========================================================================+
                                   |
                                   v
+========================================================================+
|             LAYER 2: HYBRID RAG & CROSS-ENCODER SUBSYSTEM              |
|                                                                        |
|    BM25 Sparse Index ──>  Dense Vector Index   ──>  Hybrid Score Fusion|
|    Cross-Encoder     ──>  MMR Diversity        ──>  Semantic Cache     |
+========================================================================+
                                   |
                                   v
+========================================================================+
|             LAYER 1: CORE FLASHSTORE LSM-TREE ENGINE                   |
|                                                                        |
|    WAL (CRC32 IEEE)  ──>  Concurrent MemTables ──>  SSTables (L0..LN)  |
|    LRU Block Cache   ──>  Leveled Compaction   ──>  Manifest / Version |
+========================================================================+
```

### Core Storage Hierarchy (Layer 1 Detail)

```
                      +---------------------------------------+
                      |          FlashStore Public API        |
                      |   (put, get, delete, write_batch)     |
                      +---------------------------------------+
                                    |          |
                      Write Path    |          |   Read Path
                                    v          v
                 +--------------------+      +-------------------------+
                 |  Write-Ahead Log   |      |    Active MemTable      |
                 | (current.wal CRC32)|      | (SkipList Concurrent)   |
                 +--------------------+      +-------------------------+
                                    |                  |
                                    v                  v
                             [Disk Append]   +-------------------------+
                                             |  Immutable MemTables    |
                                             |  (Vector of MemTables)  |
                                             +-------------------------+
                                                       |
                                               Flush   v
                                             +-------------------------+
                                             |      Level 0 SSTs       |
                                             |   (Overlapping Ranges)  |
                                             +-------------------------+
                                                       |
                                          Compaction   v
                                             +-------------------------+
                                             |      Level 1..N SSTs    |
                                             | (Non-Overlapping Sorted)|
                                             +-------------------------+
                                                       |
                      +--------------------------------+------------------------+
                      |                                                         |
                      v                                                         v
         +--------------------------+                              +--------------------------+
         |     LRU Block Cache      |                              |    Manifest & Version    |
         |   (Uncompressed Blocks)  |                              |    (MANIFEST FileEdits)  |
         +--------------------------+                              +--------------------------+
```

---

## 3. Core Engine Components

### 3.1 Write-Ahead Log (`src/wal/`)
- **Purpose**: Guarantees zero data loss (ACID Durability) across power loss or ungraceful shutdown.
- **Mechanism**: Every `put`, `delete`, or `write_batch` operation is formatted into framed binary records with CRC32-IEEE checksums and appended to `current.wal` before updating in-memory state.
- **Sync Mode**: Configurable via `Options::sync_wal` (`true` for synchronous `fsync` after writes, `false` for OS-buffered asynchronous writes).

### 3.2 MemTable Subsystem (`src/memtable/`)
- **Structure**: High-concurrency sorted SkipList with tower height up to 16.
- **Active vs. Immutable**:
  - `memtable`: Single active `Arc<MemTable>` wrapped in `parking_lot::RwLock` for concurrent readers and sequential write updates.
  - `imm_memtables`: `RwLock<Vec<Arc<MemTable>>>` storing read-only tables awaiting background/explicit flush.
- **Capacity**: Monitored by `approximate_size()`. When exceeding `memtable_size` (default 4MB), flush is scheduled.

### 3.3 SSTable Subsystem (`src/sstable/`)
- **Format**: Immutable disk files named `<id>.sst`.
- **Layout**:
  - Sequence of variable-length **Data Blocks** containing ordered key-value entries.
  - **Meta Index Block** storing offsets and sizes of metadata blocks.
  - **Bloom Filter Block** containing bit arrays generated via Murmur/CRC32 double-hashing (default 10 bits/key) for $O(1)$ negative lookup pruning.
  - **Index Block** indexing the last key and file offset of each data block for $O(\log N)$ block lookup.
  - **48-byte Fixed Footer** terminating with magic number `0xF1A5457072653031`.

### 3.4 In-Memory Block Cache (`src/cache/`)
- **Implementation**: `LruBlockCache` implementing `BlockCache` trait with $O(1)$ hash map index and intrusive doubly linked list for eviction.
- **Keying**: Indexed by tuple `(file_number, block_offset)`.
- **Thread Safety**: Protected by fine-grained `parking_lot::Mutex`.

### 3.5 Manifest & Version Management (`src/manifest/`)
- **Version Tracking**: Tracks current active SSTables per level and monotonic sequence numbers.
- **VersionEdit**: Atomic journal entry representing additions (new SSTs from flush/compaction) and deletions (obsolete SSTs).
- **Crash Recovery**: Manifest is read sequentially on startup to reconstruct the `VersionSet` without scanning file directories manually.

### 3.6 Leveled Compaction (`src/compaction/`)
- **Strategy**: Multi-level tiered compaction with dynamic score calculation.
- **Level 0**: Triggered when file count $\ge 4$.
- **Levels 1..N**: Triggered when total level byte size exceeds level target ($10 \text{MB} \times 10^{\text{level}-1}$).
- **Purge Rules**: Obsolete versions and tombstones are safely removed during multi-way merges.

---

## 4. Operation Lifecycles

### 4.1 Write Path (`put`, `delete`, `write_batch`)
1. Obtain next monotonic `SequenceNumber` from `version_set.next_sequence()`.
2. Construct `WalRecord` containing key, value/tombstone flag, and sequence number.
3. Acquire `wal` mutex and write/flush record to disk.
4. Acquire read lock on active `memtable` and insert key-value/tombstone pair into the SkipList.
5. If `memtable.approximate_size() >= options.memtable_size`, trigger `maybe_schedule_flush()`.

### 4.2 Read Path (`get`)
1. **Active MemTable**: Search `memtable.read()`. If found, return value or `None` (if tombstone).
2. **Immutable MemTables**: Search `imm_memtables.read()` from newest to oldest.
3. **Level 0 SSTables**: Search Level 0 SSTables in reverse order (newest file to oldest file):
   - Fast check: Bloom filter probe.
   - If present in Bloom: Binary search Index Block -> Check LRU Cache / Read Data Block -> Decode entry.
4. **Level 1..N SSTables**: For each level, binary search file metadata by key range $[smallest\_key, largest\_key]$. Since levels 1..N have strictly non-overlapping key ranges, at most 1 file is inspected per level.

### 4.3 Range Scan Path (`scan`)
1. Read entries from all levels and immutable/active memtables.
2. Deduplicate keys using max sequence number ordering.
3. Filter out tombstones and apply bounds `[start_key, end_key)`.

---

## 5. Concurrency Model & Lock Hierarchy

To avoid deadlocks and maximize throughput under heavy concurrent workloads, FlashStore maintains a strict lock ordering:

```
+-------------------------------------------------------------+
| Lock Ordering Hierarchy:                                    |
|   1. workflow_lock / task_lock   (Coarse agent scheduling)  |
|   2. flush_lock / compact_lock   (Coarse lifecycle locks)   |
|   3. memory_store lock           (Blackboard / episodic rw) |
|   4. memtable write lock         (Active table swap)        |
|   5. imm_memtables write lock    (Immutable queue update)   |
|   6. wal lock                    (Mutex<WalWriter>)         |
|   7. version_set lock / manifest (Manifest updates)         |
|   8. block_cache lock            (LRU node mutations)       |
+-------------------------------------------------------------+
```

- Readers acquire shared `RwLock` guards on memtables and run lock-free lookups on SSTables.
- Writes append to WAL sequentially under `wal.lock()` and concurrently insert into memtable SkipLists.

---

## 6. Integrated RAG & Reranker Subsystem (Layer 2)

FlashStore features a native **Retrieval-Augmented Generation (RAG) and Cross-Encoder Reranker** subsystem layered directly over the core LSM-Tree storage engine.

```
+========================================================================+
|                        RAG & RERANKER SUBSYSTEM                        |
|                                                                        |
|  +---------------------+  +--------------------+  +------------------+ |
|  | Document Ingestion  |  |  BM25 Sparse Index |  | Dense Vector Idx | |
|  | & Chunking Pipeline |  | (Inverted Index)   |  | (Cosine/Dot/L2)  | |
|  +----------+----------+  +---------+----------+  +--------+---------+ |
|             |                       |                      |           |
|             |                       +----------+-----------+           |
|             |                                  |                       |
|             |                       +----------v-----------+           |
|             |                       | Hybrid Fusion Engine |           |
|             |                       | (RRF / Linear/Borda) |           |
|             |                       +----------+-----------+           |
|             |                                  |                       |
|             |                       +----------v-----------+           |
|             |                       | Cross-Encoder Rerank |           |
|             |                       | (Lexical/Prox/MMR)   |           |
|             |                       +----------+-----------+           |
|             |                                  |                       |
|             |                       +----------v-----------+           |
|             |                       |  Context Assembler   |           |
|             |                       |  (Citations/Prompt)  |           |
|             |                       +----------+-----------+           |
|             |                                  |                       |
+=============|==================================|=======================+
              |                                  |
              | Atomically Persisted             | Key-Space Lookups
              v                                  v
+========================================================================+
|                    CORE FLASHSTORE LSM-TREE (LAYER 1)                  |
|                                                                        |
|    WAL (CRC32)  ──>  Concurrent MemTables  ──>  SSTables (L0..LN)      |
|    Block Cache  ──>  Leveled Compaction    ──>  VersionSet / Manifest  |
+========================================================================+
```

### 6.1 Storage Adapter & Atomic WriteBatch

The `RagStoreAdapter` maps high-level document domain models to FlashStore's key-value space using binary prefixes:
- `rag:doc:<doc_id>`: UTF-8 raw text payload.
- `rag:vec:<doc_id>`: Little-endian IEEE-754 vector embedding byte slice.
- `rag:meta:<doc_id>`: Serialized metadata key-value pairs.
- `rag:chunk:<doc_id>:<chunk_idx>`: Chunk boundaries, text, and offset indices.
- `rag:sys:<key>`: Schema version and subsystem statistics.

Every document ingestion is batched into a single atomic `WriteBatch`, ensuring that text, chunks, embeddings, and metadata are written together to the WAL with strict crash consistency.

### 6.2 Dual-Mode Engine Design

1. **In-Memory Mode (`RagEngine::new(config)`)**: Ultra-fast ephemeral indexing suitable for stateless workers, unit testing, and microbenchmarks.
2. **Persistent Mode (`RagEngine::with_store(config, store)` / `recover(...)`)**: Backed by a FlashStore instance, providing instant startup hydration by replaying persisted document keys and vectors from SSTables.

For detailed algorithms, vector similarity metrics, BM25 formulas, and reranking heuristics, see the comprehensive [RAG & Reranker Architecture Guide](RAG_RERANKER.md).

---

## 7. Autonomous Multi-Agent Swarm Subsystem (Layer 3)

Layer 3 provides an autonomous multi-agent task execution and workflow engine operating natively on top of the RAG and LSM-Tree storage layers.

```
+========================================================================+
|             LAYER 3: MULTI-AGENT SWARM WORKFLOW ENGINE                 |
|                                                                        |
|  +--------------------+  +--------------------+  +------------------+  |
|  | Task Graph & DAG   |  | Adaptive Router    |  | Reflection Loop  |  |
|  | Dependency Engine  |  | (Cap/Role/Load/Pri)|  | (Critique/Invars)|  |
|  +---------+----------+  +---------+----------+  +--------+---------+  |
|            |                       |                      |            |
|            +-----------------------+----------------------+            |
|                                    |                                   |
|                      +-------------v--------------+                    |
|                      |  Agent Pipeline Coordinator|                    |
|                      +-------------+--------------+                    |
|                                    |                                   |
|            +-----------------------+----------------------+            |
|            |                       |                      |            |
|  +---------v----------+  +---------v----------+  +--------v---------+  |
|  | Dynamic Tool Reg   |  | Blackboard & Memory|  | Event & Telemetry|  |
|  | (OpenAI / MCP / KV)|  | (Episodic Recall)  |  | (p50/p90/p99)    |  |
|  +--------------------+  +--------------------+  +------------------+  |
+====================================|===================================+
                                     |
                                     v Persisted via WriteBatch
+========================================================================+
|                    CORE FLASHSTORE LSM-TREE (LAYER 1)                  |
+========================================================================+
```

### 7.1 Swarm Orchestration & Execution Graph
- **DAG Execution**: Manages tasks with complex multi-stage dependency trees, validating for acyclicity and dispatching ready tasks as upstream dependencies complete.
- **State Machine**: Transitions tasks deterministically through `Pending` $\to$ `Assigned` $\to$ `InProgress` $\to$ `InReflection` $\to$ `Completed`.

### 7.2 Adaptive Task Routing
- **Multi-Factor Scoring**: Routes tasks based on required capabilities ($w_c$), role affinity ($w_a$), agent queue depth / workload ($w_w$), and task priority preemption ($w_p$).
- **Pluggable Routers**: Includes `AdaptiveTaskRouter`, `LoadBalancingRouter`, and `PriorityPreemptiveRouter`.

### 7.3 Stateful Reflection & Convergence Loops
- **Iterative Self-Correction**: Drives agent solution refinement using structured critiques across accuracy, completeness, structure, and domain invariants.
- **Convergence Guardrails**: Halts on quality threshold convergence, plateau detection ($\Delta S < \text{min\_delta}$), or max iteration limits.

### 7.4 Dynamic Tools & MCP Integration
- **Schema Validation**: Parameters conform to JSON Schema with automatic export to OpenAI Tool Calling and Anthropic Model Context Protocol (MCP) tool format.
- **Embedded Storage Tools**: Provides native tools (`kv_put`, `kv_get`, `kv_delete`, `kv_scan`) granting autonomous agents direct, safe access to the FlashStore LSM-tree engine.

### 7.5 Dual-Tier Memory & Associative Recall
- **Shared Working Memory (Blackboard)**: Fast concurrent scratchpad for intra-swarm data sharing.
- **Episodic Memory**: Recorded task outcomes and reflections indexed with decaying recency, assigned importance, and query-relevance lexical similarity.

### 7.6 Atomic Storage Adapter & Checkpoint Recovery
- All swarm states (agent registrations, tasks, messages, checkpoints, and episodic memories) are mapped to deterministic binary key prefixes (`agent:def:`, `agent:task:`, `agent:msg:`, `agent:wf:`, `agent:mem:`).
- Checkpoints commit atomically via `WriteBatch`, enabling full crash recovery via `AgentPipeline::restore_from_store`.

For deep technical specifications, mathematical scoring formulations, and tool calling schemas, see the [Multi-Agent Swarm Architecture Guide](AGENT_WORKFLOW.md).

---

## 8. Model Context Protocol (MCP) Subsystem (Layer 4)

Layer 4 provides an embedded Model Context Protocol (MCP) server adhering to protocol specification `2024-11-05`. It exposes FlashStore's storage primitives, RAG retrieval engine, and multi-agent swarm workflows over a uniform JSON-RPC 2.0 communication bus.

### 8.1 Tool Registry & Execution
- **Storage Primitives**: `flash_get`, `flash_put`, `flash_delete`, `flash_scan`, `flash_stats`.
- **RAG Subsystem**: `flash_rag_query` for hybrid dense/sparse vector retrieval with cross-encoder scoring.
- **Agent Orchestrator**: `flash_agent_run` for submitting and driving multi-agent workflows.
- All tools declare strict input schemas validated via JSON Schema Draft 7.

### 8.2 Inspectable Resources
Dynamic URI endpoints exposing real-time engine telemetry and diagnostics:
- `flash://stats`: MemTable byte sizes, immutable count, SSTable level distribution.
- `flash://manifest`: VersionSet status and multi-tier SSTable distribution.
- `flash://rag/stats` & `flash://rag/config`: RAG index size, cache hit/miss rates, and active hyperparameters.
- `flash://agent/state` & `flash://agent/agents`: Autonomous swarm task counts, blackboard state, and agent capabilities.

### 8.3 Prompt Templates & Dynamic Synthesis
Structured templates for client LLMs:
- `flash_troubleshoot`: Expert systems engineering diagnostic prompt for LSM-tree anomalies.
- `flash_rag_synthesize`: Context-grounded citation-aware answer synthesis prompt.
- `flash_agent_orchestrate`: Task decomposition and agent delegation prompt.

### 8.4 Transport & Notification Broadcasting
- **Stdio Transport**: Line-delimited JSON-RPC 2.0 over standard I/O for Claude Desktop, OpenCode, and Cursor.
- **In-Memory Channel Transport**: High-throughput crossbeam-channel transport for tests and embedded applications.
- **Notification Broadcaster**: Pub/sub engine streaming log messages (`notifications/message`) across 8 syslog-aligned log levels and resource updates (`notifications/resources/updated`).

For deep technical specifications, protocol framing, and client configuration guides, see the [MCP Server & Tool Architecture Guide](MCP_SERVER.md).

---

## 9. Subsystem Technical Guides

- 🔌 **[Model Context Protocol (MCP) Architecture Guide](MCP_SERVER.md)**: JSON-RPC 2.0 framing, tool catalog, resource URIs, and prompt templates.
- 🤖 **[Multi-Agent Swarm Architecture Guide](AGENT_WORKFLOW.md)**: Task graphs, adaptive routing, reflection loops, tools, and episodic memory.
- 🧠 **[RAG & Reranker Architecture Guide](RAG_RERANKER.md)**: Hybrid BM25/vector search, Reciprocal Rank Fusion, cross-encoders, and semantic caching.
- 💾 **[SSTable Binary Format Specification](SSTABLE_FORMAT.md)**: Data blocks, Bloom filters, index blocks, and 48-byte footers.
- 📜 **[Write-Ahead Log (WAL) & Recovery Protocol](WAL_RECOVERY.md)**: CRC32 framing, durability modes, and replay algorithms.
- 🔄 **[Leveled Compaction Architecture](COMPACTION.md)**: Multi-way merge iterators, tombstone purging, and dynamic level scoring.
- ⚡ **[Performance, Benchmarking & Tuning](BENCHMARKS.md)**: RUM trade-offs and Criterion benchmark suites.


