# 🔌 Model Context Protocol (MCP) Server & Tool Architecture Guide

## 1. Overview & Architectural Philosophy

FlashStore implements the **Model Context Protocol (MCP)** specification (Protocol Version `2024-11-05`), transforming the embedded storage engine into an active, standardized AI integration endpoint.

By exposing an embedded MCP server, FlashStore enables:
1. **Zero-Overhead Agent Tooling**: Any MCP-compliant client (Claude Desktop, Cursor, OpenCode, Zed, Continue, Hermes) can directly read, write, scan, and inspect FlashStore with strict JSON Schema parameter validation.
2. **Unified 4-Tier Access**: A single JSON-RPC 2.0 communication bus spans the Core LSM-Tree Storage (Layer 1), Hybrid RAG & Vector Retrieval (Layer 2), and Autonomous Multi-Agent Swarm Workflows (Layer 3).
3. **Inspectable Telemetry & Resources**: Real-time storage engine stats, SSTable level distribution, RAG embedding counters, and agent swarm blackboard states via URI schemes (`flash://stats`, `flash://manifest`, `flash://rag/stats`, `flash://agent/state`).
4. **Structured Diagnostic & Synthesis Prompts**: Dynamic prompt templates for automated root-cause engine diagnosis (`flash_troubleshoot`), RAG answer synthesis (`flash_rag_synthesize`), and swarm task graph orchestration (`flash_agent_orchestrate`).
5. **Bidirectional Transport Flexibility**: Standard I/O (`stdin`/`stdout`) line-delimited streaming for inter-process hosts, plus in-memory channel transports for tests and programmatic embedding.

---

## 2. 4-Tier Layered Architecture

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

---

## 3. JSON-RPC 2.0 Protocol Framing & Lifecycle

### 3.1 Request & Response Lifecycle

The protocol operates over line-delimited JSON-RPC 2.0 messages:

```
Client                                                  FlashStore MCP
  │                                                           │
  ├─── 1. initialize {"protocolVersion":"2024-11-05"} ───────>│
  │<── 2. InitializeResult {capabilities, serverInfo} ────────┤
  │                                                           │
  ├─── 3. notifications/initialized ─────────────────────────>│
  │                                                           │
  ├─── 4. tools/list ────────────────────────────────────────>│
  │<── 5. McpListToolsResult [flash_get, flash_put, ...] ─────┤
  │                                                           │
  ├─── 6. tools/call {"name":"flash_get","arguments":{...}} ─>│
  │<── 7. McpToolResult {content:[...], isError:false} ───────┤
  │                                                           │
  ├─── 8. resources/read {"uri":"flash://stats"} ────────────>│
  │<── 9. McpReadResourceResult {contents:[...]} ─────────────┤
  │                                                           │
  ├─── 10. prompts/get {"name":"flash_troubleshoot",...} ────>│
  │<── 11. McpGetPromptResult {messages:[...]} ───────────────┤
  │                                                           │
  └─── 12. Transport EOF / close ────────────────────────────>│
```

### 3.2 Standard Error Codes
All errors conform strictly to JSON-RPC 2.0:
- `PARSE_ERROR (-32700)`: Invalid JSON syntax.
- `INVALID_REQUEST (-32600)`: JSON is not a valid JSON-RPC 2.0 Request object.
- `METHOD_NOT_FOUND (-32601)`: Method does not exist or capability disabled.
- `INVALID_PARAMS (-32602)`: Arguments failed schema validation.
- `INTERNAL_ERROR (-32603)`: Unhandled internal engine failure.

---

## 4. MCP Tools Catalog & JSON Schema

All tools expose input schemas following JSON Schema Draft 7:

| Tool Name | Scope | Required Parameters | Description |
|---|---|---|---|
| `flash_get` | Storage | `key: string` | Point lookup retrieving value from MemTable, Cache, or SSTables. |
| `flash_put` | Storage | `key: string`, `value: string` | Atomically commits key-value pair to active MemTable and WAL. |
| `flash_delete` | Storage | `key: string` | Commits tombstone record into MemTable and WAL. |
| `flash_scan` | Storage | none (optional: `start_bound`, `end_bound`, `limit`) | Scans key-value pairs across the LSM-tree in sorted order. |
| `flash_stats` | Storage | none | Retrieves real-time engine statistics (bytes, memtable count, SSTables). |
| `flash_rag_query` | RAG | `query: string` (optional: `top_k`) | Runs hybrid dense/BM25 retrieval, RRF fusion, and cross-encoder reranking. |
| `flash_agent_run` | Agent | `task_id: string`, `title: string` | Submits and executes a task through the autonomous multi-agent swarm. |

---

## 5. MCP Resource Endpoints & Dynamic Routing

Resource URIs provide live inspectable data without side effects:

### `flash://stats`
```json
{
  "active_memtable_size_bytes": 4194304,
  "immutable_memtables_count": 0,
  "levels_file_count": [2, 6, 0, 0, 0, 0, 0]
}
```

### `flash://manifest`
```json
{
  "manifest_status": "active",
  "total_levels": 7,
  "levels_distribution": [2, 6, 0, 0, 0, 0, 0]
}
```

### `flash://rag/stats`
```json
{
  "total_documents": 1420,
  "dense_vectors": 1420,
  "sparse_terms": 8940,
  "total_searches": 350,
  "cache_hits": 82,
  "cache_misses": 268,
  "cache_hit_rate": 0.234,
  "mean_latency_ms": 1.45
}
```

### `flash://agent/state`
```json
{
  "workflow_status": "Completed",
  "total_tasks": 4,
  "completed_tasks": 4,
  "failed_tasks": 0,
  "active_agents": ["planner", "coder", "verifier"]
}
```

---

## 6. MCP Prompt Templates

Prompt templates encapsulate expert system engineering practices directly into LLM prompts:

- **`flash_troubleshoot`**:
  - Parameters: `issue` (required), `component` (optional).
  - Prepares comprehensive systems engineering instructions directing the model to analyze WAL corruption, Bloom filter saturation, write stalls, or compaction backlogs.
- **`flash_rag_synthesize`**:
  - Parameters: `query` (required), `context` (required).
  - Prepares strict context-bounded synthesis instructions enforcing citation numbers `[1]`, `[2]` and prohibiting hallucination.
- **`flash_agent_orchestrate`**:
  - Parameters: `objective` (required), `available_agents` (optional).
  - Prepares DAG formulation instructions for decomposing objectives into topologically ordered execution graphs.

---

## 7. Transport Implementations

FlashStore provides two high-performance implementations of `McpTransport`:

1. **`StdioTransport<R, W>`**:
   - Wraps standard input and standard output streams (`std::io::stdin`, `std::io::stdout`).
   - Line-buffered reader with lock acquisition per message frame.
   - Used by the standalone `flash-mcp` binary and `flash-cli mcp serve`.
2. **`MockMcpTransport` / `MockMcpClientSide`**:
   - Backed by crossbeam channels for thread-safe in-memory message passing.
   - Ideal for end-to-end integration tests, embedded microservices, and in-process agent drivers without OS pipe overhead.

---

## 8. Host Integration Guides

### Claude Desktop Integration
Add FlashStore to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "flash-store": {
      "command": "flash-mcp",
      "args": [
        "--dir", "./data/flash-store",
        "--rag",
        "--agent"
      ]
    }
  }
}
```

### Cursor Integration
Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "flash-store": {
      "command": "flash-mcp",
      "args": ["--dir", "./data/flash-store", "--rag"]
    }
  }
}
```

### OpenCode Integration
Configure in `opencode.json`:

```json
{
  "mcp": {
    "flash-store": {
      "type": "stdio",
      "command": "flash-mcp",
      "args": ["--dir", "./data/flash-store", "--rag", "--agent"]
    }
  }
}
```
