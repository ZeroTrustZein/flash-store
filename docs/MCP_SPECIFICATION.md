# 🔌 Model Context Protocol (MCP) Server & Client Specification

## 1. Overview & System Integration

FlashStore natively implements the **Model Context Protocol (MCP)** specification (Protocol Version `2024-11-05`), exposing the underlying Log-Structured Merge-Tree (LSM-Tree) storage engine, RAG retrieval & semantic search, and autonomous multi-agent swarm workflows over a uniform JSON-RPC 2.0 communication bus.

By exposing an embedded MCP server, FlashStore seamlessly connects with LLM applications, host clients (such as Claude Desktop, OpenCode, Zed, Cursor, and Continue), and autonomous agent swarms without custom ad-hoc network wrappers.

```
+========================================================================+
|                       MCP HOST / CLIENT APPLICATION                    |
|          (Claude Desktop, Cursor, Custom Agent Driver, etc.)           |
+========================================================================+
                                    |
                         Standard I/O (stdin / stdout)
                         or In-Memory Channel Transport
                         JSON-RPC 2.0 Protocol Framing
                                    |
                                    v
+========================================================================+
|                    LAYER 4: FLASHSTORE MCP SUBSYSTEM                   |
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
+==============+              +==============+              +==============+
|   LAYER 1    |              |   LAYER 2    |              |   LAYER 3    |
| Core Storage |              |   RAG Engine |              | Agent Swarm  |
| (FlashStore) |              |  (Vector/BM) |              |  (DAG Engine)|
+==============+              +==============+              +==============+
```

---

## 2. Protocol Specification & Framing

FlashStore MCP implements strict **JSON-RPC 2.0** message framing.

### 2.1 Request Structure
```json
{
  "jsonrpc": "2.0",
  "id": "req-101",
  "method": "tools/call",
  "params": {
    "name": "flash_get",
    "arguments": {
      "key": "user:1001"
    }
  }
}
```

### 2.2 Response Structure
```json
{
  "jsonrpc": "2.0",
  "id": "req-101",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"found\":true,\"key\":\"user:1001\",\"value\":\"Alice\"}"
      }
    ],
    "isError": false
  }
}
```

### 2.3 Error Structure
Standardized error codes conforming to JSON-RPC 2.0 and MCP extensions:
- `-32700`: Parse error (invalid JSON).
- `-32600`: Invalid request (malformed structure).
- `-32601`: Method not found.
- `-32602`: Invalid params (missing arguments or schema failure).
- `-32603`: Internal server error.
- `-32002`: Resource not found.
- `-32001`: Tool execution error.

```json
{
  "jsonrpc": "2.0",
  "id": "req-102",
  "error": {
    "code": -32602,
    "message": "Missing required field 'key'"
  }
}
```

### 2.4 Notifications Structure
One-way messages without an `id` field:
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "flash://stats"
  }
}
```

---

## 3. Server Initialization & Capabilities Negotiation

When an MCP client connects, it negotiates protocol version and capabilities via `initialize`:

### Request: `initialize`
```json
{
  "jsonrpc": "2.0",
  "id": "init-1",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "claude-code",
      "version": "1.0.0"
    }
  }
}
```

### Response: `InitializeResult`
```json
{
  "jsonrpc": "2.0",
  "id": "init-1",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": false, "listChanged": true },
      "prompts": { "listChanged": true },
      "logging": {}
    },
    "serverInfo": {
      "name": "flash-store-mcp",
      "version": "0.1.0"
    },
    "instructions": "FlashStore MCP Server provides direct access to embedded LSM-tree key-value storage, hybrid RAG vector retrieval, autonomous multi-agent orchestration, and prompt templates."
  }
}
```

---

## 4. MCP Tools Catalog & Schemas

Tools provide LLMs and clients with executable operations against FlashStore.

### 4.1 Storage Engine Tools

#### `flash_put`
Stores a UTF-8 key-value pair into the FlashStore active MemTable and Write-Ahead Log (WAL).
- **Parameters**:
  - `key` (string, required): Key identifier.
  - `value` (string, required): Value payload.
- **Example Call**:
  ```json
  { "key": "node:config", "value": "{\"port\": 8080}" }
  ```
- **Output**: `{"success": true, "key": "node:config", "bytes_written": 16}`

#### `flash_get`
Retrieves a value for a key from MemTables, immutable MemTables, Block Cache, or SSTables.
- **Parameters**:
  - `key` (string, required): Target key identifier.
- **Example Call**:
  ```json
  { "key": "node:config" }
  ```
- **Output**: `{"found": true, "key": "node:config", "value": "{\"port\": 8080}"}`

#### `flash_delete`
Writes a tombstone deletion marker for the specified key.
- **Parameters**:
  - `key` (string, required): Target key identifier.
- **Example Call**:
  ```json
  { "key": "node:config" }
  ```
- **Output**: `{"success": true, "key": "node:config"}`

#### `flash_scan`
Performs an ordered lexicographical range scan between optional `start_bound` and `end_bound`.
- **Parameters**:
  - `start_bound` (string, optional): Inclusive lower bound key.
  - `end_bound` (string, optional): Exclusive upper bound key.
  - `limit` (integer, optional, default: 100): Maximum key-value pairs to return.
- **Example Call**:
  ```json
  { "start_bound": "user:", "end_bound": "user:~", "limit": 10 }
  ```
- **Output**: `{"count": 2, "entries": [{"key": "user:1", "value": "..."}, {"key": "user:2", "value": "..."}]}`

#### `flash_stats`
Returns instantaneous engine storage metrics.
- **Parameters**: None.
- **Output**:
  ```json
  {
    "active_memtable_size_bytes": 1048576,
    "immutable_memtables_count": 1,
    "levels_file_count": [4, 10, 0, 0, 0, 0, 0]
  }
  ```

### 4.2 RAG & Vector Retrieval Tools

#### `flash_rag_query`
Queries the hybrid BM25 and dense vector retrieval subsystem with reciprocal rank fusion (RRF) and cross-encoder reranking.
- **Parameters**:
  - `query` (string, required): Natural language search query.
  - `top_k` (integer, optional, default: 5): Maximum candidate documents.
- **Example Call**:
  ```json
  { "query": "LSM-tree write-ahead log recovery", "top_k": 3 }
  ```

### 4.3 Autonomous Agent Tools

#### `flash_agent_run`
Triggers an autonomous task execution within the multi-agent swarm workflow engine.
- **Parameters**:
  - `task_id` (string, required): Unique task identifier.
  - `title` (string, required): Short task objective title.
  - `description` (string, optional): Detailed specifications or execution prompt.
- **Example Call**:
  ```json
  { "task_id": "task-audit-1", "title": "Run Invariant Audit", "description": "Verify LSM level overlap constraints" }
  ```

---

## 5. MCP Resources & Real-Time Telemetry

Resources provide inspectable endpoints mapped to uniform URI patterns:

| Resource URI | MIME Type | Description |
|---|---|---|
| `flash://stats` | `application/json` | Real-time LSM-tree engine statistics, active MemTable byte sizes, and SSTable distribution. |
| `flash://manifest` | `application/json` | VersionSet manifest status and SSTable file counts across tiers L0–LN. |
| `flash://rag/stats` | `application/json` | Total document counts, dense vector embeddings, sparse inverted index terms, cache hits/misses. |
| `flash://rag/config` | `application/json` | Active hyper-parameters: vector dimension, similarity metric, BM25 $k_1/b$, RRF constant $k$. |
| `flash://agent/state` | `application/json` | Swarm state: completed tasks, active tasks, failure counts, and blackboard entries. |
| `flash://agent/agents` | `application/json` | Registered swarm roster with agent roles, capabilities, and system prompts. |

---

## 6. MCP Prompt Templates

FlashStore provides pre-configured prompt templates for LLM workflows:

### 6.1 `flash_troubleshoot`
Generates systems-level diagnostic prompts for diagnosing engine write stalls, Bloom filter saturation, and compaction delays.
- **Arguments**:
  - `issue` (required): Description of observed anomaly.
  - `component` (optional): `memtable`, `wal`, `sstable`, or `compaction`.

### 6.2 `flash_rag_synthesize`
Generates authoritative answers to user queries grounded strictly in retrieved context chunks with citation indexes.
- **Arguments**:
  - `query` (required): The user question.
  - `context` (required): Retrieved document chunks.

### 6.3 `flash_agent_orchestrate`
Coordinates swarm task decomposition and assigns specialized agent roles.
- **Arguments**:
  - `objective` (required): High-level system goal.
  - `available_agents` (optional): Active agent definitions.

---

## 7. Standalone MCP Server Binary (`flash-mcp`)

FlashStore includes a dedicated binary `flash-mcp` designed to run as an independent background daemon over standard input/output (`stdin`/`stdout`).

### 7.1 Command Line Options
```bash
flash-mcp [OPTIONS]
```
- `-d, --dir <PATH>`: Database path (default: `./data/flash-store`).
- `--name <NAME>`: MCP server identifier (default: `flash-store-mcp`).
- `--server-version <VER>`: Override semantic server version string.
- `--disable-tools`: Disables tool execution endpoint.
- `--disable-resources`: Disables inspectable resources.
- `--disable-prompts`: Disables prompt templates.
- `--rag`: Enables native RAG hybrid retrieval tools and resources.
- `--agent`: Enables autonomous multi-agent swarm workflow tools and resources.

### 7.2 Host Configuration Example (Claude Desktop)

To wire FlashStore into Claude Desktop, add the following to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "flash-store": {
      "command": "C:/Users/Zein/auto-workspace/projects/flash-store/target/release/flash-mcp.exe",
      "args": [
        "--dir", "C:/Users/Zein/.flash-store/data",
        "--rag",
        "--agent"
      ]
    }
  }
}
```

---

## 8. CLI MCP Subcommands (`flash-cli mcp`)

FlashStore's unified CLI (`flash-cli`) provides built-in tools for interacting with and debugging the MCP subsystem:

```bash
# Display MCP server telemetry and capabilities
flash-cli mcp info

# List all registered tools
flash-cli mcp tools --format json

# Execute a tool directly with inline arguments
flash-cli mcp call flash_put -a '{"key": "cluster:node-1", "value": "healthy"}'

# Execute a tool reading arguments from a JSON file
flash-cli mcp call flash_put --file ./payload.json

# Read an inspectable resource
flash-cli mcp read flash://stats

# List available prompt templates
flash-cli mcp prompts

# Render a prompt template with arguments
flash-cli mcp get-prompt flash_troubleshoot -a '{"issue": "Compaction backlog L0"}'

# Send raw JSON-RPC 2.0 requests
flash-cli mcp request -j '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'

# Start the stdio MCP server loop from flash-cli
flash-cli mcp serve --name flash-store-mcp
```
