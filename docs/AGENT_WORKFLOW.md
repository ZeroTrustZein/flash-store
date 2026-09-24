# 🤖 Autonomous Multi-Agent Swarm Orchestrator & Workflow Subsystem

## 1. Overview & Architectural Philosophy

FlashStore's **Autonomous Multi-Agent Workflow Engine** elevates the storage engine into an active, intelligent orchestration substrate. Built directly on top of FlashStore's core Log-Structured Merge-Tree (LSM-Tree) and native RAG subsystem, the agent workflow engine provides:

1. **DAG-Based Task Graph Execution**: Deterministic Directed Acyclic Graph (DAG) scheduling with topological ordering, multi-level dependency resolution, and parallel execution candidacy.
2. **Adaptive Task Routing**: Multi-criteria heuristic task distribution balancing agent capabilities, role affinities, dynamic workloads, and priority preemption.
3. **Stateful Reflection Loops & Convergence**: Iterative self-correction where critique engines evaluate task candidate outputs against score thresholds, improvement deltas, and domain invariant verifications.
4. **Dynamic Tool Execution & MCP Integration**: Extensible tool registry supporting parameter validation, standardized JSON Schema, Model Context Protocol (MCP) tool format compliance, and direct LSM-Tree storage tools (`kv_put`, `kv_get`, `kv_delete`, `kv_scan`).
5. **Dual-Tier Memory (Blackboard & Episodic)**: Concurrent working memory (shared blackboard) for inter-agent context synthesis coupled with persistent episodic memory supporting associative semantic recall.
6. **Unified Storage Persistence & Checkpointing**: Atomic `WriteBatch` commits of agent definitions, task graphs, execution checkpoints, and episodic memories into FlashStore's WAL and SSTables with zero schema drift.
7. **Real-Time Event Streaming & Telemetry**: Decoupled pub/sub event broadcaster supporting custom handlers, latency tracking (p50, p90, p99, min, max, avg), and real-time execution statistics.

---

## 2. 3-Tier Layered Architecture

FlashStore unifies storage, retrieval, and autonomous computation in a cohesive 3-tier architecture:

```
+========================================================================+
|             LAYER 3: AUTONOMOUS MULTI-AGENT SWARM SUBSYSTEM            |
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
                                     v
+========================================================================+
|             LAYER 2: HYBRID RAG & CROSS-ENCODER SUBSYSTEM              |
|                                                                        |
|   BM25 Inverted Index  ──  Dense Vector Index  ──  Hybrid Fusion (RRF) |
|   Cross-Encoder Reranker  ──  MMR Diversity   ──  Semantic Query Cache |
+====================================|===================================+
                                     |
                                     v
+========================================================================+
|             LAYER 1: CORE FLASHSTORE LSM-TREE ENGINE                   |
|                                                                        |
|   WAL Journal (CRC32)  ──  Concurrent MemTables  ──  SSTables (L0..LN) |
|   LRU Block Cache      ──  Leveled Compaction   ──  Atomic Manifest    |
+========================================================================+
```

---

## 3. Domain Model & Core Types

The agent subsystem defines strongly-typed data structures in `src/agent/types.rs`:

### 3.1 Agent Definitions & Roles

```rust
pub struct AgentDefinition {
    pub id: AgentId,
    pub name: String,
    pub description: String,
    pub role: AgentRole,
    pub capabilities: HashSet<AgentCapability>,
    pub system_prompt: String,
    pub max_iterations: usize,
    pub timeout_secs: u64,
    pub metadata: HashMap<String, String>,
}
```

- **`AgentRole`**: Enumeration defining agent specialization:
  - `Planner`: Breaks down complex goals into DAG task graphs.
  - `Router`: Directs tasks to specialized agents based on heuristic scores.
  - `Coder`: Implements algorithms, storage routines, or transformations.
  - `Critic`: Reviews candidate solutions and provides structured critiques.
  - `Verifier`: Asserts formal invariants, schema compliance, and test suites.
  - `ToolExecutor`: Invokes registered tools, runs external APIs, and interacts with storage.
  - `MemoryAgent`: Manages working memory, blackboard entries, and episodic retrieval.
  - `Custom(String)`: User-defined custom role strings.
- **`AgentCapability`**: Granular feature flags required by tasks:
  - `TextGeneration`: Natural language generation and synthesis.
  - `ToolCalling`: Automated parameter parsing and tool execution.
  - `CodeExecution`: Logic evaluation and computation.
  - `StorageOps`: Direct FlashStore KV put, get, scan, and delete access.
  - `MemoryRetrieval`: Working memory and episodic knowledge recall.
  - `Reflection`: Self-critique, invariant verification, and iterative revision.

### 3.2 Task Lifecycle & Graph Execution

```rust
pub struct AgentTask {
    pub id: TaskId,
    pub title: String,
    pub description: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub assigned_agent: Option<AgentId>,
    pub required_capabilities: HashSet<AgentCapability>,
    pub input_context: Option<serde_json::Value>,
    pub output_result: Option<serde_json::Value>,
    pub reflection_history: Vec<ReflectionStep>,
    pub error_message: Option<String>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
}
```

- **`TaskPriority`**: `Low` (0), `Medium` (1), `High` (2), `Critical` (3). Ordered and comparable.
- **`TaskStatus` State Machine**:
  ```
  [Pending] ──> [Assigned] ──> [InProgress] ──> [InReflection] ──> [Completed]
      │             │               │                 │
      +-------------+---------------+-----------------+--------> [Failed] / [Cancelled]
  ```
- **`TaskGraph`**: Directed Acyclic Graph maintaining dependency relationships (`depends_on`).
  - Supports dependency cycles detection (`CycleDetected`).
  - Identifies ready tasks whose dependencies have all reached `Completed` status.
  - Calculates topological execution tiers for maximal concurrency.

---

## 4. Adaptive Task Routing Engine

The routing engine (`src/agent/router.rs`) assigns tasks to the most suitable available agents using configurable multi-objective scoring.

### 4.1 Routing Heuristic Formulation

For candidate agent $A$ and incoming task $T$, the aggregate suitability score $S(A, T) \in [0.0, 1.0]$ is computed as:

$$S(A, T) = w_c \cdot \text{CapMatch}(A, T) + w_a \cdot \text{RoleAffinity}(A, T) + w_w \cdot (1 - \text{Workload}(A)) + w_p \cdot \text{Priority}(T)$$

Where:
- $\text{CapMatch}(A, T) = \frac{|\text{Caps}(A) \cap \text{ReqCaps}(T)|}{|\text{ReqCaps}(T)|}$ (Strictly required: if $\text{ReqCaps}(T) \not\subseteq \text{Caps}(A)$, match is rejected or penalized).
- $\text{RoleAffinity}(A, T) \in [0.0, 1.0]$: Affinity matrix based on task keywords and role mappings.
- $\text{Workload}(A) = \min\left(1.0, \frac{\text{ActiveTasks}(A)}{\text{MaxCapacity}(A)}\right)$.
- $\text{Priority}(T) = \frac{\text{PriorityTier}(T)}{3.0}$.
- Default weights: $w_c = 0.40, w_a = 0.30, w_w = 0.20, w_p = 0.10$ (normalized such that $\sum w = 1.0$).

### 4.2 Routing Strategies

- **`AdaptiveTaskRouter`**: Evaluates all candidate agents via the weighted heuristic formula above.
- **`LoadBalancingRouter`**: Selects capable agents with the lowest current queued workload.
- **`PriorityPreemptiveRouter`**: Fast-tracks `Critical` and `High` priority tasks to top-performing agents.

---

## 5. Stateful Reflection Loops & Self-Correction

The reflection engine (`src/agent/reflection.rs`) drives iterative improvement of task solutions prior to completion.

```
                  +--------------------------------+
                  |     Initial Solution Output    |
                  +---------------+----------------+
                                  |
                                  v
+─────────────────────────────────────────────────────────────────────────────+
|                          Stateful Reflection Loop                           |
|                                                                             |
|      +───────────────────────────────────────────────────────────────+      |
|      | 1. Invariant Verification: syntax, length, forbidden terms    |      |
|      +──────────────────────────────┬────────────────────────────────+      |
|                                     v                                       |
|      +───────────────────────────────────────────────────────────────+      |
|      | 2. Criterion Evaluation: Accuracy, Completeness, Structure    |      |
|      +──────────────────────────────┬────────────────────────────────+      |
|                                     v                                       |
|      +───────────────────────────────────────────────────────────────+      |
|      | 3. Critique Formulation: Verdict (Pass, Minor, Major, Violation)     |
|      +──────────────────────────────┬────────────────────────────────+      |
|                                     v                                       |
|      +───────────────────────────────────────────────────────────────+      |
|      | 4. Convergence Check: Score >= Threshold OR Max Iterations    |      |
|      +──────────────────────────────┬────────────────────────────────+      |
+─────────────────────────────────────|───────────────────────────────────────+
                                      |
                      +---------------+---------------+
                      | Passed                        | Needs Revision
                      v                               v
            +-------------------+          +---------------------+
            | Solution Approved |          | Revision Prompt Gen |
            +-------------------+          +----------+----------+
                                                      |
                                                      v
                                           [ Re-execute Agent ]
```

### 5.1 Evaluation Verdicts & Criteria

- **`ReflectionVerdict`**:
  - `Pass`: Solution meets all quality criteria ($\ge \text{threshold}$).
  - `MinorIssues`: Usable, but minor improvements recommended.
  - `MajorRework`: Fails key criteria; requires substantial revision.
  - `InvariantViolation`: Violates critical correctness or security invariants; rejected immediately.
- **`EvaluationCriterion`**:
  - `Accuracy`: Alignment with task requirements and input context.
  - `Completeness`: Coverage of required fields, arguments, or edge cases.
  - `Structure`: Output formatting, JSON Schema adherence, or documentation style.
  - `Efficiency`: Computational or algorithmic brevity.

### 5.2 Convergence Evaluation

The loop terminates when:
1. `verdict == ReflectionVerdict::Pass` and aggregate score $\ge \text{threshold}$ (default: 0.85).
2. The score delta between consecutive iterations $\Delta S < \text{min\_improvement}$ (plateau detection).
3. The iteration count reaches `max_iterations` (default: 5).

---

## 6. Dynamic Tool Execution & MCP Integration

The tool subsystem (`src/agent/tools.rs`) allows autonomous agents to invoke local and storage operations safely.

### 6.1 Schema Compliance (OpenAI & Anthropic MCP)

Every `Tool` exposes structured metadata convertible to standard LLM tool-calling formats:

```json
{
  "name": "kv_put",
  "description": "Store a key-value pair in FlashStore LSM-Tree storage",
  "parameters": {
    "type": "object",
    "properties": {
      "key": { "type": "string", "description": "Storage key" },
      "value": { "type": "string", "description": "Payload string or JSON" }
    },
    "required": ["key", "value"]
  }
}
```

### 6.2 Standard Built-In Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `kv_put` | `key: String, value: String` | Writes a key-value pair directly into FlashStore's active MemTable and WAL. |
| `kv_get` | `key: String` | Queries a key from FlashStore (checking MemTables, SSTables, and Block Cache). |
| `kv_delete` | `key: String` | Writes a deletion tombstone into FlashStore for the specified key. |
| `kv_scan` | `start?: String, end?: String, limit?: usize` | Performs a range scan across FlashStore SSTables with sequence deduplication. |
| `calculator` | `expression: String` | Evaluates safe arithmetic expressions (`+`, `-`, `*`, `/`, `^`, parentheses). |
| `echo` | `text: String` | Echoes input text, useful for context formatting and inter-agent messages. |

---

## 7. Dual-Tier Memory: Blackboard & Episodic

The memory subsystem (`src/agent/memory.rs`) implements cognitive persistence for swarms:

### 7.1 Working Memory (Shared Blackboard)

- Thread-safe key-value store guarded by `parking_lot::RwLock`.
- Enables agents to share intermediate artifacts (e.g. planner outputs architecture specs; coder reads spec from blackboard).
- Namespace-isolated to avoid cross-workflow collisions.

### 7.2 Episodic Memory & Associative Recall

Recorded task solutions, reflections, and tool outputs are stored as `EpisodicMemoryEntry`:

```rust
pub struct EpisodicMemoryEntry {
    pub id: String,
    pub agent_id: AgentId,
    pub task_id: Option<TaskId>,
    pub content: String,
    pub importance: f32,
    pub timestamp: u64,
    pub tags: Vec<String>,
}
```

Associative recall ranks candidate memories using the **Memory Scoring Formula**:

$$S_{\text{recall}}(E, Q) = w_r \cdot \text{Recency}(E) + w_i \cdot \text{Importance}(E) + w_s \cdot \text{Relevance}(E, Q)$$

Where:
- $\text{Recency}(E) = \exp\left(-\frac{\Delta t}{\tau}\right)$, decay over time $\Delta t$.
- $\text{Importance}(E) \in [0.0, 1.0]$: Assigned significance score.
- $\text{Relevance}(E, Q) \in [0.0, 1.0]$: Lexical token overlap between query $Q$ and episode content $E$.
- Configurable weights: `MemoryScoringWeights { recency: 0.3, importance: 0.3, relevance: 0.4 }`.

---

## 8. FlashStore Storage Persistence & Checkpointing

The storage adapter (`src/agent/store.rs`) maps swarm state into FlashStore's key space:

### 8.1 Key Namespace

| Key Pattern | Description |
| :--- | :--- |
| `agent:def:<agent_id>` | Serialized `AgentDefinition` registration metadata. |
| `agent:task:<task_id>` | Serialized `AgentTask` record, lifecycle status, and outputs. |
| `agent:msg:<workflow_id>:<seq_no>` | Ordered inter-agent messages forming the communication audit log. |
| `agent:wf:<workflow_id>` | Checkpoint snapshot of `WorkflowState` (active tasks, DAG status). |
| `agent:mem:<namespace>:<key>` | Serialized blackboard entries and episodic memory collections. |

### 8.2 Atomic Checkpointing via `WriteBatch`

All tasks, agent states, messages, and memories within an execution checkpoint are bundled into an atomic `WriteBatch`. This guarantees that if a process crashes mid-workflow:
1. No partial or corrupted tasks are written to disk.
2. The entire swarm state can be reconstituted via `AgentPipeline::restore_from_store(store, workflow_id)`.

---

## 9. Event Streaming & Telemetry

The events and metrics subsystem (`src/agent/events.rs`, `src/agent/metrics.rs`) provides observability:

### 9.1 Event Broadcaster

Publish/subscribe event pipeline:
- `AgentEvent::WorkflowStarted` / `WorkflowCompleted`
- `AgentEvent::TaskSubmitted` / `TaskAssigned` / `TaskStarted` / `TaskCompleted` / `TaskFailed`
- `AgentEvent::ToolInvoked` / `ToolCompleted`
- `AgentEvent::ReflectionIteration`
- `AgentEvent::MemoryRecorded`

Handlers:
- `MetricsEventHandler`: Aggregates latency statistics on task and workflow completions.
- `CollectingEventHandler`: In-memory thread-safe event ring buffer for auditing.
- `CallbackEventHandler`: Functional closure hooks for custom logging or integrations.

### 9.2 Latency & Performance Telemetry

- **Percentile Distributions**: Computes $p50$, $p90$, and $p99$ latencies for tasks, tool invocations, and reflection loops.
- **Throughput Metrics**: Total tasks executed, completed vs. failed, average execution time, and tool call frequency.

---

## 10. Rust API Reference

### 10.1 Complete End-to-End Pipeline Example

```rust
use flash_store::prelude::*;
use std::sync::Arc;

fn main() -> Result<()> {
    // 1. Initialize FlashStore
    let dir = tempfile::tempdir().unwrap();
    let options = OptionsBuilder::new().dir(dir.path()).build();
    let store = Arc::new(FlashStore::open(options)?);

    // 2. Create Agent Pipeline
    let mut pipeline = AgentPipeline::new("swarm-demo").with_store(store.clone());

    // 3. Register Agents
    let planner = AgentDefinition::builder("agent-planner", AgentRole::Planner)
        .name("Swarm Planner")
        .capability(AgentCapability::TextGeneration)
        .build();

    let storage_coder = AgentDefinition::builder("agent-storage", AgentRole::Coder)
        .name("Storage Engineer")
        .capability(AgentCapability::StorageOps)
        .capability(AgentCapability::ToolCalling)
        .capability(AgentCapability::Reflection)
        .build();

    pipeline.register_agent(planner)?;
    pipeline.register_agent(storage_coder)?;

    // 4. Submit Tasks with Dependencies (DAG)
    let plan_task = AgentTask::builder("task-1", "Design KV Schema")
        .description("Define key-value schema for high-throughput metrics")
        .priority(TaskPriority::High)
        .require_capability(AgentCapability::TextGeneration)
        .build();

    let write_task = AgentTask::builder("task-2", "Persist Sample Metric")
        .description("Store telemetry metric in FlashStore using kv_put")
        .priority(TaskPriority::Critical)
        .require_capability(AgentCapability::StorageOps)
        .require_capability(AgentCapability::ToolCalling)
        .require_capability(AgentCapability::Reflection)
        .input_context(serde_json::json!({
            "required_tool": "kv_put",
            "tool_args": {
                "key": "metric:cpu:node_1",
                "value": "{\"usage\": 42.5}"
            }
        }))
        .build();

    pipeline.submit_task(plan_task, &[])?;
    pipeline.submit_task(write_task, &["task-1".into()])?;

    // 5. Execute Workflow to Completion
    let summary = pipeline.run()?;
    println!("Workflow completed: {} tasks, {} ms", summary.completed_tasks, summary.total_duration_ms);

    // 6. Inspect Associative Episodic Memory
    let recalled = pipeline.memory().recall("KV Schema", 5)?;
    for memory in recalled {
        println!("Memory hit: {} (score: {:.3})", memory.memory.content, memory.score);
    }

    Ok(())
}
```

---

## 11. Command Line Interface (`flash-cli agent`)

The `flash-cli agent` subcommand suite exposes the multi-agent engine:

### 11.1 Subcommands

| Subcommand | Description | Example |
| :--- | :--- | :--- |
| `run` | Execute an autonomous task | `flash-cli agent run "Benchmark MemTable" --priority high` |
| `workflow` | Run a multi-task JSON workflow file | `flash-cli agent workflow --file workflow.json` |
| `register` | Register an agent in the swarm | `flash-cli agent register --id coder-1 --role coder --capabilities code_execution,tool_calling` |
| `agents` | List active registered swarm agents | `flash-cli agent agents --format text` |
| `tasks` | List workflow tasks and statuses | `flash-cli agent tasks --status completed` |
| `get-task` | Inspect task details and reflections | `flash-cli agent get-task task-101` |
| `tools` | List registered tools and schemas | `flash-cli agent tools` |
| `call` | Invoke a registered tool directly | `flash-cli agent call kv_put '{"key":"k1","value":"v1"}'` |
| `memory` | Search or record episodic memory | `flash-cli agent memory --query "storage"` |
| `stats` | View swarm telemetry and latency | `flash-cli agent stats` |
| `checkpoint`| Save workflow checkpoint to disk | `flash-cli agent checkpoint --workflow default` |

### 11.2 Batch Workflow JSON Format

Workflows with DAG dependencies can be declared declaratively in JSON files:

```json
{
  "workflow_id": "data-pipeline",
  "tasks": [
    {
      "id": "extract",
      "title": "Extract Metrics",
      "description": "Fetch metrics from system log",
      "priority": "high",
      "capabilities": ["tool_calling"],
      "depends_on": []
    },
    {
      "id": "transform",
      "title": "Transform & Validate",
      "description": "Parse metrics into JSON and verify bounds",
      "priority": "medium",
      "capabilities": ["code_execution", "reflection"],
      "depends_on": ["extract"]
    },
    {
      "id": "load",
      "title": "Store in FlashStore",
      "description": "Persist validated records using kv_put",
      "priority": "critical",
      "capabilities": ["storage_ops", "tool_calling"],
      "input": {
        "key": "sensor:readings:001",
        "value": "{\"temp\": 23.4, \"status\": \"ok\"}"
      },
      "depends_on": ["transform"]
    }
  ]
}
```

Run with:
```bash
flash-cli agent workflow --file data_pipeline.json
```

### 11.3 Interactive REPL Commands

The interactive shell (`flash-cli repl`) provides native `AGENT` commands:

```text
flash-store [./my_db]> AGENT AGENTS
Active Swarm Agents:
  - coder-01 [coder] (Capabilities: code_execution, storage_ops, tool_calling)
  - critic-01 [critic] (Capabilities: reflection)

flash-store [./my_db]> AGENT RUN "Calculate fibonacci(10)"
Task completed in 2.3ms. Result: 55

flash-store [./my_db]> AGENT CALL kv_put {"key":"sample","value":"hello"}
Tool 'kv_put' executed successfully: {"key":"sample","status":"written"}

flash-store [./my_db]> AGENT MEMORY RECORD "Fibonacci sequence verified"
Episodic memory recorded successfully.

flash-store [./my_db]> AGENT STATS
Swarm Telemetry:
  Completed Tasks: 3
  Failed Tasks: 0
  Avg Latency: 1.8ms
  P95 Latency: 2.3ms
```
