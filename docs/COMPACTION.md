# FlashStore Leveled Compaction Architecture

## 1. Overview

As writes accumulate in an LSM-tree, new SSTables are continuously flushed to **Level 0 (L0)**. Over time, multiple SSTables accumulate overlapping key ranges, increasing read amplification and wasting disk space on stale versions or tombstones.

FlashStore implements a **Leveled Compaction Engine** based on the Classic Leveled LSM architecture. Compaction continuously merges sorted runs, bounds the number of files per level, purges dead records, and enforces strict non-overlapping key partitions in Levels $1 \dots N$.

---

## 2. Leveled Hierarchy & Compaction Triggers

```
+-------------------------------------------------------------+
| Level 0: [SST 1] [SST 2] [SST 3] [SST 4]                    |
|   - Overlapping key ranges                                  |
|   - Trigger: file count >= 4 (l0_compaction_trigger)        |
+-------------------------------------------------------------+
                            |
                     Merge  v
+-------------------------------------------------------------+
| Level 1: [SST 10: a-d] [SST 11: e-k] [SST 12: l-z]          |
|   - Non-overlapping sorted runs                             |
|   - Size Target: 10 MB (base_level_size_bytes)              |
+-------------------------------------------------------------+
                            |
                     Merge  v
+-------------------------------------------------------------+
| Level 2: [SST 20] [SST 21] [SST 22] [SST 23] [SST 24] ...   |
|   - Size Target: 100 MB (10 MB * 10^1)                      |
+-------------------------------------------------------------+
                            |
                     Merge  v
+-------------------------------------------------------------+
| Level N: [SST N0] [SST N1] ...                              |
|   - Size Target: 10 MB * 10^(N-1)                           |
+-------------------------------------------------------------+
```

### 2.1 Compaction Triggers
1. **Level 0 Trigger**:
   - Condition: `level_0_files.len() >= compactor.l0_compaction_trigger` (default: 4).
   - Target: Flushes all Level 0 files into overlapping Level 1 files.
2. **Level $L \ge 1$ Trigger**:
   - Score: $\text{Score}_L = \frac{\text{Total Bytes in Level } L}{\text{Max Bytes Target for Level } L}$.
   - Condition: $\text{Score}_L \ge 1.0$.
   - Target: Moves highest score level files down to level $L+1$.

---

## 3. Compaction Workflow

```
1. pick_compaction(levels)
       |
       v
2. Find Overlapping Files in target level L+1:
   smallest_key = min(input_files.smallest_key)
   largest_key  = max(input_files.largest_key)
   target_input_files = find_overlapping_files(level[L+1], smallest_key, largest_key)
       |
       v
3. Multi-Way Merge Sort:
   - Stream entries from all inputs in sorted user_key order.
   - For duplicate user_keys: retain only the latest sequence number.
       |
       v
4. Tombstone Handling:
   - If value_type == Tombstone AND target_level is max_level (or bottom-most level):
         Drop tombstone permanently.
   - Otherwise: write tombstone to new SSTable to shadow older keys in deeper levels.
       |
       v
5. Build New Target Level SSTable(s):
   - TableBuilder emits new SSTables (<file_num>.sst) constrained by target block size.
       |
       v
6. Commit VersionEdit to MANIFEST:
   - Record added files (new SSTables).
   - Record deleted files (old input files).
   - Atomically update VersionSet.
       |
       v
7. Garbage Collect:
   - Safely remove obsolete SSTable files from disk.
```

---

## 4. Range Overlap Algorithm

Two key ranges $[A_{min}, A_{max}]$ and $[B_{min}, B_{max}]$ overlap if and only if:
$$\neg (A_{max} < B_{min} \lor B_{max} < A_{min})$$

In Rust:
```rust
pub fn ranges_overlap(min1: &Bytes, max1: &Bytes, min2: &Bytes, max2: &Bytes) -> bool {
    !(max1 < min2 || max2 < min1)
}
```

---

## 5. Performance and Amplification Characteristics

- **Write Amplification (WA)**: Leveled compaction has higher write amplification ($\approx 10 \times \text{levels}$) compared to size-tiered compaction, but delivers significantly superior read performance and lower space overhead.
- **Read Amplification (RA)**: Minimized to $O(\text{max\_levels})$. At most 1 file is read per level $L \ge 1$.
- **Space Amplification (SA)**: Bounded to approximately $\approx 1.11 \times$ database size ($10\%$ overhead in L1..LN-1).
