# FlashStore SSTable Binary Format Specification

## 1. Overview

Each SSTable (`.sst` file) in FlashStore is an immutable on-disk file containing sorted, indexed key-value entries. The file layout is structured into contiguous Data Blocks, an optional Filter Block, Meta Index Block, Index Block, and a fixed 48-byte Footer.

---

## 2. File Layout Diagram

```
+-------------------------------------------------------------+
| Data Block 0                                                |
|   - Entry 0: [key_len][key][val_len][val][seq_no][type]     |
|   - Entry 1: ...                                            |
|   - Restart Array: [offset_0][offset_1]...[num_restarts]    |
|   - Trailer: [CompressionType (1B)] [CRC32 Checksum (4B)]   |
+-------------------------------------------------------------+
| Data Block 1 ...                                            |
+-------------------------------------------------------------+
| Filter Block (Bloom Filter bit array + hash count k)        |
+-------------------------------------------------------------+
| Meta Index Block                                            |
|   - "filter.flashstore.bloom" -> BlockHandle                |
+-------------------------------------------------------------+
| Index Block                                                 |
|   - Last Key Block 0 -> BlockHandle { offset: 0, size: S0 } |
|   - Last Key Block 1 -> BlockHandle { offset: S0, size: S1 }|
+-------------------------------------------------------------+
| Fixed Footer (48 Bytes)                                     |
|   - meta_index_offset (8B LE)                               |
|   - meta_index_size   (8B LE)                               |
|   - index_offset      (8B LE)                               |
|   - index_size        (8B LE)                               |
|   - reserved          (8B LE)                               |
|   - magic number      (8B LE) = 0xF1A5457072653031          |
+-------------------------------------------------------------+
```

---

## 3. Section Specifications

### 3.1 Data Block Format

Each Data Block is target-sized by `Options::block_size` (default 4KB).

#### Entry Encoding
Each key-value record inside a block is serialized as:
- `key_length`: 4 bytes Little-Endian unsigned int (`u32`)
- `key_bytes`: Raw key payload ($N$ bytes)
- `value_length`: 4 bytes Little-Endian unsigned int (`u32`)
- `value_bytes`: Raw value payload ($M$ bytes)
- `seq_no`: 8 bytes Little-Endian unsigned int (`u64`)
- `value_type`: 1 byte (`0x00` = Value, `0x01` = Tombstone)

#### Restart Array & Trailer
At the end of each uncompressed data block:
- `restart_offsets`: `[u32; num_restarts]` indexing entry offsets within the block.
- `num_restarts`: 4 bytes `u32` storing the number of restart offsets.
- **Block Trailer**:
  - `compression_type`: 1 byte (`0x00` = None, `0x01` = Snappy, `0x02` = Zstd, `0x03` = Lz4).
  - `checksum`: 4 bytes Little-Endian `u32` calculating CRC32-IEEE over block data and compression byte.

---

### 3.2 Bloom Filter Block

- **Algorithm**: Standard double-hashing Bloom Filter parameterized by `Options::bloom_bits_per_key` (default 10 bits/key, giving $\approx 1\%$ false positive probability).
- **Format**:
  - Raw bit array ($M$ bytes).
  - Hash function count $k$ (1 byte, typically $k = \lfloor 10 \times \ln 2 \rfloor = 7$).

---

### 3.3 Meta Index Block & Index Block

- **BlockHandle**: A reference to an on-disk slice:
  - `offset`: 8 bytes `u64`
  - `size`: 8 bytes `u64`
- **Meta Index Block**: Maps metadata block names (such as `"filter.flashstore.bloom"`) to their corresponding `BlockHandle`.
- **Index Block**: Maps the largest user key present in each Data Block to its corresponding `BlockHandle`.

---

### 3.4 Footer Specification

The footer is strictly **48 bytes** located at the end of the SSTable:

| Field Name | Type | Offset in Footer | Description |
|---|---|---|---|
| `meta_index_offset` | `u64` LE | 0..8 | Byte offset of Meta Index Block |
| `meta_index_size` | `u64` LE | 8..16 | Byte length of Meta Index Block |
| `index_offset` | `u64` LE | 16..24 | Byte offset of Index Block |
| `index_size` | `u64` LE | 24..32 | Byte length of Index Block |
| `reserved` | `u64` LE | 32..40 | Reserved zero padding |
| `magic_number` | `u64` LE | 40..48 | `0xF1A5457072653031` ("FLASHpre01") |

---

## 4. Reading an SSTable

1. Open file and seek to `file_length - 48`.
2. Read 48 bytes and decode `Footer`. Validate `magic_number == 0xF1A5457072653031`.
3. Seek to `meta_index_offset` and read Meta Index Block. Parse Bloom filter `BlockHandle`.
4. Load Bloom Filter Block. Check `filter.may_contain(target_key)`. If false, skip SSTable entirely.
5. Seek to `index_offset` and read Index Block.
6. Binary search Index Block for the first data block whose index key $\ge target\_key$.
7. Query `LruBlockCache` with key `(file_number, data_block_offset)`.
   - **Cache Hit**: Decode block in memory.
   - **Cache Miss**: Read from disk, verify CRC32 trailer, insert into cache, decode block.
8. Perform binary search across restart offsets within the Data Block, followed by linear scan to locate target key.
