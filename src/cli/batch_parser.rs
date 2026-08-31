use crate::batch::WriteBatch;
use crate::error::{FlashStoreError, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum RawBatchOp {
    Flat {
        #[serde(alias = "type", alias = "action")]
        op: String,
        key: String,
        #[serde(default)]
        value: Option<String>,
    },
    Nested {
        put: Option<KvPair>,
        delete: Option<KeyOnly>,
        del: Option<KeyOnly>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvPair {
    key: String,
    value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeyOnly {
    key: String,
}

/// Parse batch operations from a JSON string or plain text representation into a `WriteBatch`.
pub fn parse_json_ops(content: &str) -> Result<WriteBatch> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(WriteBatch::new());
    }

    // Try parsing as JSON first
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        if let Ok(raw_ops) = serde_json::from_str::<Vec<RawBatchOp>>(trimmed) {
            let mut batch = WriteBatch::new();
            for raw_op in raw_ops {
                match raw_op {
                    RawBatchOp::Flat { op, key, value } => {
                        let op_lower = op.to_lowercase();
                        match op_lower.as_str() {
                            "put" | "set" | "insert" => {
                                let val_str = value.unwrap_or_default();
                                batch.put(Bytes::from(key), Bytes::from(val_str));
                            }
                            "delete" | "del" | "remove" => {
                                batch.delete(Bytes::from(key));
                            }
                            other => {
                                return Err(FlashStoreError::InvalidArgument(format!(
                                    "Unknown batch operation: '{}'",
                                    other
                                )));
                            }
                        }
                    }
                    RawBatchOp::Nested { put, delete, del } => {
                        if let Some(p) = put {
                            batch.put(Bytes::from(p.key), Bytes::from(p.value));
                        } else if let Some(d) = delete.or(del) {
                            batch.delete(Bytes::from(d.key));
                        }
                    }
                }
            }
            return Ok(batch);
        }
    }

    // Fallback: parse as line-separated commands (e.g., PUT k v \n DEL k)
    parse_plain_lines(trimmed)
}

/// Parse line-delimited commands into a `WriteBatch`.
pub fn parse_plain_lines(input: &str) -> Result<WriteBatch> {
    let mut batch = WriteBatch::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let command = parts[0].to_uppercase();
        match command.as_str() {
            "PUT" | "SET" => {
                if parts.len() < 3 {
                    return Err(FlashStoreError::InvalidArgument(format!(
                        "Invalid PUT line, expected 'PUT <key> <value>', got: '{}'",
                        line
                    )));
                }
                let key = parts[1];
                let val = parts[2..].join(" ");
                batch.put(Bytes::from(key.to_string()), Bytes::from(val));
            }
            "DEL" | "DELETE" => {
                if parts.len() < 2 {
                    return Err(FlashStoreError::InvalidArgument(format!(
                        "Invalid DEL line, expected 'DEL <key>', got: '{}'",
                        line
                    )));
                }
                let key = parts[1];
                batch.delete(Bytes::from(key.to_string()));
            }
            other => {
                return Err(FlashStoreError::InvalidArgument(format!(
                    "Unrecognized batch command: '{}'",
                    other
                )));
            }
        }
    }

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_flat_batch() -> Result<()> {
        let json = r#"[
            {"op": "put", "key": "k1", "value": "v1"},
            {"op": "delete", "key": "k2"},
            {"type": "set", "key": "k3", "value": "v3"}
        ]"#;
        let batch = parse_json_ops(json)?;
        assert_eq!(batch.len(), 3);
        Ok(())
    }

    #[test]
    fn test_parse_plain_lines_batch() -> Result<()> {
        let text = "PUT user1 Alice\n# comment\nPUT user2 Bob Jones\nDEL user1\n";
        let batch = parse_plain_lines(text)?;
        assert_eq!(batch.len(), 3);
        Ok(())
    }

    #[test]
    fn test_parse_invalid_batch() {
        let bad_json = r#"[{"op": "invalid_op", "key": "k"}]"#;
        assert!(parse_json_ops(bad_json).is_err());

        let bad_line = "UNKNOWN_CMD foo bar";
        assert!(parse_plain_lines(bad_line).is_err());
    }
}
