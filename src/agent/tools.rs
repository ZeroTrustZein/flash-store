//! # Automated Tool Calling Suite and Dispatching Engine
//!
//! Provides deterministic tool implementations and execution dispatching for:
//! - Direct key-value storage operations against FlashStore (`KvGet`, `KvPut`, `KvDelete`, `KvScan`).
//! - Arithmetic computation and expression evaluation (`CalculatorTool`).
//! - Inter-agent handshake and debugging utilities (`EchoTool`).
//! - Parallel and sequential tool call dispatching with schema validation and latency tracking.

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use serde_json::json;

use crate::agent::types::{
    Tool, ToolCall, ToolDefinition, ToolOutput, ToolParamType, ToolParameter, ToolRegistry,
};
use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};

// ============================================================================
// Tool Dispatcher
// ============================================================================

/// High-level executor and dispatcher for tool invocations.
pub struct ToolDispatcher {
    registry: Arc<ToolRegistry>,
}

impl ToolDispatcher {
    /// Creates a new `ToolDispatcher` wrapping the given tool registry.
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }

    /// Dispatches a single tool call through the registry with validation and timing.
    pub fn dispatch(&self, call: &ToolCall) -> ToolOutput {
        self.registry.execute(call).unwrap_or_else(|e| {
            ToolOutput::error(
                call.call_id.clone(),
                call.tool_name.clone(),
                format!("Dispatcher execution failure: {}", e),
                0,
            )
        })
    }

    /// Dispatches a sequence of tool calls sequentially, collecting their outputs.
    pub fn dispatch_batch(&self, calls: &[ToolCall]) -> Vec<ToolOutput> {
        calls.iter().map(|call| self.dispatch(call)).collect()
    }
}

// ============================================================================
// FlashStore Storage Tools
// ============================================================================

/// Tool for reading values from FlashStore KV storage.
pub struct KvGetTool {
    store: Arc<FlashStore>,
}

impl KvGetTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl Tool for KvGetTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("kv_get", "Retrieves a stored value by key from FlashStore")
            .with_parameter(ToolParameter::new(
                "key",
                ToolParamType::String,
                "The unique lookup key",
                true,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let key = arguments
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing required 'key' argument".into()))?;

        match self.store.get(key)? {
            Some(bytes) => {
                let val_str = String::from_utf8(bytes.to_vec())
                    .unwrap_or_else(|_| format!("<binary {} bytes>", bytes.len()));
                Ok(json!({
                    "found": true,
                    "key": key,
                    "value": val_str,
                }))
            }
            None => Ok(json!({
                "found": false,
                "key": key,
                "value": serde_json::Value::Null,
            })),
        }
    }
}

/// Tool for writing key-value pairs into FlashStore.
pub struct KvPutTool {
    store: Arc<FlashStore>,
}

impl KvPutTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl Tool for KvPutTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("kv_put", "Inserts or updates a key-value pair in FlashStore")
            .with_parameter(ToolParameter::new(
                "key",
                ToolParamType::String,
                "The storage key to write",
                true,
            ))
            .with_parameter(ToolParameter::new(
                "value",
                ToolParamType::String,
                "The string value to associate with key",
                true,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let key = arguments
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing 'key' argument".into()))?;

        let value = arguments
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing 'value' argument".into()))?;

        self.store.put(key, value)?;

        Ok(json!({
            "success": true,
            "key": key,
            "bytes_written": value.len(),
        }))
    }
}

/// Tool for deleting keys from FlashStore.
pub struct KvDeleteTool {
    store: Arc<FlashStore>,
}

impl KvDeleteTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl Tool for KvDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("kv_delete", "Deletes a key from FlashStore by writing a tombstone")
            .with_parameter(ToolParameter::new(
                "key",
                ToolParamType::String,
                "The key to delete",
                true,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let key = arguments
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing 'key' argument".into()))?;

        self.store.delete(key)?;

        Ok(json!({
            "success": true,
            "key": key,
        }))
    }
}

/// Tool for scanning range of keys in FlashStore.
pub struct KvScanTool {
    store: Arc<FlashStore>,
}

impl KvScanTool {
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl Tool for KvScanTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("kv_scan", "Scans a range of keys starting from start_key")
            .with_parameter(ToolParameter::new(
                "start_key",
                ToolParamType::String,
                "The lower bound key to start scan from",
                true,
            ))
            .with_parameter(ToolParameter::new(
                "limit",
                ToolParamType::Integer,
                "Maximum number of entries to return (default 20)",
                false,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let start_key = arguments
            .get("start_key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing 'start_key' argument".into()))?;

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize;

        let start_bytes = Some(Bytes::copy_from_slice(start_key.as_bytes()));
        let results = self.store.scan(start_bytes, None)?;

        let mut entries = Vec::new();
        for (k, v) in results.into_iter().take(limit) {
            let key = String::from_utf8_lossy(&k).to_string();
            let val = String::from_utf8_lossy(&v).to_string();
            entries.push(json!({
                "key": key,
                "value": val,
            }));
        }

        Ok(json!({
            "count": entries.len(),
            "entries": entries,
        }))
    }
}

// ============================================================================
// Standard Utility Tools
// ============================================================================

/// Tool for performing deterministic arithmetic calculations.
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("calculator", "Performs mathematical arithmetic calculations")
            .with_parameter(ToolParameter::new(
                "operation",
                ToolParamType::String,
                "One of: add, sub, mul, div, pow, mod",
                true,
            ))
            .with_parameter(ToolParameter::new(
                "a",
                ToolParamType::Float,
                "First operand",
                true,
            ))
            .with_parameter(ToolParameter::new(
                "b",
                ToolParamType::Float,
                "Second operand",
                true,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let op = arguments
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing 'operation' argument".into()))?;

        let a = arguments
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing numeric operand 'a'".into()))?;

        let b = arguments
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| FlashStoreError::InvalidArgument("Missing numeric operand 'b'".into()))?;

        let result = match op.to_lowercase().as_str() {
            "add" | "+" => a + b,
            "sub" | "-" => a - b,
            "mul" | "*" => a * b,
            "div" | "/" => {
                if b.abs() < f64::EPSILON {
                    return Err(FlashStoreError::InvalidArgument("Division by zero".into()));
                }
                a / b
            }
            "pow" | "^" => a.powf(b),
            "mod" | "%" => a % b,
            unknown => {
                return Err(FlashStoreError::InvalidArgument(format!(
                    "Unknown operation '{}'; expected add, sub, mul, div, pow, or mod",
                    unknown
                )))
            }
        };

        Ok(json!({
            "operation": op,
            "a": a,
            "b": b,
            "result": result,
        }))
    }
}

/// Tool for echoing messages, useful for inter-agent pinging and diagnostics.
pub struct EchoTool;

impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("echo", "Echoes input back; used for diagnostics and testing")
            .with_parameter(ToolParameter::new(
                "message",
                ToolParamType::String,
                "Text message to echo back",
                true,
            ))
    }

    fn execute(&self, arguments: &serde_json::Value) -> Result<serde_json::Value> {
        let msg = arguments
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        Ok(json!({
            "echoed": msg,
            "length": msg.len(),
            "timestamp": Instant::now().elapsed().as_micros(),
        }))
    }
}

// ============================================================================
// Factory Helpers
// ============================================================================

/// Constructs a pre-configured `ToolRegistry` with standard system tools and optional FlashStore tools.
pub fn create_standard_tool_registry(store: Option<Arc<FlashStore>>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    registry.register(Arc::new(CalculatorTool));
    registry.register(Arc::new(EchoTool));

    if let Some(s) = store {
        registry.register(Arc::new(KvGetTool::new(s.clone())));
        registry.register(Arc::new(KvPutTool::new(s.clone())));
        registry.register(Arc::new(KvDeleteTool::new(s.clone())));
        registry.register(Arc::new(KvScanTool::new(s)));
    }

    registry
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptionsBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_calculator_tool_operations() {
        let calc = CalculatorTool;
        let add_res = calc
            .execute(&json!({ "operation": "add", "a": 15.0, "b": 27.0 }))
            .unwrap();
        assert_eq!(add_res["result"], 42.0);

        let div_zero = calc.execute(&json!({ "operation": "div", "a": 10.0, "b": 0.0 }));
        assert!(div_zero.is_err());
    }

    #[test]
    fn test_echo_tool() {
        let echo = EchoTool;
        let res = echo.execute(&json!({ "message": "hello agents" })).unwrap();
        assert_eq!(res["echoed"], "hello agents");
        assert_eq!(res["length"], 12);
    }

    #[test]
    fn test_kv_tools_with_flashstore() {
        let dir = tempdir().unwrap();
        let options = OptionsBuilder::new().dir(dir.path()).build();
        let store = Arc::new(FlashStore::open(options).unwrap());

        let put_tool = KvPutTool::new(store.clone());
        let get_tool = KvGetTool::new(store.clone());
        let del_tool = KvDeleteTool::new(store.clone());
        let scan_tool = KvScanTool::new(store.clone());

        // 1. Put
        let put_res = put_tool
            .execute(&json!({ "key": "agent:config:max_threads", "value": "16" }))
            .unwrap();
        assert_eq!(put_res["success"], true);

        // 2. Get
        let get_res = get_tool
            .execute(&json!({ "key": "agent:config:max_threads" }))
            .unwrap();
        assert_eq!(get_res["found"], true);
        assert_eq!(get_res["value"], "16");

        // 3. Scan
        put_tool
            .execute(&json!({ "key": "agent:config:timeout", "value": "60" }))
            .unwrap();
        let scan_res = scan_tool
            .execute(&json!({ "start_key": "agent:config:", "limit": 10 }))
            .unwrap();
        assert_eq!(scan_res["count"], 2);

        // 4. Delete
        let del_res = del_tool
            .execute(&json!({ "key": "agent:config:max_threads" }))
            .unwrap();
        assert_eq!(del_res["success"], true);

        let get_after_del = get_tool
            .execute(&json!({ "key": "agent:config:max_threads" }))
            .unwrap();
        assert_eq!(get_after_del["found"], false);
    }

    #[test]
    fn test_tool_dispatcher_batch() {
        let registry = Arc::new(create_standard_tool_registry(None));
        let dispatcher = ToolDispatcher::new(registry);

        let calls = vec![
            ToolCall::new("call-1", "calculator", json!({ "operation": "mul", "a": 6.0, "b": 7.0 })),
            ToolCall::new("call-2", "echo", json!({ "message": "ping" })),
            ToolCall::new("call-3", "unknown_tool", json!({})),
        ];

        let outputs = dispatcher.dispatch_batch(&calls);
        assert_eq!(outputs.len(), 3);
        assert!(outputs[0].is_success());
        assert_eq!(outputs[0].result["result"], 42.0);
        assert!(outputs[1].is_success());
        assert!(!outputs[2].is_success()); // unknown tool returns error
    }
}
