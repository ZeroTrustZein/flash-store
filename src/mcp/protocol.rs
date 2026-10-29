//! # JSON-RPC 2.0 Protocol Envelope and Wire Types
//!
//! Implements JSON-RPC 2.0 framing and error models required by Model Context Protocol (MCP).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// Standard JSON-RPC 2.0 Error Codes.
pub mod error_codes {
    /// Invalid JSON was received by the server. An error occurred on the server while parsing the JSON text.
    pub const PARSE_ERROR: i32 = -32700;
    /// The JSON sent is not a valid Request object.
    pub const INVALID_REQUEST: i32 = -32600;
    /// The method does not exist / is not available.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid method parameter(s).
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal JSON-RPC error.
    pub const INTERNAL_ERROR: i32 = -32603;
    /// Server-specific error range lower bound.
    pub const SERVER_ERROR_START: i32 = -32000;
    /// Server-specific error range upper bound.
    pub const SERVER_ERROR_END: i32 = -32099;
}

/// JSON-RPC 2.0 Request ID (can be string, integer, or null for notifications).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String identifier.
    String(String),
    /// Integer numeric identifier.
    Number(i64),
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestId::String(s) => write!(f, "{}", s),
            RequestId::Number(n) => write!(f, "{}", n),
        }
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

/// JSON-RPC 2.0 Request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version string, must be exactly "2.0".
    pub jsonrpc: String,
    /// Request identifier assigned by the client.
    pub id: RequestId,
    /// The name of the method to be invoked.
    pub method: String,
    /// Structured parameters passed to the method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Creates a new JSON-RPC 2.0 request.
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 Notification envelope (a request without an `id`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// JSON-RPC version string, must be exactly "2.0".
    pub jsonrpc: String,
    /// The name of the method/event to be notified.
    pub method: String,
    /// Structured parameters passed with the notification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    /// Creates a new JSON-RPC 2.0 notification.
    pub fn new(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 Error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Numeric error code.
    pub code: i32,
    /// Short human-readable summary of the error.
    pub message: String,
    /// Optional additional structured details or data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    /// Creates a new error object.
    pub fn new(code: i32, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            code,
            message: message.into(),
            data,
        }
    }

    /// Predefined parse error (-32700).
    pub fn parse_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::PARSE_ERROR, "Parse error", Some(Value::String(details.into())))
    }

    /// Predefined invalid request error (-32600).
    pub fn invalid_request(details: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_REQUEST, "Invalid Request", Some(Value::String(details.into())))
    }

    /// Predefined method not found error (-32601).
    pub fn method_not_found(method: impl Into<String>) -> Self {
        Self::new(
            error_codes::METHOD_NOT_FOUND,
            format!("Method not found: {}", method.into()),
            None,
        )
    }

    /// Predefined invalid params error (-32602).
    pub fn invalid_params(details: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_PARAMS, "Invalid params", Some(Value::String(details.into())))
    }

    /// Predefined internal error (-32603).
    pub fn internal_error(details: impl Into<String>) -> Self {
        Self::new(error_codes::INTERNAL_ERROR, "Internal error", Some(Value::String(details.into())))
    }
}

/// JSON-RPC 2.0 Response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version string, must be exactly "2.0".
    pub jsonrpc: String,
    /// The matching request ID (or null in case of parse errors).
    pub id: Option<RequestId>,
    /// Result payload if invocation succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error object if invocation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Constructs a successful JSON-RPC response.
    pub fn success(id: impl Into<RequestId>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id.into()),
            result: Some(result),
            error: None,
        }
    }

    /// Constructs an error JSON-RPC response.
    pub fn error(id: Option<RequestId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}
