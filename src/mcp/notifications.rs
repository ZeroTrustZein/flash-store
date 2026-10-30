//! # MCP Notifications and Logging Subsystem
//!
//! Provides asynchronous event publishing, structured logging notifications,
//! and resource invalidation broadcasts adhering to the Model Context Protocol.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::mcp::protocol::JsonRpcNotification;
use crate::mcp::types::McpLogLevel;

/// Notification parameters for `notifications/message` (MCP Logging).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingMessageParams {
    /// Severity level of the log entry.
    pub level: McpLogLevel,
    /// Logger category or component name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logger: Option<String>,
    /// Log payload data (string, JSON object, etc.).
    pub data: Value,
}

impl LoggingMessageParams {
    /// Creates a new logging message notification payload.
    pub fn new(level: McpLogLevel, logger: Option<String>, data: Value) -> Self {
        Self {
            level,
            logger,
            data,
        }
    }
}

/// Notification parameters for `notifications/resources/updated`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUpdatedParams {
    /// URI of the resource that was modified.
    pub uri: String,
}

/// Observer trait for receiving MCP notifications dispatched by the server.
pub trait McpNotificationObserver: Send + Sync {
    /// Called when an MCP notification is emitted.
    fn on_notification(&self, notification: &JsonRpcNotification);
}

/// Broadcaster for fan-out publishing of MCP JSON-RPC notifications.
#[derive(Default)]
pub struct McpNotificationBroadcaster {
    observers: RwLock<Vec<Arc<dyn McpNotificationObserver>>>,
    min_log_level: RwLock<McpLogLevel>,
    notification_count: AtomicUsize,
}

impl McpNotificationBroadcaster {
    /// Creates a new notification broadcaster with default minimum log level of `Info`.
    pub fn new() -> Self {
        Self {
            observers: RwLock::new(Vec::new()),
            min_log_level: RwLock::new(McpLogLevel::Info),
            notification_count: AtomicUsize::new(0),
        }
    }

    /// Sets the minimum logging level threshold for log notification emission.
    pub fn set_min_log_level(&self, level: McpLogLevel) {
        *self.min_log_level.write() = level;
    }

    /// Gets the current minimum logging level threshold.
    pub fn min_log_level(&self) -> McpLogLevel {
        *self.min_log_level.read()
    }

    /// Adds an observer to receive notifications.
    pub fn add_observer(&self, observer: Arc<dyn McpNotificationObserver>) {
        self.observers.write().push(observer);
    }

    /// Returns the number of registered observers.
    pub fn observer_count(&self) -> usize {
        self.observers.read().len()
    }

    /// Returns total notifications broadcast so far.
    pub fn total_broadcasts(&self) -> usize {
        self.notification_count.load(Ordering::Relaxed)
    }

    /// Broadcasts an arbitrary JSON-RPC notification to all subscribed observers.
    pub fn broadcast(&self, notification: &JsonRpcNotification) {
        self.notification_count.fetch_add(1, Ordering::Relaxed);
        let guard = self.observers.read();
        for obs in guard.iter() {
            obs.on_notification(notification);
        }
    }

    /// Broadcasts a log message if severity meets the configured minimum log level.
    pub fn log(&self, level: McpLogLevel, logger: Option<&str>, message: impl Into<String>) {
        let min_lvl = self.min_log_level();
        if Self::level_to_rank(level) >= Self::level_to_rank(min_lvl) {
            let params = json!({
                "level": level,
                "logger": logger,
                "data": message.into(),
            });
            let notif = JsonRpcNotification::new("notifications/message", Some(params));
            self.broadcast(&notif);
        }
    }

    /// Broadcasts a resource update notification.
    pub fn notify_resource_updated(&self, uri: &str) {
        let params = json!({ "uri": uri });
        let notif = JsonRpcNotification::new("notifications/resources/updated", Some(params));
        self.broadcast(&notif);
    }

    /// Broadcasts a tool list changed notification.
    pub fn notify_tools_changed(&self) {
        let notif = JsonRpcNotification::new("notifications/tools/list_changed", None);
        self.broadcast(&notif);
    }

    /// Broadcasts a prompt list changed notification.
    pub fn notify_prompts_changed(&self) {
        let notif = JsonRpcNotification::new("notifications/prompts/list_changed", None);
        self.broadcast(&notif);
    }

    fn level_to_rank(level: McpLogLevel) -> u8 {
        match level {
            McpLogLevel::Debug => 1,
            McpLogLevel::Info => 2,
            McpLogLevel::Notice => 3,
            McpLogLevel::Warning => 4,
            McpLogLevel::Error => 5,
            McpLogLevel::Critical => 6,
            McpLogLevel::Alert => 7,
            McpLogLevel::Emergency => 8,
        }
    }
}

/// In-memory notification collector for testing and validation.
#[derive(Default)]
pub struct CollectingNotificationObserver {
    collected: RwLock<Vec<JsonRpcNotification>>,
}

impl CollectingNotificationObserver {
    /// Creates a new collector.
    pub fn new() -> Self {
        Self {
            collected: RwLock::new(Vec::new()),
        }
    }

    /// Returns a clone of all collected notifications.
    pub fn notifications(&self) -> Vec<JsonRpcNotification> {
        self.collected.read().clone()
    }

    /// Returns count of collected notifications.
    pub fn len(&self) -> usize {
        self.collected.read().len()
    }

    /// Checks if collector is empty.
    pub fn is_empty(&self) -> bool {
        self.collected.read().is_empty()
    }
}

impl McpNotificationObserver for CollectingNotificationObserver {
    fn on_notification(&self, notification: &JsonRpcNotification) {
        self.collected.write().push(notification.clone());
    }
}
