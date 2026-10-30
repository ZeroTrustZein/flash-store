//! # Model Context Protocol (MCP) Resources and Endpoints
//!
//! Exposes structured system telemetry, database status, SSTable manifests, and memory diagnostics
//! as inspectable URI resources adhering to the MCP resource specification.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::engine::FlashStore;
use crate::error::{FlashStoreError, Result};

/// Metadata descriptor for an MCP resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceDefinition {
    /// Uniform Resource Identifier (e.g. `flash://stats`, `flash://manifest`).
    pub uri: String,
    /// Human-readable title of the resource.
    pub name: String,
    /// Detailed description of the contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type of the payload (e.g. `application/json`, `text/plain`).
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl McpResourceDefinition {
    /// Creates a new resource descriptor.
    pub fn new(
        uri: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            description: Some(description.into()),
            mime_type: Some(mime_type.into()),
        }
    }
}

/// Payload content returned when reading an MCP resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceContent {
    /// URI of the resolved resource.
    pub uri: String,
    /// MIME type of the content.
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text-based representation of the resource.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64 binary blob representation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl McpResourceContent {
    /// Constructs a text-based resource content object.
    pub fn text(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some(mime_type.into()),
            text: Some(text.into()),
            blob: None,
        }
    }

    /// Constructs a binary blob resource content object.
    pub fn blob(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        blob: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some(mime_type.into()),
            text: None,
            blob: Some(blob.into()),
        }
    }
}

/// Result returned by `resources/list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpListResourcesResult {
    /// List of available resources.
    pub resources: Vec<McpResourceDefinition>,
    /// Optional cursor for pagination.
    #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Result returned by `resources/read`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpReadResourceResult {
    /// Array of content representations for the requested URI.
    pub contents: Vec<McpResourceContent>,
}

/// Provider trait for serving dynamic MCP resources.
pub trait McpResourceProvider: Send + Sync {
    /// Lists all resource definitions served by this provider.
    fn list_resources(&self) -> Vec<McpResourceDefinition>;

    /// Reads content for the given URI.
    fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>>;
}

/// Standard MCP resource provider exposing live FlashStore telemetry and manifests.
pub struct FlashStorageResourceProvider {
    store: Arc<FlashStore>,
}

impl FlashStorageResourceProvider {
    /// Creates a new `FlashStorageResourceProvider` wrapping a FlashStore instance.
    pub fn new(store: Arc<FlashStore>) -> Self {
        Self { store }
    }
}

impl McpResourceProvider for FlashStorageResourceProvider {
    fn list_resources(&self) -> Vec<McpResourceDefinition> {
        vec![
            McpResourceDefinition::new(
                "flash://stats",
                "FlashStore Live Statistics",
                "Real-time LSM-tree engine statistics including memtable byte sizes and SSTable file counts",
                "application/json",
            ),
            McpResourceDefinition::new(
                "flash://manifest",
                "FlashStore Manifest & Level Hierarchy",
                "Overview of SSTable file counts across tiered LSM levels",
                "application/json",
            ),
        ]
    }

    fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        match uri {
            "flash://stats" => {
                let stats = self.store.stats();
                let payload = serde_json::json!({
                    "active_memtable_size_bytes": stats.active_memtable_size,
                    "immutable_memtables_count": stats.immutable_memtables_count,
                    "levels_file_count": stats.levels_file_count,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://stats",
                    "application/json",
                    text,
                )])
            }
            "flash://manifest" => {
                let stats = self.store.stats();
                let payload = serde_json::json!({
                    "manifest_status": "active",
                    "total_levels": stats.levels_file_count.len(),
                    "levels_distribution": stats.levels_file_count,
                });
                let text =
                    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string());
                Ok(vec![McpResourceContent::text(
                    "flash://manifest",
                    "application/json",
                    text,
                )])
            }
            _ => Err(FlashStoreError::InvalidArgument(format!(
                "Resource not found: {}",
                uri
            ))),
        }
    }
}

/// Registry managing and routing MCP resource lookups.
#[derive(Default)]
pub struct McpResourceRegistry {
    providers: RwLock<HashMap<String, Vec<Arc<dyn McpResourceProvider>>>>,
}

impl McpResourceRegistry {
    /// Creates a new empty resource registry.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Constructs a resource registry pre-populated with standard FlashStore resource providers.
    pub fn with_flash_store(store: Arc<FlashStore>) -> Self {
        let registry = Self::new();
        registry.register_storage_resources(store);
        registry
    }

    /// Registers the standard FlashStore resource provider under the `flash` scheme.
    pub fn register_storage_resources(&self, store: Arc<FlashStore>) {
        self.register("flash", Arc::new(FlashStorageResourceProvider::new(store)));
    }

    /// Registers a provider for a specific URI scheme or prefix.
    pub fn register(&self, prefix: impl Into<String>, provider: Arc<dyn McpResourceProvider>) {
        self.providers
            .write()
            .entry(prefix.into())
            .or_default()
            .push(provider);
    }

    /// Aggregates all resources across registered providers.
    pub fn list(&self) -> Vec<McpResourceDefinition> {
        let guard = self.providers.read();
        let mut defs = Vec::new();
        for provider_list in guard.values() {
            for provider in provider_list {
                defs.extend(provider.list_resources());
            }
        }
        defs.sort_by(|a, b| a.uri.cmp(&b.uri));
        defs
    }

    /// Resolves and reads a resource by its exact URI.
    pub fn read(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let guard = self.providers.read();
        for provider_list in guard.values() {
            for provider in provider_list {
                if let Ok(contents) = provider.read_resource(uri) {
                    if !contents.is_empty() {
                        return Ok(contents);
                    }
                }
            }
        }
        Err(FlashStoreError::InvalidArgument(format!(
            "Resource not found: {}",
            uri
        )))
    }
}
