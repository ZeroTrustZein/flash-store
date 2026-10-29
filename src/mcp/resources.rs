//! # Model Context Protocol (MCP) Resources and Endpoints
//!
//! Exposes structured system telemetry, database status, SSTable manifests, and memory diagnostics
//! as inspectable URI resources adhering to the MCP resource specification.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

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

/// Registry managing and routing MCP resource lookups.
#[derive(Default)]
pub struct McpResourceRegistry {
    providers: RwLock<HashMap<String, Arc<dyn McpResourceProvider>>>,
}

impl McpResourceRegistry {
    /// Creates a new empty resource registry.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a provider for a specific URI scheme or prefix.
    pub fn register(&self, prefix: impl Into<String>, provider: Arc<dyn McpResourceProvider>) {
        self.providers.write().insert(prefix.into(), provider);
    }

    /// Aggregates all resources across registered providers.
    pub fn list(&self) -> Vec<McpResourceDefinition> {
        let guard = self.providers.read();
        let mut defs = Vec::new();
        for provider in guard.values() {
            defs.extend(provider.list_resources());
        }
        defs.sort_by(|a, b| a.uri.cmp(&b.uri));
        defs
    }

    /// Resolves and reads a resource by its exact URI.
    pub fn read(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let guard = self.providers.read();
        for provider in guard.values() {
            if let Ok(contents) = provider.read_resource(uri) {
                if !contents.is_empty() {
                    return Ok(contents);
                }
            }
        }
        Err(FlashStoreError::InvalidArgument(format!(
            "Resource not found: {}",
            uri
        )))
    }
}
