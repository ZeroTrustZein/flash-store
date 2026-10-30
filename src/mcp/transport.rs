//! # Model Context Protocol (MCP) Transport Layer
//!
//! Provides line-delimited JSON-RPC framing over standard I/O streams and in-memory test harnesses.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::error::{FlashStoreError, Result};
use crate::mcp::protocol::{JsonRpcRequest, JsonRpcResponse};

/// Trait abstracting transport communication channels for MCP.
pub trait McpTransport: Send + Sync {
    /// Reads the next JSON-RPC request from the transport stream.
    fn read_request(&self) -> Result<Option<JsonRpcRequest>>;

    /// Writes a JSON-RPC response back to the transport stream.
    fn write_response(&self, response: &JsonRpcResponse) -> Result<()>;
}

/// Standard I/O (stdin/stdout) line-delimited transport for production MCP integration.
pub struct StdioTransport<R: Read + Send, W: Write + Send> {
    reader: parking_lot::Mutex<BufReader<R>>,
    writer: parking_lot::Mutex<BufWriter<W>>,
}

impl<R: Read + Send, W: Write + Send> StdioTransport<R, W> {
    /// Creates a new `StdioTransport` from arbitrary reader and writer streams.
    pub fn new(read_stream: R, write_stream: W) -> Self {
        Self {
            reader: parking_lot::Mutex::new(BufReader::new(read_stream)),
            writer: parking_lot::Mutex::new(BufWriter::new(write_stream)),
        }
    }
}

impl<R: Read + Send + 'static, W: Write + Send + 'static> McpTransport for StdioTransport<R, W> {
    fn read_request(&self) -> Result<Option<JsonRpcRequest>> {
        let mut line = String::new();
        let bytes_read = {
            let mut reader = self.reader.lock();
            reader.read_line(&mut line).map_err(FlashStoreError::Io)?
        };

        if bytes_read == 0 {
            return Ok(None);
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let request: JsonRpcRequest = serde_json::from_str(trimmed)?;

        Ok(Some(request))
    }

    fn write_response(&self, response: &JsonRpcResponse) -> Result<()> {
        let serialized = serde_json::to_string(response)?;

        let mut writer = self.writer.lock();
        writer
            .write_all(serialized.as_bytes())
            .map_err(FlashStoreError::Io)?;
        writer.write_all(b"\n").map_err(FlashStoreError::Io)?;
        writer.flush().map_err(FlashStoreError::Io)?;

        Ok(())
    }
}

/// In-memory bidirectional channel transport for unit tests and deterministic simulations.
pub struct MockMcpTransport {
    req_rx: Receiver<JsonRpcRequest>,
    resp_tx: Sender<JsonRpcResponse>,
    is_closed: AtomicBool,
}

impl MockMcpTransport {
    /// Creates a pair of linked transports for testing: `(client_side, server_side)`.
    pub fn create_channel_pair() -> (MockMcpClientSide, Arc<Self>) {
        let (req_tx, req_rx) = crossbeam_channel::unbounded();
        let (resp_tx, resp_rx) = crossbeam_channel::unbounded();

        let server_transport = Arc::new(Self {
            req_rx,
            resp_tx,
            is_closed: AtomicBool::new(false),
        });

        let client_side = MockMcpClientSide { req_tx, resp_rx };

        (client_side, server_transport)
    }

    /// Signals transport shutdown.
    pub fn close(&self) {
        self.is_closed.store(true, Ordering::SeqCst);
    }
}

impl McpTransport for MockMcpTransport {
    fn read_request(&self) -> Result<Option<JsonRpcRequest>> {
        loop {
            if self.is_closed.load(Ordering::SeqCst) {
                return Ok(None);
            }
            match self
                .req_rx
                .recv_timeout(std::time::Duration::from_millis(50))
            {
                Ok(req) => return Ok(Some(req)),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return Ok(None),
            }
        }
    }

    fn write_response(&self, response: &JsonRpcResponse) -> Result<()> {
        self.resp_tx.send(response.clone()).map_err(|e| {
            FlashStoreError::Mcp(format!("Failed sending response on mock channel: {}", e))
        })
    }
}

/// Client counterpart for testing interactions with `MockMcpTransport`.
pub struct MockMcpClientSide {
    req_tx: Sender<JsonRpcRequest>,
    resp_rx: Receiver<JsonRpcResponse>,
}

impl MockMcpClientSide {
    /// Sends a JSON-RPC request to the mock server.
    pub fn send_request(&self, req: JsonRpcRequest) -> Result<()> {
        self.req_tx
            .send(req)
            .map_err(|e| FlashStoreError::Mcp(format!("Mock send failure: {}", e)))
    }

    /// Receives a response from the mock server.
    pub fn receive_response(&self) -> Result<JsonRpcResponse> {
        self.resp_rx
            .recv()
            .map_err(|e| FlashStoreError::Mcp(format!("Mock recv failure: {}", e)))
    }
}
