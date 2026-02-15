//! Raft type configuration, log entry definitions, and type aliases.
//!
//! Defines the openraft [`TypeConfig`] with application-specific log entry
//! and response types, plus convenience aliases for [`NodeId`] and [`Raft`].

use crate::document::Document;
use crate::hnsw::graph::HnswConfig;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use uuid::Uuid;

/// Unique identifier for a node in the Raft cluster.
pub type NodeId = u64;

/// The openraft `Raft` instance parameterized with our [`TypeConfig`].
pub type Raft = openraft::Raft<TypeConfig>;

openraft::declare_raft_types!(
    pub TypeConfig:
        D = LogEntry,
        R = LogEntryResponse,
);

/// A replicated log entry representing a database mutation.
///
/// Each variant mirrors a write operation from the REST API. When committed
/// via Raft consensus, the entry is applied to every node's in-memory database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    /// Create a new collection with the given HNSW configuration.
    CreateCollection {
        name: String,
        dimension: usize,
        config: HnswConfig,
    },
    /// Delete a collection by name.
    DeleteCollection { name: String },
    /// Insert a single document with its embedding into a collection.
    InsertDocument {
        collection_name: String,
        document: Document,
        embedding: Vec<f32>,
    },
    /// Soft-delete a document by UUID.
    DeleteDocument {
        collection_name: String,
        document_id: Uuid,
    },
    /// Insert multiple documents atomically into a collection.
    InsertDocumentBatch {
        collection_name: String,
        documents: Vec<(Document, Vec<f32>)>,
    },
    /// Update a document (delete + re-insert with new data).
    UpdateDocument {
        collection_name: String,
        document_id: Uuid,
        document: Document,
        embedding: Vec<f32>,
    },
    /// Assign a collection to a specific node for sharding.
    AssignCollection {
        collection_name: String,
        owner_node_id: NodeId,
    },
}

/// Response returned after applying a [`LogEntry`] to the state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntryResponse {
    /// Whether the entry was applied successfully.
    pub success: bool,
}

impl LogEntryResponse {
    /// Creates a successful response.
    pub fn ok() -> Self {
        Self { success: true }
    }
}
