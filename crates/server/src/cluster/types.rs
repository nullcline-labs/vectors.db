//! Raft type configuration, log entry definitions, and type aliases.

use serde::{Deserialize, Serialize};
use std::io::Cursor;
use uuid::Uuid;
use vectorsdb_core::document::Document;
use vectorsdb_core::hnsw::graph::HnswConfig;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    CreateCollection {
        name: String,
        dimension: usize,
        config: HnswConfig,
    },
    DeleteCollection {
        name: String,
    },
    InsertDocument {
        collection_name: String,
        document: Document,
        embedding: Vec<f32>,
    },
    DeleteDocument {
        collection_name: String,
        document_id: Uuid,
    },
    InsertDocumentBatch {
        collection_name: String,
        documents: Vec<(Document, Vec<f32>)>,
    },
    UpdateDocument {
        collection_name: String,
        document_id: Uuid,
        document: Document,
        embedding: Vec<f32>,
    },
    AssignCollection {
        collection_name: String,
        owner_node_id: NodeId,
    },
}

/// Response returned after applying a [`LogEntry`] to the state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntryResponse {
    pub success: bool,
}

impl LogEntryResponse {
    pub fn ok() -> Self {
        Self { success: true }
    }
}
