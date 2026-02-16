//! Core document types for vectors.db.
//!
//! A `Document` represents a stored record with text content, a unique UUID,
//! and arbitrary key-value metadata. `MetadataValue` supports boolean, integer,
//! float, and string values for use in filtered search queries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A typed metadata value attached to a document.
///
/// Used for metadata filtering in search queries (e.g., `eq`, `gt`, `in` operators).
/// Uses the default externally-tagged serde representation for bincode compatibility.
/// The server API layer converts to/from untagged JSON at the HTTP boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    /// Boolean value (`true` / `false`).
    Boolean(bool),
    /// 64-bit signed integer.
    Integer(i64),
    /// 64-bit floating-point number.
    Float(f64),
    /// UTF-8 string.
    String(String),
}

/// A stored document with text content, unique ID, and metadata.
///
/// Documents are the primary unit of storage in a collection. Each document
/// is associated with an embedding vector (stored separately in the HNSW index)
/// and indexed for both vector search and BM25 keyword search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier (UUID v4).
    pub id: Uuid,
    /// Text content, indexed by BM25.
    pub text: String,
    /// Arbitrary key-value metadata for filtering.
    pub metadata: HashMap<String, MetadataValue>,
}

impl Document {
    /// Creates a new document with a random UUID.
    pub fn new(text: String, metadata: HashMap<String, MetadataValue>) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            metadata,
        }
    }

    /// Creates a document with a specific UUID.
    pub fn with_id(id: Uuid, text: String, metadata: HashMap<String, MetadataValue>) -> Self {
        Self { id, text, metadata }
    }
}
