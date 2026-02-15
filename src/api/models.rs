//! Request and response data transfer objects for the REST API.
//!
//! All types derive `Serialize` and/or `Deserialize` for JSON marshalling via Axum.

use crate::document::MetadataValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Request body for `POST /collections`.
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>,
    pub distance_metric: Option<String>,
}

/// Request body for `POST /collections/:name/documents`.
#[derive(Debug, Deserialize)]
pub struct InsertDocumentRequest {
    pub id: Option<Uuid>,
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
}

/// Request body for `POST /collections/:name/search`.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query_text: Option<String>,
    pub query_embedding: Option<Vec<f32>>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub offset: usize,
    pub min_similarity: Option<f32>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_fusion")]
    pub fusion_method: String,
    pub filter: Option<FilterClause>,
}

fn default_k() -> usize {
    10
}
fn default_alpha() -> f32 {
    0.7
}
fn default_fusion() -> String {
    "rrf".to_string()
}

/// Response body for document retrieval and search results.
#[derive(Debug, Serialize)]
pub struct DocumentResponse {
    pub id: Uuid,
    pub text: String,
    pub metadata: HashMap<String, MetadataValue>,
    pub score: Option<f32>,
}

/// Summary info for a collection in list responses.
#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub document_count: usize,
}

/// Response body for document insertion.
#[derive(Debug, Serialize)]
pub struct InsertResponse {
    pub id: Uuid,
}

/// Generic success message response.
#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: String,
}

/// Response body for `POST /collections` with full configuration details.
#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub message: String,
    pub name: String,
    pub dimension: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub distance_metric: String,
}

/// Response body for `GET /collections/:name/documents/count`.
#[derive(Debug, Serialize)]
pub struct CountResponse {
    pub count: usize,
}

/// Response body for search queries with pagination metadata.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<DocumentResponse>,
    pub count: usize,
    pub total: usize,
}

/// Request body for `POST /collections/:name/documents/batch`.
#[derive(Debug, Deserialize)]
pub struct BatchInsertRequest {
    pub documents: Vec<InsertDocumentRequest>,
}

/// Response body for batch document insertion.
#[derive(Debug, Serialize)]
pub struct BatchInsertResponse {
    pub ids: Vec<Uuid>,
    pub inserted: usize,
}

/// Response body for `GET /health`.
///
/// Returns `"ok"` when healthy, `"degraded"` when memory usage exceeds 90% of limit.
/// A degraded status returns HTTP 503 to signal load balancers.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub collections_count: usize,
    pub total_documents: usize,
    pub memory_used_bytes: usize,
    pub memory_limit_bytes: usize,
    pub wal_size_bytes: u64,
}

// --- Metadata Filtering types ---

/// Metadata filter clause with `must` (AND) and `must_not` (AND-NOT) conditions.
#[derive(Debug, Deserialize, Clone)]
pub struct FilterClause {
    #[serde(default)]
    pub must: Vec<FilterCondition>,
    #[serde(default)]
    pub must_not: Vec<FilterCondition>,
}

/// A single filter condition on a metadata field.
#[derive(Debug, Deserialize, Clone)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOperator,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub values: Option<Vec<serde_json::Value>>,
}

/// Comparison operator for filter conditions.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum FilterOperator {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    In,
}

// --- Document Update ---

/// Request body for `PUT /collections/:name/documents/:id`.
#[derive(Debug, Deserialize)]
pub struct UpdateDocumentRequest {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, MetadataValue>>,
}

// --- Collection Stats ---

/// Response body for `GET /collections/:name/stats`.
#[derive(Debug, Serialize)]
pub struct CollectionStatsResponse {
    pub name: String,
    pub document_count: usize,
    pub dimension: usize,
    pub estimated_memory_bytes: usize,
    pub deleted_count: usize,
}

// --- HNSW Rebuild ---

/// Response body for `POST /admin/rebuild/:name`.
#[derive(Debug, Serialize)]
pub struct RebuildResponse {
    pub message: String,
    pub document_count: usize,
    pub elapsed_ms: u128,
}

// --- Routing / Sharding ---

/// Response body for `GET /admin/routing`.
#[derive(Debug, Serialize)]
pub struct RoutingTableResponse {
    pub routing: std::collections::HashMap<String, u64>,
}

/// Request body for `POST /admin/assign`.
#[derive(Debug, Deserialize)]
pub struct AssignCollectionRequest {
    pub collection: String,
    pub node_id: u64,
}

// --- Backup / Restore ---

/// Response body for `POST /admin/backup`.
#[derive(Debug, Serialize)]
pub struct BackupResponse {
    pub message: String,
    pub files: Vec<String>,
}

/// Response body for `POST /admin/restore`.
#[derive(Debug, Serialize)]
pub struct RestoreResponse {
    pub message: String,
    pub collections_loaded: usize,
}
