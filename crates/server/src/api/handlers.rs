//! HTTP request handlers and shared application state.

use crate::api::audit::{audit_event, AuditContext};
use crate::api::errors::ApiError;
use crate::api::metrics;
use crate::api::models::*;
use crate::api::rate_limit::TokenBucket;
use crate::cluster::types::{LogEntry as RaftLogEntry, NodeId};
use crate::cluster::Raft;
use crate::replication::ReplicationState;
use crate::wal_async::WriteAheadLog;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use metrics_exporter_prometheus::PrometheusHandle;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;
use vectorsdb_core::config;
use vectorsdb_core::document::Document;
use vectorsdb_core::hnsw::distance::DistanceMetric;
use vectorsdb_core::hnsw::graph::HnswConfig;
use vectorsdb_core::storage::crypto::EncryptionKey;
use vectorsdb_core::storage::wal::WalEntry;
use vectorsdb_core::storage::{save_collection, Database};

/// Shared application state passed to every handler via Axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub data_dir: String,
    pub wal: Arc<WriteAheadLog>,
    pub wal_path: PathBuf,
    pub api_key: Option<String>,
    pub prometheus_handle: PrometheusHandle,
    pub max_memory_bytes: usize,
    pub rbac: Option<crate::api::rbac::RbacConfig>,
    pub raft: Option<Arc<Raft>>,
    pub node_id: Option<NodeId>,
    pub routing_table: Option<Arc<RwLock<HashMap<String, NodeId>>>>,
    pub peer_addrs: Option<HashMap<NodeId, String>>,
    pub start_time: Instant,
    pub key_rate_limiters: Arc<parking_lot::Mutex<HashMap<String, TokenBucket>>>,
    /// Tracks in-flight memory reservations from concurrent write requests.
    /// Prevents multiple requests from collectively exceeding the memory limit.
    pub memory_reserved: Arc<AtomicUsize>,
    /// Optional encryption key for encrypted snapshots and WAL.
    pub encryption_key: Option<Arc<EncryptionKey>>,
    /// Replication state (standby flag, WAL position).
    pub replication: ReplicationState,
}

/// Reject mutation requests when this node is in standby (read-only) mode.
fn check_standby(state: &AppState) -> Result<(), ApiError> {
    if state.replication.is_standby() {
        return Err(ApiError::ServiceUnavailable(
            "Standby node: read-only".into(),
        ));
    }
    Ok(())
}

fn validate_embedding(embedding: &[f32]) -> Result<(), ApiError> {
    if embedding.iter().any(|&v| v.is_nan() || v.is_infinite()) {
        return Err(ApiError::BadRequest("Embedding contains NaN or Inf".into()));
    }
    Ok(())
}

fn validate_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<(), ApiError> {
    if metadata.len() > config::MAX_METADATA_KEYS {
        return Err(ApiError::BadRequest(format!(
            "Metadata exceeds maximum of {} keys",
            config::MAX_METADATA_KEYS
        )));
    }
    let size = serde_json::to_vec(metadata).map(|v| v.len()).unwrap_or(0);
    if size > config::MAX_METADATA_BYTES {
        return Err(ApiError::BadRequest(format!(
            "Metadata exceeds maximum size of {} bytes",
            config::MAX_METADATA_BYTES
        )));
    }
    Ok(())
}

fn validate_collection_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() || name.len() > config::MAX_COLLECTION_NAME_LEN {
        return Err(ApiError::BadRequest(format!(
            "Collection name must be 1-{} characters",
            config::MAX_COLLECTION_NAME_LEN
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ApiError::BadRequest(
            "Collection name must contain only alphanumeric characters, '_', or '-'".into(),
        ));
    }
    Ok(())
}

/// `GET /health`
pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let collections = state.db.collections.read();
    let collections_count = collections.len();
    let total_documents: usize = collections.values().map(|c| c.document_count()).sum();
    drop(collections);

    let memory_used = state.db.total_memory_bytes();
    let memory_reserved = state.memory_reserved.load(Ordering::Relaxed);
    let wal_size = std::fs::metadata(&state.wal_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let uptime = state.start_time.elapsed().as_secs();

    // Check disk space on data directory
    let disk_available = disk_available_bytes(&state.data_dir);

    let mut warnings = Vec::new();

    let memory_degraded = state.max_memory_bytes > 0
        && memory_used + memory_reserved >= state.max_memory_bytes * 9 / 10;
    if memory_degraded {
        warnings.push(format!(
            "Memory usage at {}% of limit",
            (memory_used + memory_reserved) * 100 / state.max_memory_bytes
        ));
    }

    // Warn if disk space is low (< 100 MB)
    if disk_available < 100 * 1024 * 1024 {
        warnings.push(format!(
            "Low disk space: {} MB available",
            disk_available / (1024 * 1024)
        ));
    }

    // Warn if WAL is large (> 1 GB)
    if wal_size > 1024 * 1024 * 1024 {
        warnings.push(format!(
            "Large WAL: {} MB (consider compacting)",
            wal_size / (1024 * 1024)
        ));
    }

    let degraded = memory_degraded || disk_available < 100 * 1024 * 1024;

    let status_code = if degraded {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };

    (
        status_code,
        Json(HealthResponse {
            status: if degraded { "degraded" } else { "ok" }.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: uptime,
            collections_count,
            total_documents,
            memory_used_bytes: memory_used,
            memory_reserved_bytes: memory_reserved,
            memory_limit_bytes: state.max_memory_bytes,
            wal_size_bytes: wal_size,
            disk_available_bytes: disk_available,
            warnings,
        }),
    )
}

/// Get available disk space for the data directory.
#[allow(clippy::unnecessary_cast)]
fn disk_available_bytes(data_dir: &str) -> u64 {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let path = CString::new(data_dir).unwrap_or_else(|_| CString::new("/").unwrap());
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(path.as_ptr(), &mut stat) == 0 {
                // Cast needed: f_bavail/f_frsize are u32 on macOS, u64 on Linux
                return stat.f_bavail as u64 * stat.f_frsize as u64;
            }
        }
    }
    u64::MAX // Unknown on non-Unix platforms
}

/// `POST /collections`
pub async fn create_collection(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<CreateCollectionResponse>, ApiError> {
    check_standby(&state)?;
    validate_collection_name(&req.name)?;

    if req.dimension == 0 || req.dimension > config::MAX_DIMENSION {
        return Err(ApiError::BadRequest(format!(
            "Dimension must be 1-{}",
            config::MAX_DIMENSION
        )));
    }

    let mut config_hnsw = HnswConfig::default();
    if let Some(m) = req.m {
        if !(4..=128).contains(&m) {
            return Err(ApiError::BadRequest("m must be 4-128".into()));
        }
        config_hnsw.m = m;
        config_hnsw.m_max0 = m * 2;
    }
    if let Some(ef) = req.ef_construction {
        if !(10..=2000).contains(&ef) {
            return Err(ApiError::BadRequest(
                "ef_construction must be 10-2000".into(),
            ));
        }
        config_hnsw.ef_construction = ef;
    }
    if let Some(ef) = req.ef_search {
        if !(10..=2000).contains(&ef) {
            return Err(ApiError::BadRequest("ef_search must be 10-2000".into()));
        }
        config_hnsw.ef_search = ef;
    }
    if let Some(metric) = &req.distance_metric {
        config_hnsw.distance_metric = match metric.as_str() {
            "euclidean" => DistanceMetric::Euclidean,
            "dot" | "dot_product" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };
    }
    if let Some(store_raw) = req.store_raw_vectors {
        config_hnsw.store_raw_vectors = store_raw;
    }

    let wal_entry = WalEntry::CreateCollection {
        name: req.name.clone(),
        dimension: req.dimension,
        config: config_hnsw.clone(),
    };
    let raft_entry = RaftLogEntry::CreateCollection {
        name: req.name.clone(),
        dimension: req.dimension,
        config: config_hnsw.clone(),
    };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        state
            .db
            .create_collection(req.name.clone(), req.dimension, Some(config_hnsw.clone()))
            .map_err(ApiError::Conflict)?;
    }

    metrics::record_write_operation(&req.name, "create");
    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "create_collection",
            &req.name,
            &format!("dimension={}", req.dimension),
            "success",
        );
    }
    tracing::info!(collection = %req.name, dimension = req.dimension, "Collection created");
    Ok(Json(CreateCollectionResponse {
        message: format!("Collection '{}' created", req.name),
        name: req.name,
        dimension: req.dimension,
        m: config_hnsw.m,
        ef_construction: config_hnsw.ef_construction,
        ef_search: config_hnsw.ef_search,
        distance_metric: format!("{:?}", config_hnsw.distance_metric),
    }))
}

/// `GET /collections`
pub async fn list_collections(State(state): State<AppState>) -> Json<Vec<CollectionInfo>> {
    let collections = state.db.collections.read();
    let infos: Vec<CollectionInfo> = collections
        .values()
        .map(|c| {
            let data = c.data.read();
            CollectionInfo {
                name: data.name.clone(),
                document_count: data.documents.len(),
            }
        })
        .collect();
    Json(infos)
}

/// `DELETE /collections/:name`
pub async fn delete_collection(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let wal_entry = WalEntry::DeleteCollection { name: name.clone() };
    let raft_entry = RaftLogEntry::DeleteCollection { name: name.clone() };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
        metrics::record_write_operation(&name, "drop");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(ctx, "delete_collection", &name, "", "success");
        }
        tracing::info!(collection = %name, "Collection deleted");
        Ok(Json(MessageResponse {
            message: format!("Collection '{}' deleted", name),
        }))
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        if state.db.delete_collection(&name) {
            metrics::record_write_operation(&name, "drop");
            if let Some(Extension(ref ctx)) = audit_ctx {
                audit_event(ctx, "delete_collection", &name, "", "success");
            }
            tracing::info!(collection = %name, "Collection deleted");
            Ok(Json(MessageResponse {
                message: format!("Collection '{}' deleted", name),
            }))
        } else {
            Err(ApiError::NotFound(format!(
                "Collection '{}' not found",
                name
            )))
        }
    }
}

/// `POST /collections/:name/documents`
pub async fn insert_document(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
    Json(req): Json<InsertDocumentRequest>,
) -> Result<Json<InsertResponse>, ApiError> {
    check_standby(&state)?;
    let estimated_bytes = req.embedding.len() * 4 + req.text.len() + 256;
    let _mem_guard = check_memory_limit_reserve(&state, estimated_bytes)?;
    validate_embedding(&req.embedding)?;
    validate_metadata(&req.metadata)?;

    if req.text.is_empty() {
        return Err(ApiError::BadRequest("Text must not be empty".into()));
    }
    if req.text.len() > config::MAX_TEXT_LEN {
        return Err(ApiError::BadRequest(format!(
            "Text exceeds maximum length of {} bytes",
            config::MAX_TEXT_LEN
        )));
    }

    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let expected_dim = collection.data.read().dimension;
    if req.embedding.len() != expected_dim {
        return Err(ApiError::BadRequest(format!(
            "Expected embedding dimension {}, got {}",
            expected_dim,
            req.embedding.len()
        )));
    }

    let metadata = json_to_metadata(req.metadata);
    let doc = if let Some(id) = req.id {
        Document::with_id(id, req.text, metadata)
    } else {
        Document::new(req.text, metadata)
    };

    check_collection_locality(&state, &name)?;

    let wal_entry = WalEntry::InsertDocument {
        collection_name: name.clone(),
        document: doc.clone(),
        embedding: req.embedding.clone(),
    };
    let raft_entry = RaftLogEntry::InsertDocument {
        collection_name: name.clone(),
        document: doc.clone(),
        embedding: req.embedding.clone(),
    };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
        metrics::record_write_operation(&name, "insert");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "insert_document",
                &name,
                &format!("doc_id={}", doc.id),
                "success",
            );
        }
        tracing::info!(collection = %name, doc_id = %doc.id, "Document inserted");
        Ok(Json(InsertResponse { id: doc.id }))
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        let id = collection.insert_document(doc, req.embedding);
        metrics::record_write_operation(&name, "insert");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "insert_document",
                &name,
                &format!("doc_id={}", id),
                "success",
            );
        }
        tracing::info!(collection = %name, doc_id = %id, "Document inserted");
        Ok(Json(InsertResponse { id }))
    }
}

/// `POST /collections/:name/documents/batch`
pub async fn batch_insert_documents(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<Json<BatchInsertResponse>, ApiError> {
    check_standby(&state)?;
    let estimated_bytes: usize = req
        .documents
        .iter()
        .map(|d| d.embedding.len() * 4 + d.text.len() + 256)
        .sum();
    let _mem_guard = check_memory_limit_reserve(&state, estimated_bytes)?;
    if req.documents.is_empty() {
        return Err(ApiError::BadRequest("Batch must not be empty".into()));
    }
    if req.documents.len() > config::MAX_BATCH_SIZE {
        return Err(ApiError::BadRequest(format!(
            "Batch size exceeds maximum of {}",
            config::MAX_BATCH_SIZE
        )));
    }

    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let expected_dim = collection.data.read().dimension;

    let mut docs: Vec<(Document, Vec<f32>)> = Vec::with_capacity(req.documents.len());
    for (i, doc_req) in req.documents.into_iter().enumerate() {
        validate_embedding(&doc_req.embedding).map_err(|e| {
            ApiError::BadRequest(format!(
                "Document {}: {}",
                i,
                match e {
                    ApiError::BadRequest(msg) => msg,
                    _ => "validation error".to_string(),
                }
            ))
        })?;

        validate_metadata(&doc_req.metadata).map_err(|e| {
            ApiError::BadRequest(format!(
                "Document {}: {}",
                i,
                match e {
                    ApiError::BadRequest(msg) => msg,
                    _ => "validation error".to_string(),
                }
            ))
        })?;

        if doc_req.text.is_empty() {
            return Err(ApiError::BadRequest(format!(
                "Document {}: text must not be empty",
                i
            )));
        }
        if doc_req.text.len() > config::MAX_TEXT_LEN {
            return Err(ApiError::BadRequest(format!(
                "Document {}: text exceeds maximum length",
                i
            )));
        }
        if doc_req.embedding.len() != expected_dim {
            return Err(ApiError::BadRequest(format!(
                "Document {}: expected embedding dimension {}, got {}",
                i,
                expected_dim,
                doc_req.embedding.len()
            )));
        }

        let metadata = json_to_metadata(doc_req.metadata);
        let doc = if let Some(id) = doc_req.id {
            Document::with_id(id, doc_req.text, metadata)
        } else {
            Document::new(doc_req.text, metadata)
        };
        docs.push((doc, doc_req.embedding));
    }

    check_collection_locality(&state, &name)?;

    let wal_entry = WalEntry::InsertDocumentBatch {
        collection_name: name.clone(),
        documents: docs.clone(),
    };
    let raft_entry = RaftLogEntry::InsertDocumentBatch {
        collection_name: name.clone(),
        documents: docs.clone(),
    };

    if state.raft.is_some() {
        let ids: Vec<Uuid> = docs.iter().map(|(doc, _)| doc.id).collect();
        let inserted = ids.len();
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
        metrics::record_write_operation(&name, "batch_insert");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "batch_insert",
                &name,
                &format!("count={}", inserted),
                "success",
            );
        }
        tracing::info!(collection = %name, count = inserted, "Batch inserted");
        Ok(Json(BatchInsertResponse { ids, inserted }))
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        let ids: Vec<Uuid> = docs
            .into_iter()
            .map(|(doc, embedding)| collection.insert_document(doc, embedding))
            .collect();
        let inserted = ids.len();
        metrics::record_write_operation(&name, "batch_insert");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "batch_insert",
                &name,
                &format!("count={}", inserted),
                "success",
            );
        }
        tracing::info!(collection = %name, count = inserted, "Batch inserted");
        Ok(Json(BatchInsertResponse { ids, inserted }))
    }
}

/// `GET /collections/:name/documents/:id`
pub async fn get_document(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, Uuid)>,
) -> Result<Json<DocumentResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let doc = collection
        .get_document(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Document '{}' not found", id)))?;

    Ok(Json(DocumentResponse {
        id: doc.id,
        text: doc.text.clone(),
        metadata: metadata_to_json(&doc.metadata),
        score: None,
    }))
}

/// `DELETE /collections/:name/documents/:id`
pub async fn delete_document(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path((name, id)): Path<(String, Uuid)>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    check_collection_locality(&state, &name)?;

    let wal_entry = WalEntry::DeleteDocument {
        collection_name: name.clone(),
        document_id: id,
    };
    let raft_entry = RaftLogEntry::DeleteDocument {
        collection_name: name.clone(),
        document_id: id,
    };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
        metrics::record_write_operation(&name, "delete");
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "delete_document",
                &name,
                &format!("doc_id={}", id),
                "success",
            );
        }
        tracing::info!(collection = %name, doc_id = %id, "Document deleted");
        Ok(Json(MessageResponse {
            message: format!("Document '{}' deleted", id),
        }))
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        if collection.delete_document(&id) {
            metrics::record_write_operation(&name, "delete");
            if let Some(Extension(ref ctx)) = audit_ctx {
                audit_event(
                    ctx,
                    "delete_document",
                    &name,
                    &format!("doc_id={}", id),
                    "success",
                );
            }
            tracing::info!(collection = %name, doc_id = %id, "Document deleted");
            Ok(Json(MessageResponse {
                message: format!("Document '{}' deleted", id),
            }))
        } else {
            Err(ApiError::NotFound(format!("Document '{}' not found", id)))
        }
    }
}

/// `PUT /collections/:name/documents/:id`
pub async fn update_document(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path((name, id)): Path<(String, Uuid)>,
    Json(req): Json<UpdateDocumentRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let existing = collection
        .get_document(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Document '{}' not found", id)))?;

    let new_text = req.text.unwrap_or_else(|| existing.text.clone());
    let new_metadata = if let Some(req_meta) = req.metadata {
        validate_metadata(&req_meta)?;
        json_to_metadata(req_meta)
    } else {
        existing.metadata.clone()
    };

    if new_text.is_empty() {
        return Err(ApiError::BadRequest("Text must not be empty".into()));
    }
    if new_text.len() > config::MAX_TEXT_LEN {
        return Err(ApiError::BadRequest(format!(
            "Text exceeds maximum length of {} bytes",
            config::MAX_TEXT_LEN
        )));
    }

    let new_embedding = if let Some(emb) = req.embedding {
        validate_embedding(&emb)?;
        let expected_dim = collection.data.read().dimension;
        if emb.len() != expected_dim {
            return Err(ApiError::BadRequest(format!(
                "Expected embedding dimension {}, got {}",
                expected_dim,
                emb.len()
            )));
        }
        emb
    } else {
        let data = collection.data.read();
        let internal_id = data
            .uuid_to_internal
            .get(&id)
            .ok_or_else(|| ApiError::Internal("Internal error".into()))?;
        if data.hnsw_index.has_raw_vectors() {
            data.hnsw_index.get_raw_vector(*internal_id).to_vec()
        } else {
            let dim = data.dimension;
            let mut raw = vec![0.0f32; dim];
            data.hnsw_index.dequantize_into(*internal_id, &mut raw);
            raw
        }
    };

    let new_doc = Document::with_id(id, new_text, new_metadata);

    check_memory_limit(&state)?;
    check_collection_locality(&state, &name)?;

    let wal_entry = WalEntry::UpdateDocument {
        collection_name: name.clone(),
        document_id: id,
        document: new_doc.clone(),
        embedding: new_embedding.clone(),
    };
    let raft_entry = RaftLogEntry::UpdateDocument {
        collection_name: name.clone(),
        document_id: id,
        document: new_doc.clone(),
        embedding: new_embedding.clone(),
    };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        collection.delete_document(&id);
        collection.insert_document(new_doc, new_embedding);
    }

    metrics::record_write_operation(&name, "update");
    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "update_document",
            &name,
            &format!("doc_id={}", id),
            "success",
        );
    }
    tracing::info!(collection = %name, doc_id = %id, "Document updated");
    Ok(Json(MessageResponse {
        message: format!("Document '{}' updated", id),
    }))
}

/// `POST /collections/:name/search`
pub async fn search(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if req.k == 0 || req.k > config::MAX_K {
        return Err(ApiError::BadRequest(format!(
            "k must be 1-{}",
            config::MAX_K
        )));
    }

    if req.offset > config::MAX_OFFSET {
        return Err(ApiError::BadRequest(format!(
            "offset must be 0-{}",
            config::MAX_OFFSET
        )));
    }

    if let Some(ref emb) = req.query_embedding {
        validate_embedding(emb)?;
    }

    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let has_embedding = req.query_embedding.is_some();
    let has_text = req.query_text.is_some();

    if !has_embedding && !has_text {
        return Err(ApiError::BadRequest(
            "At least one of query_text or query_embedding is required".to_string(),
        ));
    }

    if let Some(ref emb) = req.query_embedding {
        let expected_dim = collection.data.read().dimension;
        if emb.len() != expected_dim {
            return Err(ApiError::BadRequest(format!(
                "Expected embedding dimension {}, got {}",
                expected_dim,
                emb.len()
            )));
        }
    }

    let fetch_count = req.offset.saturating_add(req.k);

    let results = if let Some(ref filter) = req.filter {
        if has_embedding && has_text {
            collection.hybrid_search_filtered(
                req.query_text.as_deref(),
                req.query_embedding.as_deref(),
                fetch_count,
                req.alpha,
                &req.fusion_method,
                req.min_similarity,
                filter,
            )
        } else if let Some(ref emb) = req.query_embedding {
            collection.vector_search_filtered(emb, fetch_count, req.min_similarity, filter)
        } else if let Some(ref text) = req.query_text {
            let raw = collection.keyword_search(text, fetch_count);
            raw.into_iter()
                .filter(|sd| {
                    vectorsdb_core::search::filter::matches_filter(&sd.document.metadata, filter)
                })
                .collect()
        } else {
            return Err(ApiError::BadRequest(
                "At least one of query_text or query_embedding is required".into(),
            ));
        }
    } else if has_embedding && has_text {
        collection.hybrid_search(
            req.query_text.as_deref(),
            req.query_embedding.as_deref(),
            fetch_count,
            req.alpha,
            &req.fusion_method,
            req.min_similarity,
        )
    } else if let Some(ref emb) = req.query_embedding {
        collection.vector_search(emb, fetch_count, req.min_similarity)
    } else if let Some(ref text) = req.query_text {
        collection.keyword_search(text, fetch_count)
    } else {
        return Err(ApiError::BadRequest(
            "At least one of query_text or query_embedding is required".into(),
        ));
    };

    let search_type = match (has_embedding, has_text) {
        (true, true) => "hybrid",
        (true, false) => "vector",
        _ => "keyword",
    };
    metrics::record_search_operation(&name, search_type);

    let total = results.len();
    let paginated: Vec<DocumentResponse> = results
        .into_iter()
        .skip(req.offset)
        .take(req.k)
        .map(|sd| DocumentResponse {
            id: sd.document.id,
            text: sd.document.text.clone(),
            metadata: metadata_to_json(&sd.document.metadata),
            score: Some(sd.score),
        })
        .collect();

    let count = paginated.len();
    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "search",
            &name,
            &format!("mode={},k={},results={}", search_type, req.k, count),
            "success",
        );
    }
    tracing::info!(collection = %name, k = req.k, mode = search_type, results = count, "Search completed");
    Ok(Json(SearchResponse {
        results: paginated,
        count,
        total,
    }))
}

/// `POST /collections/:name/save`
pub async fn save(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    save_collection(
        &collection,
        &state.data_dir,
        state.encryption_key.as_deref(),
    )
    .map_err(|e| {
        tracing::error!("Failed to save collection: {}", e);
        ApiError::Internal("Save operation failed".into())
    })?;

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(ctx, "save", &name, "", "success");
    }
    Ok(Json(MessageResponse {
        message: format!("Collection '{}' saved", name),
    }))
}

/// `POST /collections/:name/load`
pub async fn load(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let path = std::path::Path::new(&state.data_dir).join(format!("{}.vdb", name));
    if !path.exists() {
        return Err(ApiError::NotFound(format!(
            "No saved data found for collection '{}'",
            name
        )));
    }

    let collection =
        vectorsdb_core::storage::load_collection(&path, state.encryption_key.as_deref()).map_err(
            |e| {
                tracing::error!("Failed to load collection: {}", e);
                ApiError::Internal("Load operation failed".into())
            },
        )?;

    let mut collections = state.db.collections.write();
    collections.insert(name.clone(), collection);

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(ctx, "load", &name, "", "success");
    }
    Ok(Json(MessageResponse {
        message: format!("Collection '{}' loaded", name),
    }))
}

/// `POST /admin/compact`
pub async fn compact(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let _gate = state.wal.freeze();
    let collections = state.db.collections.read();
    for (name, collection) in collections.iter() {
        save_collection(collection, &state.data_dir, state.encryption_key.as_deref()).map_err(
            |e| {
                tracing::error!("Failed to save '{}': {}", name, e);
                ApiError::Internal("Save operation failed".into())
            },
        )?;
    }
    let count = collections.len();
    drop(collections);

    state.wal.truncate().map_err(|e| {
        tracing::error!("Failed to truncate WAL: {}", e);
        ApiError::Internal("Compaction failed".into())
    })?;

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "compact",
            "admin",
            &format!("collections={}", count),
            "success",
        );
    }
    Ok(Json(MessageResponse {
        message: format!("Compaction complete, {} collections saved", count),
    }))
}

/// `GET /metrics`
pub async fn metrics_endpoint(State(state): State<AppState>) -> String {
    state.prometheus_handle.render()
}

fn check_collection_locality(state: &AppState, name: &str) -> Result<(), ApiError> {
    if let (Some(rt), Some(my_id)) = (&state.routing_table, state.node_id) {
        if let Some(&owner) = rt.read().get(name) {
            if owner != my_id {
                let addr = state
                    .peer_addrs
                    .as_ref()
                    .and_then(|m| m.get(&owner))
                    .ok_or_else(|| ApiError::Internal("Internal error".into()))?;
                return Err(ApiError::Redirect(format!("http://{}", addr)));
            }
        }
    }
    Ok(())
}

async fn raft_write_or_wal(
    state: &AppState,
    raft_entry: RaftLogEntry,
    wal_entry: &WalEntry,
) -> Result<(), ApiError> {
    if let Some(ref raft) = state.raft {
        match raft.client_write(raft_entry).await {
            Ok(_) => Ok(()),
            Err(e) => {
                if let Some(fwd) = e.forward_to_leader() {
                    if let Some(ref node) = fwd.leader_node {
                        return Err(ApiError::Redirect(format!("http://{}", node.addr)));
                    }
                    return Err(ApiError::Internal("Service temporarily unavailable".into()));
                }
                Err({
                    tracing::error!("Raft write failed: {}", e);
                    ApiError::Internal("Write replication failed".into())
                })
            }
        }
    } else {
        state.wal.append(wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        Ok(())
    }
}

/// RAII guard that releases a memory reservation when dropped.
struct MemoryReservation {
    reserved: Arc<AtomicUsize>,
    amount: usize,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.amount, Ordering::Relaxed);
    }
}

/// Check memory limit and atomically reserve `extra_bytes` for this request.
/// Returns a guard that releases the reservation on drop (success or error).
fn check_memory_limit_reserve(
    state: &AppState,
    extra_bytes: usize,
) -> Result<MemoryReservation, ApiError> {
    if state.max_memory_bytes == 0 {
        return Ok(MemoryReservation {
            reserved: state.memory_reserved.clone(),
            amount: 0,
        });
    }

    // Atomically add our reservation
    let prev_reserved = state
        .memory_reserved
        .fetch_add(extra_bytes, Ordering::Relaxed);
    let used = state.db.total_memory_bytes();
    let total = used + prev_reserved + extra_bytes;

    if total > state.max_memory_bytes {
        // Over limit — release reservation and reject
        state
            .memory_reserved
            .fetch_sub(extra_bytes, Ordering::Relaxed);
        return Err(ApiError::InsufficientStorage(format!(
            "Memory limit exceeded: {} bytes committed + {} bytes reserved, {} bytes allowed",
            used,
            prev_reserved + extra_bytes,
            state.max_memory_bytes
        )));
    }

    Ok(MemoryReservation {
        reserved: state.memory_reserved.clone(),
        amount: extra_bytes,
    })
}

fn check_memory_limit(state: &AppState) -> Result<(), ApiError> {
    if state.max_memory_bytes > 0 {
        let used = state.db.total_memory_bytes();
        let reserved = state.memory_reserved.load(Ordering::Relaxed);
        if used + reserved >= state.max_memory_bytes {
            return Err(ApiError::InsufficientStorage(format!(
                "Memory limit exceeded: {} bytes used + {} bytes reserved, {} bytes allowed",
                used, reserved, state.max_memory_bytes
            )));
        }
    }
    Ok(())
}

/// `GET /collections/:name/stats`
pub async fn collection_stats(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionStatsResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let name_str = collection.data.read().name.clone();
    let dimension = collection.data.read().dimension;
    let document_count = collection.document_count();
    let deleted_count = collection.deleted_count();
    let estimated_memory_bytes = collection.estimate_memory_bytes();

    Ok(Json(CollectionStatsResponse {
        name: name_str,
        document_count,
        dimension,
        estimated_memory_bytes,
        deleted_count,
    }))
}

/// `POST /admin/rebuild/:name`
pub async fn rebuild_index(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
) -> Result<Json<RebuildResponse>, ApiError> {
    check_standby(&state)?;
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let start = std::time::Instant::now();
    let document_count = collection.rebuild_indices();
    let elapsed_ms = start.elapsed().as_millis();

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "rebuild_index",
            &name,
            &format!("docs={},ms={}", document_count, elapsed_ms),
            "success",
        );
    }
    Ok(Json(RebuildResponse {
        message: format!("Collection '{}' rebuilt", name),
        document_count,
        elapsed_ms,
    }))
}

/// `POST /admin/backup`
pub async fn backup(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
) -> Result<Json<BackupResponse>, ApiError> {
    check_standby(&state)?;
    let _gate = state.wal.freeze();
    let collections = state.db.collections.read();
    let mut files = Vec::new();
    for (name, collection) in collections.iter() {
        save_collection(collection, &state.data_dir, state.encryption_key.as_deref()).map_err(
            |e| {
                tracing::error!("Failed to save '{}': {}", name, e);
                ApiError::Internal("Save operation failed".into())
            },
        )?;
        files.push(format!("{}.vdb", name));
    }
    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "backup",
            "admin",
            &format!("collections={}", files.len()),
            "success",
        );
    }
    Ok(Json(BackupResponse {
        message: format!("Backed up {} collections", files.len()),
        files,
    }))
}

/// `GET /admin/routing`
pub async fn routing_table(
    State(state): State<AppState>,
) -> Result<Json<RoutingTableResponse>, ApiError> {
    let table = state
        .routing_table
        .as_ref()
        .map(|rt| rt.read().clone())
        .unwrap_or_default();
    Ok(Json(RoutingTableResponse { routing: table }))
}

/// `POST /admin/assign`
pub async fn assign_collection(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Json(req): Json<AssignCollectionRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let raft_entry = RaftLogEntry::AssignCollection {
        collection_name: req.collection.clone(),
        owner_node_id: req.node_id,
    };

    if let Some(ref raft) = state.raft {
        match raft.client_write(raft_entry).await {
            Ok(_) => {
                if let Some(Extension(ref ctx)) = audit_ctx {
                    audit_event(
                        ctx,
                        "assign_collection",
                        &req.collection,
                        &format!("node_id={}", req.node_id),
                        "success",
                    );
                }
                Ok(Json(MessageResponse {
                    message: format!(
                        "Collection '{}' assigned to node {}",
                        req.collection, req.node_id
                    ),
                }))
            }
            Err(e) => {
                if let Some(fwd) = e.forward_to_leader() {
                    if let Some(ref node) = fwd.leader_node {
                        return Err(ApiError::Redirect(format!("http://{}", node.addr)));
                    }
                }
                Err({
                    tracing::error!("Raft write failed: {}", e);
                    ApiError::Internal("Write replication failed".into())
                })
            }
        }
    } else {
        if let Some(ref rt) = state.routing_table {
            rt.write().insert(req.collection.clone(), req.node_id);
        }
        if let Some(Extension(ref ctx)) = audit_ctx {
            audit_event(
                ctx,
                "assign_collection",
                &req.collection,
                &format!("node_id={}", req.node_id),
                "success",
            );
        }
        Ok(Json(MessageResponse {
            message: format!(
                "Collection '{}' assigned to node {}",
                req.collection, req.node_id
            ),
        }))
    }
}

/// `GET /collections/:name/documents/count`
pub async fn document_count(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CountResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;
    Ok(Json(CountResponse {
        count: collection.document_count(),
    }))
}

/// `POST /collections/:name/clear`
pub async fn clear_collection(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    check_standby(&state)?;
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let data = collection.data.read();
    let dimension = data.dimension;
    let config = data.hnsw_index.config.clone();
    drop(data);

    state.db.delete_collection(&name);
    state
        .db
        .create_collection(name.clone(), dimension, Some(config))
        .map_err(ApiError::Internal)?;

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(ctx, "clear_collection", &name, "", "success");
    }
    tracing::info!(collection = %name, "Collection cleared");
    Ok(Json(MessageResponse {
        message: format!("Collection '{}' cleared", name),
    }))
}

/// `POST /admin/restore`
pub async fn restore_all(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
) -> Result<Json<RestoreResponse>, ApiError> {
    check_standby(&state)?;
    let loaded = vectorsdb_core::storage::load_all_collections(
        &state.data_dir,
        state.encryption_key.as_deref(),
    )
    .map_err(|e| {
        tracing::error!("Failed to load collections: {}", e);
        ApiError::Internal("Restore operation failed".into())
    })?;

    let count = loaded.len();
    let mut collections = state.db.collections.write();
    for collection in loaded {
        let name = collection.data.read().name.clone();
        collections.insert(name, collection);
    }

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "restore_all",
            "admin",
            &format!("collections={}", count),
            "success",
        );
    }
    Ok(Json(RestoreResponse {
        message: format!("Restored {} collections", count),
        collections_loaded: count,
    }))
}

/// `POST /admin/promote`
///
/// Promote a standby node to primary (read-write). Only works on standby nodes.
pub async fn promote(
    State(state): State<AppState>,
    audit_ctx: Option<Extension<AuditContext>>,
) -> Result<Json<PromoteResponse>, ApiError> {
    if !state.replication.is_standby() {
        return Err(ApiError::BadRequest("Node is not in standby mode".into()));
    }

    state.replication.promote();
    let wal_position = state.replication.wal_position();

    if let Some(Extension(ref ctx)) = audit_ctx {
        audit_event(
            ctx,
            "promote",
            "admin",
            &format!("wal_position={}", wal_position),
            "success",
        );
    }
    tracing::info!(wal_position = wal_position, "Node promoted to primary");
    Ok(Json(PromoteResponse {
        promoted: true,
        wal_position,
    }))
}
