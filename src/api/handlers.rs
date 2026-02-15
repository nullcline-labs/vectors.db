//! HTTP request handlers and shared application state.
//!
//! Each public async function corresponds to an API route registered in
//! [`create_router`](crate::api::create_router). Handlers extract path/query/body parameters via
//! Axum extractors and delegate to the [`Database`](crate::storage::Database) and
//! [`Collection`](crate::storage::Collection) methods, returning JSON responses or
//! [`ApiError`](crate::api::errors::ApiError) on failure.
//!
//! Write operations are persisted through the WAL (or replicated via Raft when
//! running in cluster mode) before mutating in-memory state.

use crate::api::errors::ApiError;
use crate::api::metrics;
use crate::api::models::*;
use crate::api::rate_limit::TokenBucket;
use crate::cluster::types::{LogEntry as RaftLogEntry, NodeId};
use crate::cluster::Raft;
use crate::config;
use crate::document::Document;
use crate::hnsw::distance::DistanceMetric;
use crate::hnsw::graph::HnswConfig;
use crate::storage::wal::WalEntry;
use crate::storage::{save_collection, Database, WriteAheadLog};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use metrics_exporter_prometheus::PrometheusHandle;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// Shared application state passed to every handler via Axum's `State` extractor.
///
/// Contains the in-memory database, WAL, authentication config, Prometheus handle,
/// optional Raft consensus instance, and sharding routing table.
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
}

fn validate_embedding(embedding: &[f32]) -> Result<(), ApiError> {
    if embedding.iter().any(|&v| v.is_nan() || v.is_infinite()) {
        return Err(ApiError::BadRequest("Embedding contains NaN or Inf".into()));
    }
    Ok(())
}

fn validate_metadata(
    metadata: &std::collections::HashMap<String, crate::document::MetadataValue>,
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

/// `GET /health` — returns server status, version, and operational metrics.
///
/// Returns HTTP 200 when healthy, 503 when degraded (memory usage >= 90% of limit).
pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let collections = state.db.collections.read();
    let collections_count = collections.len();
    let total_documents: usize = collections.values().map(|c| c.document_count()).sum();
    drop(collections);

    let memory_used = state.db.total_memory_bytes();
    let wal_size = std::fs::metadata(&state.wal_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let uptime = state.start_time.elapsed().as_secs();

    let degraded = state.max_memory_bytes > 0 && memory_used >= state.max_memory_bytes * 9 / 10;

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
            memory_limit_bytes: state.max_memory_bytes,
            wal_size_bytes: wal_size,
        }),
    )
}

/// `POST /collections` — creates a new collection with the given dimension and HNSW parameters.
///
/// Validates the collection name, dimension, and optional HNSW configuration
/// (`m`, `ef_construction`, `ef_search`, `distance_metric`). The write is
/// persisted via WAL or Raft before the in-memory collection is created.
pub async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<CreateCollectionResponse>, ApiError> {
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

/// `GET /collections` — returns a list of all collections with their document counts.
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

/// `DELETE /collections/:name` — deletes a collection and all its documents.
pub async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    let wal_entry = WalEntry::DeleteCollection { name: name.clone() };
    let raft_entry = RaftLogEntry::DeleteCollection { name: name.clone() };

    if state.raft.is_some() {
        raft_write_or_wal(&state, raft_entry, &wal_entry).await?;
        metrics::record_write_operation(&name, "drop");
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

/// `POST /collections/:name/documents` — inserts a single document with its embedding.
///
/// Validates embedding dimension matches the collection, checks memory limits,
/// and persists through WAL/Raft before inserting into HNSW and BM25 indices.
pub async fn insert_document(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<InsertDocumentRequest>,
) -> Result<Json<InsertResponse>, ApiError> {
    check_memory_limit(&state)?;
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

    let doc = if let Some(id) = req.id {
        Document::with_id(id, req.text, req.metadata)
    } else {
        Document::new(req.text, req.metadata)
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
        tracing::info!(collection = %name, doc_id = %doc.id, "Document inserted");
        Ok(Json(InsertResponse { id: doc.id }))
    } else {
        state.wal.append(&wal_entry).await.map_err(|e| {
            tracing::error!("WAL append failed: {}", e);
            ApiError::Internal("Write failed".into())
        })?;
        let id = collection.insert_document(doc, req.embedding);
        metrics::record_write_operation(&name, "insert");
        tracing::info!(collection = %name, doc_id = %id, "Document inserted");
        Ok(Json(InsertResponse { id }))
    }
}

/// `POST /collections/:name/documents/batch` — inserts multiple documents atomically.
///
/// All documents are validated before any insertion occurs. Enforces batch size
/// limits and per-document embedding/text validation.
pub async fn batch_insert_documents(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<Json<BatchInsertResponse>, ApiError> {
    check_memory_limit(&state)?;
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

    // Validate all documents before inserting any
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

        let doc = if let Some(id) = doc_req.id {
            Document::with_id(id, doc_req.text, doc_req.metadata)
        } else {
            Document::new(doc_req.text, doc_req.metadata)
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
        tracing::info!(collection = %name, count = inserted, "Batch inserted");
        Ok(Json(BatchInsertResponse { ids, inserted }))
    }
}

/// `GET /collections/:name/documents/:id` — retrieves a single document by UUID.
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
        metadata: doc.metadata.clone(),
        score: None,
    }))
}

/// `DELETE /collections/:name/documents/:id` — soft-deletes a document by UUID.
pub async fn delete_document(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, Uuid)>,
) -> Result<Json<MessageResponse>, ApiError> {
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
            tracing::info!(collection = %name, doc_id = %id, "Document deleted");
            Ok(Json(MessageResponse {
                message: format!("Document '{}' deleted", id),
            }))
        } else {
            Err(ApiError::NotFound(format!("Document '{}' not found", id)))
        }
    }
}

/// `PUT /collections/:name/documents/:id` — updates a document's text, metadata, or embedding.
///
/// Fields not provided in the request body retain their existing values. Internally
/// performs a delete + re-insert to update the HNSW and BM25 indices.
pub async fn update_document(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, Uuid)>,
    Json(req): Json<UpdateDocumentRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let existing = collection
        .get_document(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Document '{}' not found", id)))?;

    let new_text = req.text.unwrap_or_else(|| existing.text.clone());
    let new_metadata = req.metadata.unwrap_or_else(|| existing.metadata.clone());
    validate_metadata(&new_metadata)?;

    if new_text.is_empty() {
        return Err(ApiError::BadRequest("Text must not be empty".into()));
    }
    if new_text.len() > config::MAX_TEXT_LEN {
        return Err(ApiError::BadRequest(format!(
            "Text exceeds maximum length of {} bytes",
            config::MAX_TEXT_LEN
        )));
    }

    // If no new embedding provided, we need the existing raw vector
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
        // Get existing raw vector from HNSW index
        let data = collection.data.read();
        let internal_id = data
            .uuid_to_internal
            .get(&id)
            .ok_or_else(|| ApiError::Internal("Internal error".into()))?;
        data.hnsw_index.get_raw_vector(*internal_id).to_vec()
    };

    let new_doc = Document::with_id(id, new_text, new_metadata);

    // Check memory limit before update
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
    tracing::info!(collection = %name, doc_id = %id, "Document updated");
    Ok(Json(MessageResponse {
        message: format!("Document '{}' updated", id),
    }))
}

/// `POST /collections/:name/search` — performs vector, keyword, or hybrid search.
///
/// Dispatches to the appropriate search method based on which query fields are
/// provided (`query_embedding`, `query_text`, or both). Supports metadata
/// pre-filtering, pagination via `offset`/`k`, minimum similarity threshold,
/// and configurable fusion method (`rrf` or `linear`).
pub async fn search(
    State(state): State<AppState>,
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

    // Validate embedding dimension if provided
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
        // Pre-filtering: filter is applied during HNSW graph traversal
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
        } else if has_embedding {
            collection.vector_search_filtered(
                req.query_embedding.as_ref().unwrap(),
                fetch_count,
                req.min_similarity,
                filter,
            )
        } else {
            // Keyword-only with post-filter (BM25 doesn't support pre-filtering)
            let raw = collection.keyword_search(req.query_text.as_ref().unwrap(), fetch_count);
            raw.into_iter()
                .filter(|sd| crate::search::filter::matches_filter(&sd.document.metadata, filter))
                .collect()
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
    } else if has_embedding {
        collection.vector_search(
            req.query_embedding.as_ref().unwrap(),
            fetch_count,
            req.min_similarity,
        )
    } else {
        collection.keyword_search(req.query_text.as_ref().unwrap(), fetch_count)
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
            metadata: sd.document.metadata.clone(),
            score: Some(sd.score),
        })
        .collect();

    let count = paginated.len();
    tracing::info!(collection = %name, k = req.k, mode = search_type, results = count, "Search completed");
    Ok(Json(SearchResponse {
        results: paginated,
        count,
        total,
    }))
}

/// `POST /collections/:name/save` — persists a collection to disk as a `.vdb` file.
pub async fn save(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    save_collection(&collection, &state.data_dir).map_err(|e| {
        tracing::error!("Failed to save collection: {}", e);
        ApiError::Internal("Save operation failed".into())
    })?;

    Ok(Json(MessageResponse {
        message: format!("Collection '{}' saved", name),
    }))
}

/// `POST /collections/:name/load` — loads a collection from a `.vdb` file on disk.
pub async fn load(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
    let path = std::path::Path::new(&state.data_dir).join(format!("{}.vdb", name));
    if !path.exists() {
        return Err(ApiError::NotFound(format!(
            "No saved data found for collection '{}'",
            name
        )));
    }

    let collection = crate::storage::load_collection(&path).map_err(|e| {
        tracing::error!("Failed to load collection: {}", e);
        ApiError::Internal("Load operation failed".into())
    })?;

    let mut collections = state.db.collections.write();
    collections.insert(name.clone(), collection);

    Ok(Json(MessageResponse {
        message: format!("Collection '{}' loaded", name),
    }))
}

/// `POST /admin/compact` — saves all collections to disk and truncates the WAL.
///
/// Freezes the WAL to block concurrent writes during the save + truncate
/// sequence, ensuring no entries are lost between the two operations.
pub async fn compact(State(state): State<AppState>) -> Result<Json<MessageResponse>, ApiError> {
    let _gate = state.wal.freeze();
    let collections = state.db.collections.read();
    for (name, collection) in collections.iter() {
        save_collection(collection, &state.data_dir).map_err(|e| {
            tracing::error!("Failed to save '{}': {}", name, e);
            ApiError::Internal("Save operation failed".into())
        })?;
    }
    let count = collections.len();
    drop(collections);

    state.wal.truncate().map_err(|e| {
        tracing::error!("Failed to truncate WAL: {}", e);
        ApiError::Internal("Compaction failed".into())
    })?;

    Ok(Json(MessageResponse {
        message: format!("Compaction complete, {} collections saved", count),
    }))
}

/// `GET /metrics` — returns Prometheus-formatted metrics.
pub async fn metrics_endpoint(State(state): State<AppState>) -> String {
    state.prometheus_handle.render()
}

/// Check if a collection is owned by this node. Returns Err(Redirect) if it belongs elsewhere.
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

/// Submit a write operation via Raft consensus, or fall back to local WAL.
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

fn check_memory_limit(state: &AppState) -> Result<(), ApiError> {
    if state.max_memory_bytes > 0 {
        let used = state.db.total_memory_bytes();
        if used >= state.max_memory_bytes {
            return Err(ApiError::InsufficientStorage(format!(
                "Memory limit exceeded: {} bytes used, {} bytes allowed",
                used, state.max_memory_bytes
            )));
        }
    }
    Ok(())
}

/// `GET /collections/:name/stats` — returns collection statistics including document count,
/// dimension, estimated memory usage, and deleted node count.
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

/// `POST /admin/rebuild/:name` — rebuilds HNSW and BM25 indices from live documents,
/// removing soft-deleted nodes and compacting internal ID space.
pub async fn rebuild_index(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<RebuildResponse>, ApiError> {
    let collection = state
        .db
        .get_collection(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{}' not found", name)))?;

    let start = std::time::Instant::now();
    let document_count = collection.rebuild_indices();
    let elapsed_ms = start.elapsed().as_millis();

    Ok(Json(RebuildResponse {
        message: format!("Collection '{}' rebuilt", name),
        document_count,
        elapsed_ms,
    }))
}

/// `POST /admin/backup` — saves all collections to disk and returns the list of `.vdb` files.
///
/// Freezes the WAL during the backup to ensure a consistent snapshot.
pub async fn backup(State(state): State<AppState>) -> Result<Json<BackupResponse>, ApiError> {
    let _gate = state.wal.freeze();
    let collections = state.db.collections.read();
    let mut files = Vec::new();
    for (name, collection) in collections.iter() {
        save_collection(collection, &state.data_dir).map_err(|e| {
            tracing::error!("Failed to save '{}': {}", name, e);
            ApiError::Internal("Save operation failed".into())
        })?;
        files.push(format!("{}.vdb", name));
    }
    Ok(Json(BackupResponse {
        message: format!("Backed up {} collections", files.len()),
        files,
    }))
}

/// `GET /admin/routing` — returns the current collection-to-node sharding routing table.
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

/// `POST /admin/assign` — assigns a collection to a specific node in the cluster.
///
/// In cluster mode the assignment is replicated via Raft; in standalone mode
/// it updates the local routing table directly.
pub async fn assign_collection(
    State(state): State<AppState>,
    Json(req): Json<AssignCollectionRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let raft_entry = RaftLogEntry::AssignCollection {
        collection_name: req.collection.clone(),
        owner_node_id: req.node_id,
    };

    if let Some(ref raft) = state.raft {
        match raft.client_write(raft_entry).await {
            Ok(_) => Ok(Json(MessageResponse {
                message: format!(
                    "Collection '{}' assigned to node {}",
                    req.collection, req.node_id
                ),
            })),
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
        // Standalone mode: update local routing table directly
        if let Some(ref rt) = state.routing_table {
            rt.write().insert(req.collection.clone(), req.node_id);
        }
        Ok(Json(MessageResponse {
            message: format!(
                "Collection '{}' assigned to node {}",
                req.collection, req.node_id
            ),
        }))
    }
}

/// `GET /collections/:name/documents/count` — returns the number of documents in a collection.
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

/// `POST /collections/:name/clear` — removes all documents from a collection while preserving its configuration.
pub async fn clear_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MessageResponse>, ApiError> {
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

    tracing::info!(collection = %name, "Collection cleared");
    Ok(Json(MessageResponse {
        message: format!("Collection '{}' cleared", name),
    }))
}

/// `POST /admin/restore` — loads all `.vdb` files from the data directory into memory.
pub async fn restore_all(State(state): State<AppState>) -> Result<Json<RestoreResponse>, ApiError> {
    let loaded = crate::storage::load_all_collections(&state.data_dir).map_err(|e| {
        tracing::error!("Failed to load collections: {}", e);
        ApiError::Internal("Restore operation failed".into())
    })?;

    let count = loaded.len();
    let mut collections = state.db.collections.write();
    for collection in loaded {
        let name = collection.data.read().name.clone();
        collections.insert(name, collection);
    }

    Ok(Json(RestoreResponse {
        message: format!("Restored {} collections", count),
        collections_loaded: count,
    }))
}
