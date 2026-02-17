//! Python bindings for vectors.db via PyO3.
//!
//! Exposes `VectorDB`, `SearchResult`, and `CollectionInfo` classes
//! in a `vectorsdb` Python module.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use vectorsdb_core::document::{Document, MetadataValue};
use vectorsdb_core::filter_types::{FilterClause, FilterCondition, FilterOperator};
use vectorsdb_core::hnsw::distance::DistanceMetric;
use vectorsdb_core::hnsw::graph::HnswConfig;
use vectorsdb_core::storage::crypto::EncryptionKey;
use vectorsdb_core::storage::{load_collection, save_collection, Database};

// ---------------------------------------------------------------------------
// Helper: Python dict → HashMap<String, MetadataValue>
// ---------------------------------------------------------------------------

fn py_dict_to_metadata(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, MetadataValue>> {
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract()?;
        let v = py_to_metadata_value(&value)?;
        map.insert(k, v);
    }
    Ok(map)
}

fn py_to_metadata_value(obj: &Bound<'_, PyAny>) -> PyResult<MetadataValue> {
    // Order matters: bool before int (Python bool is a subclass of int)
    if obj.is_instance_of::<PyBool>() {
        return Ok(MetadataValue::Boolean(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(MetadataValue::Integer(obj.extract::<i64>()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(MetadataValue::Float(obj.extract::<f64>()?));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(MetadataValue::String(obj.extract::<String>()?));
    }
    Err(PyValueError::new_err(
        "metadata values must be bool, int, float, or str",
    ))
}

// ---------------------------------------------------------------------------
// Helper: MetadataValue → Python object
// ---------------------------------------------------------------------------

fn metadata_value_to_py(py: Python<'_>, v: &MetadataValue) -> PyObject {
    match v {
        MetadataValue::Boolean(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        MetadataValue::Integer(i) => i.into_pyobject(py).unwrap().into_any().unbind(),
        MetadataValue::Float(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
        MetadataValue::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
    }
}

fn metadata_to_py_dict(py: Python<'_>, meta: &HashMap<String, MetadataValue>) -> PyObject {
    let dict = PyDict::new(py);
    for (k, v) in meta {
        dict.set_item(k, metadata_value_to_py(py, v)).unwrap();
    }
    dict.into_pyobject(py).unwrap().into_any().unbind()
}

// ---------------------------------------------------------------------------
// Helper: Python filter dict → FilterClause
// ---------------------------------------------------------------------------

fn py_to_filter_clause(obj: &Bound<'_, PyAny>) -> PyResult<FilterClause> {
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    let must = if let Some(list) = dict.get_item("must")? {
        py_to_filter_conditions(&list)?
    } else {
        Vec::new()
    };
    let must_not = if let Some(list) = dict.get_item("must_not")? {
        py_to_filter_conditions(&list)?
    } else {
        Vec::new()
    };
    Ok(FilterClause { must, must_not })
}

fn py_to_filter_conditions(list: &Bound<'_, PyAny>) -> PyResult<Vec<FilterCondition>> {
    let py_list: &Bound<'_, PyList> = list.downcast()?;
    let mut conditions = Vec::new();
    for item in py_list.iter() {
        let d: &Bound<'_, PyDict> = item.downcast()?;
        let field: String = d
            .get_item("field")?
            .ok_or_else(|| PyValueError::new_err("filter condition missing 'field'"))?
            .extract()?;
        let op_str: String = d
            .get_item("op")?
            .ok_or_else(|| PyValueError::new_err("filter condition missing 'op'"))?
            .extract()?;
        let op = match op_str.as_str() {
            "eq" => FilterOperator::Eq,
            "ne" => FilterOperator::Ne,
            "gt" => FilterOperator::Gt,
            "lt" => FilterOperator::Lt,
            "gte" => FilterOperator::Gte,
            "lte" => FilterOperator::Lte,
            "in" => FilterOperator::In,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown filter operator: '{}'",
                    other
                )))
            }
        };
        let value: Option<serde_json::Value> = if let Some(v) = d.get_item("value")? {
            Some(py_to_json_value(&v)?)
        } else {
            None
        };
        let values: Option<Vec<serde_json::Value>> = if let Some(v) = d.get_item("values")? {
            let list: &Bound<'_, PyList> = v.downcast()?;
            let mut vals = Vec::new();
            for item in list.iter() {
                vals.push(py_to_json_value(&item)?);
            }
            Some(vals)
        } else {
            None
        };
        conditions.push(FilterCondition {
            field,
            op,
            value,
            values,
        });
    }
    Ok(conditions)
}

fn py_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(serde_json::Value::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        let i: i64 = obj.extract()?;
        return Ok(serde_json::Value::Number(i.into()));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        return Ok(serde_json::json!(f));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(serde_json::Value::String(obj.extract::<String>()?));
    }
    Err(PyValueError::new_err(
        "filter values must be None, bool, int, float, or str",
    ))
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result with document ID, text, score, and metadata.
#[pyclass(frozen)]
#[derive(Clone)]
struct SearchResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    score: f32,
    metadata_map: HashMap<String, MetadataValue>,
}

#[pymethods]
impl SearchResult {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyObject {
        metadata_to_py_dict(py, &self.metadata_map)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', text='{}', score={:.4})",
            self.id,
            if self.text.len() > 50 {
                format!("{}...", &self.text[..50])
            } else {
                self.text.clone()
            },
            self.score
        )
    }
}

// ---------------------------------------------------------------------------
// CollectionInfo
// ---------------------------------------------------------------------------

/// Information about a collection.
#[pyclass(frozen)]
struct CollectionInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    document_count: usize,
    #[pyo3(get)]
    estimated_memory_bytes: usize,
}

#[pymethods]
impl CollectionInfo {
    fn __repr__(&self) -> String {
        format!(
            "CollectionInfo(name='{}', dimension={}, documents={}, memory={})",
            self.name, self.dimension, self.document_count, self.estimated_memory_bytes
        )
    }
}

// ---------------------------------------------------------------------------
// VectorDB
// ---------------------------------------------------------------------------

/// In-process vector database with HNSW and BM25 indexing.
///
/// Create ephemeral or persistent databases:
///
///     db = VectorDB()                    # ephemeral (in-memory only)
///     db = VectorDB(data_dir="./data")   # with WAL persistence
#[pyclass]
struct VectorDB {
    db: Arc<Database>,
    data_dir: Option<String>,
}

#[pymethods]
impl VectorDB {
    #[new]
    #[pyo3(signature = (data_dir=None))]
    fn new(data_dir: Option<String>) -> PyResult<Self> {
        Ok(Self {
            db: Arc::new(Database::new()),
            data_dir,
        })
    }

    /// Create a new collection.
    ///
    /// Args:
    ///     name: Collection name.
    ///     dimension: Embedding vector dimension.
    ///     m: HNSW M parameter (default 16).
    ///     ef_construction: HNSW ef_construction (default 200).
    ///     ef_search: HNSW ef_search (default 50).
    ///     distance_metric: "cosine", "euclidean", or "dot_product" (default "cosine").
    ///     store_raw_vectors: If True, stores raw f32 vectors for exact reranking
    ///         (+0.7% recall, +59% RAM). Default False (compact mode).
    #[pyo3(signature = (name, dimension, m=None, ef_construction=None, ef_search=None, distance_metric=None, store_raw_vectors=None))]
    #[allow(clippy::too_many_arguments)]
    fn create_collection(
        &self,
        name: String,
        dimension: usize,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        distance_metric: Option<&str>,
        store_raw_vectors: Option<bool>,
    ) -> PyResult<()> {
        let metric = match distance_metric {
            Some("cosine") | None => DistanceMetric::Cosine,
            Some("euclidean") => DistanceMetric::Euclidean,
            Some("dot_product") => DistanceMetric::DotProduct,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown distance metric: '{}' (use 'cosine', 'euclidean', or 'dot_product')",
                    other
                )))
            }
        };

        let mut config = HnswConfig::default();
        if let Some(m_val) = m {
            config.m = m_val;
            config.m_max0 = m_val * 2;
        }
        if let Some(ef) = ef_construction {
            config.ef_construction = ef;
        }
        if let Some(ef) = ef_search {
            config.ef_search = ef;
        }
        config.distance_metric = metric;
        if let Some(store_raw) = store_raw_vectors {
            config.store_raw_vectors = store_raw;
        }

        self.db
            .create_collection(name, dimension, Some(config))
            .map_err(PyValueError::new_err)
    }

    /// Delete a collection by name. Returns True if it existed.
    fn delete_collection(&self, name: &str) -> bool {
        self.db.delete_collection(name)
    }

    /// List all collection names.
    fn list_collections(&self) -> Vec<String> {
        self.db.list_collections()
    }

    /// Get information about a collection.
    fn collection_info(&self, name: &str) -> PyResult<CollectionInfo> {
        let collection = self
            .db
            .get_collection(name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", name)))?;
        let data = collection.data.read();
        Ok(CollectionInfo {
            name: data.name.clone(),
            dimension: data.dimension,
            document_count: collection.document_count(),
            estimated_memory_bytes: collection.estimate_memory_bytes(),
        })
    }

    /// Insert a single document. Returns the document UUID as a string.
    ///
    /// Args:
    ///     collection: Collection name.
    ///     text: Document text content.
    ///     embedding: Embedding vector as a list of floats.
    ///     metadata: Optional metadata dict.
    ///     id: Optional UUID string. Auto-generated if omitted.
    #[pyo3(signature = (collection, text, embedding, metadata=None, id=None))]
    fn insert(
        &self,
        collection: &str,
        text: String,
        embedding: Vec<f32>,
        metadata: Option<&Bound<'_, PyDict>>,
        id: Option<&str>,
    ) -> PyResult<String> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        let meta = match metadata {
            Some(d) => py_dict_to_metadata(d)?,
            None => HashMap::new(),
        };

        let doc = match id {
            Some(id_str) => {
                let uuid = Uuid::parse_str(id_str)
                    .map_err(|e| PyValueError::new_err(format!("invalid UUID: {}", e)))?;
                Document::with_id(uuid, text, meta)
            }
            None => Document::new(text, meta),
        };

        let doc_id = col.insert_document(doc, embedding);
        Ok(doc_id.to_string())
    }

    /// Insert multiple documents. Returns a list of UUID strings.
    ///
    /// Each document dict should have keys: "text", "embedding", and optionally "metadata", "id".
    fn batch_insert(
        &self,
        collection: &str,
        documents: &Bound<'_, PyList>,
    ) -> PyResult<Vec<String>> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        let mut ids = Vec::with_capacity(documents.len());
        for item in documents.iter() {
            let dict: &Bound<'_, PyDict> = item.downcast()?;
            let text: String = dict
                .get_item("text")?
                .ok_or_else(|| PyValueError::new_err("document missing 'text'"))?
                .extract()?;
            let embedding: Vec<f32> = dict
                .get_item("embedding")?
                .ok_or_else(|| PyValueError::new_err("document missing 'embedding'"))?
                .extract()?;
            let meta = if let Some(m) = dict.get_item("metadata")? {
                let d: &Bound<'_, PyDict> = m.downcast()?;
                py_dict_to_metadata(d)?
            } else {
                HashMap::new()
            };
            let doc = if let Some(id_obj) = dict.get_item("id")? {
                let id_str: String = id_obj.extract()?;
                let uuid = Uuid::parse_str(&id_str)
                    .map_err(|e| PyValueError::new_err(format!("invalid UUID: {}", e)))?;
                Document::with_id(uuid, text, meta)
            } else {
                Document::new(text, meta)
            };

            let doc_id = col.insert_document(doc, embedding);
            ids.push(doc_id.to_string());
        }
        Ok(ids)
    }

    /// Get a document by UUID. Returns a dict with 'id', 'text', 'metadata'.
    fn get(&self, py: Python<'_>, collection: &str, id: &str) -> PyResult<PyObject> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        let uuid = Uuid::parse_str(id)
            .map_err(|e| PyValueError::new_err(format!("invalid UUID: {}", e)))?;

        let doc = col
            .get_document(&uuid)
            .ok_or_else(|| PyKeyError::new_err(format!("document '{}' not found", id)))?;

        let dict = PyDict::new(py);
        dict.set_item("id", doc.id.to_string())?;
        dict.set_item("text", &doc.text)?;
        dict.set_item("metadata", metadata_to_py_dict(py, &doc.metadata))?;
        Ok(dict.into_pyobject(py).unwrap().into_any().unbind())
    }

    /// Delete a document by UUID. Returns True if it existed.
    fn delete(&self, collection: &str, id: &str) -> PyResult<bool> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        let uuid = Uuid::parse_str(id)
            .map_err(|e| PyValueError::new_err(format!("invalid UUID: {}", e)))?;

        Ok(col.delete_document(&uuid))
    }

    /// Search a collection.
    ///
    /// Supports vector search, keyword search, and hybrid search.
    /// If both query_embedding and query_text are given, hybrid search is used.
    ///
    /// Args:
    ///     collection: Collection name.
    ///     query_embedding: Optional embedding vector for vector/hybrid search.
    ///     query_text: Optional text for keyword/hybrid search.
    ///     k: Number of results (default 10).
    ///     min_similarity: Optional minimum score threshold.
    ///     alpha: Hybrid search weight: 1.0 = all vector, 0.0 = all keyword (default 0.7).
    ///     fusion_method: "rrf" or "linear" (default "rrf").
    ///     filter: Optional filter dict with "must" / "must_not" lists.
    #[pyo3(signature = (collection, query_embedding=None, query_text=None, k=10, min_similarity=None, alpha=0.7, fusion_method="rrf", filter=None))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        collection: &str,
        query_embedding: Option<Vec<f32>>,
        query_text: Option<&str>,
        k: usize,
        min_similarity: Option<f32>,
        alpha: f32,
        fusion_method: &str,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<SearchResult>> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        if query_embedding.is_none() && query_text.is_none() {
            return Err(PyValueError::new_err(
                "at least one of query_embedding or query_text is required",
            ));
        }

        let filter_clause = match filter {
            Some(f) => Some(py_to_filter_clause(f)?),
            None => None,
        };

        let scored = if query_embedding.is_some() && query_text.is_some() {
            // Hybrid search
            match &filter_clause {
                Some(fc) => col.hybrid_search_filtered(
                    query_text,
                    query_embedding.as_deref(),
                    k,
                    alpha,
                    fusion_method,
                    min_similarity,
                    fc,
                ),
                None => col.hybrid_search(
                    query_text,
                    query_embedding.as_deref(),
                    k,
                    alpha,
                    fusion_method,
                    min_similarity,
                ),
            }
        } else if let Some(ref emb) = query_embedding {
            // Pure vector search
            match &filter_clause {
                Some(fc) => col.vector_search_filtered(emb, k, min_similarity, fc),
                None => col.vector_search(emb, k, min_similarity),
            }
        } else {
            // Pure keyword search
            col.keyword_search(query_text.unwrap(), k)
        };

        Ok(scored
            .into_iter()
            .map(|sd| SearchResult {
                id: sd.document.id.to_string(),
                text: sd.document.text.clone(),
                score: sd.score,
                metadata_map: sd.document.metadata.clone(),
            })
            .collect())
    }

    /// Save a collection snapshot to disk.
    ///
    /// Uses the data_dir passed at construction, or a custom path.
    /// If encryption_key is provided (64-char hex), the snapshot is encrypted with AES-256-GCM.
    #[pyo3(signature = (collection, path=None, encryption_key=None))]
    fn save(
        &self,
        collection: &str,
        path: Option<&str>,
        encryption_key: Option<&str>,
    ) -> PyResult<()> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;

        let dir = path
            .map(|s| s.to_string())
            .or_else(|| self.data_dir.clone())
            .ok_or_else(|| {
                PyValueError::new_err("no path specified and VectorDB was created without data_dir")
            })?;

        let enc_key = encryption_key
            .map(|hex| EncryptionKey::from_hex(hex).map_err(PyValueError::new_err))
            .transpose()?;

        save_collection(&col, &dir, enc_key.as_ref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load a collection snapshot from disk.
    ///
    /// Uses the data_dir passed at construction, or a custom path.
    /// If encryption_key is provided (64-char hex), encrypted snapshots are decrypted.
    #[pyo3(signature = (name, path=None, encryption_key=None))]
    fn load(&self, name: &str, path: Option<&str>, encryption_key: Option<&str>) -> PyResult<()> {
        let dir = path
            .map(|s| s.to_string())
            .or_else(|| self.data_dir.clone())
            .ok_or_else(|| {
                PyValueError::new_err("no path specified and VectorDB was created without data_dir")
            })?;

        let enc_key = encryption_key
            .map(|hex| EncryptionKey::from_hex(hex).map_err(PyValueError::new_err))
            .transpose()?;

        let file_path = Path::new(&dir).join(format!("{}.vdb", name));
        let collection = load_collection(&file_path, enc_key.as_ref())
            .map_err(|e| PyRuntimeError::new_err(format!("failed to load '{}': {}", name, e)))?;

        // Insert into the database (replace if exists)
        self.db.delete_collection(name);
        let data = collection.data.read();
        self.db
            .create_collection(
                data.name.clone(),
                data.dimension,
                Some(data.hnsw_index.config.clone()),
            )
            .map_err(PyRuntimeError::new_err)?;
        drop(data);

        // Replace with the loaded collection directly
        let mut collections = self.db.collections.write();
        collections.insert(name.to_string(), collection);

        Ok(())
    }

    /// Rebuild HNSW and BM25 indices for a collection, reclaiming deleted space.
    /// Returns the number of live documents in the rebuilt index.
    fn rebuild(&self, collection: &str) -> PyResult<usize> {
        let col = self
            .db
            .get_collection(collection)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", collection)))?;
        Ok(col.rebuild_indices())
    }

    /// Estimate total memory usage across all collections (bytes).
    fn total_memory_bytes(&self) -> usize {
        self.db.total_memory_bytes()
    }

    fn __repr__(&self) -> String {
        let names = self.db.list_collections();
        format!("VectorDB(collections={})", names.len())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _vectorsdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorDB>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<CollectionInfo>()?;
    Ok(())
}
