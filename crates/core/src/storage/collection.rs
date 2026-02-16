//! Collection and database data structures.
//!
//! A [`Collection`] wraps an HNSW vector index and a BM25 inverted index,
//! providing vector search, keyword search, hybrid search, and filtered variants.
//! [`Database`] manages named collections with thread-safe concurrent access.

use crate::bm25::{bm25_search, InvertedIndex};
use crate::document::Document;
use crate::filter_types::FilterClause;
use crate::hnsw::graph::HnswConfig;
use crate::hnsw::{knn_search, knn_search_filtered, HnswIndex};
use crate::quantization::QuantizedVector;
use crate::search::filter::matches_filter;
use crate::search::hybrid::{linear_fusion, rrf_fusion};
use crate::search::ScoredDocument;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Internal data for a collection, protected by a `RwLock`.
///
/// Contains the HNSW index, BM25 index, document store, and ID mappings.
#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionData {
    pub name: String,
    pub dimension: usize,
    pub documents: HashMap<Uuid, Arc<Document>>,
    pub hnsw_index: HnswIndex,
    pub bm25_index: InvertedIndex,
    /// Centralized UUID → internal u32 ID mapping (used by both HNSW and BM25).
    pub uuid_to_internal: HashMap<Uuid, u32>,
    /// Reverse map: internal u32 ID → UUID. Indexed by internal_id.
    pub internal_to_uuid: Vec<Uuid>,
}

impl CollectionData {
    /// Creates a new empty collection with the given name, dimension, and HNSW configuration.
    pub fn new(name: String, dimension: usize, hnsw_config: HnswConfig) -> Self {
        Self {
            name,
            dimension,
            documents: HashMap::new(),
            hnsw_index: HnswIndex::new(dimension, hnsw_config),
            bm25_index: InvertedIndex::new(),
            uuid_to_internal: HashMap::new(),
            internal_to_uuid: Vec::new(),
        }
    }

    /// Validate internal invariants after deserialization.
    ///
    /// Checks that all parallel arrays have consistent lengths, that ID mappings
    /// are symmetric, that entry point and neighbor IDs are in bounds, and that
    /// the HNSW dimension matches the collection dimension.
    pub fn validate(&self) -> Result<(), String> {
        let nc = self.hnsw_index.node_count as usize;
        let dim = self.dimension;

        // HNSW dimension must match collection dimension
        if self.hnsw_index.dimension != dim {
            return Err(format!(
                "HNSW dimension {} != collection dimension {}",
                self.hnsw_index.dimension, dim
            ));
        }

        // Quantized vector arena: 1 byte per dimension per node
        if self.hnsw_index.vector_data.len() != nc * dim {
            return Err(format!(
                "vector_data length {} != node_count({}) * dimension({})",
                self.hnsw_index.vector_data.len(),
                nc,
                dim
            ));
        }

        // Per-node quantization parameters
        if self.hnsw_index.vector_min.len() != nc {
            return Err(format!(
                "vector_min length {} != node_count {}",
                self.hnsw_index.vector_min.len(),
                nc
            ));
        }
        if self.hnsw_index.vector_scale.len() != nc {
            return Err(format!(
                "vector_scale length {} != node_count {}",
                self.hnsw_index.vector_scale.len(),
                nc
            ));
        }

        // Graph structure arrays
        if self.hnsw_index.neighbors.len() != nc {
            return Err(format!(
                "neighbors length {} != node_count {}",
                self.hnsw_index.neighbors.len(),
                nc
            ));
        }
        if self.hnsw_index.layers.len() != nc {
            return Err(format!(
                "layers length {} != node_count {}",
                self.hnsw_index.layers.len(),
                nc
            ));
        }
        if self.hnsw_index.deleted.len() != nc {
            return Err(format!(
                "deleted length {} != node_count {}",
                self.hnsw_index.deleted.len(),
                nc
            ));
        }

        // PQ codes must be consistent if codebook is present
        if let Some(ref cb) = self.hnsw_index.pq_codebook {
            let expected_pq_len = nc * cb.num_subspaces;
            if self.hnsw_index.pq_codes.len() != expected_pq_len {
                return Err(format!(
                    "pq_codes length {} != node_count({}) * num_subspaces({})",
                    self.hnsw_index.pq_codes.len(),
                    nc,
                    cb.num_subspaces
                ));
            }
        }

        // ID mappings must be symmetric
        if self.uuid_to_internal.len() != self.internal_to_uuid.len() {
            return Err(format!(
                "uuid_to_internal({}) != internal_to_uuid({})",
                self.uuid_to_internal.len(),
                self.internal_to_uuid.len()
            ));
        }
        if self.internal_to_uuid.len() != nc {
            return Err(format!(
                "internal_to_uuid length {} != node_count {}",
                self.internal_to_uuid.len(),
                nc
            ));
        }

        // Documents count should not exceed node count
        if self.documents.len() > nc {
            return Err(format!(
                "documents({}) > node_count({})",
                self.documents.len(),
                nc
            ));
        }

        // Entry point must be in bounds
        if let Some(ep) = self.hnsw_index.entry_point {
            if ep as usize >= nc {
                return Err(format!("entry_point {} >= node_count {}", ep, nc));
            }
        }

        // All neighbor IDs must be in bounds
        for (node_id, node_neighbors) in self.hnsw_index.neighbors.iter().enumerate() {
            for (layer, layer_neighbors) in node_neighbors.iter().enumerate() {
                for &neighbor in layer_neighbors {
                    if neighbor as usize >= nc {
                        return Err(format!(
                            "neighbor {} out of bounds (node_count={}) at node {} layer {}",
                            neighbor, nc, node_id, layer
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Allocate the next internal ID for a given UUID.
    fn assign_internal_id(&mut self, uuid: Uuid) -> u32 {
        let internal_id = self.internal_to_uuid.len() as u32;
        self.uuid_to_internal.insert(uuid, internal_id);
        self.internal_to_uuid.push(uuid);
        internal_id
    }
}

/// A thread-safe collection of documents with vector and keyword indices.
///
/// All operations acquire either a read or write lock on the internal [`CollectionData`].
/// Cloning a `Collection` produces a new handle to the same shared data.
#[derive(Debug, Clone)]
pub struct Collection {
    pub data: Arc<RwLock<CollectionData>>,
}

impl Collection {
    /// Creates a new empty collection.
    pub fn new(name: String, dimension: usize, hnsw_config: HnswConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(CollectionData::new(
                name,
                dimension,
                hnsw_config,
            ))),
        }
    }

    /// Inserts a document with its embedding into the collection.
    /// Returns the document's UUID.
    pub fn insert_document(&self, doc: Document, embedding: Vec<f32>) -> Uuid {
        let quantized = QuantizedVector::quantize(&embedding);
        let mut data = self.data.write();
        let doc_id = doc.id;

        let internal_id = data.assign_internal_id(doc_id);
        data.bm25_index.add_document(internal_id, &doc.text);
        data.hnsw_index.insert(internal_id, &embedding, quantized);
        data.documents.insert(doc_id, Arc::new(doc));
        doc_id
    }

    /// Retrieves a document by UUID, or `None` if not found.
    pub fn get_document(&self, id: &Uuid) -> Option<Arc<Document>> {
        self.data.read().documents.get(id).cloned()
    }

    /// Soft-deletes a document by UUID. Returns `true` if the document existed.
    pub fn delete_document(&self, id: &Uuid) -> bool {
        let mut data = self.data.write();
        if data.documents.remove(id).is_some() {
            if let Some(&internal_id) = data.uuid_to_internal.get(id) {
                data.hnsw_index.mark_deleted(internal_id);
                data.bm25_index.remove_document(internal_id);
            }
            true
        } else {
            false
        }
    }

    /// Performs approximate nearest neighbor search using the HNSW index.
    ///
    /// Returns up to `k` documents sorted by descending similarity.
    /// Optionally filters results below `min_similarity`.
    pub fn vector_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_similarity: Option<f32>,
    ) -> Vec<ScoredDocument> {
        let data = self.data.read();
        let results = knn_search(&data.hnsw_index, query_embedding, k);

        results
            .into_iter()
            .filter_map(|(distance, internal_id)| {
                let similarity = 1.0 - distance;
                if let Some(min_sim) = min_similarity {
                    if similarity < min_sim {
                        return None;
                    }
                }
                let uuid = data.internal_to_uuid.get(internal_id as usize)?;
                data.documents.get(uuid).map(|doc| ScoredDocument {
                    document: Arc::clone(doc),
                    score: similarity,
                })
            })
            .collect()
    }

    /// Performs BM25 keyword search on the inverted index.
    ///
    /// Returns up to `k` documents sorted by descending BM25 score.
    pub fn keyword_search(&self, query: &str, k: usize) -> Vec<ScoredDocument> {
        let data = self.data.read();
        let results = bm25_search(&data.bm25_index, query, k);

        results
            .into_iter()
            .filter_map(|(internal_id, score)| {
                let uuid = data.internal_to_uuid.get(internal_id as usize)?;
                data.documents.get(uuid).map(|doc| ScoredDocument {
                    document: Arc::clone(doc),
                    score,
                })
            })
            .collect()
    }

    /// Performs hybrid search combining vector and keyword results.
    ///
    /// `alpha` controls the weight between vector (alpha) and keyword (1-alpha) scores.
    /// `fusion_method`: `"rrf"` for Reciprocal Rank Fusion, `"linear"` for weighted combination.
    pub fn hybrid_search(
        &self,
        query_text: Option<&str>,
        query_embedding: Option<&[f32]>,
        k: usize,
        alpha: f32,
        fusion_method: &str,
        min_similarity: Option<f32>,
    ) -> Vec<ScoredDocument> {
        let data = self.data.read();

        // Vector results (internal_id, similarity)
        let vector_results: Vec<(u32, f32)> = if let Some(emb) = query_embedding {
            knn_search(&data.hnsw_index, emb, k)
                .into_iter()
                .filter(|&(_, internal_id)| !data.hnsw_index.is_deleted(internal_id))
                .map(|(distance, internal_id)| (internal_id, 1.0 - distance))
                .collect()
        } else {
            Vec::new()
        };

        // Keyword results (internal_id, score)
        let keyword_results: Vec<(u32, f32)> = if let Some(text) = query_text {
            bm25_search(&data.bm25_index, text, k)
        } else {
            Vec::new()
        };

        // Fusion
        let fused = match fusion_method {
            "linear" => linear_fusion(&vector_results, &keyword_results, alpha, k),
            _ => rrf_fusion(&vector_results, &keyword_results, k),
        };

        fused
            .into_iter()
            .filter_map(|(internal_id, score)| {
                if let Some(min_sim) = min_similarity {
                    if score < min_sim {
                        return None;
                    }
                }
                let uuid = data.internal_to_uuid.get(internal_id as usize)?;
                data.documents.get(uuid).map(|doc| ScoredDocument {
                    document: Arc::clone(doc),
                    score,
                })
            })
            .collect()
    }

    /// Vector search with metadata pre-filtering during HNSW graph traversal.
    ///
    /// Filtered nodes are still used for navigation but excluded from results,
    /// ensuring high recall even with selective filters.
    pub fn vector_search_filtered(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_similarity: Option<f32>,
        filter: &FilterClause,
    ) -> Vec<ScoredDocument> {
        let data = self.data.read();
        let filter_fn = |internal_id: u32| -> bool {
            if let Some(uuid) = data.internal_to_uuid.get(internal_id as usize) {
                if let Some(doc) = data.documents.get(uuid) {
                    return matches_filter(&doc.metadata, filter);
                }
            }
            false
        };
        let results = knn_search_filtered(&data.hnsw_index, query_embedding, k, &filter_fn);

        results
            .into_iter()
            .filter_map(|(distance, internal_id)| {
                let similarity = 1.0 - distance;
                if let Some(min_sim) = min_similarity {
                    if similarity < min_sim {
                        return None;
                    }
                }
                let uuid = data.internal_to_uuid.get(internal_id as usize)?;
                data.documents.get(uuid).map(|doc| ScoredDocument {
                    document: Arc::clone(doc),
                    score: similarity,
                })
            })
            .collect()
    }

    /// Hybrid search with metadata pre-filtering.
    ///
    /// Combines filtered vector search (pre-filtering in HNSW) with filtered
    /// keyword search (post-filtering on BM25 results).
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search_filtered(
        &self,
        query_text: Option<&str>,
        query_embedding: Option<&[f32]>,
        k: usize,
        alpha: f32,
        fusion_method: &str,
        min_similarity: Option<f32>,
        filter: &FilterClause,
    ) -> Vec<ScoredDocument> {
        let data = self.data.read();
        let filter_fn = |internal_id: u32| -> bool {
            if let Some(uuid) = data.internal_to_uuid.get(internal_id as usize) {
                if let Some(doc) = data.documents.get(uuid) {
                    return matches_filter(&doc.metadata, filter);
                }
            }
            false
        };

        // Vector results with pre-filtering
        let vector_results: Vec<(u32, f32)> = if let Some(emb) = query_embedding {
            knn_search_filtered(&data.hnsw_index, emb, k, &filter_fn)
                .into_iter()
                .map(|(distance, internal_id)| (internal_id, 1.0 - distance))
                .collect()
        } else {
            Vec::new()
        };

        // Keyword results (BM25 doesn't have pre-filtering, post-filter instead)
        let keyword_results: Vec<(u32, f32)> = if let Some(text) = query_text {
            bm25_search(&data.bm25_index, text, k)
                .into_iter()
                .filter(|(internal_id, _)| filter_fn(*internal_id))
                .collect()
        } else {
            Vec::new()
        };

        let fused = match fusion_method {
            "linear" => linear_fusion(&vector_results, &keyword_results, alpha, k),
            _ => rrf_fusion(&vector_results, &keyword_results, k),
        };

        fused
            .into_iter()
            .filter_map(|(internal_id, score)| {
                if let Some(min_sim) = min_similarity {
                    if score < min_sim {
                        return None;
                    }
                }
                let uuid = data.internal_to_uuid.get(internal_id as usize)?;
                data.documents.get(uuid).map(|doc| ScoredDocument {
                    document: Arc::clone(doc),
                    score,
                })
            })
            .collect()
    }

    /// Returns the number of live (non-deleted) documents.
    pub fn document_count(&self) -> usize {
        self.data.read().documents.len()
    }

    /// Returns the number of soft-deleted nodes in the HNSW index.
    pub fn deleted_count(&self) -> usize {
        let data = self.data.read();
        data.hnsw_index.deleted.iter().filter(|&&d| d).count()
    }

    /// Estimates the total memory usage of this collection in bytes.
    pub fn estimate_memory_bytes(&self) -> usize {
        let data = self.data.read();
        let mut total = 0usize;

        // HNSW vectors: vector_data (u8) + min/scale vecs + raw f32 vectors (if stored)
        total += data.hnsw_index.vector_data.len();
        total += data.hnsw_index.vector_min.len() * 4;
        total += data.hnsw_index.vector_scale.len() * 4;
        total += data.hnsw_index.raw_vectors.len() * 4;

        // PQ codes + codebook
        total += data.hnsw_index.pq_codes.len();
        if let Some(ref cb) = data.hnsw_index.pq_codebook {
            total += cb.centroids.len() * 4;
        }

        // HNSW neighbors graph
        for node_neighbors in &data.hnsw_index.neighbors {
            for layer in node_neighbors {
                total += layer.len() * 4 + 24; // Vec overhead
            }
            total += 24; // Vec<Vec<u32>> overhead
        }
        total += data.hnsw_index.layers.len();
        total += data.hnsw_index.deleted.len();

        // BM25 index
        for (term, postings) in &data.bm25_index.index {
            total += term.len() + 24; // String + HashMap overhead
            total += postings.len() * 8 + 24; // Posting is ~8 bytes + Vec overhead
        }
        total += data.bm25_index.doc_lengths.len() * 4;

        // Documents
        for doc in data.documents.values() {
            total += doc.text.len() + 16; // String + UUID
            total += doc.metadata.len() * 64; // rough estimate per metadata entry
            total += 64; // Arc + struct overhead
        }

        // ID mappings
        total += data.uuid_to_internal.len() * 24; // UUID(16) + u32(4) + overhead
        total += data.internal_to_uuid.len() * 16; // UUID

        total
    }

    /// Rebuild both HNSW and BM25 indices from live documents using a two-phase approach.
    ///
    /// **Phase A** (read lock): Snapshot all live documents with their raw vectors.
    /// **Phase B** (no lock): Build new indices from scratch — this is the expensive part.
    /// **Phase C** (write lock, brief): Swap in the new indices and catch up with any
    /// documents inserted concurrently during Phase B.
    pub fn rebuild_indices(&self) -> usize {
        // Phase A: snapshot under read lock
        let (live_docs, dimension, config) = {
            let data = self.data.read();
            let dim = data.dimension;
            let docs: Vec<(Uuid, Arc<Document>, Vec<f32>)> = data
                .documents
                .iter()
                .filter_map(|(uuid, doc)| {
                    let internal_id = data.uuid_to_internal.get(uuid)?;
                    let raw = if data.hnsw_index.has_raw_vectors() {
                        data.hnsw_index.get_raw_vector(*internal_id).to_vec()
                    } else {
                        let mut buf = vec![0.0f32; dim];
                        data.hnsw_index.dequantize_into(*internal_id, &mut buf);
                        buf
                    };
                    Some((*uuid, Arc::clone(doc), raw))
                })
                .collect();
            (docs, dim, data.hnsw_index.config.clone())
        }; // read lock released

        // Phase B: build new indices without any lock
        let mut new_hnsw = HnswIndex::new(dimension, config);
        let mut new_bm25 = InvertedIndex::new();
        let mut new_uuid_to_internal = HashMap::new();
        let mut new_internal_to_uuid = Vec::new();
        let rebuilt_ids: std::collections::HashSet<Uuid> =
            live_docs.iter().map(|(uuid, _, _)| *uuid).collect();

        for (uuid, doc, embedding) in &live_docs {
            let internal_id = new_internal_to_uuid.len() as u32;
            new_uuid_to_internal.insert(*uuid, internal_id);
            new_internal_to_uuid.push(*uuid);
            new_bm25.add_document(internal_id, &doc.text);
            let quantized = QuantizedVector::quantize(embedding);
            new_hnsw.insert(internal_id, embedding, quantized);
        }

        // Train PQ if enabled
        new_hnsw.train_pq();

        let doc_count = live_docs.len();

        // Phase C: swap under write lock, catch up with concurrent inserts
        {
            let mut data = self.data.write();

            // Find documents inserted during Phase B (present in data but not in rebuilt set)
            let new_docs: Vec<(Uuid, Arc<Document>, Vec<f32>)> = data
                .documents
                .iter()
                .filter_map(|(uuid, doc)| {
                    if rebuilt_ids.contains(uuid) {
                        return None;
                    }
                    // New documents have an internal_id assigned after Phase A snapshot
                    let internal_id = data.uuid_to_internal.get(uuid)?;
                    let raw = if data.hnsw_index.has_raw_vectors() {
                        data.hnsw_index.get_raw_vector(*internal_id).to_vec()
                    } else {
                        let mut buf = vec![0.0f32; dimension];
                        data.hnsw_index.dequantize_into(*internal_id, &mut buf);
                        buf
                    };
                    Some((*uuid, Arc::clone(doc), raw))
                })
                .collect();

            // Swap indices
            data.hnsw_index = new_hnsw;
            data.bm25_index = new_bm25;
            data.uuid_to_internal = new_uuid_to_internal;
            data.internal_to_uuid = new_internal_to_uuid;

            // Re-insert documents added during Phase B
            for (uuid, doc, embedding) in new_docs {
                let quantized = QuantizedVector::quantize(&embedding);
                let internal_id = data.assign_internal_id(uuid);
                data.bm25_index.add_document(internal_id, &doc.text);
                data.hnsw_index.insert(internal_id, &embedding, quantized);
            }
        }

        doc_count
    }
}

/// Database holds all collections.
#[derive(Debug, Clone, Default)]
pub struct Database {
    pub collections: Arc<RwLock<HashMap<String, Collection>>>,
}

impl Database {
    /// Creates a new empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new collection. Returns `Err` if a collection with the same name already exists.
    pub fn create_collection(
        &self,
        name: String,
        dimension: usize,
        hnsw_config: Option<HnswConfig>,
    ) -> Result<(), String> {
        let mut collections = self.collections.write();
        if collections.contains_key(&name) {
            return Err(format!("Collection '{}' already exists", name));
        }
        let config = hnsw_config.unwrap_or_default();
        collections.insert(name.clone(), Collection::new(name, dimension, config));
        Ok(())
    }

    /// Returns a cloned handle to the named collection, or `None` if not found.
    pub fn get_collection(&self, name: &str) -> Option<Collection> {
        self.collections.read().get(name).cloned()
    }

    /// Deletes a collection by name. Returns `true` if it existed.
    pub fn delete_collection(&self, name: &str) -> bool {
        self.collections.write().remove(name).is_some()
    }

    /// Returns the names of all collections.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Returns the estimated total memory usage across all collections.
    pub fn total_memory_bytes(&self) -> usize {
        self.collections
            .read()
            .values()
            .map(|c| c.estimate_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, MetadataValue};
    use crate::filter_types::{FilterClause, FilterCondition, FilterOperator};
    use crate::hnsw::graph::HnswConfig;
    use std::collections::HashMap;

    fn make_config() -> HnswConfig {
        HnswConfig {
            store_raw_vectors: true,
            ..HnswConfig::default()
        }
    }

    fn make_embedding(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|j| (((seed + 1) * 2654435761 + j * 40503) & 0xFFFF) as f32 / 65535.0)
            .collect()
    }

    fn make_doc(text: &str, meta: HashMap<String, MetadataValue>) -> Document {
        Document::new(text.to_string(), meta)
    }

    fn meta_kv(k: &str, v: MetadataValue) -> HashMap<String, MetadataValue> {
        let mut m = HashMap::new();
        m.insert(k.to_string(), v);
        m
    }

    // ── Collection basic CRUD ──────────────────────────────────────────

    #[test]
    fn test_collection_new() {
        let col = Collection::new("test".to_string(), 4, make_config());
        assert_eq!(col.document_count(), 0);
        assert_eq!(col.deleted_count(), 0);
    }

    #[test]
    fn test_insert_and_get_document() {
        let col = Collection::new("col".to_string(), 4, make_config());
        let doc = make_doc("hello world", HashMap::new());
        let id = doc.id;
        col.insert_document(doc, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(col.document_count(), 1);
        let fetched = col.get_document(&id).unwrap();
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.text, "hello world");
    }

    #[test]
    fn test_get_nonexistent_document() {
        let col = Collection::new("col".to_string(), 4, make_config());
        assert!(col.get_document(&Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_delete_document() {
        let col = Collection::new("col".to_string(), 4, make_config());
        let doc = make_doc("bye", HashMap::new());
        let id = doc.id;
        col.insert_document(doc, vec![0.0, 1.0, 0.0, 0.0]);
        assert!(col.delete_document(&id));
        assert_eq!(col.document_count(), 0);
        assert_eq!(col.deleted_count(), 1);
        assert!(col.get_document(&id).is_none());
    }

    #[test]
    fn test_delete_nonexistent_document() {
        let col = Collection::new("col".to_string(), 4, make_config());
        assert!(!col.delete_document(&Uuid::new_v4()));
    }

    // ── Vector search ──────────────────────────────────────────────────

    #[test]
    fn test_vector_search_basic() {
        let col = Collection::new("vs".to_string(), 4, make_config());
        col.insert_document(make_doc("a", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);
        col.insert_document(make_doc("b", HashMap::new()), vec![0.0, 1.0, 0.0, 0.0]);
        col.insert_document(make_doc("c", HashMap::new()), vec![0.0, 0.0, 1.0, 0.0]);

        let results = col.vector_search(&[1.0, 0.0, 0.0, 0.0], 2, None);
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // Nearest to [1,0,0,0] should be "a"
        assert_eq!(results[0].document.text, "a");
    }

    #[test]
    fn test_vector_search_min_similarity() {
        let col = Collection::new("vs".to_string(), 4, make_config());
        col.insert_document(make_doc("close", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);
        col.insert_document(make_doc("far", HashMap::new()), vec![-1.0, 0.0, 0.0, 0.0]);

        // With very high min_similarity, "far" should be excluded
        let results = col.vector_search(&[1.0, 0.0, 0.0, 0.0], 10, Some(0.9));
        for r in &results {
            assert!(r.score >= 0.9, "score {} < min 0.9", r.score);
        }
    }

    #[test]
    fn test_vector_search_empty_index() {
        let col = Collection::new("empty".to_string(), 4, make_config());
        let results = col.vector_search(&[1.0, 0.0, 0.0, 0.0], 5, None);
        assert!(results.is_empty());
    }

    // ── Keyword search ─────────────────────────────────────────────────

    #[test]
    fn test_keyword_search() {
        let col = Collection::new("ks".to_string(), 4, make_config());
        col.insert_document(
            make_doc("rust programming language", HashMap::new()),
            make_embedding(4, 0),
        );
        col.insert_document(
            make_doc("python programming language", HashMap::new()),
            make_embedding(4, 1),
        );
        col.insert_document(
            make_doc("database systems", HashMap::new()),
            make_embedding(4, 2),
        );

        let results = col.keyword_search("rust programming", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].document.text, "rust programming language");
    }

    #[test]
    fn test_keyword_search_no_match() {
        let col = Collection::new("ks".to_string(), 4, make_config());
        col.insert_document(
            make_doc("hello world", HashMap::new()),
            make_embedding(4, 0),
        );
        let results = col.keyword_search("nonexistent_query_xyz", 5);
        assert!(results.is_empty());
    }

    // ── Hybrid search ──────────────────────────────────────────────────

    #[test]
    fn test_hybrid_search_rrf() {
        let col = Collection::new("hs".to_string(), 4, make_config());
        col.insert_document(
            make_doc("machine learning algorithms", HashMap::new()),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        col.insert_document(
            make_doc("deep learning neural networks", HashMap::new()),
            vec![0.9, 0.1, 0.0, 0.0],
        );
        col.insert_document(
            make_doc("database indexing systems", HashMap::new()),
            vec![0.0, 0.0, 1.0, 0.0],
        );

        let results = col.hybrid_search(
            Some("machine learning"),
            Some(&[1.0, 0.0, 0.0, 0.0]),
            3,
            0.5,
            "rrf",
            None,
        );
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_search_linear() {
        let col = Collection::new("hs".to_string(), 4, make_config());
        col.insert_document(
            make_doc("alpha text", HashMap::new()),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        col.insert_document(
            make_doc("beta text", HashMap::new()),
            vec![0.0, 1.0, 0.0, 0.0],
        );

        let results = col.hybrid_search(
            Some("alpha"),
            Some(&[1.0, 0.0, 0.0, 0.0]),
            2,
            0.5,
            "linear",
            None,
        );
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_search_vector_only() {
        let col = Collection::new("hs".to_string(), 4, make_config());
        col.insert_document(make_doc("hello", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);

        let results = col.hybrid_search(None, Some(&[1.0, 0.0, 0.0, 0.0]), 2, 1.0, "rrf", None);
        // Should work with only vector query
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_hybrid_search_keyword_only() {
        let col = Collection::new("hs".to_string(), 4, make_config());
        col.insert_document(
            make_doc("hello world", HashMap::new()),
            make_embedding(4, 0),
        );

        let results = col.hybrid_search(Some("hello world"), None, 2, 0.0, "rrf", None);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_hybrid_search_min_similarity() {
        let col = Collection::new("hs".to_string(), 4, make_config());
        col.insert_document(make_doc("hello", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);

        let results = col.hybrid_search(
            Some("hello"),
            Some(&[1.0, 0.0, 0.0, 0.0]),
            5,
            0.5,
            "rrf",
            Some(999.0),
        );
        assert!(
            results.is_empty(),
            "extreme min_similarity should filter all"
        );
    }

    // ── Filtered search ────────────────────────────────────────────────

    #[test]
    fn test_vector_search_filtered() {
        let col = Collection::new("fs".to_string(), 4, make_config());
        col.insert_document(
            make_doc(
                "cat",
                meta_kv("type", MetadataValue::String("animal".into())),
            ),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        col.insert_document(
            make_doc(
                "dog",
                meta_kv("type", MetadataValue::String("animal".into())),
            ),
            vec![0.9, 0.1, 0.0, 0.0],
        );
        col.insert_document(
            make_doc(
                "car",
                meta_kv("type", MetadataValue::String("vehicle".into())),
            ),
            vec![0.8, 0.2, 0.0, 0.0],
        );

        let filter = FilterClause {
            must: vec![FilterCondition {
                field: "type".to_string(),
                op: FilterOperator::Eq,
                value: Some(serde_json::Value::String("animal".into())),
                values: None,
            }],
            must_not: vec![],
        };

        let results = col.vector_search_filtered(&[1.0, 0.0, 0.0, 0.0], 5, None, &filter);
        for r in &results {
            match r.document.metadata.get("type") {
                Some(MetadataValue::String(s)) => assert_eq!(s, "animal"),
                other => panic!("expected String(\"animal\"), got {:?}", other),
            }
        }
    }

    #[test]
    fn test_hybrid_search_filtered() {
        let col = Collection::new("hf".to_string(), 4, make_config());
        col.insert_document(
            make_doc(
                "fast car racing",
                meta_kv("category", MetadataValue::String("sports".into())),
            ),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        col.insert_document(
            make_doc(
                "fast food restaurant",
                meta_kv("category", MetadataValue::String("food".into())),
            ),
            vec![0.9, 0.1, 0.0, 0.0],
        );

        let filter = FilterClause {
            must: vec![FilterCondition {
                field: "category".to_string(),
                op: FilterOperator::Eq,
                value: Some(serde_json::Value::String("sports".into())),
                values: None,
            }],
            must_not: vec![],
        };

        let results = col.hybrid_search_filtered(
            Some("fast"),
            Some(&[1.0, 0.0, 0.0, 0.0]),
            5,
            0.5,
            "rrf",
            None,
            &filter,
        );
        for r in &results {
            match r.document.metadata.get("category") {
                Some(MetadataValue::String(s)) => assert_eq!(s, "sports"),
                other => panic!("expected String(\"sports\"), got {:?}", other),
            }
        }
    }

    // ── Rebuild, memory, deleted_count ──────────────────────────────────

    #[test]
    fn test_rebuild_indices() {
        let col = Collection::new("rb".to_string(), 4, make_config());
        let doc1 = make_doc("keep this", HashMap::new());
        let doc2 = make_doc("delete this", HashMap::new());
        let id2 = doc2.id;
        col.insert_document(doc1, vec![1.0, 0.0, 0.0, 0.0]);
        col.insert_document(doc2, vec![0.0, 1.0, 0.0, 0.0]);
        col.delete_document(&id2);
        assert_eq!(col.document_count(), 1);
        assert_eq!(col.deleted_count(), 1);

        let rebuilt = col.rebuild_indices();
        assert_eq!(rebuilt, 1);
        // After rebuild, deleted count resets to 0
        assert_eq!(col.deleted_count(), 0);
        assert_eq!(col.document_count(), 1);
    }

    #[test]
    fn test_estimate_memory_bytes() {
        let col = Collection::new("mem".to_string(), 4, make_config());
        let before = col.estimate_memory_bytes();
        col.insert_document(make_doc("hello", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);
        let after = col.estimate_memory_bytes();
        assert!(after > before, "memory should increase after insert");
    }

    // ── CollectionData::validate ───────────────────────────────────────

    #[test]
    fn test_validate_ok() {
        let col = Collection::new("v".to_string(), 4, make_config());
        col.insert_document(make_doc("a", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);
        let data = col.data.read();
        assert!(data.validate().is_ok());
    }

    #[test]
    fn test_validate_dimension_mismatch() {
        let mut cd = CollectionData::new("bad".to_string(), 4, HnswConfig::default());
        cd.hnsw_index.dimension = 8; // mismatch
        assert!(cd.validate().is_err());
    }

    // ── Database CRUD ──────────────────────────────────────────────────

    #[test]
    fn test_database_create_and_get() {
        let db = Database::new();
        assert!(db.create_collection("c1".into(), 4, None).is_ok());
        assert!(db.get_collection("c1").is_some());
        assert!(db.get_collection("nope").is_none());
    }

    #[test]
    fn test_database_duplicate_collection() {
        let db = Database::new();
        db.create_collection("dup".into(), 4, None).unwrap();
        let err = db.create_collection("dup".into(), 4, None);
        assert!(err.is_err());
    }

    #[test]
    fn test_database_delete_collection() {
        let db = Database::new();
        db.create_collection("del".into(), 4, None).unwrap();
        assert!(db.delete_collection("del"));
        assert!(!db.delete_collection("del"));
        assert!(db.get_collection("del").is_none());
    }

    #[test]
    fn test_database_list_collections() {
        let db = Database::new();
        db.create_collection("a".into(), 4, None).unwrap();
        db.create_collection("b".into(), 4, None).unwrap();
        let names = db.list_collections();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_database_total_memory_bytes() {
        let db = Database::new();
        db.create_collection("m".into(), 4, None).unwrap();
        let col = db.get_collection("m").unwrap();
        col.insert_document(make_doc("hi", HashMap::new()), vec![1.0, 0.0, 0.0, 0.0]);
        let mem = db.total_memory_bytes();
        assert!(mem > 0);
    }
}
