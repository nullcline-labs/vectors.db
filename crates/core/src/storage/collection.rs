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

        // HNSW vectors: vector_data (u8) + min/scale vecs (raw_vectors no longer stored)
        total += data.hnsw_index.vector_data.len();
        total += data.hnsw_index.vector_min.len() * 4;
        total += data.hnsw_index.vector_scale.len() * 4;

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
                    let mut raw = vec![0.0f32; dim];
                    data.hnsw_index.dequantize_into(*internal_id, &mut raw);
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
                    let mut raw = vec![0.0f32; dimension];
                    data.hnsw_index.dequantize_into(*internal_id, &mut raw);
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
