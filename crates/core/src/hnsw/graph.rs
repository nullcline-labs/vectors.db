//! HNSW graph structure and configuration.
//!
//! [`HnswConfig`] defines tuning parameters (M, ef_construction, ef_search, distance metric).
//! [`HnswIndex`] stores the graph using Struct-of-Arrays layout for cache efficiency.

use crate::config;
use crate::hnsw::distance::DistanceMetric;
use crate::quantization::pq::PqCodebook;
use crate::quantization::scalar::VectorRef;
use crate::quantization::QuantizedVector;
use serde::{Deserialize, Serialize};

/// Configuration parameters for an HNSW index.
///
/// Controls the trade-off between build speed, search speed, recall, and memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bidirectional links per node (except layer 0, which uses `m_max0`).
    pub m: usize,
    /// Maximum links per node at layer 0 (typically `2 * m`).
    pub m_max0: usize,
    /// Candidate list size during index construction.
    pub ef_construction: usize,
    /// Candidate list size during search (higher = better recall, slower).
    pub ef_search: usize,
    /// Maximum number of layers in the graph.
    pub max_layers: usize,
    /// Distance function for similarity computation.
    pub distance_metric: DistanceMetric,
    /// Number of PQ subspaces (0 = PQ disabled, use scalar quantization only).
    pub pq_subspaces: usize,
    /// When true, stores raw f32 vectors for exact reranking (+0.7% recall, +59% RAM).
    /// When false (default), uses only scalar-quantized u8 vectors (compact mode).
    #[serde(default)]
    pub store_raw_vectors: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: config::HNSW_DEFAULT_M,
            m_max0: config::HNSW_DEFAULT_M * 2,
            ef_construction: config::HNSW_DEFAULT_EF_CONSTRUCTION,
            ef_search: config::HNSW_DEFAULT_EF_SEARCH,
            max_layers: config::HNSW_DEFAULT_MAX_LAYERS,
            distance_metric: DistanceMetric::Cosine,
            pq_subspaces: config::PQ_DEFAULT_SUBSPACES,
            store_raw_vectors: false,
        }
    }
}

/// HNSW Index using Struct-of-Arrays (SoA) layout for cache-friendly access.
/// Vector data is stored contiguously in an arena. No HnswNode struct.
#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    pub config: HnswConfig,
    // SoA: quantized vector arena — all vector bytes contiguous
    pub vector_data: Vec<u8>,
    pub vector_min: Vec<f32>,
    pub vector_scale: Vec<f32>,
    // SoA: raw f32 vector arena (populated only when store_raw_vectors=true)
    #[serde(default)]
    pub raw_vectors: Vec<f32>,
    // SoA: graph structure
    pub neighbors: Vec<Vec<Vec<u32>>>, // [node_id][layer][neighbor_ids]
    pub layers: Vec<u8>,
    pub deleted: Vec<bool>,
    // PQ quantization (optional, trained via train_pq)
    /// PQ codebook (None until training is performed).
    pub pq_codebook: Option<PqCodebook>,
    /// PQ codes arena: M bytes per node, contiguous.
    pub pq_codes: Vec<u8>,
    // Index metadata
    pub entry_point: Option<u32>,
    pub max_layer: usize,
    pub dimension: usize,
    pub node_count: u32,
}

/// Portable software prefetch hint (L1 cache, read).
/// No-op on unsupported platforms.
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

impl HnswIndex {
    /// Creates a new empty HNSW index with the given dimension and configuration.
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        Self {
            config,
            vector_data: Vec::new(),
            vector_min: Vec::new(),
            vector_scale: Vec::new(),
            raw_vectors: Vec::new(),
            neighbors: Vec::new(),
            layers: Vec::new(),
            deleted: Vec::new(),
            pq_codebook: None,
            pq_codes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            dimension,
            node_count: 0,
        }
    }

    /// Creates a new empty HNSW index with default configuration (cosine, M=16, ef_c=200).
    pub fn with_default_config(dimension: usize) -> Self {
        Self::new(dimension, HnswConfig::default())
    }

    /// Returns the number of non-deleted nodes in the index.
    pub fn len(&self) -> usize {
        self.deleted.iter().filter(|&&d| !d).count()
    }

    /// Returns `true` if the index contains no non-deleted nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate a random layer for a new node using exponential distribution.
    pub fn random_level(&self) -> usize {
        let ml = 1.0 / (self.config.m as f64).ln();
        let r: f64 = rand::random();
        let level = (-r.ln() * ml).floor() as usize;
        level.min(self.config.max_layers - 1)
    }

    /// Get a VectorRef for the given node. O(1) slice into contiguous arena.
    #[inline]
    pub fn get_vector_ref(&self, id: u32) -> VectorRef<'_> {
        let idx = id as usize;
        let start = idx * self.dimension;
        VectorRef {
            data: &self.vector_data[start..start + self.dimension],
            min: self.vector_min[idx],
            scale: self.vector_scale[idx],
        }
    }

    /// Returns `true` if the node with the given internal ID has been soft-deleted.
    #[inline]
    pub fn is_deleted(&self, id: u32) -> bool {
        self.deleted[id as usize]
    }

    /// Returns the layer assignment of the given node.
    #[inline]
    pub fn get_layer(&self, id: u32) -> u8 {
        self.layers[id as usize]
    }

    /// Store a new quantized vector in the arena. Returns nothing — data appended in place.
    pub fn push_vector(&mut self, vector: &QuantizedVector) {
        self.vector_data.extend_from_slice(&vector.data);
        self.vector_min.push(vector.min);
        self.vector_scale.push(vector.scale);
    }

    /// Store a raw f32 vector in the arena (only when store_raw_vectors=true).
    pub fn push_raw_vector(&mut self, raw: &[f32]) {
        self.raw_vectors.extend_from_slice(raw);
    }

    /// Get a raw f32 vector slice for the given node.
    /// Only valid when raw_vectors are populated (store_raw_vectors=true).
    #[inline]
    pub fn get_raw_vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dimension;
        &self.raw_vectors[start..start + self.dimension]
    }

    /// Returns true if raw f32 vectors are available for exact distance computation.
    #[inline]
    pub fn has_raw_vectors(&self) -> bool {
        !self.raw_vectors.is_empty()
    }

    /// Dequantize a node's vector into the provided buffer (no allocation).
    #[inline]
    pub fn dequantize_into(&self, id: u32, buf: &mut [f32]) {
        let vref = self.get_vector_ref(id);
        for (i, &b) in vref.data.iter().enumerate() {
            buf[i] = vref.min + b as f32 * vref.scale;
        }
    }

    /// Free legacy raw_vectors memory (called after loading old snapshots).
    pub fn free_raw_vectors(&mut self) {
        self.raw_vectors.clear();
        self.raw_vectors.shrink_to_fit();
    }

    /// Mark a node as deleted by internal ID.
    pub fn mark_deleted(&mut self, internal_id: u32) -> bool {
        if (internal_id as usize) < self.deleted.len() {
            self.deleted[internal_id as usize] = true;
            true
        } else {
            false
        }
    }

    /// Prefetch quantized vector data for a node (u8 arena + min/scale metadata).
    /// Prefetches two cache lines for vectors > 64 bytes (dim > 64).
    #[inline(always)]
    pub fn prefetch_vector(&self, id: u32) {
        let start = id as usize * self.dimension;
        if start < self.vector_data.len() {
            let ptr = unsafe { self.vector_data.as_ptr().add(start) };
            prefetch_read(ptr);
            if self.dimension > 64 {
                prefetch_read(unsafe { ptr.add(64) });
            }
            prefetch_read(unsafe { self.vector_min.as_ptr().add(id as usize) as *const u8 });
        }
    }

    /// Prefetch raw f32 vector data for a node into L1 cache.
    /// Prefetches two cache lines for vectors > 16 floats (dim > 16).
    #[inline(always)]
    pub fn prefetch_raw_vector(&self, id: u32) {
        let start = id as usize * self.dimension;
        if start < self.raw_vectors.len() {
            let ptr = unsafe { self.raw_vectors.as_ptr().add(start) as *const u8 };
            prefetch_read(ptr);
            if self.dimension > 16 {
                prefetch_read(unsafe { ptr.add(64) });
            }
        }
    }

    /// Get PQ codes for a node (M bytes).
    #[inline]
    pub fn get_pq_codes(&self, id: u32) -> &[u8] {
        let m = self.pq_codebook.as_ref().unwrap().num_subspaces;
        let start = id as usize * m;
        &self.pq_codes[start..start + m]
    }

    /// Prefetch PQ codes for a node into L1 cache.
    #[inline(always)]
    pub fn prefetch_pq_codes(&self, id: u32) {
        if let Some(ref cb) = self.pq_codebook {
            let start = id as usize * cb.num_subspaces;
            if start < self.pq_codes.len() {
                prefetch_read(unsafe { self.pq_codes.as_ptr().add(start) });
            }
        }
    }

    /// Train PQ codebook on current vectors and encode all nodes.
    /// Uses raw f32 vectors when available, otherwise dequantizes from scalar quantization.
    /// Does nothing if pq_subspaces is 0 or the index is empty.
    pub fn train_pq(&mut self) {
        let m = self.config.pq_subspaces;
        if m == 0 || self.node_count == 0 {
            return;
        }
        let n = self.node_count as usize;
        let dim = self.dimension;
        if self.has_raw_vectors() {
            let data = &self.raw_vectors[..n * dim];
            let codebook = PqCodebook::train(data, dim, m, config::PQ_NUM_CENTROIDS);
            self.pq_codes = codebook.encode_batch(data, dim);
            self.pq_codebook = Some(codebook);
        } else {
            let mut deq = vec![0.0f32; n * dim];
            for id in 0..n {
                self.dequantize_into(id as u32, &mut deq[id * dim..(id + 1) * dim]);
            }
            let codebook = PqCodebook::train(&deq, dim, m, config::PQ_NUM_CENTROIDS);
            self.pq_codes = codebook.encode_batch(&deq, dim);
            self.pq_codebook = Some(codebook);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantizedVector;

    fn make_index(dim: usize) -> HnswIndex {
        HnswIndex::new(dim, HnswConfig::default())
    }

    fn make_raw_index(dim: usize) -> HnswIndex {
        HnswIndex::new(
            dim,
            HnswConfig {
                store_raw_vectors: true,
                ..HnswConfig::default()
            },
        )
    }

    fn insert_one(index: &mut HnswIndex, id: u32, raw: &[f32]) {
        let q = QuantizedVector::quantize(raw);
        index.push_vector(&q);
        if index.config.store_raw_vectors {
            index.push_raw_vector(raw);
        }
        index.neighbors.push(vec![Vec::new()]);
        index.layers.push(0);
        index.deleted.push(false);
        index.node_count += 1;
        if index.entry_point.is_none() {
            index.entry_point = Some(id);
        }
    }

    #[test]
    fn test_new_empty_index() {
        let idx = make_index(128);
        assert_eq!(idx.dimension, 128);
        assert_eq!(idx.node_count, 0);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.entry_point.is_none());
    }

    #[test]
    fn test_with_default_config() {
        let idx = HnswIndex::with_default_config(64);
        assert_eq!(idx.dimension, 64);
        assert!(matches!(idx.config.distance_metric, DistanceMetric::Cosine));
    }

    #[test]
    fn test_push_and_get_vector_ref() {
        let mut idx = make_index(4);
        let raw = vec![0.5, 0.25, 0.75, 1.0];
        insert_one(&mut idx, 0, &raw);
        let vref = idx.get_vector_ref(0);
        assert_eq!(vref.data.len(), 4);
        // Dequantized should be close to original
        let deq: Vec<f32> = vref
            .data
            .iter()
            .map(|&b| vref.min + b as f32 * vref.scale)
            .collect();
        for (a, b) in raw.iter().zip(deq.iter()) {
            assert!((a - b).abs() < 0.02, "deq mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_raw_vectors() {
        let mut idx = make_raw_index(4);
        let raw = vec![1.0, 2.0, 3.0, 4.0];
        insert_one(&mut idx, 0, &raw);
        assert!(idx.has_raw_vectors());
        assert_eq!(idx.get_raw_vector(0), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_no_raw_vectors() {
        let idx = make_index(4);
        assert!(!idx.has_raw_vectors());
    }

    #[test]
    fn test_free_raw_vectors() {
        let mut idx = make_raw_index(4);
        insert_one(&mut idx, 0, &[1.0, 2.0, 3.0, 4.0]);
        assert!(idx.has_raw_vectors());
        idx.free_raw_vectors();
        assert!(!idx.has_raw_vectors());
        assert!(idx.raw_vectors.is_empty());
    }

    #[test]
    fn test_dequantize_into() {
        let mut idx = make_index(4);
        let raw = vec![0.1, 0.5, 0.9, 0.3];
        insert_one(&mut idx, 0, &raw);
        let mut buf = vec![0.0f32; 4];
        idx.dequantize_into(0, &mut buf);
        for (a, b) in raw.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 0.02, "deq mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_mark_deleted() {
        let mut idx = make_index(4);
        insert_one(&mut idx, 0, &[1.0, 0.0, 0.0, 0.0]);
        insert_one(&mut idx, 1, &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_deleted(0));
        assert!(idx.mark_deleted(0));
        assert!(idx.is_deleted(0));
        assert_eq!(idx.len(), 1);
        // Out-of-bounds mark_deleted returns false
        assert!(!idx.mark_deleted(99));
    }

    #[test]
    fn test_get_layer() {
        let mut idx = make_index(4);
        insert_one(&mut idx, 0, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.get_layer(0), 0);
    }

    #[test]
    fn test_random_level() {
        let idx = make_index(4);
        // Just check it doesn't panic and respects max_layers
        for _ in 0..100 {
            let level = idx.random_level();
            assert!(level < idx.config.max_layers);
        }
    }

    #[test]
    fn test_prefetch_no_panic() {
        let mut idx = make_raw_index(4);
        insert_one(&mut idx, 0, &[1.0, 0.0, 0.0, 0.0]);
        // Just exercise prefetch — no crash = pass
        idx.prefetch_vector(0);
        idx.prefetch_raw_vector(0);
    }

    #[test]
    fn test_prefetch_large_dim() {
        let mut idx = make_raw_index(128);
        let raw: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        insert_one(&mut idx, 0, &raw);
        // Exercises the second-cache-line prefetch branch (dim > 64 / dim > 16)
        idx.prefetch_vector(0);
        idx.prefetch_raw_vector(0);
    }

    #[test]
    fn test_train_pq_with_raw_vectors() {
        let dim = 16;
        let mut idx = HnswIndex::new(
            dim,
            HnswConfig {
                store_raw_vectors: true,
                pq_subspaces: 2,
                ..HnswConfig::default()
            },
        );
        // Need > 256 vectors for meaningful PQ training
        for i in 0..300u32 {
            let raw: Vec<f32> = (0..dim)
                .map(|j| ((i as usize * 7 + j * 13) % 97) as f32 / 97.0)
                .collect();
            insert_one(&mut idx, i, &raw);
        }
        idx.train_pq();
        assert!(idx.pq_codebook.is_some());
        assert_eq!(idx.pq_codes.len(), 300 * 2);
        let codes = idx.get_pq_codes(0);
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn test_train_pq_without_raw_dequantizes() {
        let dim = 16;
        let mut idx = HnswIndex::new(
            dim,
            HnswConfig {
                store_raw_vectors: false,
                pq_subspaces: 2,
                ..HnswConfig::default()
            },
        );
        for i in 0..300u32 {
            let raw: Vec<f32> = (0..dim)
                .map(|j| ((i as usize * 7 + j * 13) % 97) as f32 / 97.0)
                .collect();
            insert_one(&mut idx, i, &raw);
        }
        idx.train_pq();
        assert!(idx.pq_codebook.is_some());
        assert_eq!(idx.pq_codes.len(), 300 * 2);
    }

    #[test]
    fn test_train_pq_noop_when_disabled() {
        let mut idx = make_index(4);
        insert_one(&mut idx, 0, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.config.pq_subspaces, 0);
        idx.train_pq();
        assert!(idx.pq_codebook.is_none());
    }

    #[test]
    fn test_train_pq_noop_when_empty() {
        let mut idx = HnswIndex::new(
            8,
            HnswConfig {
                pq_subspaces: 2,
                ..HnswConfig::default()
            },
        );
        idx.train_pq();
        assert!(idx.pq_codebook.is_none());
    }
}
