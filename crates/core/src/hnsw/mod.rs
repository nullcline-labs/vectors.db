//! Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.
//!
//! This module implements the HNSW algorithm for fast approximate nearest neighbor search.
//! Vectors are always stored using scalar quantization (f32 → u8) for 4× memory reduction.
//! When `store_raw_vectors` is false (default, compact mode), all distances use
//! asymmetric f32-query-vs-u8-stored computation. When true, raw f32 vectors are
//! also stored for exact distance computation during search and reranking (+0.7% recall,
//! +59% RAM).
//!
//! The graph uses a Struct-of-Arrays (SoA) layout for cache-friendly access:
//! all vector bytes are stored contiguously in an arena, with separate arrays for
//! quantization parameters, neighbor lists, and layer assignments.

/// Distance metrics: cosine, euclidean, and dot product.
pub mod distance;
/// HNSW graph structure, configuration, and data storage.
pub mod graph;
/// HNSW insertion algorithm with bidirectional connections and heuristic pruning.
pub mod insert;
/// HNSW search: single-layer search, multi-layer KNN, and filtered search.
pub mod search;
/// Generation-based visited set for efficient graph traversal.
pub mod visited;

pub use distance::DistanceMetric;
pub use graph::{HnswConfig, HnswIndex};
pub use search::{knn_search, knn_search_filtered};
