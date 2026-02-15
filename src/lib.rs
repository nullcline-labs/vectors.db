//! # vectors.db
//!
//! An in-memory vector database with HNSW approximate nearest neighbor search,
//! BM25 full-text search, and hybrid retrieval.
//!
//! ## Features
//!
//! - **HNSW vector search** with scalar quantization (f32 → u8) for 4× memory reduction
//! - **BM25 keyword search** using Okapi BM25 scoring with inverted index
//! - **Hybrid search** combining vector and keyword results via RRF or linear fusion
//! - **Metadata filtering** with pre-filtering during HNSW graph traversal
//! - **Write-Ahead Log** with CRC32 checksums and fsync for crash recovery
//! - **Raft consensus** for multi-node replication (via openraft)
//! - **Collection-level sharding** with routing table and request redirection
//! - **REST API** built on Axum with authentication, RBAC, rate limiting, and TLS
//!
//! ## Architecture
//!
//! ```text
//! HTTP API (Axum) → Database → Collection → { HNSW Index, BM25 Index }
//!                                          → Hybrid Fusion (RRF / Linear)
//! Persistence: WAL (append + CRC32 + fsync) → Snapshots (bincode)
//! Clustering:  Raft (openraft) → Routing Table → Request Forwarding
//! ```

/// REST API layer: Axum router, HTTP handlers, request/response models, authentication, and metrics.
pub mod api;
/// BM25 full-text search: inverted index, Okapi BM25 scoring, and whitespace tokenizer.
pub mod bm25;
/// Raft consensus clustering: log store, state machine, network transport, and internal API routes.
pub mod cluster;
/// Global configuration constants: limits, defaults, and tuning parameters.
pub mod config;
/// Core document types: `Document` struct and `MetadataValue` enum.
pub mod document;
/// HNSW approximate nearest neighbor index: graph structure, search, insertion, and distance metrics.
pub mod hnsw;
/// Scalar quantization: f32 → u8 compression with per-vector min/scale calibration.
pub mod quantization;
/// Search primitives: scored results, metadata filtering, and hybrid fusion strategies.
pub mod search;
/// Storage layer: collections, database, write-ahead log, and disk persistence.
pub mod storage;
