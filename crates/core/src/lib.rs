//! # vectorsdb-core
//!
//! Embeddable in-memory vector database engine with HNSW approximate nearest
//! neighbor search, BM25 full-text search, and hybrid retrieval.
//!
//! This is the core library crate with zero async dependencies — suitable for
//! embedding directly in Rust, Python (via PyO3), or other language bindings.

/// BM25 full-text search: inverted index, Okapi BM25 scoring, and whitespace tokenizer.
pub mod bm25;
/// Global configuration constants: limits, defaults, and tuning parameters.
pub mod config;
/// Core document types: `Document` struct and `MetadataValue` enum.
pub mod document;
/// Filter types used by search and storage layers.
pub mod filter_types;
/// HNSW approximate nearest neighbor index: graph structure, search, insertion, and distance metrics.
pub mod hnsw;
/// Scalar quantization: f32 → u8 compression with per-vector min/scale calibration.
pub mod quantization;
/// Search primitives: scored results, metadata filtering, and hybrid fusion strategies.
pub mod search;
/// Storage layer: collections, database, write-ahead log, and disk persistence.
pub mod storage;
