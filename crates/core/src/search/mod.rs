//! Search primitives: scored results, metadata filtering, and hybrid fusion.
//!
//! This module provides the building blocks for combining vector search and
//! keyword search results, applying metadata filters, and representing scored documents.

/// Metadata filtering predicates for search queries.
pub mod filter;
/// Hybrid fusion strategies: Reciprocal Rank Fusion (RRF) and linear combination.
pub mod hybrid;
/// Scored document types.
pub mod types;

pub use hybrid::{linear_fusion, rrf_fusion};
pub use types::ScoredDocument;
