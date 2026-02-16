//! Scored document types for search results.

use crate::document::Document;
use std::sync::Arc;

/// A document with an associated relevance score from a search query.
///
/// Used as the return type for vector, keyword, and hybrid search methods.
/// The `score` semantics depend on the search type:
/// - **Vector search**: cosine similarity (higher = more similar)
/// - **Keyword search**: BM25 score (higher = more relevant)
/// - **Hybrid search**: fused score (RRF or linear combination)
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    /// The matched document (shared reference).
    pub document: Arc<Document>,
    /// Relevance score (interpretation depends on search type).
    pub score: f32,
}
