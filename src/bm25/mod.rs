//! BM25 full-text search engine.
//!
//! Implements Okapi BM25 scoring with an inverted index for keyword search.
//! Documents are tokenized using a whitespace tokenizer with stop word removal
//! (English + Italian). No stemming is applied.

/// Inverted index data structure with postings lists.
pub mod inverted_index;
/// BM25 Okapi scoring and query execution.
pub mod scorer;
/// Whitespace tokenizer with stop word filtering.
pub mod tokenizer;

pub use inverted_index::InvertedIndex;
pub use scorer::bm25_search;
