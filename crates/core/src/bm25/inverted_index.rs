//! Inverted index for BM25 full-text search.
//!
//! Maps terms to postings lists (document ID + term frequency). Documents are
//! identified by internal u32 IDs for memory efficiency.

use crate::bm25::tokenizer::tokenize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in a term's postings list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posting {
    /// Internal u32 document ID.
    pub doc_id: u32,
    /// Number of times the term appears in this document.
    pub term_frequency: u32,
}

/// Inverted index mapping terms to postings lists.
///
/// Supports incremental document addition and removal. Document lengths
/// are tracked for BM25 length normalization.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InvertedIndex {
    /// term → list of postings
    pub index: HashMap<String, Vec<Posting>>,
    /// internal_id → document length (number of tokens). Indexed by u32 internal ID.
    pub doc_lengths: Vec<u32>,
    /// Total number of documents indexed
    pub doc_count: u32,
    /// Sum of all document lengths (for average calculation)
    pub total_doc_length: u64,
}

impl InvertedIndex {
    /// Creates a new empty inverted index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a document's text using its internal u32 ID.
    pub fn add_document(&mut self, internal_id: u32, text: &str) {
        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;

        // Grow doc_lengths vec if needed
        let idx = internal_id as usize;
        if idx >= self.doc_lengths.len() {
            self.doc_lengths.resize(idx + 1, 0);
        }
        self.doc_lengths[idx] = doc_len;
        self.doc_count += 1;
        self.total_doc_length += doc_len as u64;

        // Count term frequencies for this doc
        let mut tf_map: HashMap<&str, u32> = HashMap::new();
        for token in tokens.iter() {
            *tf_map.entry(token).or_insert(0) += 1;
        }

        for (term, tf) in tf_map {
            self.index
                .entry(term.to_string())
                .or_default()
                .push(Posting {
                    doc_id: internal_id,
                    term_frequency: tf,
                });
        }
    }

    /// Remove a document from the index by internal ID.
    pub fn remove_document(&mut self, internal_id: u32) {
        let idx = internal_id as usize;
        if idx < self.doc_lengths.len() && self.doc_lengths[idx] > 0 {
            let doc_len = self.doc_lengths[idx];
            self.doc_lengths[idx] = 0;
            self.doc_count -= 1;
            self.total_doc_length -= doc_len as u64;

            // Remove postings for this document
            self.index.retain(|_, postings| {
                postings.retain(|p| p.doc_id != internal_id);
                !postings.is_empty()
            });
        }
    }

    /// Returns the average document length across all indexed documents.
    pub fn average_doc_length(&self) -> f32 {
        if self.doc_count == 0 {
            return 0.0;
        }
        self.total_doc_length as f32 / self.doc_count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_document_updates_index() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "the quick brown fox");
        assert_eq!(idx.doc_count, 1);
        assert!(idx.index.contains_key("quick"));
        assert!(idx.index.contains_key("brown"));
        assert!(idx.index.contains_key("fox"));
        // "the" is a stop word, should not be indexed
        assert!(!idx.index.contains_key("the"));
    }

    #[test]
    fn test_term_frequency() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "hello hello hello world");
        let postings = idx.index.get("hello").unwrap();
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].term_frequency, 3);
    }

    #[test]
    fn test_multiple_documents() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "rust programming language");
        idx.add_document(1, "python programming language");
        assert_eq!(idx.doc_count, 2);
        let postings = idx.index.get("programming").unwrap();
        assert_eq!(postings.len(), 2);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "hello world");
        idx.add_document(1, "hello rust");
        assert_eq!(idx.doc_count, 2);
        idx.remove_document(0);
        assert_eq!(idx.doc_count, 1);
        // "world" only appeared in doc 0, should be gone
        assert!(!idx.index.contains_key("world"));
        // "hello" still in doc 1
        let postings = idx.index.get("hello").unwrap();
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].doc_id, 1);
    }

    #[test]
    fn test_remove_nonexistent_doc() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "hello world");
        idx.remove_document(99); // should not panic
        assert_eq!(idx.doc_count, 1);
    }

    #[test]
    fn test_average_doc_length() {
        let mut idx = InvertedIndex::new();
        assert_eq!(idx.average_doc_length(), 0.0);
        idx.add_document(0, "one two three"); // 3 tokens (after stop word removal: depends on which are stop words)
        idx.add_document(1, "four five six seven eight"); // 5 tokens
        let avg = idx.average_doc_length();
        assert!(avg > 0.0, "average doc length should be > 0");
    }

    #[test]
    fn test_doc_lengths_tracked() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "hello world");
        assert!(idx.doc_lengths[0] > 0);
        idx.remove_document(0);
        assert_eq!(idx.doc_lengths[0], 0);
    }
}
