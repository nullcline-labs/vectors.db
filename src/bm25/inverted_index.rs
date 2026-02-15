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
