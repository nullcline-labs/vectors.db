//! BM25 Okapi scoring engine.
//!
//! Scores documents against a query using the BM25 formula with configurable
//! `k1` and `b` parameters (see [`crate::config`]).

use crate::bm25::inverted_index::InvertedIndex;
use crate::bm25::tokenizer::tokenize;
use crate::config;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// BM25 Okapi scoring for a query against the inverted index.
/// Returns scored documents (internal_id, score) sorted by descending score.
pub fn bm25_search(index: &InvertedIndex, query: &str, k: usize) -> Vec<(u32, f32)> {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() || index.doc_count == 0 {
        return Vec::new();
    }

    let avgdl = index.average_doc_length();
    let n = index.doc_count as f32;
    let k1 = config::BM25_K1;
    let b = config::BM25_B;

    let mut scores: HashMap<u32, f32> = HashMap::with_capacity(256.min(index.doc_count as usize));

    for token in query_tokens.iter() {
        if let Some(postings) = index.index.get(token) {
            let df = postings.len() as f32;
            // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for posting in postings {
                let dl = if (posting.doc_id as usize) < index.doc_lengths.len() {
                    index.doc_lengths[posting.doc_id as usize] as f32
                } else {
                    0.0
                };
                let tf = posting.term_frequency as f32;

                // BM25 score for this term-document pair
                let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl));
                let score = idf * tf_norm;

                *scores.entry(posting.doc_id).or_insert(0.0) += score;
            }
        }
    }

    // Partial sort: O(n log k) via min-heap of size k
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>> = BinaryHeap::with_capacity(k + 1);
    for (id, score) in scores {
        heap.push(Reverse((OrderedFloat(score), id)));
        if heap.len() > k {
            heap.pop();
        }
    }
    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((s, id))| (id, s.0))
        .collect();
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}
