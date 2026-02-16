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
    let mut results: Vec<(u32, f32)> = heap.into_iter().map(|Reverse((s, id))| (id, s.0)).collect();
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::inverted_index::InvertedIndex;

    fn build_corpus() -> InvertedIndex {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "rust programming systems language fast");
        idx.add_document(1, "python programming scripting easy");
        idx.add_document(2, "java enterprise programming verbose");
        idx.add_document(3, "rust memory safety zero cost abstractions");
        idx
    }

    #[test]
    fn test_bm25_empty_query() {
        let idx = build_corpus();
        let results = bm25_search(&idx, "", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_empty_index() {
        let idx = InvertedIndex::new();
        let results = bm25_search(&idx, "rust", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_finds_matching_docs() {
        let idx = build_corpus();
        let results = bm25_search(&idx, "rust", 10);
        assert!(!results.is_empty());
        let ids: Vec<u32> = results.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&0), "doc 0 contains 'rust'");
        assert!(ids.contains(&3), "doc 3 contains 'rust'");
    }

    #[test]
    fn test_bm25_ranking_order() {
        let mut idx = InvertedIndex::new();
        idx.add_document(0, "rust rust rust"); // high TF for "rust"
        idx.add_document(1, "rust programming"); // lower TF
        let results = bm25_search(&idx, "rust", 10);
        assert!(results.len() >= 2);
        assert_eq!(results[0].0, 0, "doc with higher TF should rank first");
    }

    #[test]
    fn test_bm25_no_match() {
        let idx = build_corpus();
        let results = bm25_search(&idx, "nonexistent_xyz_term", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_k_truncation() {
        let idx = build_corpus();
        let results = bm25_search(&idx, "programming", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_bm25_scores_positive() {
        let idx = build_corpus();
        let results = bm25_search(&idx, "rust programming", 10);
        for &(_, score) in &results {
            assert!(score > 0.0, "BM25 scores should be positive, got {score}");
        }
    }
}
