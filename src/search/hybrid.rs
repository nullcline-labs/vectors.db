//! Hybrid search fusion strategies.
//!
//! Combines vector search and keyword search results into a single ranked list.
//! Two strategies are available:
//! - **RRF** (Reciprocal Rank Fusion): rank-based, parameter-free combination
//! - **Linear**: score-based combination with min-max normalization and alpha weighting

use crate::config;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Reciprocal Rank Fusion: combines ranked lists.
/// score(d) = sum(1 / (k + rank_i(d)))
pub fn rrf_fusion(
    vector_results: &[(u32, f32)],
    keyword_results: &[(u32, f32)],
    k: usize,
) -> Vec<(u32, f32)> {
    let rrf_k = config::RRF_K;
    let mut scores: HashMap<u32, f32> =
        HashMap::with_capacity(vector_results.len() + keyword_results.len());

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
    }

    for (rank, (id, _)) in keyword_results.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
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

/// Linear combination with min-max normalization.
/// score(d) = alpha * norm_vector(d) + (1 - alpha) * norm_keyword(d)
pub fn linear_fusion(
    vector_results: &[(u32, f32)],
    keyword_results: &[(u32, f32)],
    alpha: f32,
    k: usize,
) -> Vec<(u32, f32)> {
    let mut scores: HashMap<u32, f32> =
        HashMap::with_capacity(vector_results.len() + keyword_results.len());

    // Normalize and accumulate vector scores inline (avoids intermediate Vec)
    if let Some((min_v, max_v)) = min_max(vector_results) {
        let range = max_v - min_v;
        for &(id, score) in vector_results {
            let norm = if range < f32::EPSILON {
                1.0
            } else {
                (score - min_v) / range
            };
            *scores.entry(id).or_insert(0.0) += alpha * norm;
        }
    }

    // Normalize and accumulate keyword scores inline
    if let Some((min_k, max_k)) = min_max(keyword_results) {
        let range = max_k - min_k;
        for &(id, score) in keyword_results {
            let norm = if range < f32::EPSILON {
                1.0
            } else {
                (score - min_k) / range
            };
            *scores.entry(id).or_insert(0.0) += (1.0 - alpha) * norm;
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

/// Single-pass min/max computation.
fn min_max(results: &[(u32, f32)]) -> Option<(f32, f32)> {
    if results.is_empty() {
        return None;
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &(_, s) in results {
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }
    Some((min, max))
}
