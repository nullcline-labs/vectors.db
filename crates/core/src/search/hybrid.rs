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
    let mut results: Vec<(u32, f32)> = heap.into_iter().map(|Reverse((s, id))| (id, s.0)).collect();
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
    let mut results: Vec<(u32, f32)> = heap.into_iter().map(|Reverse((s, id))| (id, s.0)).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_disjoint_lists() {
        let vec_results = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let kw_results = vec![(3, 5.0), (4, 4.0), (5, 3.0)];
        let fused = rrf_fusion(&vec_results, &kw_results, 6);
        assert_eq!(fused.len(), 6);
        let ids: Vec<u32> = fused.iter().map(|&(id, _)| id).collect();
        for id in 0..=5 {
            assert!(ids.contains(&id), "missing id {id} in fused results");
        }
    }

    #[test]
    fn test_rrf_overlapping_boosts_score() {
        let vec_results = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let kw_results = vec![(1, 5.0), (3, 4.0), (0, 3.0)];
        let fused = rrf_fusion(&vec_results, &kw_results, 4);
        // IDs 0 and 1 appear in both lists so should have higher RRF scores
        let top_ids: Vec<u32> = fused.iter().take(2).map(|&(id, _)| id).collect();
        assert!(
            top_ids.contains(&0) || top_ids.contains(&1),
            "overlapping IDs should rank higher"
        );
    }

    #[test]
    fn test_rrf_empty_inputs() {
        let fused = rrf_fusion(&[], &[], 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_one_empty() {
        let vec_results = vec![(0, 0.9), (1, 0.8)];
        let fused = rrf_fusion(&vec_results, &[], 10);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_linear_fusion_alpha_1() {
        let vec_results = vec![(0, 0.9), (1, 0.5)];
        let kw_results = vec![(2, 10.0), (3, 5.0)];
        let fused = linear_fusion(&vec_results, &kw_results, 1.0, 10);
        let top_id = fused[0].0;
        assert!(
            top_id == 0 || top_id == 1,
            "alpha=1.0 should prioritize vector results, got top={top_id}"
        );
    }

    #[test]
    fn test_linear_fusion_alpha_0() {
        let vec_results = vec![(0, 0.9), (1, 0.5)];
        let kw_results = vec![(2, 10.0), (3, 5.0)];
        let fused = linear_fusion(&vec_results, &kw_results, 0.0, 10);
        let top_id = fused[0].0;
        assert!(
            top_id == 2 || top_id == 3,
            "alpha=0.0 should prioritize keyword results, got top={top_id}"
        );
    }

    #[test]
    fn test_linear_fusion_truncates_to_k() {
        let vec_results: Vec<(u32, f32)> = (0..20).map(|i| (i, 1.0 - i as f32 / 20.0)).collect();
        let kw_results: Vec<(u32, f32)> = (20..40)
            .map(|i| (i, 1.0 - (i - 20) as f32 / 20.0))
            .collect();
        let fused = linear_fusion(&vec_results, &kw_results, 0.5, 5);
        assert_eq!(fused.len(), 5);
    }

    #[test]
    fn test_linear_fusion_empty() {
        let fused = linear_fusion(&[], &[], 0.5, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_min_max_helper() {
        assert_eq!(min_max(&[]), None);
        assert_eq!(min_max(&[(0, 3.0), (1, 1.0), (2, 5.0)]), Some((1.0, 5.0)));
        assert_eq!(min_max(&[(0, 2.0)]), Some((2.0, 2.0)));
    }
}
