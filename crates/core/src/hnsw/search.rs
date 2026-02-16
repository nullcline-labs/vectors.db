//! HNSW search algorithms: single-layer search and multi-layer KNN.
//!
//! Supports optional pre-filtering via a predicate `Fn(u32) -> bool` applied during
//! graph traversal. Filtered nodes are still used for navigation but excluded from results.
//! Uses asymmetric u8 quantized distance for fast navigation, then reranks with exact f32.

use crate::hnsw::graph::HnswIndex;
use crate::hnsw::visited::VisitedSet;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

/// A candidate during search: (negative distance, internal_id).
/// BinaryHeap is a max-heap; we use negative distance for min-heap behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    neg_distance: OrderedFloat<f32>,
    id: u32,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_distance.cmp(&other.neg_distance)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A result entry: (distance, internal_id). Max-heap by distance for pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResultEntry {
    distance: OrderedFloat<f32>,
    id: u32,
}

impl Ord for ResultEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for ResultEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute distance from query to a node in the index.
/// When `exact` is true, uses f32-vs-f32 raw vectors (no quantization loss).
/// When `exact` is false, uses f32-vs-u8 asymmetric distance (faster, approximate).
#[inline]
fn compute_distance(
    index: &HnswIndex,
    query: &[f32],
    node_id: u32,
    query_norm_sq: f32,
    exact: bool,
) -> f32 {
    if exact {
        let raw = index.get_raw_vector(node_id);
        index.config.distance_metric.distance_exact(query, raw)
    } else {
        let vref = index.get_vector_ref(node_id);
        index
            .config
            .distance_metric
            .distance_asym_prenorm(query, vref, query_norm_sq)
    }
}

/// Search a single layer of the HNSW graph.
/// Returns ef closest non-deleted nodes to the query at the given layer.
/// `visited` is a reusable VisitedSet (cleared at the start of each call).
/// `query_norm_sq` is the precomputed sum of squares of the query vector.
/// `exact`: when true uses f32 raw vectors for distance (higher quality, for construction);
///          when false uses u8 quantized asymmetric distance (faster, for search).
#[allow(clippy::too_many_arguments)]
pub fn search_layer<F: Fn(u32) -> bool>(
    index: &HnswIndex,
    query: &[f32],
    entry_points: &[u32],
    ef: usize,
    layer: usize,
    visited: &mut VisitedSet,
    query_norm_sq: f32,
    exact: bool,
    filter_fn: &F,
) -> Vec<(f32, u32)> {
    visited.clear();
    let estimated_visits = ef * 2;
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(estimated_visits);
    let mut results: BinaryHeap<ResultEntry> = BinaryHeap::with_capacity(ef + 1);
    // Cached worst distance — avoids repeated heap peeks in the hot loop
    let mut worst_dist = f32::MAX;

    for &ep in entry_points {
        if visited.insert(ep) {
            let dist = compute_distance(index, query, ep, query_norm_sq, exact);
            candidates.push(Candidate {
                neg_distance: OrderedFloat(-dist),
                id: ep,
            });
            if !index.is_deleted(ep) && filter_fn(ep) {
                results.push(ResultEntry {
                    distance: OrderedFloat(dist),
                    id: ep,
                });
                if results.len() >= ef {
                    worst_dist = results.peek().map_or(f32::MAX, |r| r.distance.0);
                }
            }
        }
    }

    while let Some(candidate) = candidates.pop() {
        let c_dist = -candidate.neg_distance.0;

        // If the closest candidate is farther than the worst result, stop
        if results.len() >= ef && c_dist > worst_dist {
            break;
        }

        let node_id = candidate.id as usize;
        if layer >= index.neighbors[node_id].len() {
            continue;
        }

        let neighbor_list = &index.neighbors[node_id][layer];
        for i in 0..neighbor_list.len() {
            let neighbor_id = neighbor_list[i];

            // Prefetch next neighbor's vector data while processing current
            if i + 1 < neighbor_list.len() {
                let next_id = neighbor_list[i + 1];
                if exact {
                    index.prefetch_raw_vector(next_id);
                } else {
                    index.prefetch_vector(next_id);
                }
            }

            if !visited.insert(neighbor_id) {
                continue;
            }

            let dist = compute_distance(index, query, neighbor_id, query_norm_sq, exact);

            let should_add = results.len() < ef || dist < worst_dist;

            if should_add {
                candidates.push(Candidate {
                    neg_distance: OrderedFloat(-dist),
                    id: neighbor_id,
                });
                if !index.is_deleted(neighbor_id) && filter_fn(neighbor_id) {
                    results.push(ResultEntry {
                        distance: OrderedFloat(dist),
                        id: neighbor_id,
                    });
                    if results.len() > ef {
                        results.pop(); // remove worst
                    }
                    // Update cached worst distance
                    worst_dist = results.peek().map_or(f32::MAX, |r| r.distance.0);
                }
            }
        }
    }

    results
        .into_sorted_vec()
        .into_iter()
        .map(|r| (r.distance.0, r.id))
        .collect()
}

/// Multi-layer KNN search through the HNSW graph.
/// Uses u8 quantized distance for fast navigation, then reranks with exact f32 distances.
pub fn knn_search(index: &HnswIndex, query: &[f32], k: usize) -> Vec<(f32, u32)> {
    knn_search_filtered(index, query, k, &|_: u32| true)
}

/// Multi-layer KNN search with a filter predicate applied during graph traversal.
/// Nodes that don't pass the filter are still used for navigation but excluded from results.
pub fn knn_search_filtered<F: Fn(u32) -> bool>(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
    filter_fn: &F,
) -> Vec<(f32, u32)> {
    let entry_point = match index.entry_point {
        Some(ep) => ep,
        None => return Vec::new(),
    };

    // Precompute query norm squared (constant for this query)
    let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();

    // Allocate VisitedSet once, reuse across all layers
    let mut visited = VisitedSet::new(index.node_count as usize);

    let mut current_ep = entry_point;

    // Traverse from top layer down to layer 1, using ef=1 (quantized, fast)
    // Upper layers use no-op filter — filtering only matters at layer 0
    let no_filter = |_: u32| true;
    for layer in (1..=index.max_layer).rev() {
        let results = search_layer(
            index,
            query,
            std::slice::from_ref(&current_ep),
            1,
            layer,
            &mut visited,
            query_norm_sq,
            false,
            &no_filter,
        );
        if let Some(&(_, nearest)) = results.first() {
            current_ep = nearest;
        }
    }

    // Search layer 0 with ef_search (quantized, fast) — apply filter here
    let ef = index.config.ef_search.max(k);
    let mut results = search_layer(
        index,
        query,
        std::slice::from_ref(&current_ep),
        ef,
        0,
        &mut visited,
        query_norm_sq,
        false,
        filter_fn,
    );

    // Rerank all candidates using exact f32 distances (no quantization loss)
    for i in 0..results.len() {
        if i + 1 < results.len() {
            index.prefetch_raw_vector(results[i + 1].1);
        }
        let raw = index.get_raw_vector(results[i].1);
        results[i].0 = index.config.distance_metric.distance_exact(query, raw);
    }
    results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    results.truncate(k);
    results
}
