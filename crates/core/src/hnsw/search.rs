//! HNSW search algorithms: single-layer search and multi-layer KNN.
//!
//! Supports optional pre-filtering via a predicate `Fn(u32) -> bool` applied during
//! graph traversal. Filtered nodes are still used for navigation but excluded from results.
//! Uses asymmetric f32-vs-u8 quantized distance (or PQ ADC when enabled).

use crate::hnsw::graph::HnswIndex;
use crate::hnsw::visited::VisitedSet;
use crate::quantization::pq::PqDistanceTable;
use ordered_float::OrderedFloat;
use std::cell::RefCell;
use std::collections::BinaryHeap;

thread_local! {
    /// Thread-local VisitedSet pool for search operations.
    /// Eliminates per-query 2MB allocations by reusing across searches on the same thread.
    static SEARCH_VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
}

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
/// When PQ table is available, uses M table lookups (fastest approximate).
/// When raw f32 vectors are stored, uses exact f32-vs-f32 distance.
/// Otherwise, uses f32-vs-u8 asymmetric distance (SIMD-optimized).
#[inline]
fn compute_distance(
    index: &HnswIndex,
    query: &[f32],
    node_id: u32,
    query_norm_sq: f32,
    pq_table: Option<&PqDistanceTable>,
) -> f32 {
    if let Some(table) = pq_table {
        let codes = index.get_pq_codes(node_id);
        table.distance(codes)
    } else if index.has_raw_vectors() {
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
#[allow(clippy::too_many_arguments)]
pub fn search_layer<F: Fn(u32) -> bool>(
    index: &HnswIndex,
    query: &[f32],
    entry_points: &[u32],
    ef: usize,
    layer: usize,
    visited: &mut VisitedSet,
    query_norm_sq: f32,
    filter_fn: &F,
    pq_table: Option<&PqDistanceTable>,
) -> Vec<(f32, u32)> {
    visited.clear();
    let estimated_visits = ef * 2;
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(estimated_visits);
    let mut results: BinaryHeap<ResultEntry> = BinaryHeap::with_capacity(ef + 1);
    // Cached worst distance — avoids repeated heap peeks in the hot loop
    let mut worst_dist = f32::MAX;

    for &ep in entry_points {
        if visited.insert(ep) {
            let dist = compute_distance(index, query, ep, query_norm_sq, pq_table);
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

            // Prefetch next neighbor's data while processing current
            if i + 1 < neighbor_list.len() {
                let next_id = neighbor_list[i + 1];
                if pq_table.is_some() {
                    index.prefetch_pq_codes(next_id);
                } else if index.has_raw_vectors() {
                    index.prefetch_raw_vector(next_id);
                } else {
                    index.prefetch_vector(next_id);
                }
            }

            if !visited.insert(neighbor_id) {
                continue;
            }

            let dist = compute_distance(index, query, neighbor_id, query_norm_sq, pq_table);

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
/// Uses asymmetric f32-vs-u8 distance (or PQ ADC), with asymmetric reranking.
pub fn knn_search(index: &HnswIndex, query: &[f32], k: usize) -> Vec<(f32, u32)> {
    knn_search_filtered(index, query, k, &|_: u32| true)
}

/// Multi-layer KNN search with a filter predicate applied during graph traversal.
/// Nodes that don't pass the filter are still used for navigation but excluded from results.
/// Uses adaptive ef oversampling: if the initial search yields fewer than k results,
/// retries with progressively larger ef to handle low-selectivity filters.
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

    SEARCH_VISITED.with(|cell| {
        let mut visited = cell.borrow_mut();
        visited.ensure_capacity(index.node_count as usize);

        // Precompute query norm squared (constant for this query)
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();

        // Build PQ distance table if codebook is available
        let pq_table = index
            .pq_codebook
            .as_ref()
            .map(|cb| cb.build_distance_table(query, index.config.distance_metric));
        let pq_table_ref = pq_table.as_ref();

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
                &mut *visited,
                query_norm_sq,
                &no_filter,
                pq_table_ref,
            );
            if let Some(&(_, nearest)) = results.first() {
                current_ep = nearest;
            }
        }

        // Search layer 0 with ef_search — apply filter here.
        // Adaptive oversampling: if results < k, retry with larger ef (up to 4x).
        let base_ef = index.config.ef_search.max(k);
        let max_ef = (base_ef * 4).min(index.node_count as usize);
        let mut ef = base_ef;
        let mut results;

        loop {
            // search_layer calls visited.clear() at its start — no reallocation needed
            results = search_layer(
                index,
                query,
                std::slice::from_ref(&current_ep),
                ef,
                0,
                &mut *visited,
                query_norm_sq,
                filter_fn,
                pq_table_ref,
            );

            if results.len() >= k || ef >= max_ef {
                break;
            }

            // Double ef for next attempt
            ef = (ef * 2).min(max_ef);
        }

        // Rerank candidates using exact f32 distance (when available) or asymmetric f32-vs-u8
        if index.has_raw_vectors() {
            for i in 0..results.len() {
                if i + 1 < results.len() {
                    index.prefetch_raw_vector(results[i + 1].1);
                }
                let raw = index.get_raw_vector(results[i].1);
                results[i].0 = index.config.distance_metric.distance_exact(query, raw);
            }
        } else {
            for i in 0..results.len() {
                if i + 1 < results.len() {
                    index.prefetch_vector(results[i + 1].1);
                }
                let vref = index.get_vector_ref(results[i].1);
                results[i].0 =
                    index
                        .config
                        .distance_metric
                        .distance_asym_prenorm(query, vref, query_norm_sq);
            }
        }
        results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        results.truncate(k);
        results
    })
}
