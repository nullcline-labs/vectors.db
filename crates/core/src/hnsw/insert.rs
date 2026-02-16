//! HNSW insertion algorithm.
//!
//! Inserts a vector into the HNSW graph with bidirectional connections and
//! heuristic neighbor pruning (Algorithm 4 from the HNSW paper).
//! Uses exact f32-vs-f32 distance when raw vectors are stored,
//! or cached dequantization + f32-vs-f32 distance in compact mode.

use crate::hnsw::graph::HnswIndex;
use crate::hnsw::search::search_layer;
use crate::hnsw::visited::VisitedSet;
use crate::quantization::QuantizedVector;
use std::cell::RefCell;

thread_local! {
    /// Thread-local VisitedSet for insert operations.
    /// Eliminates per-insert allocation (~2MB for 1M-node index).
    static INSERT_VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
}

impl HnswIndex {
    /// Insert a new vector into the HNSW index (SoA layout).
    /// Uses exact f32 distance when store_raw_vectors=true, asymmetric f32-vs-u8 otherwise.
    /// internal_id must equal node_count before this call.
    pub fn insert(&mut self, internal_id: u32, raw_vector: &[f32], vector: QuantizedVector) {
        let level = self.random_level();

        // First node — push SoA fields and return
        if self.entry_point.is_none() {
            self.push_vector(&vector);
            if self.config.store_raw_vectors {
                self.push_raw_vector(raw_vector);
            }
            if let Some(ref cb) = self.pq_codebook {
                let codes = cb.encode(raw_vector);
                self.pq_codes.extend_from_slice(&codes);
            }
            let mut layer_neighbors = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                layer_neighbors.push(Vec::new());
            }
            self.neighbors.push(layer_neighbors);
            self.layers.push(level as u8);
            self.deleted.push(false);
            self.node_count += 1;
            self.entry_point = Some(internal_id);
            self.max_layer = level;
            return;
        }

        let entry_point = self
            .entry_point
            .expect("entry_point is Some after is_none() guard");

        // Precompute query norm squared for this raw_vector
        let query_norm_sq: f32 = raw_vector.iter().map(|&x| x * x).sum();
        let top = level.min(self.max_layer);

        // Phases 1 & 2: search_layer borrows &self, so visited must be external.
        // Use thread-local VisitedSet to avoid per-insert allocation.
        let node_neighbors = INSERT_VISITED.with(|cell| {
            let mut visited = cell.borrow_mut();
            visited.ensure_capacity(self.node_count as usize);

            let mut current_ep = entry_point;

            // Phase 1: Greedily traverse from top layer down to node's level + 1
            let no_filter = |_: u32| true;
            for layer in (level + 1..=self.max_layer).rev() {
                let results = search_layer(
                    self,
                    raw_vector,
                    std::slice::from_ref(&current_ep),
                    1,
                    layer,
                    &mut visited,
                    query_norm_sq,
                    &no_filter,
                    None,
                );
                if let Some(&(_, nearest)) = results.first() {
                    current_ep = nearest;
                }
            }

            // Phase 2: Search each layer and collect neighbors for the new node.
            let mut node_neighbors: Vec<Vec<u32>> = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                node_neighbors.push(Vec::new());
            }

            let mut layer_eps: Vec<u32> = vec![current_ep];
            for layer in (0..=top).rev() {
                let ef = self.config.ef_construction;
                let candidates = search_layer(
                    self,
                    raw_vector,
                    &layer_eps,
                    ef,
                    layer,
                    &mut visited,
                    query_norm_sq,
                    &no_filter,
                    None,
                );

                let m_max = if layer == 0 {
                    self.config.m_max0
                } else {
                    self.config.m
                };

                let selected = select_neighbors_heuristic(self, &candidates, m_max);
                node_neighbors[layer] = selected.iter().map(|&(_, id)| id).collect();

                // Update entry points for next (lower) layer
                layer_eps.clear();
                layer_eps.extend(candidates.iter().map(|&(_, id)| id));
                if layer_eps.is_empty() {
                    layer_eps.push(entry_point);
                }
            }

            node_neighbors
        });

        // Push the new node's SoA fields
        self.push_vector(&vector);
        if self.config.store_raw_vectors {
            self.push_raw_vector(raw_vector);
        }
        if let Some(ref cb) = self.pq_codebook {
            let codes = cb.encode(raw_vector);
            self.pq_codes.extend_from_slice(&codes);
        }
        self.neighbors.push(node_neighbors);
        self.layers.push(level as u8);
        self.deleted.push(false);
        self.node_count += 1;

        // Phase 3: Add bidirectional connections and prune over-capacity neighbors
        let metric = self.config.distance_metric;
        let dim = self.dimension;
        let use_exact = self.has_raw_vectors();
        let mut base_buf = vec![0.0f32; dim];
        for layer in 0..=top {
            let m_max = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            let my_neighbors: Vec<u32> = self.neighbors[internal_id as usize][layer].clone();
            for &neighbor_id in &my_neighbors {
                let nid = neighbor_id as usize;

                // Ensure neighbor has enough layer vecs
                while self.neighbors[nid].len() <= layer {
                    self.neighbors[nid].push(Vec::new());
                }
                self.neighbors[nid][layer].push(internal_id);

                // Prune if over capacity
                if self.neighbors[nid][layer].len() > m_max {
                    let neighbor_ids: Vec<u32> = self.neighbors[nid][layer].clone();
                    let candidates: Vec<(f32, u32)> = if use_exact {
                        neighbor_ids
                            .iter()
                            .map(|&cid| {
                                let dist = metric.distance_exact(
                                    self.get_raw_vector(neighbor_id),
                                    self.get_raw_vector(cid),
                                );
                                (dist, cid)
                            })
                            .collect()
                    } else {
                        self.dequantize_into(neighbor_id, &mut base_buf);
                        neighbor_ids
                            .iter()
                            .map(|&cid| {
                                let cid_ref = self.get_vector_ref(cid);
                                let dist = metric.distance_asym(&base_buf, cid_ref);
                                (dist, cid)
                            })
                            .collect()
                    };
                    let pruned = select_neighbors_heuristic(self, &candidates, m_max);
                    self.neighbors[nid][layer] = pruned.iter().map(|&(_, id)| id).collect();
                }
            }
        }

        // Update entry point if new node has higher layer
        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(internal_id);
        }
    }
}

/// Heuristic neighbor selection (Algorithm 4 from the HNSW paper).
/// Prefers diverse neighbors: a candidate is selected only if it is closer to the base node
/// than to any already-selected neighbor. This avoids redundant clusters of near-identical
/// neighbors and ensures better graph connectivity, especially for cosine distance.
/// Uses exact f32-vs-f32 when raw vectors are available, otherwise dequantizes candidates
/// on-the-fly and caches selected vectors for f32-vs-f32 SIMD distance.
fn select_neighbors_heuristic(
    index: &HnswIndex,
    candidates: &[(f32, u32)],
    m: usize,
) -> Vec<(f32, u32)> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<(f32, u32)> = Vec::with_capacity(m);
    let metric = index.config.distance_metric;
    let use_exact = index.has_raw_vectors();

    // In compact mode, cache dequantized f32 vectors of selected neighbors
    // to use fast f32-vs-f32 SIMD distance instead of repeated u8 asymmetric distance.
    let dim = index.dimension;
    let mut selected_bufs: Vec<Vec<f32>> = if use_exact {
        Vec::new()
    } else {
        Vec::with_capacity(m)
    };
    let mut cid_buf = vec![0.0f32; dim];

    for &(dist_to_base, cid) in &sorted {
        if selected.len() >= m {
            break;
        }

        let is_diverse = if use_exact {
            let cid_raw = index.get_raw_vector(cid);
            selected.iter().all(|&(_, sid)| {
                let sid_raw = index.get_raw_vector(sid);
                let dist_to_selected = metric.distance_exact(cid_raw, sid_raw);
                dist_to_base <= dist_to_selected
            })
        } else {
            index.dequantize_into(cid, &mut cid_buf);
            selected_bufs.iter().all(|sid_f32| {
                let dist_to_selected = metric.distance_exact(&cid_buf, sid_f32);
                dist_to_base <= dist_to_selected
            })
        };

        if is_diverse {
            selected.push((dist_to_base, cid));
            if !use_exact {
                selected_bufs.push(cid_buf.clone());
            }
        }
    }

    // If heuristic didn't fill M slots, fill remaining with closest unused candidates
    if selected.len() < m {
        let selected_ids: std::collections::HashSet<u32> =
            selected.iter().map(|&(_, id)| id).collect();
        for &(dist, cid) in &sorted {
            if selected.len() >= m {
                break;
            }
            if !selected_ids.contains(&cid) {
                selected.push((dist, cid));
            }
        }
    }

    selected
}
