//! HNSW insertion algorithm.
//!
//! Inserts a vector into the HNSW graph with bidirectional connections and
//! heuristic neighbor pruning (Algorithm 4 from the HNSW paper).
//! Uses asymmetric f32-vs-u8 distance during construction (no raw_vectors stored).

use crate::hnsw::graph::HnswIndex;
use crate::hnsw::search::search_layer;
use crate::hnsw::visited::VisitedSet;
use crate::quantization::QuantizedVector;

impl HnswIndex {
    /// Insert a new vector into the HNSW index (SoA layout).
    /// Uses asymmetric f32-vs-u8 distance for graph construction.
    /// Only the quantized vector is stored (no raw f32 vector).
    /// internal_id must equal node_count before this call.
    pub fn insert(&mut self, internal_id: u32, raw_vector: &[f32], vector: QuantizedVector) {
        let level = self.random_level();

        // First node — push SoA fields and return
        if self.entry_point.is_none() {
            self.push_vector(&vector);
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
        let mut current_ep = entry_point;

        // Precompute query norm squared for this raw_vector
        let query_norm_sq: f32 = raw_vector.iter().map(|&x| x * x).sum();

        // Allocate VisitedSet once, reuse across all search_layer calls
        let mut visited = VisitedSet::new(self.node_count as usize);

        // Phase 1: Greedily traverse from top layer down to node's level + 1
        // Uses asymmetric f32-vs-u8 distance (query is the new node's raw f32 vector)
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
        // We collect all neighbor lists first, then push the node.
        let top = level.min(self.max_layer);
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

        // Push the new node's SoA fields
        self.push_vector(&vector);
        if let Some(ref cb) = self.pq_codebook {
            let codes = cb.encode(raw_vector);
            self.pq_codes.extend_from_slice(&codes);
        }
        self.neighbors.push(node_neighbors);
        self.layers.push(level as u8);
        self.deleted.push(false);
        self.node_count += 1;

        // Phase 3: Add bidirectional connections and prune over-capacity neighbors
        // Dequantizes base node to f32, then uses asymmetric f32-vs-u8 for pruning
        let metric = self.config.distance_metric;
        let dim = self.dimension;
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

                // Prune if over capacity — dequantize base, asymmetric f32-vs-u8
                if self.neighbors[nid][layer].len() > m_max {
                    self.dequantize_into(neighbor_id, &mut base_buf);
                    let neighbor_ids: Vec<u32> = self.neighbors[nid][layer].clone();
                    let candidates: Vec<(f32, u32)> = neighbor_ids
                        .iter()
                        .map(|&cid| {
                            let cid_ref = self.get_vector_ref(cid);
                            let dist = metric.distance_asym(&base_buf, cid_ref);
                            (dist, cid)
                        })
                        .collect();
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
/// Dequantizes each candidate to f32, then uses asymmetric f32-vs-u8 for pairwise distances.
fn select_neighbors_heuristic(
    index: &HnswIndex,
    candidates: &[(f32, u32)],
    m: usize,
) -> Vec<(f32, u32)> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<(f32, u32)> = Vec::with_capacity(m);
    let metric = index.config.distance_metric;
    let mut cid_buf = vec![0.0f32; index.dimension];

    for &(dist_to_base, cid) in &sorted {
        if selected.len() >= m {
            break;
        }

        // Dequantize candidate once, then compare against all selected neighbors
        index.dequantize_into(cid, &mut cid_buf);
        let is_diverse = selected.iter().all(|&(_, sid)| {
            let sid_ref = index.get_vector_ref(sid);
            let dist_to_selected = metric.distance_asym(&cid_buf, sid_ref);
            dist_to_base <= dist_to_selected
        });

        if is_diverse {
            selected.push((dist_to_base, cid));
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
