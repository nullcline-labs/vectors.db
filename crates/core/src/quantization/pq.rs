//! Product Quantization (PQ) for fast approximate distance computation.
//!
//! Splits vectors into M subspaces and learns K=256 centroids per subspace
//! via k-means. Each vector is encoded as M bytes (one centroid ID per subspace).
//! Distance computation uses a precomputed lookup table: M table lookups + M additions
//! instead of D multiply-adds.

use crate::config;
use crate::hnsw::distance::DistanceMetric;
use serde::{Deserialize, Serialize};

/// PQ codebook: M subspaces × K centroids × sub_dim floats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    pub num_subspaces: usize,
    pub num_centroids: usize,
    pub sub_dim: usize,
    /// Flat centroid array: centroids[m * K * sub_dim + k * sub_dim .. + sub_dim]
    pub centroids: Vec<f32>,
}

/// Precomputed distance table for a single query: [M][K] partial distances.
pub struct PqDistanceTable {
    pub table: Vec<f32>,
    pub num_subspaces: usize,
}

impl PqCodebook {
    /// Train a PQ codebook on a set of raw f32 vectors.
    ///
    /// `vectors` is a contiguous arena of n vectors, each `dim` floats.
    /// `m` is the number of subspaces, `k` is the number of centroids (must be 256).
    pub fn train(vectors: &[f32], dim: usize, m: usize, k: usize) -> Self {
        assert!(
            dim.is_multiple_of(m),
            "dimension must be divisible by num_subspaces"
        );
        assert!(k == 256, "PQ requires exactly 256 centroids for u8 codes");
        let sub_dim = dim / m;
        let n = vectors.len() / dim;
        assert!(n > 0, "need at least one vector to train PQ");

        let mut centroids = vec![0.0f32; m * k * sub_dim];

        for sub in 0..m {
            // Extract sub-vectors for this subspace
            let mut sub_vectors = vec![0.0f32; n * sub_dim];
            for i in 0..n {
                let src_start = i * dim + sub * sub_dim;
                let dst_start = i * sub_dim;
                sub_vectors[dst_start..dst_start + sub_dim]
                    .copy_from_slice(&vectors[src_start..src_start + sub_dim]);
            }

            // Run k-means
            let effective_k = k.min(n);
            let sub_centroids = kmeans(&sub_vectors, sub_dim, effective_k);

            // Copy centroids (pad with zeros if n < k)
            let out_start = sub * k * sub_dim;
            let copy_len = effective_k * sub_dim;
            centroids[out_start..out_start + copy_len].copy_from_slice(&sub_centroids[..copy_len]);
        }

        Self {
            num_subspaces: m,
            num_centroids: k,
            sub_dim,
            centroids,
        }
    }

    /// Encode a single vector into M PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subspaces);
        for sub in 0..self.num_subspaces {
            let sub_vec = &vector[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            codes.push(self.find_nearest_centroid(sub, sub_vec));
        }
        codes
    }

    /// Encode all vectors in a contiguous arena. Returns n×M bytes.
    pub fn encode_batch(&self, vectors: &[f32], dim: usize) -> Vec<u8> {
        let n = vectors.len() / dim;
        let mut codes = Vec::with_capacity(n * self.num_subspaces);
        for i in 0..n {
            let vec = &vectors[i * dim..(i + 1) * dim];
            for sub in 0..self.num_subspaces {
                let sub_vec = &vec[sub * self.sub_dim..(sub + 1) * self.sub_dim];
                codes.push(self.find_nearest_centroid(sub, sub_vec));
            }
        }
        codes
    }

    /// Build distance lookup table for a query vector and distance metric.
    /// Returns a table of shape [M][K] where table[m*K + k] is the partial
    /// distance from query subvector m to centroid k.
    pub fn build_distance_table(&self, query: &[f32], metric: DistanceMetric) -> PqDistanceTable {
        let k = self.num_centroids;
        let mut table = vec![0.0f32; self.num_subspaces * k];

        for sub in 0..self.num_subspaces {
            let q_start = sub * self.sub_dim;
            let q_sub = &query[q_start..q_start + self.sub_dim];
            let table_start = sub * k;

            for ci in 0..k {
                let c_start = sub * k * self.sub_dim + ci * self.sub_dim;
                let centroid = &self.centroids[c_start..c_start + self.sub_dim];

                table[table_start + ci] = match metric {
                    DistanceMetric::Euclidean => {
                        let mut sum = 0.0f32;
                        for d in 0..self.sub_dim {
                            let diff = q_sub[d] - centroid[d];
                            sum += diff * diff;
                        }
                        sum
                    }
                    DistanceMetric::DotProduct => {
                        let mut sum = 0.0f32;
                        for d in 0..self.sub_dim {
                            sum += q_sub[d] * centroid[d];
                        }
                        -sum
                    }
                    DistanceMetric::Cosine => {
                        // Use dot product as proxy; reranking with exact f32 fixes final order
                        let mut sum = 0.0f32;
                        for d in 0..self.sub_dim {
                            sum += q_sub[d] * centroid[d];
                        }
                        -sum
                    }
                };
            }
        }

        PqDistanceTable {
            table,
            num_subspaces: self.num_subspaces,
        }
    }

    /// Find nearest centroid in a subspace. Returns centroid index (0-255).
    #[inline]
    fn find_nearest_centroid(&self, subspace: usize, sub_vec: &[f32]) -> u8 {
        let k = self.num_centroids;
        let base = subspace * k * self.sub_dim;
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for ci in 0..k {
            let c_start = base + ci * self.sub_dim;
            let centroid = &self.centroids[c_start..c_start + self.sub_dim];
            let mut dist = 0.0f32;
            for d in 0..self.sub_dim {
                let diff = sub_vec[d] - centroid[d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = ci as u8;
            }
        }
        best_idx
    }
}

impl PqDistanceTable {
    /// Compute approximate distance for a PQ-encoded vector.
    /// `codes` is M bytes, one per subspace.
    #[inline]
    pub fn distance(&self, codes: &[u8]) -> f32 {
        let k = 256;
        let mut dist = 0.0f32;
        for m in 0..self.num_subspaces {
            dist += unsafe {
                *self
                    .table
                    .get_unchecked(m * k + *codes.get_unchecked(m) as usize)
            };
        }
        dist
    }
}

/// K-means clustering with k-means++ initialization.
/// Returns k × sub_dim centroids as flat Vec<f32>.
fn kmeans(data: &[f32], sub_dim: usize, k: usize) -> Vec<f32> {
    let n = data.len() / sub_dim;
    if n <= k {
        // Fewer points than centroids: each point is its own centroid
        let mut centroids = vec![0.0f32; k * sub_dim];
        centroids[..n * sub_dim].copy_from_slice(&data[..n * sub_dim]);
        return centroids;
    }

    // K-means++ initialization
    let mut centroids = vec![0.0f32; k * sub_dim];
    let mut rng = SimpleRng::new();

    // First centroid: random point
    let first = rng.next_usize() % n;
    centroids[..sub_dim].copy_from_slice(&data[first * sub_dim..(first + 1) * sub_dim]);

    // Distance from each point to its nearest centroid
    let mut min_dists = vec![f32::MAX; n];

    for ci in 1..k {
        // Update min distances with the last added centroid
        let last_centroid = &centroids[(ci - 1) * sub_dim..ci * sub_dim];
        let mut total = 0.0f64;
        for i in 0..n {
            let point = &data[i * sub_dim..(i + 1) * sub_dim];
            let d = sq_dist(point, last_centroid);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
            total += min_dists[i] as f64;
        }

        // Weighted random selection proportional to distance²
        if total < 1e-30 {
            // All points coincide with existing centroids
            let idx = rng.next_usize() % n;
            centroids[ci * sub_dim..(ci + 1) * sub_dim]
                .copy_from_slice(&data[idx * sub_dim..(idx + 1) * sub_dim]);
            continue;
        }
        let threshold = rng.next_f64() * total;
        let mut cumulative = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate().take(n) {
            cumulative += d as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids[ci * sub_dim..(ci + 1) * sub_dim]
            .copy_from_slice(&data[chosen * sub_dim..(chosen + 1) * sub_dim]);
    }

    // K-means iterations
    let mut assignments = vec![0u8; n];
    let iterations = config::PQ_KMEANS_ITERATIONS;

    for _ in 0..iterations {
        // Assign each point to nearest centroid
        for i in 0..n {
            let point = &data[i * sub_dim..(i + 1) * sub_dim];
            let mut best = 0u8;
            let mut best_dist = f32::MAX;
            for ci in 0..k {
                let centroid = &centroids[ci * sub_dim..(ci + 1) * sub_dim];
                let d = sq_dist(point, centroid);
                if d < best_dist {
                    best_dist = d;
                    best = ci as u8;
                }
            }
            assignments[i] = best;
        }

        // Update centroids
        let mut counts = vec![0u32; k];
        centroids.fill(0.0);
        for i in 0..n {
            let ci = assignments[i] as usize;
            counts[ci] += 1;
            let point = &data[i * sub_dim..(i + 1) * sub_dim];
            let c = &mut centroids[ci * sub_dim..(ci + 1) * sub_dim];
            for d in 0..sub_dim {
                c[d] += point[d];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let c = &mut centroids[ci * sub_dim..(ci + 1) * sub_dim];
                let inv = 1.0 / counts[ci] as f32;
                for val in c.iter_mut() {
                    *val *= inv;
                }
            }
        }
    }

    centroids
}

/// Squared Euclidean distance between two sub-vectors.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Minimal deterministic PRNG (xorshift64) to avoid pulling in rand for PQ training.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        // Seed from memory address for basic entropy
        let seed = &0u8 as *const u8 as u64;
        Self {
            state: seed ^ 0x517cc1b727220a95,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
