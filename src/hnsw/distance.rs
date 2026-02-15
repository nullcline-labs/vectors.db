//! Distance metric implementations for HNSW search.
//!
//! Supports three distance functions: cosine, euclidean (L2), and dot product.
//! Each metric has four variants: symmetric u8-vs-u8 (fast), asymmetric f32-vs-u8 (accurate),
//! asymmetric with precomputed query norm (optimized), and exact f32-vs-f32 (reranking).

use crate::quantization::VectorRef;

/// Distance metric used for vector similarity computation.
///
/// Determines how distances are calculated between vectors in the HNSW index.
/// All metrics return a distance value where **lower is better** (more similar).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: `1 - cosine_similarity`. Range: \[0, 2\].
    Cosine,
    /// Squared Euclidean distance (L2²). Range: \[0, ∞).
    Euclidean,
    /// Negative dot product: `-dot(a, b)`. Lower = higher similarity.
    DotProduct,
}

impl DistanceMetric {
    /// Compute distance between two vector references (symmetric u8-vs-u8).
    /// Uses SIMD-friendly chunked loops internally.
    pub fn distance_ref(&self, a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                1.0 - crate::quantization::scalar::cosine_similarity_ref(a, b)
            }
            DistanceMetric::Euclidean => {
                crate::quantization::scalar::euclidean_distance_sq_ref(a, b)
            }
            DistanceMetric::DotProduct => -crate::quantization::scalar::dot_product_ref(a, b),
        }
    }

    /// Asymmetric distance: f32 query vs u8 stored.
    /// More accurate than symmetric u8-vs-u8 since the query keeps full f32 precision.
    pub fn distance_asym(&self, query: &[f32], stored: VectorRef<'_>) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                1.0 - crate::quantization::scalar::cosine_similarity_asym(query, stored)
            }
            DistanceMetric::Euclidean => {
                crate::quantization::scalar::euclidean_distance_sq_asym(query, stored)
            }
            DistanceMetric::DotProduct => {
                -crate::quantization::scalar::dot_product_asym(query, stored)
            }
        }
    }

    /// Exact f32-vs-f32 distance for reranking. No quantization loss.
    pub fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                if denom < 1e-10 {
                    return 1.0;
                }
                1.0 - dot / denom
            }
            DistanceMetric::Euclidean => {
                let mut sum = 0.0f32;
                for i in 0..a.len() {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                sum
            }
            DistanceMetric::DotProduct => {
                let mut sum = 0.0f32;
                for i in 0..a.len() {
                    sum += a[i] * b[i];
                }
                -sum
            }
        }
    }

    /// Asymmetric distance with precomputed query norm squared.
    /// For Cosine uses the prenorm variant (skips redundant query norm calculation).
    /// For Euclidean/DotProduct delegates to the normal asymmetric version.
    pub fn distance_asym_prenorm(
        &self,
        query: &[f32],
        stored: VectorRef<'_>,
        query_norm_sq: f32,
    ) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                1.0 - crate::quantization::scalar::cosine_similarity_asym_prenorm(
                    query,
                    stored,
                    query_norm_sq,
                )
            }
            DistanceMetric::Euclidean => {
                crate::quantization::scalar::euclidean_distance_sq_asym(query, stored)
            }
            DistanceMetric::DotProduct => {
                -crate::quantization::scalar::dot_product_asym(query, stored)
            }
        }
    }
}
