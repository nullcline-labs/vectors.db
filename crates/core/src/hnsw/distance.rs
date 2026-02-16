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
            DistanceMetric::Cosine => 1.0 - crate::quantization::simd::cosine_asym(query, stored),
            DistanceMetric::Euclidean => {
                crate::quantization::simd::euclidean_sq_asym(query, stored)
            }
            DistanceMetric::DotProduct => {
                -crate::quantization::simd::dot_product_asym(query, stored)
            }
        }
    }

    /// Exact f32-vs-f32 distance for reranking. No quantization loss.
    pub fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => 1.0 - crate::quantization::simd::cosine_f32(a, b),
            DistanceMetric::Euclidean => crate::quantization::simd::euclidean_sq_f32(a, b),
            DistanceMetric::DotProduct => -crate::quantization::simd::dot_product_f32(a, b),
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
                1.0 - crate::quantization::simd::cosine_asym_prenorm(query, stored, query_norm_sq)
            }
            DistanceMetric::Euclidean => {
                crate::quantization::simd::euclidean_sq_asym(query, stored)
            }
            DistanceMetric::DotProduct => {
                -crate::quantization::simd::dot_product_asym(query, stored)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantizedVector;

    fn make_vec(vals: &[f32]) -> (Vec<f32>, QuantizedVector) {
        let v = vals.to_vec();
        let q = QuantizedVector::quantize(&v);
        (v, q)
    }

    #[test]
    fn test_cosine_distance_ref_identical() {
        let (_, qa) = make_vec(&[1.0, 2.0, 3.0, 4.0]);
        let d = DistanceMetric::Cosine.distance_ref(qa.as_ref(), qa.as_ref());
        assert!(d < 0.02, "self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_euclidean_distance_ref_zero() {
        let (_, qa) = make_vec(&[1.0, 2.0, 3.0, 4.0]);
        let d = DistanceMetric::Euclidean.distance_ref(qa.as_ref(), qa.as_ref());
        assert!(d < 0.01, "self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_dot_product_distance_ref() {
        let (_, qa) = make_vec(&[1.0, 0.0, 0.0, 0.0]);
        let (_, qb) = make_vec(&[0.0, 1.0, 0.0, 0.0]);
        let d = DistanceMetric::DotProduct.distance_ref(qa.as_ref(), qb.as_ref());
        assert!(
            d.abs() < 0.1,
            "orthogonal dot product distance ~ 0, got {d}"
        );
    }

    #[test]
    fn test_distance_asym_cosine() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let (_, qs) = make_vec(&[1.0, 2.0, 3.0, 4.0]);
        let d = DistanceMetric::Cosine.distance_asym(&query, qs.as_ref());
        assert!(d < 0.02, "asym self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_distance_asym_prenorm_matches_asym() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let (_, qs) = make_vec(&[4.0, 3.0, 2.0, 1.0]);
        let d1 = DistanceMetric::Cosine.distance_asym(&query, qs.as_ref());
        let d2 = DistanceMetric::Cosine.distance_asym_prenorm(&query, qs.as_ref(), query_norm_sq);
        assert!(
            (d1 - d2).abs() < 0.01,
            "prenorm should match asym: {d1} vs {d2}"
        );
    }

    #[test]
    fn test_distance_exact_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = DistanceMetric::Cosine.distance_exact(&a, &b);
        assert!(
            (d - 1.0).abs() < 0.001,
            "orthogonal cosine distance = 1.0, got {d}"
        );
    }

    #[test]
    fn test_distance_exact_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = DistanceMetric::Euclidean.distance_exact(&a, &b);
        assert!(
            (d - 25.0).abs() < 0.001,
            "squared euclidean should be 25, got {d}"
        );
    }

    #[test]
    fn test_distance_exact_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = DistanceMetric::DotProduct.distance_exact(&a, &b);
        assert!(
            (d - (-32.0)).abs() < 0.001,
            "negative dot product should be -32, got {d}"
        );
    }

    #[test]
    fn test_asym_accuracy_vs_exact() {
        let query = vec![0.5, -0.3, 0.8, 0.1, 0.9, -0.2, 0.6, 0.4];
        let stored_raw = vec![0.7, 0.2, -0.5, 0.3, 0.1, 0.8, -0.4, 0.6];
        let qs = QuantizedVector::quantize(&stored_raw);
        let exact_cos = DistanceMetric::Cosine.distance_exact(&query, &stored_raw);
        let asym_cos = DistanceMetric::Cosine.distance_asym(&query, qs.as_ref());
        assert!(
            (exact_cos - asym_cos).abs() < 0.1,
            "asym vs exact cosine gap too large: exact={exact_cos}, asym={asym_cos}"
        );
    }

    #[test]
    fn test_euclidean_asym_vs_exact() {
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let stored_raw = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let qs = QuantizedVector::quantize(&stored_raw);
        let exact = DistanceMetric::Euclidean.distance_exact(&query, &stored_raw);
        let asym = DistanceMetric::Euclidean.distance_asym(&query, qs.as_ref());
        let rel_err = (exact - asym).abs() / exact.max(1.0);
        assert!(
            rel_err < 0.15,
            "euclidean asym relative error too large: exact={exact}, asym={asym}"
        );
    }
}
