//! Scalar quantization implementation.
//!
//! Each f32 vector is compressed to u8 by finding the min and max values,
//! then linearly mapping each component to \[0, 255\]. The `min` and `scale`
//! parameters are stored per vector for dequantization.
//!
//! Distance functions use SIMD-friendly chunked loops with i32 or f32 inner
//! accumulators and f64 outer accumulation to minimize rounding error.

use serde::{Deserialize, Serialize};

/// Scalar-quantized vector: f32 → u8 with min/max for reconstruction.
/// `scale` is precomputed as (max - min) / 255.0 to avoid redundant division in hot paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    pub data: Vec<u8>,
    pub min: f32,
    pub max: f32,
    pub scale: f32,
}

/// Lightweight reference to quantized vector data. Zero allocation.
/// Used by SoA HnswIndex for efficient distance computation.
#[derive(Debug, Clone, Copy)]
pub struct VectorRef<'a> {
    pub data: &'a [u8],
    pub min: f32,
    pub scale: f32,
}

impl QuantizedVector {
    /// Quantize a f32 vector to u8 using min-max scaling.
    pub fn quantize(vector: &[f32]) -> Self {
        if vector.is_empty() {
            return Self {
                data: Vec::new(),
                min: 0.0,
                max: 0.0,
                scale: 0.0,
            };
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in vector {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let range = max - min;
        let (data, scale) = if range < f32::EPSILON {
            (vec![128u8; vector.len()], 0.0)
        } else {
            let inv_scale = 255.0 / range;
            let data = vector
                .iter()
                .map(|&v| ((v - min) * inv_scale).round().clamp(0.0, 255.0) as u8)
                .collect();
            (data, range / 255.0)
        };

        Self {
            data,
            min,
            max,
            scale,
        }
    }

    /// Dequantize back to f32. Lossy.
    pub fn dequantize(&self) -> Vec<f32> {
        if self.scale == 0.0 {
            return vec![self.min; self.data.len()];
        }
        self.data
            .iter()
            .map(|&v| self.min + (v as f32) * self.scale)
            .collect()
    }

    /// Returns the dimensionality of the quantized vector.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Create a VectorRef borrowing this vector's data.
    pub fn as_ref(&self) -> VectorRef<'_> {
        VectorRef {
            data: &self.data,
            min: self.min,
            scale: self.scale,
        }
    }
}

/// SIMD-friendly chunk size. 32 elements of u8×u8 (max 65025 per product)
/// sum of 32 products fits in i32 (max 2,080,800 << 2,147,483,647).
const CHUNK: usize = 32;

/// Cosine similarity on VectorRef using SIMD-friendly i32 chunked accumulation.
/// Inner loop uses i32 arithmetic in fixed-size chunks for auto-vectorization.
pub fn cosine_similarity_ref(a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
    debug_assert_eq!(a.data.len(), b.data.len());

    if a.scale == 0.0 || b.scale == 0.0 {
        return 0.0;
    }

    let len = a.data.len();
    let mut dot = 0i64;
    let mut norm_a = 0i64;
    let mut norm_b = 0i64;
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;

    // Process in chunks of CHUNK for auto-vectorization (i32 inner loop)
    let full_chunks = len / CHUNK;
    for c in 0..full_chunks {
        let base = c * CHUNK;
        let mut cd = 0i32;
        let mut cna = 0i32;
        let mut cnb = 0i32;
        let mut csa = 0i32;
        let mut csb = 0i32;

        for j in 0..CHUNK {
            let ai = a.data[base + j] as i32;
            let bi = b.data[base + j] as i32;
            cd += ai * bi;
            cna += ai * ai;
            cnb += bi * bi;
            csa += ai;
            csb += bi;
        }

        dot += cd as i64;
        norm_a += cna as i64;
        norm_b += cnb as i64;
        sum_a += csa as i64;
        sum_b += csb as i64;
    }

    // Remainder
    for i in (full_chunks * CHUNK)..len {
        let ai = a.data[i] as i64;
        let bi = b.data[i] as i64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
        sum_a += ai;
        sum_b += bi;
    }

    let n = len as f64;
    let a_scale = a.scale as f64;
    let b_scale = b.scale as f64;
    let a_min = a.min as f64;
    let b_min = b.min as f64;

    let real_dot = a_scale * b_scale * dot as f64
        + a_min * b_scale * sum_b as f64
        + b_min * a_scale * sum_a as f64
        + a_min * b_min * n;

    let real_norm_a = a_scale * a_scale * norm_a as f64
        + 2.0 * a_min * a_scale * sum_a as f64
        + a_min * a_min * n;

    let real_norm_b = b_scale * b_scale * norm_b as f64
        + 2.0 * b_min * b_scale * sum_b as f64
        + b_min * b_min * n;

    let denom = real_norm_a.sqrt() * real_norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (real_dot / denom) as f32
}

/// Squared Euclidean distance on VectorRefs with SIMD-friendly i32 chunked accumulation.
/// Expands `sum((a_min + ai*a_scale - b_min - bi*b_scale)^2)` into integer accumulators
/// for auto-vectorization, then reconstructs the f64 result from closed-form sums.
pub fn euclidean_distance_sq_ref(a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
    debug_assert_eq!(a.data.len(), b.data.len());

    let len = a.data.len();
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_a2 = 0i64;
    let mut sum_b2 = 0i64;
    let mut sum_ab = 0i64;

    let full_chunks = len / CHUNK;
    for c in 0..full_chunks {
        let base = c * CHUNK;
        let mut csa = 0i32;
        let mut csb = 0i32;
        let mut csa2 = 0i32;
        let mut csb2 = 0i32;
        let mut csab = 0i32;

        for j in 0..CHUNK {
            let ai = a.data[base + j] as i32;
            let bi = b.data[base + j] as i32;
            csa += ai;
            csb += bi;
            csa2 += ai * ai;
            csb2 += bi * bi;
            csab += ai * bi;
        }

        sum_a += csa as i64;
        sum_b += csb as i64;
        sum_a2 += csa2 as i64;
        sum_b2 += csb2 as i64;
        sum_ab += csab as i64;
    }

    for i in (full_chunks * CHUNK)..len {
        let ai = a.data[i] as i64;
        let bi = b.data[i] as i64;
        sum_a += ai;
        sum_b += bi;
        sum_a2 += ai * ai;
        sum_b2 += bi * bi;
        sum_ab += ai * bi;
    }

    let n = len as f64;
    let a_scale = a.scale as f64;
    let b_scale = b.scale as f64;
    let a_min = a.min as f64;
    let b_min = b.min as f64;
    let offset = a_min - b_min;

    // sum((offset + ai*a_scale - bi*b_scale)^2)
    // = n*offset^2 + 2*offset*(a_scale*sum_a - b_scale*sum_b)
    //   + a_scale^2*sum_a2 - 2*a_scale*b_scale*sum_ab + b_scale^2*sum_b2
    let result = n * offset * offset
        + 2.0 * offset * (a_scale * sum_a as f64 - b_scale * sum_b as f64)
        + a_scale * a_scale * sum_a2 as f64
        - 2.0 * a_scale * b_scale * sum_ab as f64
        + b_scale * b_scale * sum_b2 as f64;

    result as f32
}

/// Dot product on VectorRefs with SIMD-friendly i32 chunked accumulation.
/// Expands `sum((a_min + ai*a_scale) * (b_min + bi*b_scale))` into integer accumulators
/// for auto-vectorization, then reconstructs the f64 result from closed-form sums.
pub fn dot_product_ref(a: VectorRef<'_>, b: VectorRef<'_>) -> f32 {
    debug_assert_eq!(a.data.len(), b.data.len());

    let len = a.data.len();
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    let mut sum_ab = 0i64;

    let full_chunks = len / CHUNK;
    for c in 0..full_chunks {
        let base = c * CHUNK;
        let mut csa = 0i32;
        let mut csb = 0i32;
        let mut csab = 0i32;

        for j in 0..CHUNK {
            let ai = a.data[base + j] as i32;
            let bi = b.data[base + j] as i32;
            csa += ai;
            csb += bi;
            csab += ai * bi;
        }

        sum_a += csa as i64;
        sum_b += csb as i64;
        sum_ab += csab as i64;
    }

    for i in (full_chunks * CHUNK)..len {
        let ai = a.data[i] as i64;
        let bi = b.data[i] as i64;
        sum_a += ai;
        sum_b += bi;
        sum_ab += ai * bi;
    }

    let n = len as f64;
    let a_scale = a.scale as f64;
    let b_scale = b.scale as f64;
    let a_min = a.min as f64;
    let b_min = b.min as f64;

    // sum((a_min + ai*a_scale) * (b_min + bi*b_scale))
    // = n*a_min*b_min + a_min*b_scale*sum_b + b_min*a_scale*sum_a + a_scale*b_scale*sum_ab
    let result = n * a_min * b_min
        + a_min * b_scale * sum_b as f64
        + b_min * a_scale * sum_a as f64
        + a_scale * b_scale * sum_ab as f64;

    result as f32
}

/// SIMD-friendly chunk size for f32 asymmetric loops.
/// 8 × f32 = 256 bit = one AVX register.
const CHUNK_F32: usize = 8;

/// Asymmetric cosine similarity: f32 query vs u8 stored.
/// Uses SIMD-friendly f32 chunked inner loop with f64 accumulation at chunk boundaries.
#[allow(clippy::needless_range_loop)]
pub fn cosine_similarity_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    debug_assert_eq!(query.len(), stored.data.len());

    if stored.scale == 0.0 {
        return 0.0;
    }

    let len = query.len();
    let s_min = stored.min;
    let s_scale = stored.scale;

    let mut dot = 0.0f64;
    let mut norm_q = 0.0f64;
    let mut norm_s = 0.0f64;

    let full_chunks = len / CHUNK_F32;
    for c in 0..full_chunks {
        let base = c * CHUNK_F32;
        let mut cd = 0.0f32;
        let mut cnq = 0.0f32;
        let mut cns = 0.0f32;
        for j in 0..CHUNK_F32 {
            let q = query[base + j];
            let s = s_min + stored.data[base + j] as f32 * s_scale;
            cd += q * s;
            cnq += q * q;
            cns += s * s;
        }
        dot += cd as f64;
        norm_q += cnq as f64;
        norm_s += cns as f64;
    }

    for i in (full_chunks * CHUNK_F32)..len {
        let q = query[i] as f64;
        let s = stored.min as f64 + stored.data[i] as f64 * stored.scale as f64;
        dot += q * s;
        norm_q += q * q;
        norm_s += s * s;
    }

    let denom = norm_q.sqrt() * norm_s.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (dot / denom) as f32
}

/// Asymmetric cosine similarity with precomputed query norm squared.
/// Avoids recomputing `norm_q` when the same query is used against many stored vectors.
#[allow(clippy::needless_range_loop)]
pub fn cosine_similarity_asym_prenorm(
    query: &[f32],
    stored: VectorRef<'_>,
    query_norm_sq: f32,
) -> f32 {
    debug_assert_eq!(query.len(), stored.data.len());

    if stored.scale == 0.0 || query_norm_sq < 1e-10 {
        return 0.0;
    }

    let len = query.len();
    let s_min = stored.min;
    let s_scale = stored.scale;

    let mut dot = 0.0f64;
    let mut norm_s = 0.0f64;

    let full_chunks = len / CHUNK_F32;
    for c in 0..full_chunks {
        let base = c * CHUNK_F32;
        let mut cd = 0.0f32;
        let mut cns = 0.0f32;
        for j in 0..CHUNK_F32 {
            let q = query[base + j];
            let s = s_min + stored.data[base + j] as f32 * s_scale;
            cd += q * s;
            cns += s * s;
        }
        dot += cd as f64;
        norm_s += cns as f64;
    }

    for i in (full_chunks * CHUNK_F32)..len {
        let q = query[i] as f64;
        let s = stored.min as f64 + stored.data[i] as f64 * stored.scale as f64;
        dot += q * s;
        norm_s += s * s;
    }

    let denom = (query_norm_sq as f64).sqrt() * norm_s.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (dot / denom) as f32
}

/// Asymmetric squared Euclidean distance: f32 query vs u8 stored.
/// Uses SIMD-friendly f32 chunked inner loop with f64 accumulation at chunk boundaries.
#[allow(clippy::needless_range_loop)]
pub fn euclidean_distance_sq_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    debug_assert_eq!(query.len(), stored.data.len());

    let len = query.len();
    let s_min = stored.min;
    let s_scale = stored.scale;

    let mut sum = 0.0f64;

    let full_chunks = len / CHUNK_F32;
    for c in 0..full_chunks {
        let base = c * CHUNK_F32;
        let mut chunk_acc = 0.0f32;
        for j in 0..CHUNK_F32 {
            let q = query[base + j];
            let s = s_min + stored.data[base + j] as f32 * s_scale;
            let diff = q - s;
            chunk_acc += diff * diff;
        }
        sum += chunk_acc as f64;
    }

    for i in (full_chunks * CHUNK_F32)..len {
        let q = query[i] as f64;
        let s = stored.min as f64 + stored.data[i] as f64 * stored.scale as f64;
        let diff = q - s;
        sum += diff * diff;
    }

    sum as f32
}

/// Asymmetric dot product: f32 query vs u8 stored.
/// Uses SIMD-friendly f32 chunked inner loop with f64 accumulation at chunk boundaries.
#[allow(clippy::needless_range_loop)]
pub fn dot_product_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    debug_assert_eq!(query.len(), stored.data.len());

    let len = query.len();
    let s_min = stored.min;
    let s_scale = stored.scale;

    let mut sum = 0.0f64;

    let full_chunks = len / CHUNK_F32;
    for c in 0..full_chunks {
        let base = c * CHUNK_F32;
        let mut chunk_acc = 0.0f32;
        for j in 0..CHUNK_F32 {
            let q = query[base + j];
            let s = s_min + stored.data[base + j] as f32 * s_scale;
            chunk_acc += q * s;
        }
        sum += chunk_acc as f64;
    }

    for i in (full_chunks * CHUNK_F32)..len {
        let q = query[i] as f64;
        let s = stored.min as f64 + stored.data[i] as f64 * stored.scale as f64;
        sum += q * s;
    }

    sum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let v = vec![0.0, 0.5, 1.0, -1.0, 0.25];
        let q = QuantizedVector::quantize(&v);
        assert!(
            q.scale > 0.0,
            "scale should be positive for non-constant vectors"
        );
        let d = q.dequantize();
        for (orig, deq) in v.iter().zip(d.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={orig}, deq={deq}");
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let q = QuantizedVector::quantize(&v);
        let sim = cosine_similarity_ref(q.as_ref(), q.as_ref());
        assert!(sim > 0.99, "self-similarity should be ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let qa = QuantizedVector::quantize(&a);
        let qb = QuantizedVector::quantize(&b);
        let sim = cosine_similarity_ref(qa.as_ref(), qb.as_ref());
        assert!(sim.abs() < 0.15, "orthogonal sim should be ~0, got {sim}");
    }
}
