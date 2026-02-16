//! SIMD-accelerated distance functions.
//!
//! Provides NEON (aarch64) and AVX2+FMA (x86_64) implementations of
//! f32-vs-f32 and f32-vs-u8 distance computations. Falls back to scalar
//! on unsupported platforms or when AVX2 is unavailable at runtime.

use super::scalar::VectorRef;

// ============================================================================
// Public dispatch functions
// ============================================================================

/// Cosine similarity between two f32 slices. Returns value in [-1, 1].
#[inline]
#[allow(unreachable_code)]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_cosine_f32(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_cosine_f32(a, b) };
        }
    }
    scalar_cosine_f32(a, b)
}

/// Squared Euclidean distance between two f32 slices.
#[inline]
#[allow(unreachable_code)]
pub fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_euclidean_sq_f32(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_euclidean_sq_f32(a, b) };
        }
    }
    scalar_euclidean_sq_f32(a, b)
}

/// Dot product between two f32 slices.
#[inline]
#[allow(unreachable_code)]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_dot_product_f32(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_dot_product_f32(a, b) };
        }
    }
    scalar_dot_product_f32(a, b)
}

/// Asymmetric cosine similarity with precomputed query norm: f32 query vs u8 stored.
#[inline]
#[allow(unreachable_code)]
pub fn cosine_asym_prenorm(query: &[f32], stored: VectorRef<'_>, query_norm_sq: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_cosine_asym_prenorm(query, stored, query_norm_sq) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_cosine_asym_prenorm(query, stored, query_norm_sq) };
        }
    }
    super::scalar::cosine_similarity_asym_prenorm(query, stored, query_norm_sq)
}

/// Asymmetric squared Euclidean distance: f32 query vs u8 stored.
#[inline]
#[allow(unreachable_code)]
pub fn euclidean_sq_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_euclidean_sq_asym(query, stored) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_euclidean_sq_asym(query, stored) };
        }
    }
    super::scalar::euclidean_distance_sq_asym(query, stored)
}

/// Asymmetric dot product: f32 query vs u8 stored.
#[inline]
#[allow(unreachable_code)]
pub fn dot_product_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_dot_product_asym(query, stored) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2_dot_product_asym(query, stored) };
        }
    }
    super::scalar::dot_product_asym(query, stored)
}

/// Asymmetric cosine similarity (computes query norm internally): f32 query vs u8 stored.
#[inline]
#[allow(unreachable_code)]
pub fn cosine_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
    cosine_asym_prenorm(query, stored, query_norm_sq)
}

// ============================================================================
// Scalar fallbacks (f32 vs f32)
// ============================================================================

fn scalar_cosine_f32(a: &[f32], b: &[f32]) -> f32 {
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
        return 0.0;
    }
    dot / denom
}

fn scalar_euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn scalar_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// NEON implementations (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
unsafe fn neon_cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut na0 = vdupq_n_f32(0.0);
    let mut na1 = vdupq_n_f32(0.0);
    let mut nb0 = vdupq_n_f32(0.0);
    let mut nb1 = vdupq_n_f32(0.0);

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let a0 = vld1q_f32(a_ptr.add(base));
        let a1 = vld1q_f32(a_ptr.add(base + 4));
        let b0 = vld1q_f32(b_ptr.add(base));
        let b1 = vld1q_f32(b_ptr.add(base + 4));
        dot0 = vfmaq_f32(dot0, a0, b0);
        dot1 = vfmaq_f32(dot1, a1, b1);
        na0 = vfmaq_f32(na0, a0, a0);
        na1 = vfmaq_f32(na1, a1, a1);
        nb0 = vfmaq_f32(nb0, b0, b0);
        nb1 = vfmaq_f32(nb1, b1, b1);
    }

    let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut norm_a = vaddvq_f32(vaddq_f32(na0, na1));
    let mut norm_b = vaddvq_f32(vaddq_f32(nb0, nb1));

    for i in (chunks * 8)..len {
        let ai = *a_ptr.add(i);
        let bi = *b_ptr.add(i);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    dot / denom
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);

    let chunks = len / 16;
    for i in 0..chunks {
        let base = i * 16;
        let d0 = vsubq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
        let d1 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 4)),
            vld1q_f32(b_ptr.add(base + 4)),
        );
        let d2 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 8)),
            vld1q_f32(b_ptr.add(base + 8)),
        );
        let d3 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 12)),
            vld1q_f32(b_ptr.add(base + 12)),
        );
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }

    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));

    for i in (chunks * 16)..len {
        let d = *a_ptr.add(i) - *b_ptr.add(i);
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);

    let chunks = len / 16;
    for i in 0..chunks {
        let base = i * 16;
        s0 = vfmaq_f32(s0, vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
        s1 = vfmaq_f32(
            s1,
            vld1q_f32(a_ptr.add(base + 4)),
            vld1q_f32(b_ptr.add(base + 4)),
        );
        s2 = vfmaq_f32(
            s2,
            vld1q_f32(a_ptr.add(base + 8)),
            vld1q_f32(b_ptr.add(base + 8)),
        );
        s3 = vfmaq_f32(
            s3,
            vld1q_f32(a_ptr.add(base + 12)),
            vld1q_f32(b_ptr.add(base + 12)),
        );
    }

    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));

    for i in (chunks * 16)..len {
        sum += *a_ptr.add(i) * *b_ptr.add(i);
    }
    sum
}

/// NEON helper: convert 8 u8 values to 2 x float32x4_t and dequantize.
/// Returns (low 4 dequantized, high 4 dequantized).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_u8x8_to_f32_deq(
    ptr: *const u8,
    min_vec: float32x4_t,
    scale_vec: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    let u8x8 = vld1_u8(ptr);
    let u16x8 = vmovl_u8(u8x8);
    let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16x8)));
    let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16x8)));
    (
        vfmaq_f32(min_vec, lo, scale_vec),
        vfmaq_f32(min_vec, hi, scale_vec),
    )
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_cosine_asym_prenorm(
    query: &[f32],
    stored: VectorRef<'_>,
    query_norm_sq: f32,
) -> f32 {
    if stored.scale == 0.0 || query_norm_sq < 1e-10 {
        return 0.0;
    }

    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = vdupq_n_f32(stored.min);
    let scale_vec = vdupq_n_f32(stored.scale);

    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut ns0 = vdupq_n_f32(0.0);
    let mut ns1 = vdupq_n_f32(0.0);

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let (s_lo, s_hi) = neon_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        let q_lo = vld1q_f32(q_ptr.add(base));
        let q_hi = vld1q_f32(q_ptr.add(base + 4));
        dot0 = vfmaq_f32(dot0, q_lo, s_lo);
        dot1 = vfmaq_f32(dot1, q_hi, s_hi);
        ns0 = vfmaq_f32(ns0, s_lo, s_lo);
        ns1 = vfmaq_f32(ns1, s_hi, s_hi);
    }

    let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut norm_s = vaddvq_f32(vaddq_f32(ns0, ns1));

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        dot += q * s;
        norm_s += s * s;
    }

    let denom = (query_norm_sq as f64).sqrt() * (norm_s as f64).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    (dot as f64 / denom) as f32
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_euclidean_sq_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = vdupq_n_f32(stored.min);
    let scale_vec = vdupq_n_f32(stored.scale);

    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let (deq_lo, deq_hi) = neon_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        let d0 = vsubq_f32(vld1q_f32(q_ptr.add(base)), deq_lo);
        let d1 = vsubq_f32(vld1q_f32(q_ptr.add(base + 4)), deq_hi);
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
    }

    let mut sum = vaddvq_f32(vaddq_f32(s0, s1));

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        let d = q - s;
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot_product_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = vdupq_n_f32(stored.min);
    let scale_vec = vdupq_n_f32(stored.scale);

    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let (deq_lo, deq_hi) = neon_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        s0 = vfmaq_f32(s0, vld1q_f32(q_ptr.add(base)), deq_lo);
        s1 = vfmaq_f32(s1, vld1q_f32(q_ptr.add(base + 4)), deq_hi);
    }

    let mut sum = vaddvq_f32(vaddq_f32(s0, s1));

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        sum += q * s;
    }
    sum
}

// ============================================================================
// AVX2+FMA implementations (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal sum of 8 f32 values in a __m256 register.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_f32x8(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut na0 = _mm256_setzero_ps();
    let mut na1 = _mm256_setzero_ps();
    let mut nb0 = _mm256_setzero_ps();
    let mut nb1 = _mm256_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let base = i * 16;
        let a0 = _mm256_loadu_ps(a_ptr.add(base));
        let a1 = _mm256_loadu_ps(a_ptr.add(base + 8));
        let b0 = _mm256_loadu_ps(b_ptr.add(base));
        let b1 = _mm256_loadu_ps(b_ptr.add(base + 8));
        dot0 = _mm256_fmadd_ps(a0, b0, dot0);
        dot1 = _mm256_fmadd_ps(a1, b1, dot1);
        na0 = _mm256_fmadd_ps(a0, a0, na0);
        na1 = _mm256_fmadd_ps(a1, a1, na1);
        nb0 = _mm256_fmadd_ps(b0, b0, nb0);
        nb1 = _mm256_fmadd_ps(b1, b1, nb1);
    }

    let mut dot = hsum_f32x8(_mm256_add_ps(dot0, dot1));
    let mut norm_a = hsum_f32x8(_mm256_add_ps(na0, na1));
    let mut norm_b = hsum_f32x8(_mm256_add_ps(nb0, nb1));

    for i in (chunks * 16)..len {
        let ai = *a_ptr.add(i);
        let bi = *b_ptr.add(i);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    dot / denom
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let base = i * 16;
        let d0 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
        );
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base + 8)),
            _mm256_loadu_ps(b_ptr.add(base + 8)),
        );
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }

    let mut sum = hsum_f32x8(_mm256_add_ps(s0, s1));

    for i in (chunks * 16)..len {
        let d = *a_ptr.add(i) - *b_ptr.add(i);
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let base = i * 16;
        s0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
            s0,
        );
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(base + 8)),
            _mm256_loadu_ps(b_ptr.add(base + 8)),
            s1,
        );
    }

    let mut sum = hsum_f32x8(_mm256_add_ps(s0, s1));

    for i in (chunks * 16)..len {
        sum += *a_ptr.add(i) * *b_ptr.add(i);
    }
    sum
}

/// AVX2 helper: load 8 u8 values, convert to __m256 f32, dequantize.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_u8x8_to_f32_deq(ptr: *const u8, min_vec: __m256, scale_vec: __m256) -> __m256 {
    let u8x8 = _mm_loadl_epi64(ptr as *const __m128i);
    let i32x8 = _mm256_cvtepu8_epi32(u8x8);
    let f32x8 = _mm256_cvtepi32_ps(i32x8);
    _mm256_fmadd_ps(f32x8, scale_vec, min_vec)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_cosine_asym_prenorm(
    query: &[f32],
    stored: VectorRef<'_>,
    query_norm_sq: f32,
) -> f32 {
    if stored.scale == 0.0 || query_norm_sq < 1e-10 {
        return 0.0;
    }

    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = _mm256_set1_ps(stored.min);
    let scale_vec = _mm256_set1_ps(stored.scale);

    let mut dot0 = _mm256_setzero_ps();
    let mut ns0 = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let deq = avx2_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        let q = _mm256_loadu_ps(q_ptr.add(base));
        dot0 = _mm256_fmadd_ps(q, deq, dot0);
        ns0 = _mm256_fmadd_ps(deq, deq, ns0);
    }

    let mut dot = hsum_f32x8(dot0);
    let mut norm_s = hsum_f32x8(ns0);

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        dot += q * s;
        norm_s += s * s;
    }

    let denom = (query_norm_sq as f64).sqrt() * (norm_s as f64).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    (dot as f64 / denom) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_euclidean_sq_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = _mm256_set1_ps(stored.min);
    let scale_vec = _mm256_set1_ps(stored.scale);

    let mut s0 = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let deq = avx2_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        let d = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(base)), deq);
        s0 = _mm256_fmadd_ps(d, d, s0);
    }

    let mut sum = hsum_f32x8(s0);

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        let d = q - s;
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_product_asym(query: &[f32], stored: VectorRef<'_>) -> f32 {
    let len = query.len();
    let q_ptr = query.as_ptr();
    let s_ptr = stored.data.as_ptr();
    let min_vec = _mm256_set1_ps(stored.min);
    let scale_vec = _mm256_set1_ps(stored.scale);

    let mut s0 = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let deq = avx2_u8x8_to_f32_deq(s_ptr.add(base), min_vec, scale_vec);
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q_ptr.add(base)), deq, s0);
    }

    let mut sum = hsum_f32x8(s0);

    for i in (chunks * 8)..len {
        let q = *q_ptr.add(i);
        let s = stored.min + stored.data[i] as f32 * stored.scale;
        sum += q * s;
    }
    sum
}
