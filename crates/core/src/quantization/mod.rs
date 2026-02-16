//! Scalar quantization for memory-efficient vector storage.
//!
//! Compresses f32 vectors to u8 using per-vector min-max scaling,
//! achieving 4× memory reduction. Includes SIMD-friendly distance
//! functions for both symmetric (u8 vs u8) and asymmetric (f32 vs u8) computation.

/// Scalar quantization: f32 → u8 with min/scale calibration and distance functions.
pub mod scalar;
/// SIMD-accelerated distance functions (NEON / AVX2 / scalar fallback).
pub mod simd;

pub use scalar::{QuantizedVector, VectorRef};
