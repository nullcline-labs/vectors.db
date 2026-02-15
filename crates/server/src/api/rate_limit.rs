//! Per-key token bucket rate limiter.
//!
//! Provides a simple token bucket implementation for per-API-key rate limiting.
//! Each key gets its own bucket with a configurable rate (tokens per second)
//! and a burst capacity equal to the rate.

use std::time::Instant;

/// A token bucket rate limiter.
///
/// Refills at `rate` tokens per second up to `capacity`.
/// Each [`try_acquire`](TokenBucket::try_acquire) call consumes one token.
pub struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,
    rate: f64,
}

impl TokenBucket {
    /// Create a new token bucket with the given rate (requests per second).
    /// Burst capacity is set equal to the rate.
    pub fn new(rate: u64) -> Self {
        let rate_f = rate as f64;
        Self {
            tokens: rate_f,
            last_refill: Instant::now(),
            capacity: rate_f,
            rate: rate_f,
        }
    }

    /// Try to acquire one token. Returns `true` if allowed, `false` if rate limited.
    pub fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}
