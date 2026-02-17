//! WAL streaming replication for active-passive HA.
//!
//! A **primary** node streams WAL frames in real-time to **standby** nodes
//! over a persistent TCP connection. Standbys are read-only until manually
//! promoted via `POST /admin/promote`.

pub mod primary;
pub mod protocol;
pub mod standby;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Shared replication state, visible to both the API layer and the replication tasks.
#[derive(Clone)]
pub struct ReplicationState {
    inner: Arc<ReplicationInner>,
}

struct ReplicationInner {
    /// True when this node is a standby (read-only).
    is_standby: AtomicBool,
    /// Monotonically increasing WAL position counter.
    wal_position: AtomicU64,
}

impl ReplicationState {
    /// Create state for a primary or standalone node (read-write).
    pub fn new_primary() -> Self {
        Self {
            inner: Arc::new(ReplicationInner {
                is_standby: AtomicBool::new(false),
                wal_position: AtomicU64::new(0),
            }),
        }
    }

    /// Create state for a standby node (read-only).
    pub fn new_standby() -> Self {
        Self {
            inner: Arc::new(ReplicationInner {
                is_standby: AtomicBool::new(true),
                wal_position: AtomicU64::new(0),
            }),
        }
    }

    /// Returns `true` if this node is in standby (read-only) mode.
    pub fn is_standby(&self) -> bool {
        self.inner.is_standby.load(Ordering::Acquire)
    }

    /// Promote this standby to a primary (read-write).
    pub fn promote(&self) {
        self.inner.is_standby.store(false, Ordering::Release);
    }

    /// Current WAL position (number of framed byte batches applied).
    pub fn wal_position(&self) -> u64 {
        self.inner.wal_position.load(Ordering::Acquire)
    }

    /// Increment WAL position by one batch and return the new value.
    pub fn advance_wal_position(&self) -> u64 {
        self.inner.wal_position.fetch_add(1, Ordering::AcqRel) + 1
    }
}
