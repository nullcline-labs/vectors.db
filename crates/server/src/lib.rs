//! vectorsdb-server — HTTP server for vectors.db.
//!
//! Provides the REST API, Raft clustering, and async WAL.
//! Core database logic lives in `vectorsdb-core`.

/// REST API layer: Axum router, HTTP handlers, models, auth, metrics.
pub mod api;
/// Raft consensus clustering.
pub mod cluster;
/// WAL streaming replication for active-passive HA.
pub mod replication;
/// Async Write-Ahead Log with group commit (tokio-based).
pub mod wal_async;
