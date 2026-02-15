//! Raft consensus clustering for multi-node replication and sharding.
//!
//! Built on [openraft](https://docs.rs/openraft), this module provides:
//! - **types**: Raft type configuration, log entry definitions, and type aliases.
//! - **store**: In-memory log store and state machine that applies entries to the [`Database`](crate::storage::Database).
//! - **network**: HTTP-based RPC transport between Raft peers.
//! - **api**: Axum routes for Raft protocol endpoints (vote, append, snapshot, init, membership).

/// Axum routes for Raft protocol RPCs and cluster management.
pub mod api;
/// HTTP-based Raft RPC network transport using reqwest.
pub mod network;
/// In-memory Raft log store and state machine.
pub mod store;
/// Raft type configuration, log entry definitions, and type aliases.
pub mod types;

pub use types::{LogEntry, LogEntryResponse, NodeId, Raft, TypeConfig};
