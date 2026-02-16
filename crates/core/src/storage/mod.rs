//! Storage layer: collections, database, write-ahead log, and disk persistence.
//!
//! Data lives in-memory in `Collection` instances grouped by a `Database`.
//! Durability is provided by a `SyncWriteAheadLog` (CRC32 + fsync) and
//! bincode snapshots (atomic temp-file + rename).

/// Collection and database data structures.
pub mod collection;
/// Disk persistence: snapshot save/load with atomic writes.
pub mod persistence;
/// Write-Ahead Log with CRC32 checksums.
pub mod wal;

pub use collection::{Collection, Database};
pub use persistence::{load_all_collections, load_collection, save_collection};
pub use wal::{ReplayStats, SyncWriteAheadLog, WalEntry};
