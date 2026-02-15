//! Global configuration constants for vectors.db.
//!
//! All tuning parameters, input validation limits, and server defaults are defined here.
//! These are compile-time constants; runtime configuration is handled via CLI arguments
//! and environment variables in `main.rs`.

/// BM25 Okapi term frequency saturation parameter.
///
/// Controls how quickly term frequency saturates. Higher values allow TF to grow more.
/// Standard value is 1.2 (range: 1.0–2.0).
pub const BM25_K1: f32 = 1.2;

/// BM25 Okapi document length normalization parameter.
///
/// Controls the impact of document length on scoring. 0.0 = no normalization,
/// 1.0 = full normalization. Standard value is 0.75.
pub const BM25_B: f32 = 0.75;

/// Default number of bidirectional links per HNSW node.
///
/// Higher values improve recall but increase memory and build time.
/// Typical range: 8–64. Default: 16.
pub const HNSW_DEFAULT_M: usize = 16;

/// Default ef parameter during HNSW index construction.
///
/// Controls the size of the dynamic candidate list during insertion.
/// Higher values produce a better graph but slow down build time.
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Default ef parameter during HNSW search.
///
/// Controls the size of the dynamic candidate list during query.
/// Higher values improve recall at the cost of latency.
pub const HNSW_DEFAULT_EF_SEARCH: usize = 50;

/// Maximum number of layers in the HNSW graph.
pub const HNSW_DEFAULT_MAX_LAYERS: usize = 16;

/// Level generation multiplier for HNSW layer assignment.
///
/// Nodes are assigned to layer `floor(-ln(uniform) * LEVEL_MULTIPLIER)`.
/// A value of 1.0 corresponds to `1/ln(M)` normalization.
pub const HNSW_LEVEL_MULTIPLIER: f64 = 1.0;

/// Reciprocal Rank Fusion (RRF) constant `k`.
///
/// Used in the formula `1 / (k + rank)` to combine ranked lists.
/// Standard value is 60.0 (from the original RRF paper).
pub const RRF_K: f32 = 60.0;

/// Maximum allowed embedding dimension.
pub const MAX_DIMENSION: usize = 4096;

/// Maximum number of results (`k`) per search request.
pub const MAX_K: usize = 10_000;

/// Maximum length of a collection name in characters.
pub const MAX_COLLECTION_NAME_LEN: usize = 128;

/// Maximum length of document text in bytes.
pub const MAX_TEXT_LEN: usize = 1_000_000;

/// Maximum number of documents per batch insert request.
pub const MAX_BATCH_SIZE: usize = 1_000;

/// Default HTTP server port.
pub const DEFAULT_PORT: u16 = 3030;

/// Default directory for WAL and snapshot files.
pub const DEFAULT_DATA_DIR: &str = "./data";

/// Maximum pagination offset for search results.
pub const MAX_OFFSET: usize = 100_000;

/// Default interval (in seconds) between automatic snapshots. 0 = disabled.
pub const DEFAULT_SNAPSHOT_INTERVAL_SECS: u64 = 300;

/// Per-request timeout in seconds.
pub const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Global rate limit in requests per second.
pub const RATE_LIMIT_RPS: u64 = 100;

/// Maximum HTTP request body size in bytes (10 MB).
pub const MAX_REQUEST_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Maximum number of metadata keys per document.
pub const MAX_METADATA_KEYS: usize = 64;

/// Maximum total serialized size of metadata in bytes (64 KB).
pub const MAX_METADATA_BYTES: usize = 65_536;

/// Maximum number of concurrent in-flight requests.
pub const MAX_CONCURRENT_REQUESTS: usize = 512;

/// Maximum entries per WAL group commit batch before forcing a flush.
pub const WAL_GROUP_COMMIT_MAX_BATCH: usize = 128;

/// Maximum wait time (microseconds) to accumulate WAL entries before flushing.
pub const WAL_GROUP_COMMIT_MAX_WAIT_US: u64 = 1000;
