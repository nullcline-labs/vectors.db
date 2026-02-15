# CLAUDE.md — vectors.db

## Project Overview

In-memory vector database written in Rust. Supports HNSW approximate nearest neighbor search, BM25 full-text search, and hybrid retrieval with RRF/linear fusion. Designed for RAG applications and semantic search.

## Build & Test Commands

```bash
cargo build --release          # Build optimized binary
cargo test                     # Run all tests (6 unit + 45 integration)
cargo clippy -- -D warnings    # Lint (zero warnings policy)
cargo fmt --check              # Format check
cargo bench                    # Run all 4 benchmarks (GloVe-25, SIFT-128, GloVe-100, BM25 NFCorpus)
cargo bench --bench ann_glove25   # Single benchmark
```

## Run

```bash
# Standalone
cargo run --release -- --port 3030 --data-dir ./data

# With auth
VECTORS_DB_API_KEY=secret cargo run --release

# With RBAC
VECTORS_DB_API_KEYS='[{"key":"admin-key","role":"admin"},{"key":"read-key","role":"read"}]' cargo run --release

# With TLS
cargo run --release -- --tls-cert cert.pem --tls-key key.pem

# With memory limit and auto-snapshots
cargo run --release -- --max-memory-mb 4096 --snapshot-interval 300

# Cluster mode (Raft)
cargo run --release -- --node-id 1 --peers "2=host2:3030,3=host3:3030"
```

## Architecture

```
src/
├── main.rs                 # CLI args, WAL replay, Raft init, graceful shutdown
├── lib.rs                  # Module declarations
├── config.rs               # Constants (limits, defaults, BM25/HNSW params)
├── document.rs             # Document struct, MetadataValue enum
├── api/
│   ├── mod.rs              # Axum router, middleware stack (auth, metrics, request-id, rate-limit, timeout, body-limit)
│   ├── handlers.rs         # All HTTP handlers, AppState struct
│   ├── models.rs           # Request/response DTOs
│   ├── errors.rs           # ApiError enum → HTTP status codes
│   ├── rbac.rs             # Role-based access control (Read/Write/Admin)
│   └── metrics.rs          # Prometheus metrics (http_requests_total, etc.)
├── hnsw/
│   ├── graph.rs            # HnswIndex struct, HnswConfig, layer management
│   ├── search.rs           # search_layer (generic filter_fn), knn_search, knn_search_filtered
│   ├── insert.rs           # HNSW insertion with layer selection
│   ├── distance.rs         # Cosine, Euclidean, DotProduct (f32 + quantized u8)
│   └── visited.rs          # VisitedSet for graph traversal
├── bm25/
│   ├── inverted_index.rs   # InvertedIndex with postings lists
│   ├── scorer.rs           # BM25 Okapi scoring (k1=1.2, b=0.75)
│   └── tokenizer.rs        # Whitespace tokenizer (no stemming/stopwords)
├── quantization/
│   └── scalar.rs           # f32→u8 scalar quantization with min/scale per vector
├── search/
│   ├── types.rs            # ScoredDocument
│   ├── filter.rs           # matches_filter() for metadata predicates
│   └── hybrid.rs           # rrf_fusion(), linear_fusion()
├── storage/
│   ├── collection.rs       # Collection, CollectionData, Database structs
│   ├── wal.rs              # Write-Ahead Log (CRC32 + fsync)
│   └── persistence.rs      # Bincode snapshot save/load, atomic writes
└── cluster/
    ├── types.rs            # openraft TypeConfig, LogEntry enum, NodeId
    ├── store.rs            # InMemoryLogStore, StateMachineStore (applies to Database)
    ├── network.rs          # HTTP-based Raft RPC (reqwest)
    └── api.rs              # Raft internal routes (/raft/vote, /raft/append, etc.)
```

## Key Design Decisions

- **In-memory only**: All data lives in RAM. WAL + snapshots provide durability, not disk-based storage.
- **Scalar quantization**: f32→u8 reduces memory 4x. Asymmetric distance (query f32 vs stored u8) for search, f32 reranking for accuracy.
- **Pre-filtering in HNSW**: Filter predicate `Fn(u32) -> bool` applied during graph traversal (not post-retrieval). Filtered nodes still used for navigation.
- **Centralized ID mapping**: `uuid_to_internal: HashMap<Uuid, u32>` and `internal_to_uuid: Vec<Uuid>` shared by HNSW and BM25 indices.
- **parking_lot::RwLock**: Low-contention read-write locks. Collection-level granularity.
- **openraft 0.9 with storage-v2**: Split `RaftLogStorage` + `RaftStateMachine` traits. BTreeMap-based in-memory log store.

## API Routes

### Public (no auth)
- `GET /health` — `{"status":"ok","version":"0.1.0"}`
- `GET /metrics` — Prometheus text format

### Collections (auth required)
- `POST /collections` — Create: `{"name","dimension","m?","ef_construction?","ef_search?","distance_metric?"}`
- `GET /collections` — List all
- `DELETE /collections/:name` — Delete

### Documents (auth required)
- `POST /collections/:name/documents` — Insert: `{"text","embedding","metadata?","id?"}`
- `POST /collections/:name/documents/batch` — Batch insert (max 1000): `{"documents":[...]}`
- `GET /collections/:name/documents/:id` — Get by UUID
- `PUT /collections/:name/documents/:id` — Update: `{"text?","embedding?","metadata?"}`
- `DELETE /collections/:name/documents/:id` — Delete

### Search (auth required)
- `POST /collections/:name/search` — `{"query_text?","query_embedding?","k","offset?","min_similarity?","alpha?","fusion_method?","filter?"}`
  - Vector-only: provide `query_embedding`
  - Keyword-only: provide `query_text`
  - Hybrid: provide both
  - Filter example: `{"filter":{"must":[{"field":"category","op":"eq","value":"science"}]}}`
  - Operators: `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in`

### Admin (auth required, Admin role in RBAC)
- `POST /collections/:name/save` — Snapshot to disk
- `POST /collections/:name/load` — Load from snapshot
- `GET /collections/:name/stats` — Memory, doc count, dimension, deleted count
- `POST /admin/compact` — Save all + truncate WAL
- `POST /admin/rebuild/:name` — Rebuild HNSW+BM25 indices (compaction)
- `POST /admin/backup` — Backup all collections
- `POST /admin/restore` — Restore from backups
- `GET /admin/routing` — Cluster routing table
- `POST /admin/assign` — Assign collection to node: `{"collection","node_id"}`

### Raft Internal (no auth, cluster mode only)
- `POST /raft/vote`, `/raft/append`, `/raft/snapshot`
- `POST /raft/init`, `/raft/add-learner`, `/raft/change-membership`

## Configuration Constants (src/config.rs)

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_DIMENSION` | 4096 | Max embedding dimensions |
| `MAX_K` | 10,000 | Max results per search |
| `MAX_BATCH_SIZE` | 1,000 | Max docs per batch insert |
| `MAX_TEXT_LEN` | 1,000,000 | Max document text bytes |
| `MAX_OFFSET` | 100,000 | Max pagination offset |
| `MAX_REQUEST_BODY_BYTES` | 10 MB | HTTP body size limit |
| `REQUEST_TIMEOUT_SECS` | 30 | Per-request timeout |
| `RATE_LIMIT_RPS` | 100 | Requests per second |
| `BM25_K1` / `BM25_B` | 1.2 / 0.75 | BM25 Okapi parameters |
| `HNSW_DEFAULT_M` | 16 | Default HNSW connections |
| `HNSW_DEFAULT_EF_CONSTRUCTION` | 200 | Default build quality |
| `HNSW_DEFAULT_EF_SEARCH` | 50 | Default search quality |

## Testing

- Integration tests in `tests/api_tests.rs` — spawn full Axum server per test with `spawn_app()` / `spawn_app_full()`
- Tests use `reqwest::Client` against actual HTTP endpoints
- `spawn_app_full()` takes auth config; `spawn_app()` is unauthenticated
- AppState in tests: `raft: None, node_id: None, routing_table: None, peer_addrs: None`
- All tests must pass: `cargo test` (51 tests currently)

## Benchmark Data

Datasets in `benchmarks/data/` (HDF5 format, ~2GB total). Download via `benches/convert_hdf5.py`.

| Benchmark | Dataset | Vectors | Dims | Metric |
|-----------|---------|---------|------|--------|
| ann_glove25 | GloVe-25 | 1.18M | 25 | Cosine |
| ann_glove100 | GloVe-100 | 1.18M | 100 | Cosine |
| ann_sift128 | SIFT-128 | 1M | 128 | Euclidean |
| bm25_nfcorpus | NFCorpus (BEIR) | 3,633 docs | — | BM25 |

## Code Conventions

- Zero clippy warnings (`cargo clippy -- -D warnings`)
- `cargo fmt` enforced
- Error types: use `ApiError` variants, never panic in handlers
- Locks: `parking_lot::RwLock`, prefer read locks, minimize write lock scope
- IDs: UUID externally, u32 internally (via `uuid_to_internal`/`internal_to_uuid`)
- Serialization: `serde` for API (JSON), `bincode` for persistence
- No disk-based storage — the database is in-memory by design
