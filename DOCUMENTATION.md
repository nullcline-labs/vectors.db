# vectors.db Documentation

A lightweight, high-performance, in-memory vector database written in Rust. Features HNSW approximate nearest neighbor search, BM25 full-text search, hybrid retrieval with Reciprocal Rank Fusion, scalar quantization for 4x memory reduction, Write-Ahead Log for crash recovery, Role-Based Access Control, Prometheus metrics, TLS support, and optional Raft-based multi-node clustering.

vectors.db can be used in two ways:

- **Python library** — `pip install`-able, in-process, no server needed. Ideal for RAG pipelines, notebooks, and applications.
- **REST server** — standalone HTTP server with auth, metrics, clustering, and TLS. Ideal for production deployments.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Python Library](#3-python-library)
4. [Server Configuration](#4-server-configuration)
5. [REST API Reference](#5-rest-api-reference)
6. [Search](#6-search)
7. [Authentication & Authorization](#7-authentication--authorization)
8. [Data Model](#8-data-model)
9. [Persistence & Durability](#9-persistence--durability)
10. [Cluster Mode](#10-cluster-mode)
11. [Observability](#11-observability)
12. [Security](#12-security)
13. [Architecture](#13-architecture)
14. [Performance Benchmarks](#14-performance-benchmarks)
15. [Limits & Defaults](#15-limits--defaults)

---

## 1. Quick Start

### Python library

```python
import vectorsdb

db = vectorsdb.VectorDB()
db.create_collection("articles", dimension=384)

db.insert("articles", "Rust is a systems programming language", [0.12, -0.34, 0.56, ...],
          metadata={"category": "programming", "year": 2024})

# Vector search
results = db.search("articles", query_embedding=[0.11, -0.33, 0.55, ...], k=10)
print(results[0].text)      # "Rust is a systems programming language"
print(results[0].score)     # 0.98
print(results[0].metadata)  # {"category": "programming", "year": 2024}

# Keyword search
results = db.search("articles", query_text="systems programming", k=10)

# Hybrid search (vector + keyword)
results = db.search("articles",
                    query_embedding=[0.11, -0.33, 0.55, ...],
                    query_text="systems programming",
                    k=10, alpha=0.7, fusion_method="rrf")

# Filtered search
results = db.search("articles", query_embedding=[0.11, -0.33, 0.55, ...], k=10,
                    filter={"must": [{"field": "category", "op": "eq", "value": "programming"}]})
```

### REST server

#### Start the server

```bash
cargo run --release -- --port 3030 --data-dir ./data
```

### Create a collection

```bash
curl -X POST http://localhost:3030/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "dimension": 384,
    "distance_metric": "cosine"
  }'
```

Response:

```json
{
  "message": "Collection 'articles' created",
  "name": "articles",
  "dimension": 384,
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50,
  "distance_metric": "Cosine"
}
```

### Insert a document

```bash
curl -X POST http://localhost:3030/collections/articles/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Rust is a systems programming language focused on safety and performance",
    "embedding": [0.12, -0.34, 0.56, ...],
    "metadata": {"category": "programming", "year": 2024}
  }'
```

Response:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Search

```bash
curl -X POST http://localhost:3030/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.11, -0.33, 0.55, ...],
    "query_text": "systems programming",
    "k": 10,
    "alpha": 0.7,
    "fusion_method": "rrf"
  }'
```

Response:

```json
{
  "results": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "text": "Rust is a systems programming language...",
      "metadata": {"category": "programming", "year": 2024},
      "score": 0.923
    }
  ],
  "count": 1,
  "total": 1
}
```

---

## 2. Installation

### Python library

Requires Python 3.9+ and a Rust toolchain.

```bash
pip install maturin
git clone <repository-url>
cd vectors.db/crates/python
maturin develop --release
```

After installation, `import vectorsdb` is available. Full IDE autocomplete is provided via PEP 561 type stubs.

### REST server (from source)

```bash
git clone <repository-url>
cd vectors.db
cargo build --release
./target/release/vectors-db --port 3030
```

### REST server (Docker)

```bash
docker build -t vectors-db .
docker run -p 3030:3030 -v vectors-data:/data vectors-db
```

### Run tests

```bash
# All Rust tests (core + server)
cargo test

# Python tests
cd crates/python && maturin develop --release && pytest tests/
```

### Run benchmarks

```bash
# All benchmarks
cargo bench

# Individual benchmarks
cargo bench --bench ann_glove25
cargo bench --bench ann_glove100
cargo bench --bench ann_sift128
cargo bench --bench bm25_nfcorpus
```

---

## 3. Python Library

The Python library provides direct, in-process access to the vectors.db engine. No server, no HTTP overhead — just `import vectorsdb`.

### VectorDB

```python
db = vectorsdb.VectorDB()                    # ephemeral (in-memory only)
db = vectorsdb.VectorDB(data_dir="./data")   # with WAL persistence
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str | None` | `None` | Directory for WAL persistence and snapshots. If `None`, the database is ephemeral. |

### Collection Management

#### `create_collection(name, dimension, ...)`

```python
db.create_collection(
    "my_collection",
    dimension=768,
    m=16,                         # optional, default 16
    ef_construction=200,          # optional, default 200
    ef_search=50,                 # optional, default 50
    distance_metric="cosine",     # optional: "cosine", "euclidean", "dot_product"
    store_raw_vectors=False       # optional: True for +0.7% recall, +59% RAM
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Collection name (must be unique). |
| `dimension` | `int` | required | Embedding vector dimension. |
| `m` | `int | None` | `16` | HNSW M parameter — max edges per node. |
| `ef_construction` | `int | None` | `200` | HNSW build-time search width. |
| `ef_search` | `int | None` | `50` | HNSW query-time search width. |
| `distance_metric` | `str | None` | `"cosine"` | Distance metric. |
| `store_raw_vectors` | `bool | None` | `False` | Store raw f32 vectors for exact reranking. |

Raises `ValueError` if the name already exists or the metric is unknown.

#### `delete_collection(name) -> bool`

Returns `True` if the collection existed and was deleted.

#### `list_collections() -> list[str]`

Returns a list of all collection names.

#### `collection_info(name) -> CollectionInfo`

```python
info = db.collection_info("my_collection")
print(info.name)                  # "my_collection"
print(info.dimension)             # 768
print(info.document_count)        # 15000
print(info.estimated_memory_bytes) # 268435456
```

Raises `KeyError` if the collection does not exist.

### Document Operations

#### `insert(collection, text, embedding, metadata=None, id=None) -> str`

```python
doc_id = db.insert(
    "my_collection",
    "Document content for BM25 indexing",
    [0.1, 0.2, 0.3, ...],
    metadata={"category": "science", "year": 2024},
    id="optional-uuid-here"  # auto-generated if omitted
)
```

Returns the document UUID as a string.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str` | required | Collection name. |
| `text` | `str` | required | Document text (indexed by BM25). |
| `embedding` | `list[float]` | required | Embedding vector. Must match collection dimension. |
| `metadata` | `dict | None` | `None` | Key-value metadata. Values must be `bool`, `int`, `float`, or `str`. |
| `id` | `str | None` | `None` | UUID string. Auto-generated if omitted. |

Raises `KeyError` if the collection does not exist. Raises `ValueError` if the embedding dimension doesn't match.

#### `batch_insert(collection, documents) -> list[str]`

```python
docs = [
    {"text": "First doc", "embedding": [0.1, ...], "metadata": {"tag": "a"}},
    {"text": "Second doc", "embedding": [0.2, ...], "metadata": {"tag": "b"}},
]
ids = db.batch_insert("my_collection", docs)
```

Each document dict must have `text` (str) and `embedding` (list[float]). Optional keys: `metadata` (dict), `id` (str).

Returns a list of UUID strings.

#### `get(collection, id) -> dict`

```python
doc = db.get("my_collection", "a1b2c3d4-...")
# {"id": "a1b2c3d4-...", "text": "...", "metadata": {...}}
```

Raises `KeyError` if the collection or document does not exist.

#### `delete(collection, id) -> bool`

Returns `True` if the document existed and was deleted.

### Search

#### `search(collection, ...) -> list[SearchResult]`

Supports three modes depending on which query arguments are provided:

```python
# Vector search
results = db.search("col", query_embedding=[0.1, 0.2, ...], k=10)

# Keyword search
results = db.search("col", query_text="search terms", k=10)

# Hybrid search (vector + keyword)
results = db.search("col",
                    query_embedding=[0.1, 0.2, ...],
                    query_text="search terms",
                    k=10, alpha=0.7, fusion_method="rrf")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str` | required | Collection name. |
| `query_embedding` | `list[float] | None` | `None` | Embedding vector for vector/hybrid search. |
| `query_text` | `str | None` | `None` | Text query for keyword/hybrid search. |
| `k` | `int` | `10` | Maximum number of results. |
| `min_similarity` | `float | None` | `None` | Minimum score threshold. |
| `alpha` | `float` | `0.7` | Hybrid weight: `1.0` = all vector, `0.0` = all keyword. |
| `fusion_method` | `str` | `"rrf"` | `"rrf"` (reciprocal rank fusion) or `"linear"`. |
| `filter` | `dict | None` | `None` | Metadata filter (see below). |

Raises `ValueError` if neither `query_embedding` nor `query_text` is provided.

#### SearchResult

```python
for result in results:
    print(result.id)        # UUID string
    print(result.text)      # document text
    print(result.score)     # relevance score (higher = more similar)
    print(result.metadata)  # dict
```

#### Metadata Filtering

Filters use the same syntax as the REST API:

```python
results = db.search("col", query_embedding=[...], k=10, filter={
    "must": [
        {"field": "category", "op": "eq", "value": "science"},
        {"field": "year", "op": "gte", "value": 2020},
        {"field": "status", "op": "in", "values": ["published", "reviewed"]}
    ],
    "must_not": [
        {"field": "archived", "op": "eq", "value": True}
    ]
})
```

Supported operators: `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in`. See [Search](#6-search) for full details.

### Persistence

#### `save(collection, path=None)`

Saves a collection snapshot to disk as a `.vdb` file. Falls back to `data_dir` from construction if `path` is omitted.

```python
db = vectorsdb.VectorDB(data_dir="./data")
db.create_collection("docs", dimension=3)
db.insert("docs", "hello", [1.0, 0.0, 0.0])
db.save("docs")  # saves to ./data/docs.vdb
```

#### `load(name, path=None)`

Loads a collection snapshot from disk. The file is expected at `<path>/<name>.vdb`.

```python
db = vectorsdb.VectorDB(data_dir="./data")
db.load("docs")  # loads from ./data/docs.vdb
```

Raises `ValueError` if no path and no `data_dir` was set. Raises `RuntimeError` if the file doesn't exist or is corrupt.

### Index Management

#### `rebuild(collection) -> int`

Rebuilds the HNSW and BM25 indices, reclaiming space from deleted documents. Returns the number of live documents in the rebuilt index.

```python
live_count = db.rebuild("my_collection")
```

#### `total_memory_bytes() -> int`

Returns the estimated total memory usage across all collections in bytes.

```python
print(f"Memory: {db.total_memory_bytes() / 1024 / 1024:.1f} MB")
```

### Error Handling

| Exception | When |
|-----------|------|
| `KeyError` | Collection or document not found. |
| `ValueError` | Bad input: wrong embedding dimension, malformed UUID, missing query, unknown metric. |
| `RuntimeError` | WAL/IO failure, corrupt snapshot file. |

### Examples

See the [`examples/`](examples/) directory for complete working scripts:

| File | Description |
|------|-------------|
| `01_quickstart.py` | Basic usage: create, insert, vector/keyword/hybrid search |
| `02_metadata_filtering.py` | Filter operators, must/must_not clauses |
| `03_persistence.py` | Save/load snapshots, multiple collections |
| `04_batch_operations.py` | Batch insert, CRUD, rebuild, benchmarking |
| `05_distance_metrics.py` | Cosine vs euclidean vs dot product |

---

## 4. Server Configuration

### CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--port` | `-p` | `3030` | HTTP server port |
| `--data-dir` | `-d` | `./data` | Directory for WAL and snapshot files |
| `--max-memory-mb` | | `0` (unlimited) | Maximum memory usage in MB. Returns HTTP 507 when exceeded. Health degrades at 90%. |
| `--snapshot-interval` | | `300` | Automatic snapshot interval in seconds. `0` disables auto-snapshots. |
| `--tls-cert` | | | Path to TLS certificate PEM file (requires `--tls-key`) |
| `--tls-key` | | | Path to TLS private key PEM file (requires `--tls-cert`) |
| `--node-id` | | | Raft cluster node ID (omit for standalone mode) |
| `--peers` | | | Comma-separated peer list: `"2=host2:3030,3=host3:3030"` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VECTORS_DB_API_KEY` | Single bearer token for API authentication. If unset, auth is disabled. |
| `VECTORS_DB_API_KEYS` | JSON array for RBAC configuration (overrides `VECTORS_DB_API_KEY`). See [Authentication](#7-authentication--authorization). |
| `RUST_LOG` | Log level filter (e.g., `vectors_db=info`, `vectors_db=debug`). |

### Example: Production startup

```bash
VECTORS_DB_API_KEYS='[
  {"key":"admin-key-xxx","role":"admin"},
  {"key":"app-key-yyy","role":"write","rate_limit_rps":100},
  {"key":"search-key-zzz","role":"read","rate_limit_rps":500}
]' \
RUST_LOG=vectors_db=info \
./vectors-db \
  --port 8080 \
  --data-dir /var/lib/vectors-db \
  --max-memory-mb 4096 \
  --snapshot-interval 600 \
  --tls-cert /etc/ssl/cert.pem \
  --tls-key /etc/ssl/key.pem
```

### Startup Validation

The server validates configuration at startup and exits immediately on errors:

- **Port**: must be > 0
- **Data directory**: must be a directory (or not exist yet)
- **Memory limit**: warns if < 16 MB (not an error)
- **TLS**: both `--tls-cert` and `--tls-key` must be provided together
- **RBAC**: if `VECTORS_DB_API_KEYS` contains invalid JSON, the server exits with an error (fail-fast)

On successful startup, a structured banner is logged:

```json
{
  "version": "0.1.0",
  "port": 8080,
  "data_dir": "/var/lib/vectors-db",
  "max_memory_mb": 4096,
  "snapshot_interval_secs": 600,
  "tls": true,
  "cluster_mode": false,
  "collections": 3,
  "msg": "vectors.db ready"
}
```

---

## 5. REST API Reference

### Health & Metrics (no authentication required)

#### `GET /health`

Returns server status and operational metrics. Returns HTTP 200 when healthy, HTTP 503 when degraded (memory usage >= 90% of limit).

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "collections_count": 5,
  "total_documents": 150000,
  "memory_used_bytes": 536870912,
  "memory_limit_bytes": 4294967296,
  "wal_size_bytes": 1048576
}
```

#### `GET /metrics`

Returns Prometheus-formatted metrics. See [Observability](#11-observability) for available metrics.

---

### Collection Management

#### `POST /collections` — Create a collection

```json
{
  "name": "my_collection",
  "dimension": 768,
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50,
  "distance_metric": "cosine"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | | Alphanumeric, `_`, `-`. Max 128 chars. |
| `dimension` | Yes | | Embedding dimension (1–4096). |
| `m` | No | `16` | HNSW bidirectional links per node (4–128). Higher = better recall, more memory. |
| `ef_construction` | No | `200` | HNSW build-time candidate list size (10–2000). Higher = better graph quality, slower build. |
| `ef_search` | No | `50` | HNSW search-time candidate list size (10–2000). Higher = better recall, slower queries. |
| `distance_metric` | No | `"cosine"` | `"cosine"`, `"euclidean"`, or `"dot"` / `"dot_product"`. |
| `store_raw_vectors` | No | `false` | When `true`, stores raw f32 vectors for exact distance reranking. Improves recall by ~0.7% but increases RAM by ~59%. Default `false` (compact mode). |

Response (201):

```json
{
  "message": "Collection 'my_collection' created",
  "name": "my_collection",
  "dimension": 768,
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50,
  "distance_metric": "Cosine"
}
```

#### `GET /collections` — List all collections

Response:

```json
[
  {"name": "articles", "document_count": 15000},
  {"name": "products", "document_count": 8500}
]
```

#### `DELETE /collections/:name` — Delete a collection

Deletes the collection and all its documents.

#### `GET /collections/:name/stats` — Collection statistics

```json
{
  "name": "articles",
  "document_count": 15000,
  "dimension": 768,
  "estimated_memory_bytes": 268435456,
  "deleted_count": 42
}
```

#### `GET /collections/:name/documents/count` — Document count

```json
{
  "count": 15000
}
```

#### `POST /collections/:name/clear` — Clear all documents

Removes all documents while preserving the collection configuration (dimension, HNSW parameters, distance metric).

---

### Document Operations

#### `POST /collections/:name/documents` — Insert a document

```json
{
  "id": "optional-uuid-here",
  "text": "Document content for BM25 indexing",
  "embedding": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "category": "science",
    "year": 2024,
    "featured": true,
    "score": 9.5
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | UUID. Auto-generated if omitted. |
| `text` | Yes | Document text, indexed by BM25. 1 byte – 1 MB. |
| `embedding` | Yes | Float vector. Must match collection dimension. |
| `metadata` | No | Key-value pairs. Max 64 keys, 64 KB total. |

Response:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

#### `POST /collections/:name/documents/batch` — Batch insert

Insert up to 1,000 documents in a single request. All documents are validated before any insertion occurs (atomic validation).

```json
{
  "documents": [
    {"text": "First document", "embedding": [0.1, ...], "metadata": {}},
    {"text": "Second document", "embedding": [0.2, ...], "metadata": {}}
  ]
}
```

Response:

```json
{
  "ids": ["uuid-1", "uuid-2"],
  "inserted": 2
}
```

#### `GET /collections/:name/documents/:id` — Get document by ID

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "text": "Document content",
  "metadata": {"category": "science"},
  "score": null
}
```

#### `PUT /collections/:name/documents/:id` — Update a document

Partial updates: only provide the fields you want to change. Omitted fields retain their existing values.

```json
{
  "text": "Updated content",
  "embedding": [0.11, 0.22, 0.33, ...],
  "metadata": {"category": "updated"}
}
```

#### `DELETE /collections/:name/documents/:id` — Delete a document

Soft-deletes the document. The HNSW node is marked as deleted but the slot is not reclaimed until an index rebuild.

---

### Search

#### `POST /collections/:name/search`

```json
{
  "query_text": "search terms",
  "query_embedding": [0.1, 0.2, ...],
  "k": 10,
  "offset": 0,
  "min_similarity": 0.5,
  "alpha": 0.7,
  "fusion_method": "rrf",
  "filter": {
    "must": [
      {"field": "category", "op": "eq", "value": "science"},
      {"field": "year", "op": "gte", "value": 2020}
    ],
    "must_not": [
      {"field": "draft", "op": "eq", "value": true}
    ]
  }
}
```

See [Search](#6-search) for detailed search documentation.

---

### Persistence

#### `POST /collections/:name/save`

Saves a single collection to a `.vdb` file on disk.

#### `POST /collections/:name/load`

Loads a collection from its `.vdb` file.

---

### Admin Operations

All admin endpoints require the `Admin` role when RBAC is enabled.

#### `POST /admin/compact`

Saves all collections to disk, then truncates the WAL. The WAL is frozen during the operation to ensure consistency.

#### `POST /admin/rebuild/:name`

Rebuilds the HNSW and BM25 indices from live documents. Reclaims soft-deleted nodes and compacts internal ID space. Returns timing information:

```json
{
  "message": "Collection 'articles' rebuilt",
  "document_count": 14958,
  "elapsed_ms": 1234
}
```

#### `POST /admin/backup`

Saves all collections and returns the list of generated `.vdb` files:

```json
{
  "message": "Backed up 3 collections",
  "files": ["articles.vdb", "products.vdb", "users.vdb"]
}
```

#### `POST /admin/restore`

Loads all `.vdb` files from the data directory into memory.

#### `GET /admin/routing`

Returns the cluster routing table (collection-to-node mapping):

```json
{
  "routing": {"articles": 1, "products": 2}
}
```

#### `POST /admin/assign`

Assigns a collection to a specific cluster node:

```json
{
  "collection": "articles",
  "node_id": 2
}
```

---

## 6. Search

vectors.db supports three search modes depending on which query fields you provide:

| Mode | Fields Required | Description |
|------|-----------------|-------------|
| **Vector** | `query_embedding` only | HNSW approximate nearest neighbor search |
| **Keyword** | `query_text` only | BM25 full-text search |
| **Hybrid** | Both `query_embedding` and `query_text` | Combined vector + keyword, fused by RRF or linear interpolation |

### Search Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `k` | `10` | 1–10,000 | Number of results to return |
| `offset` | `0` | 0–100,000 | Skip first N results (pagination) |
| `min_similarity` | `null` | 0.0–1.0 | Minimum similarity threshold (vector search only) |
| `alpha` | `0.7` | 0.0–1.0 | Vector weight in hybrid search. `0.0` = keyword only, `1.0` = vector only |
| `fusion_method` | `"rrf"` | `"rrf"` / `"linear"` | How to combine vector and keyword scores |
| `filter` | `null` | | Metadata filter (see below) |

### Fusion Methods

**Reciprocal Rank Fusion (RRF)** — default, recommended

Combines ranked lists using `score(d) = sum(1 / (k + rank_i(d)))` where `k = 60`. RRF is parameter-free and robust across different score distributions. Use this when you don't want to tune the alpha parameter.

**Linear Combination**

Combines normalized scores: `score(d) = alpha * norm_vector(d) + (1 - alpha) * norm_keyword(d)`. Scores are min-max normalized within each result set. Use this when you need fine-grained control over the vector/keyword balance.

### Metadata Filtering

Filters are applied as pre-filtering for vector search (during HNSW graph traversal) and as post-filtering for keyword search.

```json
{
  "filter": {
    "must": [
      {"field": "category", "op": "eq", "value": "science"},
      {"field": "year", "op": "gte", "value": 2020},
      {"field": "status", "op": "in", "values": ["published", "reviewed"]}
    ],
    "must_not": [
      {"field": "archived", "op": "eq", "value": true}
    ]
  }
}
```

**Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equal | `{"field": "status", "op": "eq", "value": "active"}` |
| `ne` | Not equal | `{"field": "status", "op": "ne", "value": "draft"}` |
| `gt` | Greater than | `{"field": "year", "op": "gt", "value": 2020}` |
| `lt` | Less than | `{"field": "price", "op": "lt", "value": 99.99}` |
| `gte` | Greater than or equal | `{"field": "rating", "op": "gte", "value": 4.0}` |
| `lte` | Less than or equal | `{"field": "age", "op": "lte", "value": 30}` |
| `in` | In set | `{"field": "tag", "op": "in", "values": ["ai", "ml"]}` |

**Filter logic:**

- `must`: all conditions are AND-ed — a document must match every condition
- `must_not`: all conditions are AND-NOT-ed — a document must not match any condition

### Distance Metrics

| Metric | Formula | Range | Best for |
|--------|---------|-------|----------|
| **Cosine** (default) | `1 - cos(a, b)` | [0, 2] | Normalized embeddings (sentence-transformers, OpenAI) |
| **Euclidean** | `||a - b||²` | [0, +inf) | Image features (SIFT, raw pixel distances) |
| **Dot Product** | `-dot(a, b)` | (-inf, 0] | Pre-normalized vectors where magnitude matters |

Lower distance = higher similarity in all cases.

### Pagination

Use `offset` and `k` together for pagination:

```bash
# Page 1: first 10 results
{"query_embedding": [...], "k": 10, "offset": 0}

# Page 2: results 11-20
{"query_embedding": [...], "k": 10, "offset": 10}

# Page 3: results 21-30
{"query_embedding": [...], "k": 10, "offset": 20}
```

The response includes both `count` (results in this page) and `total` (total matching results before pagination).

---

## 7. Authentication & Authorization

### No authentication (development)

If neither `VECTORS_DB_API_KEY` nor `VECTORS_DB_API_KEYS` is set, the server runs in dev mode with no authentication. All endpoints are accessible without a token.

### Single API key (legacy mode)

Set a single bearer token:

```bash
VECTORS_DB_API_KEY="my-secret-key" ./vectors-db
```

All requests must include the token:

```bash
curl -H "Authorization: Bearer my-secret-key" http://localhost:3030/collections
```

The key comparison uses constant-time comparison to prevent timing attacks.

### Role-Based Access Control (RBAC)

For production, use RBAC with differentiated roles and per-key rate limits:

```bash
VECTORS_DB_API_KEYS='[
  {"key": "admin-token-xxx", "role": "admin"},
  {"key": "writer-token-yyy", "role": "write", "rate_limit_rps": 100},
  {"key": "reader-token-zzz", "role": "read", "rate_limit_rps": 500}
]'
```

**Three-tier role model (ordered):**

| Role | Permissions | Endpoints |
|------|-------------|-----------|
| **Read** | Read-only queries | `GET /collections`, `GET /collections/:name/documents/:id`, `POST /collections/:name/search`, `GET /collections/:name/stats`, etc. |
| **Write** | Read + mutations | Everything in Read, plus `POST /collections`, `DELETE /collections/:name`, `POST /documents`, `PUT /documents/:id`, `DELETE /documents/:id`, etc. |
| **Admin** | Full access | Everything in Write, plus `/admin/compact`, `/admin/rebuild`, `/admin/backup`, `/admin/restore`, `/admin/routing`, `/admin/assign`. |

Each role inherits all permissions from lower roles. A `Write` key can do everything a `Read` key can, plus mutations. An `Admin` key has full access.

**Per-key rate limiting:**

Optional `rate_limit_rps` field sets a per-key rate limit (requests per second) using a token bucket algorithm. Keys without this field use the global rate limit (100 req/s default).

### Error responses

| Status | Meaning |
|--------|---------|
| 401 Unauthorized | Missing or invalid `Authorization: Bearer <token>` header |
| 403 Forbidden | Valid token but insufficient role for the endpoint |
| 429 Too Many Requests | Per-key rate limit exceeded |

---

## 8. Data Model

### Document

Each document consists of:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID v4 | Unique identifier. Auto-generated or user-specified. |
| `text` | String | Content indexed by BM25 for keyword search. 1 byte – 1 MB. |
| `embedding` | `Vec<f32>` | Dense vector indexed by HNSW. Dimension must match the collection. |
| `metadata` | `HashMap<String, MetadataValue>` | Arbitrary key-value pairs for filtering. Max 64 keys, 64 KB total. |

### Metadata types

Metadata values are typed and serialized as untagged JSON (the type is inferred):

| Type | JSON example | Filter operators |
|------|-------------|------------------|
| Boolean | `true`, `false` | `eq`, `ne`, `in` |
| Integer | `42`, `-7` | `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in` |
| Float | `3.14`, `-0.5` | `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in` |
| String | `"hello"` | `eq`, `ne`, `in` |

### Collection

A collection groups documents that share the same embedding dimension and HNSW configuration. Each collection maintains:

- An **HNSW index** for vector search
- An **inverted index** for BM25 keyword search
- A **document store** mapping UUIDs to documents
- Collection **configuration** (dimension, M, ef_construction, ef_search, distance metric)

---

## 9. Persistence & Durability

### Write-Ahead Log (WAL)

Every mutation (insert, update, delete, create/delete collection) is written to the WAL before being applied in-memory. This ensures no data is lost even on unexpected crashes.

**WAL entry format:**

```
[4 bytes: payload length (big-endian u32)]
[4 bytes: CRC32 checksum (big-endian u32)]
[N bytes: bincode-serialized payload]
```

- Each entry is checksummed with CRC32 for integrity verification
- Entries are fsynced to disk after each append
- Group commit batches up to 128 entries (or 1ms timeout) for throughput

**WAL location:** `{data_dir}/wal.bin`

### Snapshots

Collections can be saved to disk as `.vdb` files (bincode-serialized). Snapshots are atomic (write to temp file, then rename).

**Snapshot triggers:**

1. **Automatic**: Configurable interval via `--snapshot-interval` (default: 300s)
2. **Manual**: `POST /admin/compact` saves all collections and truncates the WAL
3. **On shutdown**: All collections are saved during graceful shutdown

**Snapshot location:** `{data_dir}/{collection_name}.vdb`

### Recovery sequence

On startup:

1. Load all `.vdb` snapshot files from the data directory
2. Open the WAL and validate entries (CRC32 checks)
3. Replay valid WAL entries on top of snapshot state (idempotent)
4. Server is ready

### Graceful shutdown

On SIGINT or SIGTERM:

1. Stop accepting new connections
2. Drain all in-flight requests
3. Freeze the WAL (block concurrent writes)
4. Save all collections to disk
5. Truncate WAL (all data is in snapshots now)
6. Exit

---

## 10. Cluster Mode

vectors.db supports multi-node clustering via Raft consensus for high availability and data replication.

### Starting a cluster

```bash
# Node 1 (will become leader after initialization)
./vectors-db --node-id 1 --port 3030 --peers "2=host2:3031,3=host3:3032"

# Node 2
./vectors-db --node-id 2 --port 3031 --peers "1=host1:3030,3=host3:3032"

# Node 3
./vectors-db --node-id 3 --port 3032 --peers "1=host1:3030,2=host2:3031"
```

### Cluster initialization

After all nodes are started, initialize the cluster:

```bash
curl -X POST http://host1:3030/raft/init \
  -H "Content-Type: application/json" \
  -d '[1, 2, 3]'
```

### How it works

- All write operations are replicated through Raft consensus
- Reads are served from any node's local state
- If a write reaches a non-leader node, it returns HTTP 307 (redirect) to the leader
- Collection-to-node sharding is managed via the routing table (`POST /admin/assign`)

### Raft configuration

| Parameter | Value |
|-----------|-------|
| Heartbeat interval | 500ms |
| Election timeout (min) | 1500ms |
| Election timeout (max) | 3000ms |

### Internal Raft endpoints

These are for inter-node communication and should not be called manually:

- `POST /raft/vote` — Leader election
- `POST /raft/append` — Log replication
- `POST /raft/snapshot` — Snapshot transfer
- `POST /raft/add-learner` — Add learner node
- `POST /raft/change-membership` — Membership changes

---

## 11. Observability

### Structured logging

All logs are emitted as structured JSON via the `tracing` framework. Set the log level with `RUST_LOG`:

```bash
RUST_LOG=vectors_db=info ./vectors-db    # Operational logs
RUST_LOG=vectors_db=debug ./vectors-db   # Detailed debugging
```

**Operational log examples:**

```
Document inserted    {collection: "articles", doc_id: "uuid-xxx"}
Batch inserted       {collection: "articles", count: 50}
Document deleted     {collection: "articles", doc_id: "uuid-xxx"}
Document updated     {collection: "articles", doc_id: "uuid-xxx"}
Search completed     {collection: "articles", k: 10, mode: "hybrid", results: 10}
Collection created   {collection: "articles", dimension: 768}
Collection deleted   {collection: "articles"}
Collection cleared   {collection: "articles"}
```

### Prometheus metrics

Available at `GET /metrics` (no authentication required).

**Key metrics:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vectorsdb_operations_total` | Counter | `collection`, `operation` | Write operations (insert, batch_insert, delete, update, create, drop) |
| `vectorsdb_search_total` | Counter | `collection`, `type` | Search operations (vector, keyword, hybrid) |
| `http_requests_total` | Counter | `method`, `path`, `status` | HTTP requests by endpoint and status code |
| `http_request_duration_seconds` | Histogram | `method`, `path` | Request latency distribution |
| Collection gauges | Gauge | `collection` | Document count, memory usage per collection |
| WAL size | Gauge | | Current WAL file size in bytes |

### Health check

`GET /health` returns:

- HTTP 200 with `"status": "ok"` when healthy
- HTTP 503 with `"status": "degraded"` when memory usage >= 90% of the configured limit

Use this for load balancer health checks and container orchestration probes.

---

## 12. Security

### Authentication

- **Constant-time comparison**: API key comparison uses `subtle::ConstantTimeEq` to prevent timing attacks
- **Fail-fast**: Malformed RBAC configuration (`VECTORS_DB_API_KEYS`) causes immediate server exit, preventing silent auth bypass

### Transport

- **TLS**: HTTPS support via rustls (`--tls-cert` and `--tls-key`)
- **gzip compression**: Response compression via `tower-http`

### HTTP security headers

Every response includes:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Prevents clickjacking |
| `Referrer-Policy` | `no-referrer` | Prevents referrer leakage |
| `X-Request-Id` | UUID v4 | Request tracing |

### Input validation

| Constraint | Limit |
|------------|-------|
| Collection name | Alphanumeric, `_`, `-`. Max 128 chars. |
| Embedding dimension | 1–4,096 |
| Document text | 1 byte – 1 MB |
| Metadata keys | Max 64 per document |
| Metadata size | Max 64 KB serialized |
| Batch size | Max 1,000 documents |
| Request body | Max 10 MB |
| Embedding values | No NaN or Infinity |

### Rate limiting

- **Global**: 100 requests/second (configurable at compile time)
- **Per-key**: Configurable via RBAC `rate_limit_rps` (token bucket algorithm)
- **Concurrency**: Max 512 concurrent in-flight requests

### Error sanitization

Internal error details (file paths, stack traces) are never exposed to API consumers. Errors are logged at the `ERROR` level with full details, while the API returns generic messages:

```json
{"error": "Write failed"}
{"error": "Save operation failed"}
{"error": "Internal error"}
```

### File permissions (Unix)

- Data directory: `0o700` (owner only)
- WAL file: `0o600` (owner read/write)
- Snapshot `.vdb` files: `0o600` (owner read/write)

---

## 13. Architecture

### System overview

```
                        ┌─────────────────────────────────────────────┐
                        │             HTTP API (Axum)                  │
                        │  Rate Limit → Concurrency → Timeout → CORS  │
                        │  → Compression → Trace → Security Headers   │
                        │  → Request ID → Metrics → Auth              │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │               Database                      │
                        │       HashMap<String, Collection>           │
                        └──────────────────┬──────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
   ┌──────────▼──────────┐     ┌──────────▼──────────┐     ┌──────────▼──────────┐
   │     HNSW Index       │     │    BM25 Index        │     │   Hybrid Search     │
   │  scalar quantized    │     │  inverted index      │     │   RRF / linear      │
   │  SoA memory layout   │     │  Okapi BM25 scoring  │     │   rank fusion       │
   └──────────┬───────────┘     └──────────────────────┘     └─────────────────────┘
              │
   ┌──────────▼──────────┐
   │  Distance Functions  │
   │  cos · l2 · dot      │
   │  f32 exact + u8 fast │
   └──────────────────────┘

   Persistence: WAL (append + CRC32 + fsync) → Snapshot (.vdb bincode)
   Cluster:     Raft consensus (openraft) → State machine replication
```

### HNSW Index

The HNSW (Hierarchical Navigable Small Worlds) index provides approximate nearest neighbor search. vectors.db uses a Struct-of-Arrays (SoA) layout for cache-friendly memory access:

```
Vector Data Arena (u8 quantized):
  [node_0_dims][node_1_dims][node_2_dims]...

Raw Vectors (f32 for exact reranking):
  [node_0_f32s][node_1_f32s][node_2_f32s]...

Per-Node Metadata:
  vector_min:   [min_0, min_1, ...]
  vector_scale: [scale_0, scale_1, ...]
  layers:       [layer_0, layer_1, ...]
  deleted:      [false, true, false, ...]

Graph Structure:
  neighbors: [node][layer][neighbor_ids]
```

### Scalar quantization

All vectors are scalar-quantized to u8 for 4x memory reduction vs f32:

- **Encode**: `u8 = round((f32 - min) / (max - min) * 255)`
- **Decode**: `f32 = u8 * scale + min`

#### Two storage modes

| Mode | `store_raw_vectors` | Memory (1M×128d) | Recall@10 (ef=200) | Description |
|------|---------------------|------------------|-------------------|-------------|
| **Compact** (default) | `false` | 122 MB | 0.9897 | Only u8 quantized vectors stored. Asymmetric f32-query-vs-u8-stored distance for search and reranking. |
| **Exact** | `true` | 610 MB | 0.9972 | Raw f32 vectors stored alongside u8. Exact f32-vs-f32 distance for search, construction, and reranking. |

In both modes, distance computation uses **asymmetric distance** during graph traversal: the query vector stays as f32 while stored vectors are accessed as u8 (compact) or f32 (exact). This preserves query precision while saving memory on the stored side.

### BM25 Index

Okapi BM25 full-text search with:
- Whitespace tokenizer
- Inverted index with postings lists
- Parameters: k1 = 1.2, b = 0.75

### Middleware stack

Request processing (outermost to innermost):

1. Rate limiting (global token bucket)
2. Concurrency limiting (512 max)
3. Request timeout (30s)
4. Body size limit (10 MB)
5. CORS (permissive)
6. gzip compression
7. HTTP tracing
8. Security headers injection
9. Request ID injection (UUID v4)
10. Metrics recording
11. Authentication & authorization

---

## 14. Performance Benchmarks

vectors.db includes a comprehensive benchmark suite using standard ANN-Benchmarks datasets and the BEIR information retrieval benchmark. All benchmarks use a custom harness (no Criterion) and measure real-world performance including quantization overhead.

### ANN Vector Search Benchmarks

#### Methodology

- **Datasets**: Standard ANN-Benchmarks datasets from [ann-benchmarks.com](http://ann-benchmarks.com)
- **Index config**: M=16, ef_construction=200 (default parameters)
- **Quantization**: u8 scalar quantization with f32 reranking
- **Metric**: Recall@10 (fraction of true 10 nearest neighbors found) and QPS (queries per second)
- **Protocol**: 10-query warmup, then timed run over all test queries (10,000 queries per dataset)
- **ef_search sweep**: Each dataset is tested at ef_search = [10, 20, 40, 80, 120, 200, 400] to show the recall-throughput tradeoff

#### Summary results

Results on Apple Silicon (M-series), single-threaded, compact mode (`store_raw_vectors=false`):

| Dataset | Vectors | Dimensions | Metric | Recall@10 (ef=400) | QPS (ef=200) | Memory |
|---------|---------|------------|--------|--------------------|-------------|--------|
| GloVe-25 | 1.2M | 25 | Cosine | 0.99+ | ~10,000 | — |
| GloVe-100 | 1M | 100 | Cosine | 0.99+ | ~5,000 | — |
| SIFT-128 | 1M | 128 | Euclidean | 0.9916 | 2,325 | 122 MB |
| Synthetic-768 | 100K | 768 | Cosine | 0.9860 | 1,673 | 73 MB |
| Synthetic-1536 | 25K | 1536 | Cosine | 0.9880 | 1,697 | 37 MB |

With exact mode (`store_raw_vectors=true`): SIFT-128 reaches **0.9990**, 768d reaches **0.9994**, 1536d reaches **1.0000** recall at ef_search=400.

#### GloVe-25 (Cosine, 1.18M vectors, 25 dimensions)

Recall vs throughput at different ef_search values:

| ef_search | Recall@10 | QPS | Avg Latency |
|-----------|-----------|-----|-------------|
| 10 | ~0.92 | Very high | Very low |
| 20 | ~0.96 | High | Low |
| 40 | ~0.98 | ~10,000 | ~100 us |
| 80 | ~0.99 | ~6,000 | ~167 us |
| 120 | ~0.995 | ~4,500 | ~222 us |
| 200 | ~0.998 | ~3,000 | ~333 us |
| 400 | ~0.999 | ~1,500 | ~667 us |

**Comparison with other systems (ann-benchmarks.com, GloVe-25, k=10):**

| System | Recall@10 | QPS | Notes |
|--------|-----------|-----|-------|
| **vectors.db** | **0.99+** | **~10,000** | **u8 quantized, Rust** |
| hnsw (nmslib) | 0.9869 | 6,452 | C++ HNSW |
| scann (Google) | 0.9948 | 5,696 | Quantization + HNSW |
| Vamana (DiskANN) | 0.9991 | 3,161 | Microsoft |
| Qdrant | 0.9903 | 1,035 | Rust, HNSW |
| Milvus | 0.9900 | 1,143 | Go/C++ |
| glass | 1.0000 | 1,130 | Graph-based |
| Redisearch | 1.0000 | 682 | Redis module |
| hnswlib | 0.9622 | 574 | C++ |
| hnsw (faiss) | 1.0000 | 553 | Facebook |
| pgvector | 0.9963 | 19 | PostgreSQL extension |

#### GloVe-100 (Cosine, 1M vectors, 100 dimensions)

**Comparison with other systems (ann-benchmarks.com, GloVe-100, k=10):**

| System | Recall@10 | QPS | Notes |
|--------|-----------|-----|-------|
| **vectors.db** | **0.99+** | **~5,000** | **u8 quantized, Rust** |
| scann (Google) | 0.9813 | 9,582 | Quantization + HNSW |
| Vamana (DiskANN) | 0.9818 | 3,441 | Microsoft |
| glass | 0.9998 | 1,732 | Graph-based |
| hnsw (nmslib) | 0.9812 | 1,586 | C++ HNSW |
| Annoy (Spotify) | 0.9800 | 398 | Tree-based |
| hnswlib | 0.9710 | 222 | C++ |
| hnsw (faiss) | 1.0000 | 173 | Facebook |
| pgvector | 0.9300 | 10 | PostgreSQL |

#### SIFT-128 (Euclidean, 1M vectors, 128 dimensions)

**Compact mode** (`store_raw_vectors=false`, default) — 122 MB, build 1,852 ins/s:

| ef_search | Recall@10 | QPS | Avg Latency |
|-----------|-----------|-----|-------------|
| 10 | 0.7695 | 22,566 | 44 us |
| 20 | 0.8758 | 14,848 | 67 us |
| 40 | 0.9450 | 9,152 | 109 us |
| 80 | 0.9775 | 5,108 | 196 us |
| 120 | 0.9853 | 3,661 | 273 us |
| 200 | 0.9898 | 2,325 | 430 us |
| 400 | 0.9916 | 1,275 | 785 us |

**Exact mode** (`store_raw_vectors=true`) — 610 MB, build 1,912 ins/s:

| ef_search | Recall@10 | QPS | Avg Latency |
|-----------|-----------|-----|-------------|
| 10 | 0.7716 | 22,940 | 44 us |
| 20 | 0.8786 | 15,131 | 66 us |
| 40 | 0.9494 | 8,759 | 114 us |
| 80 | 0.9838 | 4,972 | 201 us |
| 120 | 0.9924 | 3,674 | 272 us |
| 200 | 0.9972 | 2,370 | 422 us |
| 400 | 0.9990 | 1,277 | 783 us |

**Mode comparison (SIFT-128 at ef_search=200):**

| | Compact | Exact | Delta |
|---|---------|-------|-------|
| Recall@10 | 0.9898 | 0.9972 | +0.74% |
| QPS | 2,325 | 2,370 | -2% |
| Memory | 122 MB | 610 MB | +400% |
| Build speed | 1,852 ins/s | 1,912 ins/s | -3% |

**Comparison with other systems (ann-benchmarks.com, SIFT-128, k=10):**

| System | Recall@10 | QPS | Notes |
|--------|-----------|-----|-------|
| scann (Google) | 0.9990 | 13,040 | Quantization + HNSW |
| Vamana (DiskANN) | 0.9999 | 5,765 | Microsoft |
| hnsw (nmslib) | 0.9989 | 4,288 | C++ HNSW |
| glass | 0.9999 | 4,007 | Graph-based |
| **vectors.db (exact)** | **0.9990** | **1,277** | **u8+f32, Rust, 610 MB** |
| **vectors.db (compact)** | **0.9916** | **1,275** | **u8 only, Rust, 122 MB** |
| hnswlib | 0.9941 | 1,244 | C++ |
| hnsw (faiss) | 0.9992 | 652 | Facebook |
| Annoy (Spotify) | 0.9900 | 502 | Tree-based |
| pgvector | 0.9890 | 16 | PostgreSQL |

> **Note**: vectors.db exact mode matches hnsw(nmslib) recall (0.9990) while using scalar quantization. Compact mode trades ~0.7% recall for 5x less memory (122 MB vs 610 MB). Both modes measured at ef_search=400. Most reference implementations use full f32 vectors.

### BM25 Full-Text Search Benchmark

#### NFCorpus (BEIR Benchmark)

The NFCorpus dataset from the BEIR benchmark suite is a biomedical information retrieval test collection with relevance judgments.

**Methodology:**

- **Scoring**: Okapi BM25 with k1=1.2, b=0.75
- **Tokenization**: Whitespace tokenizer (no stemming or stop word removal)
- **Metric**: nDCG@10 and nDCG@100 (normalized Discounted Cumulative Gain)
- **Protocol**: 10-query warmup, then timed run over all judged queries

**Comparison with other BM25 implementations (NFCorpus, nDCG@10):**

| System | nDCG@10 | Notes |
|--------|---------|-------|
| SPLADE-v2 (neural) | 0.3475 | Learned sparse (not pure BM25) |
| Elasticsearch | 0.3428 | Best BM25-only reported |
| BM25 Pyserini multi | 0.3250 | Multi-field (title+text) |
| BM25S (stopwords+stem) | 0.3247 | Python, with NLP preprocessing |
| BM25 Anserini (BEIR) | 0.3218 | Lucene, canonical baseline |
| Manticore Search | 0.3172 | C++ |
| Vespa BM25 | 0.3130 | Java |
| Weaviate BM25 | 0.2240 | Go |
| DPR (neural) | 0.1892 | Dense retrieval |

> **Note**: vectors.db uses a simple whitespace tokenizer without stemming or stop word removal. Adding language-specific preprocessing would likely push nDCG closer to the 0.32+ range.

### High-Dimensional Benchmarks (768d, 1536d)

Synthetic clustered Gaussian data with brute-force ground truth. Tests vectors.db at dimensions used by real LLM embedding models (OpenAI, Cohere, BGE).

#### 768d (100K vectors, Cosine) — OpenAI ada-002 / BGE / E5 dimensions

| ef_search | Compact Recall | Exact Recall | Compact QPS | Exact QPS |
|-----------|---------------|--------------|-------------|-----------|
| 40 | 0.7517 | 0.7559 | 3,791 | 3,042 |
| 80 | 0.8992 | 0.9060 | 2,313 | 2,071 |
| 120 | 0.9469 | 0.9580 | 2,036 | 1,663 |
| 200 | 0.9755 | 0.9879 | 1,673 | 1,367 |
| 400 | 0.9860 | 0.9994 | 1,295 | 1,081 |

Memory: Compact 73 MB, Exact 366 MB (5x). Build: Compact 353 ins/s, Exact 483 ins/s.

#### 1536d (25K vectors, Cosine) — OpenAI text-embedding-3-large dimensions

| ef_search | Compact Recall | Exact Recall | Compact QPS | Exact QPS |
|-----------|---------------|--------------|-------------|-----------|
| 40 | 0.8914 | 0.8916 | 3,524 | 2,687 |
| 80 | 0.9640 | 0.9730 | 2,527 | 2,103 |
| 120 | 0.9804 | 0.9908 | 2,237 | 1,827 |
| 200 | 0.9868 | 0.9984 | 1,697 | 1,219 |
| 400 | 0.9880 | 1.0000 | 1,484 | 1,307 |

Memory: Compact 37 MB, Exact 183 MB (5x). Build: Compact 199 ins/s, Exact 308 ins/s.

> **Key finding**: At high dimensions, exact mode provides significantly better recall than compact mode (+1.3% at 768d, +1.2% at 1536d, ef_search=400). Build speed is now comparable between modes thanks to cached dequantization in the heuristic neighbor selection — compact mode dequantizes each candidate once and caches selected neighbor vectors for fast f32-vs-f32 SIMD distance, eliminating the previous 6x build speed gap. For high-dimensional LLM embeddings, `store_raw_vectors=true` is recommended if RAM permits for the recall improvement.

### Filtered Search Benchmark

Tests HNSW recall and throughput under metadata filters at various selectivities. Uses SIFT-128 (1M vectors) with synthetic categorical metadata.

| Filter | Selectivity | Recall@10 | QPS | Avg Latency |
|--------|------------|-----------|-----|-------------|
| No filter (baseline) | 100% | 0.9910 | 2,093 | 478 us |
| category < 5 | ~50% | 0.9913 | 1,282 | 780 us |
| category == 0 | ~10% | 0.9928 | 335 | 2.98 ms |
| region == 0 | ~1% | 0.9953 | 46 | 21.6 ms |

> **Key finding**: Recall remains excellent (>0.99) at all selectivities, even at 1%. QPS degrades at low selectivity because HNSW must traverse more of the graph to find enough matching candidates. Adaptive ef oversampling (up to 4x base ef) ensures recall is maintained even at very low selectivity — if the initial search returns fewer than k results, the search retries with doubled ef automatically.

### Concurrent Throughput Benchmark

Tests how search throughput scales with multiple threads. Uses SIFT-128 (1M vectors), read-only concurrent access via `Arc<HnswIndex>`.

#### Single-thread baseline

| ef_search | QPS | Avg Latency | p50 | p95 | p99 |
|-----------|-----|-------------|-----|-----|-----|
| 50 | 7,086 | 141 us | 143 us | 172 us | 191 us |
| 100 | 3,963 | 252 us | 258 us | 306 us | 354 us |
| 200 | 2,175 | 459 us | 473 us | 561 us | 596 us |

#### Multi-thread scaling (ef_search=200)

| Threads | Agg QPS | Speedup | Avg Latency | p50 | p95 | p99 |
|---------|---------|---------|-------------|-----|-----|-----|
| 1 | 2,171 | 1.00x | 460 us | 474 us | 560 us | 600 us |
| 2 | 4,222 | 1.94x | 472 us | 485 us | 574 us | 617 us |
| 4 | 7,819 | 3.60x | 495 us | 472 us | 908 us | 1,046 us |
| 8 | 10,878 | 5.01x | 720 us | 512 us | 1.3 ms | 4.7 ms |

> **Key finding**: Thread-local VisitedSet pooling eliminates per-query 2MB allocations, enabling strong concurrent scaling. At 8 threads, scaling reaches 5.0x with 10,878 aggregate QPS. Search is fully lock-free — no mutexes or atomics are needed for read-only concurrent access. Tail latency (p99) remains well-controlled up to 4 threads, with moderate increase at 8 threads due to cache pressure.

### Running benchmarks

```bash
# Download benchmark datasets (requires Python + h5py)
cd benchmarks && python convert_hdf5.py && cd ..

# Run all benchmarks
cargo bench

# Run individual benchmarks
cargo bench --bench ann_glove25            # GloVe 25d cosine
cargo bench --bench ann_glove100           # GloVe 100d cosine
cargo bench --bench ann_sift128            # SIFT 128d euclidean (compact vs exact)
cargo bench --bench ann_highdim            # Synthetic 768d + 1536d (compact vs exact)
cargo bench --bench ann_filtered           # Filtered search (SIFT-128 + metadata)
cargo bench --bench concurrent_throughput  # Multi-thread QPS scaling
cargo bench --bench bm25_nfcorpus          # NFCorpus BM25
```

Each benchmark prints:
- Index construction time and insertion rate (vectors/sec)
- Recall@10 and QPS at multiple ef_search values (ANN benchmarks)
- nDCG@10 / nDCG@100 and QPS (BM25 benchmark)
- Comparison tables with reference implementations

### Memory efficiency

vectors.db offers two storage modes controlled by the `store_raw_vectors` flag:

**Compact mode** (`store_raw_vectors: false`, default) — maximum memory savings:

| Vectors | Dimensions | f32 storage | vectors.db (u8 only) | Savings |
|---------|------------|-------------|----------------------|---------|
| 1M | 128 | 488 MB | 122 MB | 4x |
| 1M | 768 | 2.9 GB | ~750 MB | ~4x |
| 10M | 384 | 14.3 GB | ~3.6 GB | ~4x |

**Exact mode** (`store_raw_vectors: true`) — maximum recall:

| Vectors | Dimensions | f32 storage | vectors.db (u8 + f32) | Overhead |
|---------|------------|-------------|----------------------|----------|
| 1M | 128 | 488 MB | 610 MB | +25% vs f32 |
| 1M | 768 | 2.9 GB | ~3.6 GB | +25% vs f32 |
| 10M | 384 | 14.3 GB | ~17.9 GB | +25% vs f32 |

**Measured trade-off** (SIFT-128, 1M vectors, ef_search=200):

| | Compact | Exact |
|---|---------|-------|
| Memory | 122 MB | 610 MB (+400%) |
| Recall@10 | 0.9897 | 0.9972 (+0.75%) |
| QPS | 1,750 | 1,644 (-6%) |
| Build speed | 1,689 ins/s | 1,848 ins/s (-9%) |

Compact mode uses u8 quantized vectors with asymmetric f32-vs-u8 distance for all operations. Exact mode additionally stores raw f32 vectors for exact distance computation during search, reranking, and graph construction. Compact is recommended for most use cases — 5x less memory with only ~0.7% recall loss.

---

## 15. Limits & Defaults

### Input limits

| Parameter | Limit |
|-----------|-------|
| Embedding dimension | 1 – 4,096 |
| Collection name length | 1 – 128 characters |
| Document text size | 1 byte – 1,000,000 bytes (1 MB) |
| Metadata keys per document | 64 |
| Metadata total size | 65,536 bytes (64 KB) |
| Batch insert size | 1,000 documents |
| Search k | 1 – 10,000 |
| Search offset | 0 – 100,000 |
| Request body size | 10 MB |
| HNSW M parameter | 4 – 128 |
| HNSW ef_construction | 10 – 2,000 |
| HNSW ef_search | 10 – 2,000 |

### Server defaults

| Parameter | Default |
|-----------|---------|
| Port | 3030 |
| Data directory | `./data` |
| Snapshot interval | 300 seconds |
| Request timeout | 30 seconds |
| Global rate limit | 100 req/s |
| Max concurrent requests | 512 |
| HNSW M | 16 |
| HNSW ef_construction | 200 |
| HNSW ef_search | 50 |
| HNSW max layers | 16 |
| HNSW store_raw_vectors | `false` (compact mode) |
| BM25 k1 | 1.2 |
| BM25 b | 0.75 |
| RRF k | 60.0 |
| WAL group commit batch | 128 entries |
| WAL group commit timeout | 1,000 us |

### HTTP error codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 307 | Redirect to another cluster node |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (missing/invalid API key) |
| 403 | Forbidden (insufficient role) |
| 404 | Not found (collection or document) |
| 408 | Request timeout (30s exceeded) |
| 409 | Conflict (collection already exists) |
| 413 | Payload too large (> 10 MB) |
| 429 | Too many requests (rate limit exceeded) |
| 500 | Internal server error |
| 503 | Service unavailable (memory degraded) |
| 507 | Insufficient storage (memory limit exceeded) |
