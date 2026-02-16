<p align="center">
  <img src="assets/icon.png" width="128" height="128" alt="vectors.db icon">
</p>

# vectors.db

A lightweight, in-memory vector database with HNSW indexing, BM25 full-text search, and hybrid retrieval.

[![CI](https://github.com/nullcline-labs/vectors.db/actions/workflows/ci.yml/badge.svg)](https://github.com/nullcline-labs/vectors.db/actions/workflows/ci.yml)

## Features

- **HNSW vector search** with configurable M, ef_construction, ef_search
- **BM25 keyword search** with Okapi BM25 scoring
- **Hybrid search** combining vector + keyword via RRF or linear fusion
- **Scalar quantization** (f32 -> u8) for memory-efficient SIMD-friendly storage, with optional raw f32 vectors for maximum recall
- **Write-Ahead Log (WAL)** with CRC32 checksums and fsync for crash recovery
- **Bearer token authentication** via `VECTORS_DB_API_KEY`
- **Prometheus metrics** at `/metrics`
- **Request timeout** (30s) and **rate limiting** (100 req/s)
- **Batch insert** up to 1000 documents per request
- **Multiple distance metrics**: Cosine, Euclidean, Dot Product

## Quick Start

### Python library

```bash
pip install maturin
cd crates/python && maturin develop --release
```

```python
import vectorsdb

db = vectorsdb.VectorDB()
db.create_collection("docs", dimension=3)

db.insert("docs", "hello world", [1.0, 0.0, 0.0], metadata={"tag": "greeting"})
db.insert("docs", "goodbye moon", [0.0, 1.0, 0.0])

# Vector search
results = db.search("docs", query_embedding=[1.0, 0.0, 0.0], k=5)
print(results[0].text)      # "hello world"
print(results[0].score)     # 1.0
print(results[0].metadata)  # {"tag": "greeting"}

# Keyword search
results = db.search("docs", query_text="moon", k=5)

# Hybrid search (vector + keyword)
results = db.search("docs", query_embedding=[1.0, 0.0, 0.0], query_text="hello", k=5)

# Filtered search
results = db.search("docs", query_embedding=[1.0, 0.0, 0.0], k=5, filter={
    "must": [{"field": "tag", "op": "eq", "value": "greeting"}]
})

# Persistence
db.save("docs", path="./data")
db.load("docs", path="./data")
```

Full IDE autocomplete and type hints are included via PEP 561 stubs. See [`examples/`](examples/) for more usage patterns.

### REST server

#### From source

```bash
cargo run --release -- --port 3030 --data-dir ./data
```

#### Docker

```bash
docker build -t vectors-db .
docker run -p 3030:3030 -v vectors-data:/data vectors-db
```

#### First steps

```bash
# Create a collection
curl -X POST http://localhost:3030/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_collection", "dimension": 3}'

# Insert a document
curl -X POST http://localhost:3030/collections/my_collection/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "embedding": [0.1, 0.2, 0.3]}'

# Search
curl -X POST http://localhost:3030/collections/my_collection/search \
  -H "Content-Type: application/json" \
  -d '{"query_embedding": [0.1, 0.2, 0.3], "k": 5}'
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (no auth required) |
| GET | `/metrics` | Prometheus metrics (no auth required) |
| POST | `/collections` | Create a collection |
| GET | `/collections` | List all collections |
| DELETE | `/collections/:name` | Delete a collection |
| POST | `/collections/:name/documents` | Insert a document |
| POST | `/collections/:name/documents/batch` | Batch insert (max 1000) |
| GET | `/collections/:name/documents/:id` | Get document by ID |
| DELETE | `/collections/:name/documents/:id` | Delete document |
| POST | `/collections/:name/search` | Vector, keyword, or hybrid search |
| POST | `/collections/:name/save` | Save collection snapshot to disk |
| POST | `/collections/:name/load` | Load collection snapshot from disk |
| POST | `/admin/compact` | Save all collections and truncate WAL |

### Create Collection

```json
POST /collections
{
  "name": "my_collection",
  "dimension": 768,
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50,
  "distance_metric": "cosine",
  "store_raw_vectors": false
}
```

`store_raw_vectors` (default `false`): when `true`, stores raw f32 vectors alongside quantized u8 for exact distance reranking (+0.75% recall, 5x RAM). See [Benchmarks](#benchmarks) for detailed comparison.

### Insert Document

```json
POST /collections/:name/documents
{
  "text": "document content",
  "embedding": [0.1, 0.2, ...],
  "metadata": {"key": "value"}
}
```

### Search

```json
POST /collections/:name/search
{
  "query_text": "search query",
  "query_embedding": [0.1, 0.2, ...],
  "k": 10,
  "min_similarity": 0.5,
  "alpha": 0.7,
  "fusion_method": "rrf"
}
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VECTORS_DB_API_KEY` | Bearer token for API authentication. If unset, auth is disabled. |
| `RUST_LOG` | Log level filter (e.g., `vectors_db=info`) |

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port`, `-p` | 3030 | Port to listen on |
| `--data-dir`, `-d` | `./data` | Directory for WAL and snapshots |

## Architecture

```
                        ┌─────────────────────────────────┐
                        │          HTTP API (Axum)         │
                        │  auth · metrics · rate-limit     │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │           Database               │
                        │   HashMap<String, Collection>    │
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
   ┌──────────▼─────────┐  ┌──────────▼─────────┐  ┌──────────▼─────────┐
   │    HNSW Index       │  │   BM25 Index       │  │   Hybrid Search    │
   │  scalar quantized   │  │  inverted index    │  │   RRF / linear     │
   └──────────┬──────────┘  └────────────────────┘  └────────────────────┘
              │
   ┌──────────▼──────────┐
   │    Distance Metrics  │
   │  cos · l2 · dot     │
   └─────────────────────┘

   Persistence: WAL (append + CRC32 + fsync) → Snapshot (.vdb bincode)
```

## Benchmarks

Results on Apple Silicon (M-series), single-threaded, SIFT-128 (1M vectors, 128d, Euclidean):

#### Compact mode (`store_raw_vectors=false`, default)

| ef_search | Recall@10 | QPS | Memory |
|-----------|-----------|-----|--------|
| 10 | 0.7695 | 22,566 | 122 MB |
| 40 | 0.9450 | 9,152 | |
| 120 | 0.9853 | 3,661 | |
| 200 | 0.9898 | 2,325 | |
| 400 | 0.9916 | 1,275 | |

Build: 1,852 inserts/s

#### Exact mode (`store_raw_vectors=true`)

| ef_search | Recall@10 | QPS | Memory |
|-----------|-----------|-----|--------|
| 10 | 0.7716 | 22,940 | 610 MB |
| 40 | 0.9494 | 8,759 | |
| 120 | 0.9924 | 3,674 | |
| 200 | 0.9972 | 2,370 | |
| 400 | 0.9990 | 1,277 | |

Build: 1,912 inserts/s

Compact mode uses **5x less memory** with only ~0.7% recall loss. Exact mode matches hnsw(nmslib) at 0.9990 recall.

#### High-dimensional (768d, 1536d)

Synthetic data at LLM embedding dimensions. Compact vs exact at ef_search=400:

| Dimension | Compact Recall | Exact Recall | Compact QPS | Exact QPS |
|-----------|---------------|--------------|-------------|-----------|
| 768d (100K) | 0.9860 | 0.9994 | 1,295 | 1,081 |
| 1536d (25K) | 0.9880 | 1.0000 | 1,484 | 1,307 |

At high dimensions, exact mode is recommended for maximum recall (+1.3% at 768d). Build speed is comparable between modes thanks to cached dequantization during construction.

#### Filtered search & concurrency

| Benchmark | Result |
|-----------|--------|
| Filtered 50% selectivity | 0.9913 recall, 1,282 QPS |
| Filtered 1% selectivity | 0.9953 recall, 46 QPS |
| 8-thread concurrent | 10,878 QPS (5.0x scaling) |

Run benchmarks:

```bash
cargo bench
```

## License

AGPL-3.0
