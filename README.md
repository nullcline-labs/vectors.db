<p align="center">
  <img src="assets/icon.png" width="128" height="128" alt="vectors.db icon">
</p>

# vectors.db

A lightweight, in-memory vector database with HNSW indexing, BM25 full-text search, and hybrid retrieval.

[![CI](https://github.com/memoclaudio/vectors.db/actions/workflows/ci.yml/badge.svg)](https://github.com/memoclaudio/vectors.db/actions/workflows/ci.yml)

## Features

- **HNSW vector search** with configurable M, ef_construction, ef_search
- **BM25 keyword search** with Okapi BM25 scoring
- **Hybrid search** combining vector + keyword via RRF or linear fusion
- **Scalar quantization** (f32 -> u8) for memory-efficient SIMD-friendly storage
- **Write-Ahead Log (WAL)** with CRC32 checksums and fsync for crash recovery
- **Bearer token authentication** via `VECTORS_DB_API_KEY`
- **Prometheus metrics** at `/metrics`
- **Request timeout** (30s) and **rate limiting** (100 req/s)
- **Batch insert** up to 1000 documents per request
- **Multiple distance metrics**: Cosine, Euclidean, Dot Product

## Quick Start

### From source

```bash
cargo run --release -- --port 3030 --data-dir ./data
```

### Docker

```bash
docker build -t vectors-db .
docker run -p 3030:3030 -v vectors-data:/data vectors-db
```

### First steps

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
  "distance_metric": "cosine"
}
```

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

Results on Apple Silicon (M-series), single-threaded:

| Dataset | Dimensions | Recall@10 | QPS |
|---------|-----------|-----------|-----|
| GloVe-25 | 25 | 0.99+ | ~10k |
| GloVe-100 | 100 | 0.99+ | ~5k |
| SIFT-128 | 128 | 0.99+ | ~4k |

Run benchmarks:

```bash
cargo bench
```

## License

MIT
