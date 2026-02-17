# ── Stage 1: dependency cache layer ──────────────────────────────────
FROM rust:1.88 AS deps
WORKDIR /app

# Copy only manifests to cache dependency builds
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/server/Cargo.toml crates/server/Cargo.toml
COPY crates/python/Cargo.toml crates/python/Cargo.toml

# Create stub source files so cargo can resolve the workspace
RUN mkdir -p crates/core/src crates/server/src crates/python/src && \
    echo "fn main() {}" > crates/server/src/main.rs && \
    touch crates/core/src/lib.rs crates/server/src/lib.rs crates/python/src/lib.rs

RUN cargo build --release -p vectorsdb-server 2>/dev/null || true
# Remove stubs so real source is used next
RUN rm -rf crates/

# ── Stage 2: build ──────────────────────────────────────────────────
FROM deps AS builder
COPY crates/ crates/

# Recreate python stub (src/ excluded by .dockerignore)
RUN mkdir -p crates/python/src && touch crates/python/src/lib.rs

# Touch to force recompile of our crates (not deps)
RUN touch crates/core/src/lib.rs crates/server/src/lib.rs crates/server/src/main.rs
RUN cargo build --release -p vectorsdb-server

# ── Stage 3: runtime ────────────────────────────────────────────────
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="vectors.db" \
      org.opencontainers.image.description="Lightweight vector database with HNSW, BM25, and WAL streaming replication" \
      org.opencontainers.image.source="https://github.com/nullcline-labs/vectors.db" \
      org.opencontainers.image.licenses="AGPL-3.0"

# Install curl for healthcheck, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 vectorsdb && \
    useradd --uid 1000 --gid vectorsdb --shell /usr/sbin/nologin --create-home vectorsdb

COPY --from=builder /app/target/release/vectors-db /usr/local/bin/

# Data directory with correct ownership
RUN mkdir -p /data && chown vectorsdb:vectorsdb /data

USER vectorsdb

# HTTP API
EXPOSE 3030
# Replication port (configure with --replication-port)
EXPOSE 3031

VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3030/health || exit 1

ENTRYPOINT ["vectors-db", "--data-dir", "/data"]
