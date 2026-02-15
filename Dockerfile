# Stage 1: Build
FROM rust:1.83-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build --release -p vectorsdb-server

# Stage 2: Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/vectors-db /usr/local/bin/
EXPOSE 3030
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:3030/health || exit 1
ENTRYPOINT ["vectors-db", "--data-dir", "/data"]
