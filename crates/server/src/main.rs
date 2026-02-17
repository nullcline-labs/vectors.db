use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;
use vectorsdb_core::config;
use vectorsdb_core::storage::crypto::EncryptionKey;
use vectorsdb_core::storage::{load_all_collections, save_collection, Database};
use vectorsdb_server::api::create_router;
use vectorsdb_server::api::handlers::AppState;
use vectorsdb_server::api::metrics;
use vectorsdb_server::api::rbac::{ApiKeyEntry, RbacConfig};
use vectorsdb_server::cluster;
use vectorsdb_server::wal_async::WriteAheadLog;

#[derive(Parser)]
#[command(name = "vectors-db", about = "In-memory vector database")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value_t = config::DEFAULT_PORT)]
    port: u16,

    /// Data directory for persistence
    #[arg(short, long, default_value = config::DEFAULT_DATA_DIR)]
    data_dir: String,

    /// Maximum memory in MB (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_memory_mb: usize,

    /// Snapshot interval in seconds (0 = disabled)
    #[arg(long, default_value_t = config::DEFAULT_SNAPSHOT_INTERVAL_SECS)]
    snapshot_interval: u64,

    /// TLS certificate file path
    #[arg(long)]
    tls_cert: Option<String>,

    /// TLS private key file path
    #[arg(long)]
    tls_key: Option<String>,

    /// Node ID for Raft cluster mode (omit for standalone)
    #[arg(long)]
    node_id: Option<u64>,

    /// Graceful shutdown timeout in seconds
    #[arg(long, default_value_t = config::DEFAULT_SHUTDOWN_TIMEOUT_SECS)]
    shutdown_timeout: u64,

    /// Fail startup if WAL replay encounters errors (strict mode)
    #[arg(long, default_value_t = false)]
    wal_strict: bool,

    /// Auto-compaction threshold (0.0 = disabled, default 0.2 = rebuild when >20% deleted)
    #[arg(long, default_value_t = config::DEFAULT_AUTO_COMPACT_RATIO)]
    auto_compact_ratio: f32,

    /// Path to encryption key file (32 raw bytes or 64-char hex). Overrides VECTORS_DB_ENCRYPTION_KEY env var.
    #[arg(long)]
    encryption_key_file: Option<String>,

    /// Comma-separated peer addresses (id=addr format, e.g. "2=host2:3030,3=host3:3030")
    #[arg(long)]
    peers: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(
                    "vectorsdb_server=info"
                        .parse()
                        .expect("valid directive literal"),
                )
                .add_directive(
                    "vectorsdb_core=info"
                        .parse()
                        .expect("valid directive literal"),
                ),
        )
        .init();

    let args = Args::parse();

    if args.port == 0 {
        eprintln!("Error: port must be > 0");
        std::process::exit(1);
    }
    if args.max_memory_mb > 0 && args.max_memory_mb < 16 {
        tracing::warn!(
            "max_memory_mb={} is very low (minimum recommended: 16)",
            args.max_memory_mb
        );
    }
    let data_path = std::path::Path::new(&args.data_dir);
    if data_path.exists() && !data_path.is_dir() {
        eprintln!(
            "Error: data_dir '{}' exists but is not a directory",
            args.data_dir
        );
        std::process::exit(1);
    }

    // Load encryption key: --encryption-key-file takes precedence over env var
    let encryption_key: Option<Arc<EncryptionKey>> = if let Some(ref key_file) =
        args.encryption_key_file
    {
        let key = EncryptionKey::from_file(std::path::Path::new(key_file)).unwrap_or_else(|e| {
            eprintln!(
                "Error: failed to read encryption key file '{}': {}",
                key_file, e
            );
            std::process::exit(1);
        });
        tracing::info!("Encryption at rest enabled (key file: {})", key_file);
        Some(Arc::new(key))
    } else if let Ok(hex) = std::env::var("VECTORS_DB_ENCRYPTION_KEY") {
        let key = EncryptionKey::from_hex(&hex).unwrap_or_else(|e| {
            eprintln!("Error: invalid VECTORS_DB_ENCRYPTION_KEY: {}", e);
            std::process::exit(1);
        });
        tracing::info!("Encryption at rest enabled (env var)");
        Some(Arc::new(key))
    } else {
        None
    };

    let db = Database::new();

    // Load existing collection snapshots from disk
    match load_all_collections(&args.data_dir, encryption_key.as_deref()) {
        Ok(collections) => {
            let mut db_collections = db.collections.write();
            for collection in collections {
                let name = collection.data.read().name.clone();
                tracing::info!("Restored collection '{}'", name);
                db_collections.insert(name, collection);
            }
        }
        Err(e) => {
            tracing::warn!("Could not load collections: {}", e);
        }
    }

    // Initialize WAL and replay pending entries
    let wal = Arc::new(WriteAheadLog::new(&args.data_dir, encryption_key.clone())?);

    match wal.replay() {
        Ok((entries, stats)) => {
            let has_errors = stats.skipped > 0 || stats.crc_errors > 0 || stats.truncated;
            if has_errors {
                tracing::warn!(
                    "WAL replay stats: {} ok, {} skipped, {} CRC errors, truncated={}",
                    stats.success,
                    stats.skipped,
                    stats.crc_errors,
                    stats.truncated
                );
                if args.wal_strict {
                    eprintln!(
                        "Error: WAL replay encountered errors (strict mode). \
                         {} CRC errors, {} skipped, truncated={}. \
                         Fix the WAL or restart without --wal-strict.",
                        stats.crc_errors, stats.skipped, stats.truncated
                    );
                    std::process::exit(1);
                }
            }
            if !entries.is_empty() {
                tracing::info!("Replaying {} WAL entries", entries.len());
                let applied = db.replay_wal(&entries);
                tracing::info!(
                    "WAL replay complete: {applied}/{} entries applied",
                    entries.len()
                );
            }
        }
        Err(e) => {
            if args.wal_strict {
                eprintln!(
                    "Error: WAL replay failed (strict mode): {}. \
                     Fix the WAL or restart without --wal-strict.",
                    e
                );
                std::process::exit(1);
            }
            tracing::warn!("WAL replay failed: {}", e);
        }
    }

    // Parse RBAC configuration
    let rbac = parse_rbac_config();
    let api_key = if rbac.is_some() {
        tracing::info!("RBAC authentication enabled");
        None
    } else {
        let key = std::env::var("VECTORS_DB_API_KEY").ok();
        if key.is_some() {
            tracing::info!("API key authentication enabled");
        } else {
            tracing::info!("No API key set — running in dev mode (no auth)");
        }
        key
    };

    let prometheus_handle =
        metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder()?;

    let max_memory_bytes = args.max_memory_mb * 1024 * 1024;

    let peer_addrs: Option<std::collections::HashMap<u64, String>> = args.peers.as_ref().map(|p| {
        p.split(',')
            .filter_map(|entry| {
                let parts: Vec<&str> = entry.splitn(2, '=').collect();
                if parts.len() == 2 {
                    parts[0]
                        .parse::<u64>()
                        .ok()
                        .map(|id| (id, parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect()
    });

    // Initialize Raft if --node-id is provided
    let (raft, routing_table) = if let Some(node_id) = args.node_id {
        tracing::info!("Cluster mode: node_id={}", node_id);
        let db_arc = Arc::new(db.clone());
        let sm = Arc::new(cluster::store::StateMachineStore::new(db_arc));
        let routing = sm.routing_table.clone();
        let log_store = cluster::store::LogStore::default();
        let network = cluster::network::NetworkFactory::new();

        let raft_config = Arc::new(
            openraft::Config {
                heartbeat_interval: 500,
                election_timeout_min: 1500,
                election_timeout_max: 3000,
                ..Default::default()
            }
            .validate()?,
        );

        let raft = cluster::Raft::new(node_id, raft_config, network, log_store, sm).await?;

        tracing::info!("Raft node {} initialized", node_id);
        (Some(Arc::new(raft)), Some(routing))
    } else {
        (None, None)
    };

    let wal_path = PathBuf::from(&args.data_dir).join("wal.bin");

    let state = AppState {
        db: db.clone(),
        data_dir: args.data_dir.clone(),
        wal: wal.clone(),
        wal_path: wal_path.clone(),
        api_key,
        prometheus_handle,
        max_memory_bytes,
        rbac,
        raft: raft.clone(),
        node_id: args.node_id,
        routing_table: routing_table.clone(),
        peer_addrs: peer_addrs.clone(),
        start_time: Instant::now(),
        key_rate_limiters: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        memory_reserved: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        encryption_key: encryption_key.clone(),
    };

    let mut app = create_router(state);

    if let Some(ref raft_instance) = raft {
        let raft_state = cluster::api::RaftState {
            raft: raft_instance.clone(),
        };
        app = app.merge(cluster::api::raft_router(raft_state));
    }
    let addr = format!("0.0.0.0:{}", args.port);
    let collections_count = db.collections.read().len();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        port = args.port,
        data_dir = %args.data_dir,
        max_memory_mb = args.max_memory_mb,
        snapshot_interval_secs = args.snapshot_interval,
        tls = args.tls_cert.is_some(),
        cluster_mode = args.node_id.is_some(),
        collections = collections_count,
        "vectors.db ready"
    );

    // Spawn collection metrics background task
    let metrics_db = db.clone();
    let metrics_wal_path = wal_path.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            metrics::update_collection_metrics(&metrics_db);
            metrics::update_wal_metrics(&metrics_wal_path);
        }
    });

    // Spawn auto-snapshot background task
    let auto_compact_ratio = args.auto_compact_ratio;
    if args.snapshot_interval > 0 {
        let snap_db = db.clone();
        let snap_wal = wal.clone();
        let snap_data_dir = args.data_dir.clone();
        let snap_interval = args.snapshot_interval;
        let snap_enc_key = encryption_key.clone();
        tracing::info!("Auto-snapshots enabled every {}s", snap_interval);
        if auto_compact_ratio > 0.0 {
            tracing::info!(
                "Auto-compaction enabled (threshold: {:.0}% deleted)",
                auto_compact_ratio * 100.0
            );
        }
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(snap_interval));
            interval.tick().await;
            loop {
                interval.tick().await;
                tracing::info!("Running periodic snapshot...");
                let _gate = snap_wal.freeze();
                let collections = snap_db.collections.read();
                let mut all_saved = true;
                for (name, collection) in collections.iter() {
                    if let Err(e) =
                        save_collection(collection, &snap_data_dir, snap_enc_key.as_deref())
                    {
                        tracing::error!("Snapshot failed for '{}': {}", name, e);
                        all_saved = false;
                    }
                }
                drop(collections);
                if all_saved {
                    if let Err(e) = snap_wal.truncate() {
                        tracing::error!("WAL truncate after snapshot failed: {}", e);
                    } else {
                        tracing::info!("Periodic snapshot complete, WAL truncated");
                    }
                }

                // Auto-compaction: rebuild indices when deletion ratio exceeds threshold
                if auto_compact_ratio > 0.0 {
                    let collections = snap_db.collections.read();
                    let candidates: Vec<(String, f32)> = collections
                        .iter()
                        .filter_map(|(name, col)| {
                            let ratio = col.deletion_ratio();
                            if ratio > auto_compact_ratio {
                                Some((name.clone(), ratio))
                            } else {
                                None
                            }
                        })
                        .collect();
                    drop(collections);

                    for (name, ratio) in candidates {
                        tracing::info!(
                            "Auto-compaction: '{}' has {:.1}% deleted, rebuilding",
                            name,
                            ratio * 100.0
                        );
                        if let Some(col) = snap_db.get_collection(&name) {
                            let count = col.rebuild_indices();
                            tracing::info!(
                                "Auto-compaction: '{}' rebuilt with {} live docs",
                                name,
                                count
                            );
                            if let Err(e) =
                                save_collection(&col, &snap_data_dir, snap_enc_key.as_deref())
                            {
                                tracing::error!(
                                    "Auto-compaction: failed to save '{}': {}",
                                    name,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    let shutdown_timeout = args.shutdown_timeout;
    match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => {
            tracing::info!("TLS enabled");
            let tls_config =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key).await?;
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                wait_for_signal().await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(shutdown_timeout)));
            });
            axum_server::bind_rustls(addr.parse()?, tls_config)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        (None, None) => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, app)
                .with_graceful_shutdown(wait_for_signal())
                .await?;
        }
        _ => {
            eprintln!("Error: Both --tls-cert and --tls-key must be provided together");
            std::process::exit(1);
        }
    }

    flush_and_shutdown(
        &db,
        &wal,
        &args.data_dir,
        shutdown_timeout,
        encryption_key.as_deref(),
    );

    Ok(())
}

fn parse_rbac_config() -> Option<RbacConfig> {
    if let Ok(json_str) = std::env::var("VECTORS_DB_API_KEYS") {
        match serde_json::from_str::<Vec<ApiKeyEntry>>(&json_str) {
            Ok(entries) if !entries.is_empty() => {
                return Some(RbacConfig::from_entries(entries));
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error: VECTORS_DB_API_KEYS contains invalid JSON: {}", e);
                std::process::exit(1);
            }
        }
    }
    None
}

async fn wait_for_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to install Ctrl+C handler: {}", e);
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("Failed to install SIGTERM handler: {}", e);
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT"),
        _ = terminate => tracing::info!("Received SIGTERM"),
    }

    tracing::info!("Shutting down gracefully, draining in-flight requests...");
}

fn flush_and_shutdown(
    db: &Database,
    wal: &WriteAheadLog,
    data_dir: &str,
    timeout_secs: u64,
    encryption_key: Option<&EncryptionKey>,
) {
    tracing::info!("All requests drained, flushing data...");

    let _gate = wal.freeze();

    let start = Instant::now();
    let deadline = Duration::from_secs(timeout_secs);
    let collections = db.collections.read();
    let mut all_saved = true;
    for (name, collection) in collections.iter() {
        if start.elapsed() > deadline {
            tracing::error!(
                "Shutdown flush timeout ({}s) exceeded — aborting remaining saves",
                timeout_secs
            );
            all_saved = false;
            break;
        }
        match save_collection(collection, data_dir, encryption_key) {
            Ok(()) => tracing::info!("Saved collection '{}' on shutdown", name),
            Err(e) => {
                tracing::error!("Failed to save collection '{}': {}", name, e);
                all_saved = false;
            }
        }
    }

    if all_saved {
        if let Err(e) = wal.truncate() {
            tracing::error!("Failed to truncate WAL: {}", e);
        } else {
            tracing::info!("WAL truncated after successful flush");
        }
    } else {
        tracing::warn!("Some collections failed to save — WAL preserved for recovery");
    }
}
