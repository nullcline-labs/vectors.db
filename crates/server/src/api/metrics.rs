//! Prometheus metrics recording and background collection.

use metrics::{counter, gauge, histogram};
use std::path::Path;
use std::time::Duration;
use vectorsdb_core::storage::Database;

/// Records HTTP request metrics.
pub fn record_request(method: &str, path: &str, status: u16, duration: Duration) {
    let labels = [
        ("method", method.to_string()),
        ("path", path.to_string()),
        ("status", status.to_string()),
    ];
    counter!("http_requests_total", &labels).increment(1);
    histogram!("http_request_duration_seconds", &labels).record(duration.as_secs_f64());
}

/// Records a write operation metric.
pub fn record_write_operation(collection: &str, operation: &str) {
    counter!(
        "vectorsdb_operations_total",
        "collection" => collection.to_string(),
        "operation" => operation.to_string()
    )
    .increment(1);
}

/// Records a search operation metric.
pub fn record_search_operation(collection: &str, search_type: &str) {
    counter!(
        "vectorsdb_search_total",
        "collection" => collection.to_string(),
        "type" => search_type.to_string()
    )
    .increment(1);
}

/// Updates collection-level Prometheus gauges.
pub fn update_collection_metrics(db: &Database) {
    let collections = db.collections.read();
    gauge!("vectorsdb_collections_total").set(collections.len() as f64);
    for (name, collection) in collections.iter() {
        let labels = [("collection", name.clone())];
        gauge!("vectorsdb_documents_total", &labels).set(collection.document_count() as f64);
        gauge!("vectorsdb_deleted_total", &labels).set(collection.deleted_count() as f64);
        gauge!("vectorsdb_collection_memory_bytes", &labels)
            .set(collection.estimate_memory_bytes() as f64);
    }
}

/// Updates the `vectorsdb_wal_size_bytes` gauge.
pub fn update_wal_metrics(wal_path: &Path) {
    if let Ok(meta) = std::fs::metadata(wal_path) {
        gauge!("vectorsdb_wal_size_bytes").set(meta.len() as f64);
    }
}
