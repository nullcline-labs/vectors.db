//! Disk persistence for collections using bincode serialization.
//!
//! Collections are serialized to `.vdb` files using bincode. Writes use
//! atomic temp-file + rename to prevent corruption on crash.
//! A CRC32 checksum is appended as a 4-byte footer for integrity verification.

use crate::storage::collection::{Collection, CollectionData};
use parking_lot::RwLock;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Magic bytes appended before the CRC32 footer to distinguish checksummed snapshots.
const SNAPSHOT_CRC_MAGIC: &[u8; 4] = b"VCR1";

/// Save a collection to disk using bincode serialization with atomic write.
/// Appends a CRC32 checksum footer: [magic "VCR1"][u32 CRC32 BE].
pub fn save_collection(collection: &Collection, dir: &str) -> io::Result<()> {
    let data = collection.data.read();
    let bytes = bincode::serialize(&*data).map_err(|e| io::Error::other(e.to_string()))?;

    // Compute CRC32 over the bincode payload
    let crc = crc32fast::hash(&bytes);

    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    }
    let path = Path::new(dir).join(format!("{}.vdb", data.name));
    let tmp_path = Path::new(dir).join(format!("{}.vdb.tmp", data.name));

    // Write: [bincode payload][magic 4 bytes][CRC32 4 bytes BE]
    let mut output = Vec::with_capacity(bytes.len() + 8);
    output.extend_from_slice(&bytes);
    output.extend_from_slice(SNAPSHOT_CRC_MAGIC);
    output.extend_from_slice(&crc.to_be_bytes());

    // Atomic write: write to temp, then rename
    fs::write(&tmp_path, &output)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
    }
    fs::rename(&tmp_path, &path)?;

    tracing::info!(
        "Saved collection '{}' ({} bytes, CRC32={:#010x})",
        data.name,
        bytes.len(),
        crc
    );
    Ok(())
}

/// Load a collection from disk, verifying CRC32 integrity if present.
/// Supports both legacy (no CRC) and new (CRC-checksummed) snapshots.
pub fn load_collection(path: &Path) -> io::Result<Collection> {
    let raw = fs::read(path)?;

    // Check for CRC32 footer: last 8 bytes = [magic "VCR1"][CRC32 BE]
    let bytes = if raw.len() >= 8 && &raw[raw.len() - 8..raw.len() - 4] == SNAPSHOT_CRC_MAGIC {
        let payload = &raw[..raw.len() - 8];
        let stored_crc = u32::from_be_bytes([
            raw[raw.len() - 4],
            raw[raw.len() - 3],
            raw[raw.len() - 2],
            raw[raw.len() - 1],
        ]);
        let computed_crc = crc32fast::hash(payload);
        if computed_crc != stored_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Snapshot CRC32 mismatch: expected {:#010x}, got {:#010x}. File may be corrupted: {:?}",
                    stored_crc, computed_crc, path
                ),
            ));
        }
        tracing::debug!("Snapshot CRC32 verified: {:#010x}", stored_crc);
        payload
    } else {
        // Legacy snapshot without CRC — accept but warn
        tracing::warn!("Snapshot {:?} has no CRC32 checksum (legacy format)", path);
        &raw
    };

    let mut data: CollectionData = bincode::deserialize(bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // Free legacy raw_vectors from old snapshots when compact mode is active
    if !data.hnsw_index.config.store_raw_vectors {
        data.hnsw_index.free_raw_vectors();
    }

    data.validate().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("snapshot validation failed: {}", e),
        )
    })?;

    tracing::info!(
        "Loaded collection '{}' ({} documents)",
        data.name,
        data.documents.len()
    );

    Ok(Collection {
        data: Arc::new(RwLock::new(data)),
    })
}

/// Load all .vdb files from a directory.
pub fn load_all_collections(dir: &str) -> io::Result<Vec<Collection>> {
    let path = Path::new(dir);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut collections = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();
        if file_path.extension().and_then(|s| s.to_str()) == Some("vdb") {
            match load_collection(&file_path) {
                Ok(collection) => collections.push(collection),
                Err(e) => {
                    tracing::warn!("Failed to load {:?}: {}", file_path, e);
                }
            }
        }
    }
    Ok(collections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::graph::HnswConfig;
    use crate::storage::collection::Collection;

    fn tmp_dir() -> String {
        let id = uuid::Uuid::new_v4();
        let dir = std::env::temp_dir().join(format!("vdb_persist_{id}"));
        dir.to_string_lossy().to_string()
    }

    fn cleanup(dir: &str) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn make_collection(name: &str) -> Collection {
        let col = Collection::new(name.to_string(), 4, HnswConfig::default());
        let doc = crate::document::Document::new("hello world".into(), Default::default());
        col.insert_document(doc, vec![1.0, 0.0, 0.0, 0.0]);
        let doc2 = crate::document::Document::new("foo bar".into(), Default::default());
        col.insert_document(doc2, vec![0.0, 1.0, 0.0, 0.0]);
        col
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tmp_dir();
        let col = make_collection("roundtrip");
        save_collection(&col, &dir).unwrap();

        let path = std::path::Path::new(&dir).join("roundtrip.vdb");
        let loaded = load_collection(&path).unwrap();
        let data = loaded.data.read();
        assert_eq!(data.name, "roundtrip");
        assert_eq!(data.documents.len(), 2);
        assert_eq!(data.dimension, 4);
        assert_eq!(data.hnsw_index.node_count, 2);
        cleanup(&dir);
    }

    #[test]
    fn test_crc_verification_on_load() {
        let dir = tmp_dir();
        let col = make_collection("crc_test");
        save_collection(&col, &dir).unwrap();

        let path = std::path::Path::new(&dir).join("crc_test.vdb");
        // File should load successfully with valid CRC
        let loaded = load_collection(&path);
        assert!(loaded.is_ok());
        cleanup(&dir);
    }

    #[test]
    fn test_corrupted_snapshot_detected() {
        let dir = tmp_dir();
        let col = make_collection("corrupt");
        save_collection(&col, &dir).unwrap();

        let path = std::path::Path::new(&dir).join("corrupt.vdb");
        let mut data = std::fs::read(&path).unwrap();
        // Corrupt a byte in the middle of the bincode payload
        if data.len() > 20 {
            data[20] ^= 0xFF;
        }
        std::fs::write(&path, &data).unwrap();

        let result = load_collection(&path);
        assert!(result.is_err(), "corrupted snapshot should fail to load");
    }

    #[test]
    fn test_load_all_collections() {
        let dir = tmp_dir();
        let col1 = make_collection("alpha");
        let col2 = make_collection("beta");
        save_collection(&col1, &dir).unwrap();
        save_collection(&col2, &dir).unwrap();

        let loaded = load_all_collections(&dir).unwrap();
        assert_eq!(loaded.len(), 2);
        let names: Vec<String> = loaded.iter().map(|c| c.data.read().name.clone()).collect();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        cleanup(&dir);
    }

    #[test]
    fn test_load_all_empty_dir() {
        let dir = tmp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let loaded = load_all_collections(&dir).unwrap();
        assert!(loaded.is_empty());
        cleanup(&dir);
    }

    #[test]
    fn test_load_all_nonexistent_dir() {
        let loaded = load_all_collections("/tmp/nonexistent_vdb_dir_xyz").unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_validation_on_load() {
        let dir = tmp_dir();
        let col = make_collection("validate");
        save_collection(&col, &dir).unwrap();
        let path = std::path::Path::new(&dir).join("validate.vdb");
        let loaded = load_collection(&path).unwrap();
        let data = loaded.data.read();
        // Validation should have passed during load
        assert!(data.validate().is_ok());
        cleanup(&dir);
    }
}
