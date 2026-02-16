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
