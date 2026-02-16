//! Disk persistence for collections using bincode serialization.
//!
//! Collections are serialized to `.vdb` files using bincode. Writes use
//! atomic temp-file + rename to prevent corruption on crash.

use crate::storage::collection::{Collection, CollectionData};
use parking_lot::RwLock;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Save a collection to disk using bincode serialization with atomic write.
pub fn save_collection(collection: &Collection, dir: &str) -> io::Result<()> {
    let data = collection.data.read();
    let bytes = bincode::serialize(&*data).map_err(|e| io::Error::other(e.to_string()))?;

    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    }
    let path = Path::new(dir).join(format!("{}.vdb", data.name));
    let tmp_path = Path::new(dir).join(format!("{}.vdb.tmp", data.name));

    // Atomic write: write to temp, then rename
    fs::write(&tmp_path, &bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
    }
    fs::rename(&tmp_path, &path)?;

    tracing::info!("Saved collection '{}' ({} bytes)", data.name, bytes.len());
    Ok(())
}

/// Load a collection from disk.
pub fn load_collection(path: &Path) -> io::Result<Collection> {
    let bytes = fs::read(path)?;
    let mut data: CollectionData = bincode::deserialize(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // Free legacy raw_vectors from old snapshots (no longer used)
    data.hnsw_index.free_raw_vectors();

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
