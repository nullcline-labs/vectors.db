//! Disk persistence for collections using bincode serialization.
//!
//! Collections are serialized to `.vdb` files using bincode. Writes use
//! atomic temp-file + rename to prevent corruption on crash.
//! A CRC32 checksum is appended as a 4-byte footer for integrity verification.

use crate::storage::collection::{Collection, CollectionData};
use crate::storage::crypto::EncryptionKey;
use parking_lot::RwLock;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Magic bytes appended before the CRC32 footer to distinguish checksummed snapshots.
const SNAPSHOT_CRC_MAGIC: &[u8; 4] = b"VCR1";

/// Magic bytes at the start of an encrypted snapshot.
const SNAPSHOT_ENCRYPTED_MAGIC: &[u8; 4] = b"VCE1";

/// Save a collection to disk using bincode serialization with atomic write.
///
/// If an encryption key is provided, writes encrypted format: `[magic "VCE1"][nonce || ciphertext+tag]`.
/// Otherwise writes plaintext with CRC32: `[bincode][magic "VCR1"][CRC32 BE]`.
pub fn save_collection(
    collection: &Collection,
    dir: &str,
    encryption_key: Option<&EncryptionKey>,
) -> io::Result<()> {
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

    let output = if let Some(key) = encryption_key {
        // Encrypted: [magic "VCE1" 4B][nonce 12B][ciphertext + GCM tag 16B]
        let encrypted = key.encrypt(&bytes);
        let mut out = Vec::with_capacity(4 + encrypted.len());
        out.extend_from_slice(SNAPSHOT_ENCRYPTED_MAGIC);
        out.extend_from_slice(&encrypted);
        tracing::info!(
            "Saved collection '{}' ({} bytes, encrypted)",
            data.name,
            bytes.len(),
        );
        out
    } else {
        // Plaintext: [bincode payload][magic "VCR1" 4B][CRC32 4B BE]
        let crc = crc32fast::hash(&bytes);
        let mut out = Vec::with_capacity(bytes.len() + 8);
        out.extend_from_slice(&bytes);
        out.extend_from_slice(SNAPSHOT_CRC_MAGIC);
        out.extend_from_slice(&crc.to_be_bytes());
        tracing::info!(
            "Saved collection '{}' ({} bytes, CRC32={:#010x})",
            data.name,
            bytes.len(),
            crc
        );
        out
    };

    // Atomic write: write to temp, then rename
    fs::write(&tmp_path, &output)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
    }
    fs::rename(&tmp_path, &path)?;

    Ok(())
}

/// Load a collection from disk, verifying integrity.
///
/// Supports three formats:
/// 1. Encrypted: `[magic "VCE1"][nonce || ciphertext+tag]` — requires encryption key
/// 2. CRC-checksummed: `[bincode][magic "VCR1"][CRC32 BE]`
/// 3. Legacy: plain bincode (no magic, no CRC)
///
/// An encrypted snapshot loaded without a key returns an error.
/// An unencrypted snapshot loaded with a key is accepted (backward compatible).
pub fn load_collection(
    path: &Path,
    encryption_key: Option<&EncryptionKey>,
) -> io::Result<Collection> {
    let raw = fs::read(path)?;

    // Detect format by checking magic bytes
    let bytes: Vec<u8> = if raw.len() >= 4 && &raw[..4] == SNAPSHOT_ENCRYPTED_MAGIC {
        // Encrypted snapshot
        let key = encryption_key.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "snapshot {:?} is encrypted but no encryption key was provided",
                    path
                ),
            )
        })?;
        let decrypted = key.decrypt(&raw[4..])?;
        tracing::debug!("Snapshot {:?} decrypted successfully", path);
        decrypted
    } else if raw.len() >= 8 && &raw[raw.len() - 8..raw.len() - 4] == SNAPSHOT_CRC_MAGIC {
        // CRC32 checksummed snapshot
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
        payload.to_vec()
    } else {
        // Legacy snapshot without CRC — accept but warn
        tracing::warn!("Snapshot {:?} has no CRC32 checksum (legacy format)", path);
        raw
    };

    let mut data: CollectionData = bincode::deserialize(&bytes)
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
pub fn load_all_collections(
    dir: &str,
    encryption_key: Option<&EncryptionKey>,
) -> io::Result<Vec<Collection>> {
    let path = Path::new(dir);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut collections = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();
        if file_path.extension().and_then(|s| s.to_str()) == Some("vdb") {
            match load_collection(&file_path, encryption_key) {
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

    fn test_key() -> EncryptionKey {
        EncryptionKey::from_hex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
            .unwrap()
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tmp_dir();
        let col = make_collection("roundtrip");
        save_collection(&col, &dir, None).unwrap();

        let path = std::path::Path::new(&dir).join("roundtrip.vdb");
        let loaded = load_collection(&path, None).unwrap();
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
        save_collection(&col, &dir, None).unwrap();

        let path = std::path::Path::new(&dir).join("crc_test.vdb");
        let loaded = load_collection(&path, None);
        assert!(loaded.is_ok());
        cleanup(&dir);
    }

    #[test]
    fn test_corrupted_snapshot_detected() {
        let dir = tmp_dir();
        let col = make_collection("corrupt");
        save_collection(&col, &dir, None).unwrap();

        let path = std::path::Path::new(&dir).join("corrupt.vdb");
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 20 {
            data[20] ^= 0xFF;
        }
        std::fs::write(&path, &data).unwrap();

        let result = load_collection(&path, None);
        assert!(result.is_err(), "corrupted snapshot should fail to load");
    }

    #[test]
    fn test_load_all_collections() {
        let dir = tmp_dir();
        let col1 = make_collection("alpha");
        let col2 = make_collection("beta");
        save_collection(&col1, &dir, None).unwrap();
        save_collection(&col2, &dir, None).unwrap();

        let loaded = load_all_collections(&dir, None).unwrap();
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
        let loaded = load_all_collections(&dir, None).unwrap();
        assert!(loaded.is_empty());
        cleanup(&dir);
    }

    #[test]
    fn test_load_all_nonexistent_dir() {
        let loaded = load_all_collections("/tmp/nonexistent_vdb_dir_xyz", None).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_validation_on_load() {
        let dir = tmp_dir();
        let col = make_collection("validate");
        save_collection(&col, &dir, None).unwrap();
        let path = std::path::Path::new(&dir).join("validate.vdb");
        let loaded = load_collection(&path, None).unwrap();
        let data = loaded.data.read();
        assert!(data.validate().is_ok());
        cleanup(&dir);
    }

    // ── Encryption tests ───────────────────────────────────────────────

    #[test]
    fn test_encrypted_save_and_load_roundtrip() {
        let dir = tmp_dir();
        let key = test_key();
        let col = make_collection("enc_roundtrip");
        save_collection(&col, &dir, Some(&key)).unwrap();

        let path = std::path::Path::new(&dir).join("enc_roundtrip.vdb");
        let loaded = load_collection(&path, Some(&key)).unwrap();
        let data = loaded.data.read();
        assert_eq!(data.name, "enc_roundtrip");
        assert_eq!(data.documents.len(), 2);
        assert_eq!(data.dimension, 4);
        cleanup(&dir);
    }

    #[test]
    fn test_load_unencrypted_with_key_works() {
        // Backward compatibility: unencrypted file should load even with key present
        let dir = tmp_dir();
        let key = test_key();
        let col = make_collection("compat");
        save_collection(&col, &dir, None).unwrap();

        let path = std::path::Path::new(&dir).join("compat.vdb");
        let loaded = load_collection(&path, Some(&key)).unwrap();
        let data = loaded.data.read();
        assert_eq!(data.name, "compat");
        assert_eq!(data.documents.len(), 2);
        cleanup(&dir);
    }

    #[test]
    fn test_load_encrypted_without_key_fails() {
        let dir = tmp_dir();
        let key = test_key();
        let col = make_collection("nokey");
        save_collection(&col, &dir, Some(&key)).unwrap();

        let path = std::path::Path::new(&dir).join("nokey.vdb");
        let result = load_collection(&path, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("encrypted"),
            "error should mention encryption: {err}"
        );
        cleanup(&dir);
    }

    #[test]
    fn test_load_encrypted_wrong_key_fails() {
        let dir = tmp_dir();
        let key1 = test_key();
        let key2 = EncryptionKey::from_hex(
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
        .unwrap();
        let col = make_collection("wrongkey");
        save_collection(&col, &dir, Some(&key1)).unwrap();

        let path = std::path::Path::new(&dir).join("wrongkey.vdb");
        let result = load_collection(&path, Some(&key2));
        assert!(result.is_err());
        cleanup(&dir);
    }

    #[test]
    fn test_encrypted_snapshot_has_magic_header() {
        let dir = tmp_dir();
        let key = test_key();
        let col = make_collection("magic_check");
        save_collection(&col, &dir, Some(&key)).unwrap();

        let path = std::path::Path::new(&dir).join("magic_check.vdb");
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..4], b"VCE1");
        cleanup(&dir);
    }

    #[test]
    fn test_load_all_encrypted() {
        let dir = tmp_dir();
        let key = test_key();
        let col1 = make_collection("enc_a");
        let col2 = make_collection("enc_b");
        save_collection(&col1, &dir, Some(&key)).unwrap();
        save_collection(&col2, &dir, Some(&key)).unwrap();

        let loaded = load_all_collections(&dir, Some(&key)).unwrap();
        assert_eq!(loaded.len(), 2);
        cleanup(&dir);
    }
}
