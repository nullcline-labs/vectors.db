//! Synchronous Write-Ahead Log (WAL) for crash recovery.
//!
//! Every mutation is appended to the WAL before being applied in memory.
//! Each entry is framed as `[u32 length BE][u32 CRC32 BE][bincode payload]`
//! and durably flushed with `fsync`.
//!
//! This is the synchronous variant for use in the embeddable core library
//! (no tokio dependency). Uses `parking_lot::Mutex` for thread-safe access.

use crate::document::Document;
use crate::hnsw::graph::HnswConfig;
use crate::storage::crypto::EncryptionKey;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Sentinel CRC value that marks an encrypted WAL frame.
/// Real CRC32 values are extremely unlikely to be all-ones.
const ENCRYPTED_FRAME_SENTINEL: u32 = 0xFFFF_FFFF;

/// A single mutation entry in the write-ahead log.
///
/// Each variant represents an atomic operation that can be replayed on startup.
#[derive(Debug, Serialize, Deserialize)]
pub enum WalEntry {
    /// Create a new collection.
    CreateCollection {
        name: String,
        dimension: usize,
        config: HnswConfig,
    },
    /// Delete a collection by name.
    DeleteCollection { name: String },
    /// Insert a single document with its embedding.
    InsertDocument {
        collection_name: String,
        document: Document,
        embedding: Vec<f32>,
    },
    /// Delete a document by UUID.
    DeleteDocument {
        collection_name: String,
        document_id: Uuid,
    },
    /// Batch insert multiple documents.
    InsertDocumentBatch {
        collection_name: String,
        documents: Vec<(Document, Vec<f32>)>,
    },
    /// Update a document (delete + re-insert with new content/embedding).
    UpdateDocument {
        collection_name: String,
        document_id: Uuid,
        document: Document,
        embedding: Vec<f32>,
    },
}

/// Diagnostic statistics from a WAL replay.
#[derive(Debug, Default)]
pub struct ReplayStats {
    /// Number of entries successfully deserialized.
    pub success: usize,
    /// Number of entries skipped due to deserialization errors (CRC was valid).
    pub skipped: usize,
    /// Number of CRC mismatches encountered (replay stopped).
    pub crc_errors: usize,
    /// Whether replay was terminated by a truncated entry.
    pub truncated: bool,
}

/// Synchronous append-only write-ahead log with CRC32 integrity checks.
///
/// Thread-safe via `parking_lot::Mutex`. Each [`append`](SyncWriteAheadLog::append)
/// call serializes, writes, flushes, and fsyncs the entry to disk before returning.
pub struct SyncWriteAheadLog {
    /// Mutex-protected buffered writer for the WAL file.
    writer: Mutex<BufWriter<File>>,
    /// Write gate: freeze() takes exclusive, append() takes shared.
    write_gate: parking_lot::RwLock<()>,
    /// Path to WAL file (needed for replay/truncate).
    path: PathBuf,
    /// Optional encryption key for encrypting WAL entries.
    encryption_key: Option<Arc<EncryptionKey>>,
}

impl SyncWriteAheadLog {
    /// Open or create the WAL file in append mode.
    pub fn new(data_dir: &str, encryption_key: Option<Arc<EncryptionKey>>) -> io::Result<Self> {
        fs::create_dir_all(data_dir)?;
        let path = PathBuf::from(data_dir).join("wal.bin");
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let file = opts.open(&path)?;
        let writer = Mutex::new(BufWriter::new(file));
        let write_gate = parking_lot::RwLock::new(());

        Ok(Self {
            writer,
            write_gate,
            path,
            encryption_key,
        })
    }

    /// Append a WAL entry synchronously.
    ///
    /// Serializes the entry, writes, flushes, and fsyncs to disk before returning.
    pub fn append(&self, entry: &WalEntry) -> io::Result<()> {
        let framed = match &self.encryption_key {
            Some(key) => serialize_and_frame_encrypted(entry, key)?,
            None => serialize_and_frame(entry)?,
        };

        let _gate = self.write_gate.read();
        let mut w = self.writer.lock();
        w.write_all(&framed)?;
        w.flush()?;
        w.get_mut().sync_all()?;
        Ok(())
    }

    /// Read all entries from the WAL file sequentially, verifying integrity.
    ///
    /// Supports both encrypted (sentinel `0xFFFFFFFF`) and plaintext (CRC32) frames.
    /// Returns the successfully deserialized entries and diagnostic statistics.
    pub fn replay(&self) -> io::Result<(Vec<WalEntry>, ReplayStats)> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut stats = ReplayStats::default();
        let mut header_buf = [0u8; 8];

        loop {
            match reader.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let len =
                u32::from_be_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]])
                    as usize;
            let stored_crc =
                u32::from_be_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);
            let mut data = vec![0u8; len];
            match reader.read_exact(&mut data) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    tracing::warn!("WAL truncated mid-entry, stopping replay");
                    stats.truncated = true;
                    break;
                }
                Err(e) => return Err(e),
            }

            // Detect encrypted vs plaintext frame
            let payload = if stored_crc == ENCRYPTED_FRAME_SENTINEL {
                // Encrypted frame: data = nonce || ciphertext+tag
                match &self.encryption_key {
                    Some(key) => match key.decrypt(&data) {
                        Ok(plaintext) => plaintext,
                        Err(e) => {
                            tracing::warn!("WAL encrypted entry decryption failed: {}", e);
                            stats.crc_errors += 1;
                            break;
                        }
                    },
                    None => {
                        tracing::warn!(
                            "WAL contains encrypted entries but no encryption key provided"
                        );
                        stats.crc_errors += 1;
                        break;
                    }
                }
            } else {
                // Plaintext frame: verify CRC32
                let computed_crc = crc32fast::hash(&data);
                if computed_crc != stored_crc {
                    tracing::warn!("WAL entry CRC mismatch, stopping replay");
                    stats.crc_errors += 1;
                    break;
                }
                data
            };

            match bincode::deserialize::<WalEntry>(&payload) {
                Ok(entry) => {
                    entries.push(entry);
                    stats.success += 1;
                }
                Err(e) => {
                    tracing::warn!("WAL entry deserialization failed, skipping: {}", e);
                    stats.skipped += 1;
                }
            }
        }

        Ok((entries, stats))
    }

    /// Acquire an exclusive write gate, blocking all [`append`](SyncWriteAheadLog::append) calls.
    ///
    /// Hold the returned guard while performing snapshot + truncate.
    pub fn freeze(&self) -> parking_lot::RwLockWriteGuard<'_, ()> {
        self.write_gate.write()
    }

    /// Truncate the WAL file, fsync, and reopen in append mode.
    pub fn truncate(&self) -> io::Result<()> {
        let mut writer = self.writer.lock();
        let truncated = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        truncated.sync_all()?;
        *writer = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?,
        );
        Ok(())
    }
}

/// Serialize a WAL entry into its on-disk frame format:
/// `[u32 len BE][u32 crc32 BE][bincode payload]`.
pub(crate) fn serialize_and_frame(entry: &WalEntry) -> io::Result<Vec<u8>> {
    let bytes = bincode::serialize(entry).map_err(|e| io::Error::other(e.to_string()))?;
    let len = bytes.len() as u32;
    let crc = crc32fast::hash(&bytes);

    let mut framed = Vec::with_capacity(8 + bytes.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(&crc.to_be_bytes());
    framed.extend_from_slice(&bytes);
    Ok(framed)
}

/// Serialize a WAL entry into an encrypted on-disk frame:
/// `[u32 len BE][0xFFFFFFFF sentinel BE][nonce 12B][ciphertext + GCM tag 16B]`.
pub(crate) fn serialize_and_frame_encrypted(
    entry: &WalEntry,
    key: &EncryptionKey,
) -> io::Result<Vec<u8>> {
    let bytes = bincode::serialize(entry).map_err(|e| io::Error::other(e.to_string()))?;
    let encrypted = key.encrypt(&bytes); // nonce || ciphertext+tag
    let len = encrypted.len() as u32;

    let mut framed = Vec::with_capacity(8 + encrypted.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(&ENCRYPTED_FRAME_SENTINEL.to_be_bytes());
    framed.extend_from_slice(&encrypted);
    Ok(framed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::hnsw::graph::HnswConfig;
    use std::collections::HashMap;

    fn tmp_dir() -> String {
        let id = uuid::Uuid::new_v4();
        let dir = std::env::temp_dir().join(format!("vdb_test_{id}"));
        dir.to_string_lossy().to_string()
    }

    fn cleanup(dir: &str) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn test_key() -> Arc<EncryptionKey> {
        Arc::new(
            EncryptionKey::from_hex(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_append_and_replay() {
        let dir = tmp_dir();
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::CreateCollection {
                name: "test".into(),
                dimension: 128,
                config: HnswConfig::default(),
            })
            .unwrap();
            wal.append(&WalEntry::DeleteCollection {
                name: "test".into(),
            })
            .unwrap();

            let (entries, stats) = wal.replay().unwrap();
            assert_eq!(stats.success, 2);
            assert_eq!(stats.skipped, 0);
            assert_eq!(stats.crc_errors, 0);
            assert_eq!(entries.len(), 2);
            match &entries[0] {
                WalEntry::CreateCollection {
                    name, dimension, ..
                } => {
                    assert_eq!(name, "test");
                    assert_eq!(*dimension, 128);
                }
                _ => panic!("expected CreateCollection"),
            }
        }
        cleanup(&dir);
    }

    #[test]
    fn test_insert_document_roundtrip() {
        let dir = tmp_dir();
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            let doc = Document::new("hello world".into(), HashMap::new());
            let doc_id = doc.id;
            wal.append(&WalEntry::InsertDocument {
                collection_name: "col1".into(),
                document: doc,
                embedding: vec![1.0, 2.0, 3.0],
            })
            .unwrap();

            let (entries, stats) = wal.replay().unwrap();
            assert_eq!(stats.success, 1);
            match &entries[0] {
                WalEntry::InsertDocument {
                    collection_name,
                    document,
                    embedding,
                } => {
                    assert_eq!(collection_name, "col1");
                    assert_eq!(document.id, doc_id);
                    assert_eq!(embedding, &[1.0, 2.0, 3.0]);
                }
                _ => panic!("expected InsertDocument"),
            }
        }
        cleanup(&dir);
    }

    #[test]
    fn test_truncate_clears_wal() {
        let dir = tmp_dir();
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "x".into() })
                .unwrap();
            wal.truncate().unwrap();
            let (entries, _) = wal.replay().unwrap();
            assert!(entries.is_empty(), "WAL should be empty after truncate");
        }
        cleanup(&dir);
    }

    #[test]
    fn test_crc_corruption_detected() {
        let dir = tmp_dir();
        let wal_path;
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "a".into() })
                .unwrap();
            wal_path = PathBuf::from(&dir).join("wal.bin");
        }
        let mut data = std::fs::read(&wal_path).unwrap();
        if data.len() > 10 {
            data[10] ^= 0xFF;
        }
        std::fs::write(&wal_path, &data).unwrap();
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            let (_entries, stats) = wal.replay().unwrap();
            assert!(
                stats.crc_errors > 0 || stats.skipped > 0,
                "corruption should be detected"
            );
        }
        cleanup(&dir);
    }

    #[test]
    fn test_freeze_and_truncate() {
        let dir = tmp_dir();
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "b".into() })
                .unwrap();
            let _gate = wal.freeze();
            wal.truncate().unwrap();
        }
        {
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "c".into() })
                .unwrap();
            let (entries, _) = wal.replay().unwrap();
            assert_eq!(entries.len(), 1);
        }
        cleanup(&dir);
    }

    #[test]
    fn test_serialize_and_frame_format() {
        let entry = WalEntry::DeleteCollection {
            name: "test".into(),
        };
        let framed = serialize_and_frame(&entry).unwrap();
        let len = u32::from_be_bytes([framed[0], framed[1], framed[2], framed[3]]) as usize;
        let stored_crc = u32::from_be_bytes([framed[4], framed[5], framed[6], framed[7]]);
        let payload = &framed[8..];
        assert_eq!(payload.len(), len);
        assert_eq!(crc32fast::hash(payload), stored_crc);
    }

    // ── Encrypted WAL tests ────────────────────────────────────────────

    #[test]
    fn test_encrypted_append_and_replay() {
        let dir = tmp_dir();
        let key = test_key();
        {
            let wal = SyncWriteAheadLog::new(&dir, Some(key.clone())).unwrap();
            wal.append(&WalEntry::CreateCollection {
                name: "enc_test".into(),
                dimension: 64,
                config: HnswConfig::default(),
            })
            .unwrap();
            wal.append(&WalEntry::DeleteCollection {
                name: "enc_test".into(),
            })
            .unwrap();

            let (entries, stats) = wal.replay().unwrap();
            assert_eq!(stats.success, 2);
            assert_eq!(stats.crc_errors, 0);
            assert_eq!(entries.len(), 2);
            match &entries[0] {
                WalEntry::CreateCollection {
                    name, dimension, ..
                } => {
                    assert_eq!(name, "enc_test");
                    assert_eq!(*dimension, 64);
                }
                _ => panic!("expected CreateCollection"),
            }
        }
        cleanup(&dir);
    }

    #[test]
    fn test_encrypted_wal_has_sentinel() {
        let entry = WalEntry::DeleteCollection {
            name: "test".into(),
        };
        let key = test_key();
        let framed = serialize_and_frame_encrypted(&entry, &key).unwrap();
        let stored_crc = u32::from_be_bytes([framed[4], framed[5], framed[6], framed[7]]);
        assert_eq!(stored_crc, ENCRYPTED_FRAME_SENTINEL);
    }

    #[test]
    fn test_encrypted_wal_without_key_fails_replay() {
        let dir = tmp_dir();
        let key = test_key();
        {
            let wal = SyncWriteAheadLog::new(&dir, Some(key)).unwrap();
            wal.append(&WalEntry::DeleteCollection {
                name: "secret".into(),
            })
            .unwrap();
        }
        {
            // Replay without key
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            let (_entries, stats) = wal.replay().unwrap();
            assert!(stats.crc_errors > 0, "should fail without key");
        }
        cleanup(&dir);
    }

    #[test]
    fn test_mixed_encrypted_and_plaintext_replay() {
        let dir = tmp_dir();
        let key = test_key();
        {
            // Write plaintext entry
            let wal = SyncWriteAheadLog::new(&dir, None).unwrap();
            wal.append(&WalEntry::CreateCollection {
                name: "plain".into(),
                dimension: 32,
                config: HnswConfig::default(),
            })
            .unwrap();
        }
        {
            // Append encrypted entry to same WAL file
            let wal = SyncWriteAheadLog::new(&dir, Some(key.clone())).unwrap();
            wal.append(&WalEntry::DeleteCollection {
                name: "plain".into(),
            })
            .unwrap();
        }
        {
            // Replay with key — should read both
            let wal = SyncWriteAheadLog::new(&dir, Some(key)).unwrap();
            let (entries, stats) = wal.replay().unwrap();
            assert_eq!(stats.success, 2);
            assert_eq!(entries.len(), 2);
        }
        cleanup(&dir);
    }
}
