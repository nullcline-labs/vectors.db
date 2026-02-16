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
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use uuid::Uuid;

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
}

impl SyncWriteAheadLog {
    /// Open or create the WAL file in append mode.
    pub fn new(data_dir: &str) -> io::Result<Self> {
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
        })
    }

    /// Append a WAL entry synchronously.
    ///
    /// Serializes the entry, writes, flushes, and fsyncs to disk before returning.
    pub fn append(&self, entry: &WalEntry) -> io::Result<()> {
        let framed = serialize_and_frame(entry)?;

        let _gate = self.write_gate.read();
        let mut w = self.writer.lock();
        w.write_all(&framed)?;
        w.flush()?;
        w.get_mut().sync_all()?;
        Ok(())
    }

    /// Read all entries from the WAL file sequentially, verifying CRC32 checksums.
    ///
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
            let computed_crc = crc32fast::hash(&data);
            if computed_crc != stored_crc {
                tracing::warn!("WAL entry CRC mismatch, stopping replay");
                stats.crc_errors += 1;
                break;
            }
            match bincode::deserialize::<WalEntry>(&data) {
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
fn serialize_and_frame(entry: &WalEntry) -> io::Result<Vec<u8>> {
    let bytes = bincode::serialize(entry).map_err(|e| io::Error::other(e.to_string()))?;
    let len = bytes.len() as u32;
    let crc = crc32fast::hash(&bytes);

    let mut framed = Vec::with_capacity(8 + bytes.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(&crc.to_be_bytes());
    framed.extend_from_slice(&bytes);
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

    #[test]
    fn test_append_and_replay() {
        let dir = tmp_dir();
        {
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
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
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
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
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
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
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "a".into() })
                .unwrap();
            wal_path = PathBuf::from(&dir).join("wal.bin");
        }
        // Corrupt one byte in the payload
        let mut data = std::fs::read(&wal_path).unwrap();
        if data.len() > 10 {
            data[10] ^= 0xFF;
        }
        std::fs::write(&wal_path, &data).unwrap();
        {
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
            let (_entries, stats) = wal.replay().unwrap();
            // Either CRC error or deserialization failure
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
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
            wal.append(&WalEntry::DeleteCollection { name: "b".into() })
                .unwrap();
            let _gate = wal.freeze();
            wal.truncate().unwrap();
            // After drop of gate, appends should work again
        }
        {
            let wal = SyncWriteAheadLog::new(&dir).unwrap();
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
        // First 4 bytes = length (BE), next 4 = CRC32 (BE)
        let len = u32::from_be_bytes([framed[0], framed[1], framed[2], framed[3]]) as usize;
        let stored_crc = u32::from_be_bytes([framed[4], framed[5], framed[6], framed[7]]);
        let payload = &framed[8..];
        assert_eq!(payload.len(), len);
        assert_eq!(crc32fast::hash(payload), stored_crc);
    }
}
