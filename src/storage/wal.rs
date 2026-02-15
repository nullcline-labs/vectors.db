//! Write-Ahead Log (WAL) for crash recovery.
//!
//! Every mutation is appended to the WAL before being applied in memory.
//! Each entry is framed as `[u32 length BE][u32 CRC32 BE][bincode payload]`
//! and durably flushed with `fsync`.
//!
//! Uses **group commit** to batch multiple concurrent appends into a single
//! write + fsync cycle, dramatically improving throughput under load.
//! On startup, the WAL is replayed to recover any mutations not yet
//! persisted to snapshots.

use crate::config;
use crate::document::Document;
use crate::hnsw::graph::HnswConfig;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
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
    DeleteCollection {
        name: String,
    },
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

/// A request from a caller to append an entry to the WAL.
struct GroupCommitRequest {
    /// Pre-serialized + framed bytes: `[u32 len BE][u32 crc32 BE][bincode payload]`.
    framed_bytes: Vec<u8>,
    /// Caller awaits this to learn whether the write + fsync succeeded.
    result_tx: oneshot::Sender<io::Result<()>>,
}

/// Append-only write-ahead log with CRC32 integrity checks and group commit.
///
/// Concurrent [`append`](WriteAheadLog::append) calls are batched into a single
/// write + fsync cycle by a background task, reducing I/O syscalls under load.
///
/// The `write_gate` allows callers to [`freeze`](WriteAheadLog::freeze) the WAL,
/// blocking all appends while a snapshot + truncate is performed atomically.
pub struct WriteAheadLog {
    /// Channel to submit entries to the background batch writer.
    submit_tx: mpsc::Sender<GroupCommitRequest>,
    /// Write gate: freeze() takes exclusive, batch writer takes shared.
    write_gate: Arc<parking_lot::RwLock<()>>,
    /// Path to WAL file (needed for replay/truncate).
    path: PathBuf,
    /// Writer mutex, shared with background task and used by truncate().
    writer: Arc<Mutex<BufWriter<File>>>,
}

impl WriteAheadLog {
    /// Open or create the WAL file in append mode and spawn the background
    /// batch writer task.
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
        let writer = Arc::new(Mutex::new(BufWriter::new(file)));
        let write_gate = Arc::new(parking_lot::RwLock::new(()));

        let (submit_tx, submit_rx) = mpsc::channel::<GroupCommitRequest>(4096);

        let task_writer = Arc::clone(&writer);
        let task_gate = Arc::clone(&write_gate);

        tokio::spawn(async move {
            batch_writer_loop(submit_rx, task_writer, task_gate).await;
        });

        Ok(Self {
            submit_tx,
            write_gate,
            path,
            writer,
        })
    }

    /// Append a WAL entry using group commit.
    ///
    /// Serializes the entry and submits it to the background batch writer.
    /// Returns when the entry (and its batch) has been durably fsynced to disk.
    /// Under high concurrency, multiple entries share a single fsync call.
    pub async fn append(&self, entry: &WalEntry) -> io::Result<()> {
        let framed = serialize_and_frame(entry)?;

        let (result_tx, result_rx) = oneshot::channel();
        self.submit_tx
            .send(GroupCommitRequest {
                framed_bytes: framed,
                result_tx,
            })
            .await
            .map_err(|_| {
                io::Error::new(io::ErrorKind::BrokenPipe, "WAL batch writer stopped")
            })?;

        result_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL batch result lost"))?
    }

    /// Read all entries from the WAL file sequentially, verifying CRC32 checksums.
    ///
    /// Returns the successfully deserialized entries and diagnostic statistics.
    /// - On CRC mismatch: stops replay (cannot determine next entry boundary).
    /// - On truncated entry: stops replay (expected at end of WAL after crash).
    /// - On deserialization error with valid CRC: skips the entry and continues
    ///   (the entry boundary is known from the length header).
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

    /// Acquire an exclusive write gate, blocking all [`append`](WriteAheadLog::append) calls.
    ///
    /// Waits for any in-flight batch to complete, then blocks the background
    /// writer from starting new batches. Hold the returned guard while
    /// performing snapshot + truncate.
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

/// Background task that collects WAL entries and writes them in batches.
///
/// Under low load (single entry, no contention): flushes immediately with no delay.
/// Under high load: batches up to `WAL_GROUP_COMMIT_MAX_BATCH` entries or waits
/// up to `WAL_GROUP_COMMIT_MAX_WAIT_US` for more entries before flushing.
async fn batch_writer_loop(
    mut rx: mpsc::Receiver<GroupCommitRequest>,
    writer: Arc<Mutex<BufWriter<File>>>,
    write_gate: Arc<parking_lot::RwLock<()>>,
) {
    let max_batch = config::WAL_GROUP_COMMIT_MAX_BATCH;
    let max_wait = Duration::from_micros(config::WAL_GROUP_COMMIT_MAX_WAIT_US);
    let mut batch: Vec<GroupCommitRequest> = Vec::with_capacity(max_batch);

    loop {
        // Wait for the first request
        let first = match rx.recv().await {
            Some(req) => req,
            None => break, // Channel closed, WAL is being dropped
        };
        batch.push(first);

        // Non-blocking drain of anything already queued
        while batch.len() < max_batch {
            match rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        // If we got multiple entries, wait briefly for more (coalescing)
        if batch.len() > 1 && batch.len() < max_batch {
            let deadline = tokio::time::Instant::now() + max_wait;
            while batch.len() < max_batch {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    _ => break,
                }
            }
        }

        // Flush the entire batch with a single fsync
        flush_batch(&mut batch, &writer, &write_gate);
    }
}

/// Write all entries in the batch, fsync once, and notify all callers.
fn flush_batch(
    batch: &mut Vec<GroupCommitRequest>,
    writer: &Arc<Mutex<BufWriter<File>>>,
    write_gate: &Arc<parking_lot::RwLock<()>>,
) {
    // Acquire write gate in shared mode (blocks only if freeze() holds exclusive)
    let _gate = write_gate.read();
    let mut w = writer.lock();

    // Write all frames
    let mut write_err: Option<io::Error> = None;
    for req in batch.iter() {
        if write_err.is_none() {
            if let Err(e) = w.write_all(&req.framed_bytes) {
                write_err = Some(e);
            }
        }
    }

    // Single flush + fsync for the entire batch
    if write_err.is_none() {
        if let Err(e) = w.flush() {
            write_err = Some(e);
        }
    }
    if write_err.is_none() {
        if let Err(e) = w.get_mut().sync_all() {
            write_err = Some(e);
        }
    }

    // Notify all callers
    if let Some(ref e) = write_err {
        for req in batch.drain(..) {
            let _ = req
                .result_tx
                .send(Err(io::Error::new(e.kind(), e.to_string())));
        }
    } else {
        for req in batch.drain(..) {
            let _ = req.result_tx.send(Ok(()));
        }
    }
}
