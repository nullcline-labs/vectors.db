//! In-memory Raft log store and state machine.
//!
//! Provides `LogStore` (implements `RaftLogStorage`) for persisting Raft log entries
//! and votes in memory, and `StateMachineStore` (implements `RaftStateMachine`) that
//! applies committed entries to the shared [`Database`](crate::storage::Database) and routing table.

use crate::cluster::types::{LogEntry, LogEntryResponse, NodeId, TypeConfig};
use crate::storage::Database;
use openraft::storage::{LogFlushed, LogState, RaftLogStorage, RaftStateMachine, Snapshot};
use openraft::{
    BasicNode, Entry, EntryPayload, LogId, RaftLogReader, RaftSnapshotBuilder, SnapshotMeta,
    StorageError, StorageIOError, StoredMembership, Vote,
};
use parking_lot::RwLock as ParkingRwLock;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

// ---- Log Store ----

/// In-memory Raft log storage backed by a `BTreeMap`.
///
/// Stores log entries, votes, and committed log IDs. Implements openraft's
/// `RaftLogStorage` and `RaftLogReader` traits.
#[derive(Debug, Clone, Default)]
pub struct LogStore {
    inner: Arc<Mutex<LogStoreInner>>,
}

#[derive(Debug, Default)]
struct LogStoreInner {
    last_purged_log_id: Option<LogId<NodeId>>,
    log: BTreeMap<u64, Entry<TypeConfig>>,
    committed: Option<LogId<NodeId>>,
    vote: Option<Vote<NodeId>>,
}

impl RaftLogReader<TypeConfig> for LogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        let entries = inner.log.range(range).map(|(_, v)| v.clone()).collect();
        Ok(entries)
    }
}

impl RaftLogStorage<TypeConfig> for LogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        let last = inner
            .log
            .iter()
            .next_back()
            .map(|(_, e)| e.log_id)
            .or(inner.last_purged_log_id);
        Ok(LogState {
            last_purged_log_id: inner.last_purged_log_id,
            last_log_id: last,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.vote = Some(*vote);
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        Ok(inner.vote)
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<TypeConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>>,
    {
        let mut inner = self.inner.lock().await;
        for entry in entries {
            inner.log.insert(entry.log_id.index, entry);
        }
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        let keys: Vec<u64> = inner.log.range(log_id.index..).map(|(k, _)| *k).collect();
        for key in keys {
            inner.log.remove(&key);
        }
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.last_purged_log_id = Some(log_id);
        let keys: Vec<u64> = inner.log.range(..=log_id.index).map(|(k, _)| *k).collect();
        for key in keys {
            inner.log.remove(&key);
        }
        Ok(())
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.committed = committed;
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        Ok(inner.committed)
    }
}

// ---- State Machine Store ----

/// Raft state machine that applies committed log entries to the database.
///
/// Maintains the shared [`Database`] reference and collection-to-node routing table.
/// Supports snapshot creation and installation for log compaction.
#[derive(Debug)]
pub struct StateMachineStore {
    pub db: Arc<Database>,
    pub routing_table: Arc<ParkingRwLock<HashMap<String, NodeId>>>,
    last_applied_log: RwLock<Option<LogId<NodeId>>>,
    last_membership: RwLock<StoredMembership<NodeId, BasicNode>>,
    snapshot_idx: AtomicU64,
    current_snapshot: RwLock<Option<StoredSnapshot>>,
}

#[derive(Debug)]
struct StoredSnapshot {
    meta: SnapshotMeta<NodeId, BasicNode>,
    data: Vec<u8>,
}

impl StateMachineStore {
    /// Creates a new state machine store wrapping the given database.
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            routing_table: Arc::new(ParkingRwLock::new(HashMap::new())),
            last_applied_log: RwLock::new(None),
            last_membership: RwLock::new(StoredMembership::default()),
            snapshot_idx: AtomicU64::new(0),
            current_snapshot: RwLock::new(None),
        }
    }

    fn apply_entry(&self, entry: &LogEntry) -> LogEntryResponse {
        match entry {
            LogEntry::CreateCollection {
                name,
                dimension,
                config,
            } => {
                let _ = self
                    .db
                    .create_collection(name.clone(), *dimension, Some(config.clone()));
                LogEntryResponse::ok()
            }
            LogEntry::DeleteCollection { name } => {
                self.db.delete_collection(name);
                LogEntryResponse::ok()
            }
            LogEntry::InsertDocument {
                collection_name,
                document,
                embedding,
            } => {
                if let Some(collection) = self.db.get_collection(collection_name) {
                    collection.insert_document(document.clone(), embedding.clone());
                }
                LogEntryResponse::ok()
            }
            LogEntry::DeleteDocument {
                collection_name,
                document_id,
            } => {
                if let Some(collection) = self.db.get_collection(collection_name) {
                    collection.delete_document(document_id);
                }
                LogEntryResponse::ok()
            }
            LogEntry::InsertDocumentBatch {
                collection_name,
                documents,
            } => {
                if let Some(collection) = self.db.get_collection(collection_name) {
                    for (doc, emb) in documents {
                        collection.insert_document(doc.clone(), emb.clone());
                    }
                }
                LogEntryResponse::ok()
            }
            LogEntry::UpdateDocument {
                collection_name,
                document_id,
                document,
                embedding,
            } => {
                if let Some(collection) = self.db.get_collection(collection_name) {
                    collection.delete_document(document_id);
                    collection.insert_document(document.clone(), embedding.clone());
                }
                LogEntryResponse::ok()
            }
            LogEntry::AssignCollection {
                collection_name,
                owner_node_id,
            } => {
                self.routing_table
                    .write()
                    .insert(collection_name.clone(), *owner_node_id);
                LogEntryResponse::ok()
            }
        }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for Arc<StateMachineStore> {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<NodeId>> {
        let last_applied = *self.last_applied_log.read().await;
        let last_membership = self.last_membership.read().await.clone();

        // Serialize the routing table as the snapshot data
        let routing = self.routing_table.read().clone();
        let data =
            serde_json::to_vec(&routing).map_err(|e| StorageIOError::read_state_machine(&e))?;

        let idx = self.snapshot_idx.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot_id = if let Some(last) = last_applied {
            format!("{}-{}-{}", last.leader_id, last.index, idx)
        } else {
            format!("--{}", idx)
        };

        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership,
            snapshot_id,
        };

        let mut current = self.current_snapshot.write().await;
        *current = Some(StoredSnapshot {
            meta: meta.clone(),
            data: data.clone(),
        });

        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(data)),
        })
    }
}

impl RaftStateMachine<TypeConfig> for Arc<StateMachineStore> {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership<NodeId, BasicNode>), StorageError<NodeId>>
    {
        let last = *self.last_applied_log.read().await;
        let membership = self.last_membership.read().await.clone();
        Ok((last, membership))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<LogEntryResponse>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + Send,
    {
        let mut responses = Vec::new();
        for entry in entries {
            *self.last_applied_log.write().await = Some(entry.log_id);

            let resp = match &entry.payload {
                EntryPayload::Blank => LogEntryResponse::ok(),
                EntryPayload::Normal(log_entry) => self.apply_entry(log_entry),
                EntryPayload::Membership(mem) => {
                    *self.last_membership.write().await =
                        StoredMembership::new(Some(entry.log_id), mem.clone());
                    LogEntryResponse::ok()
                }
            };
            responses.push(resp);
        }
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let data = snapshot.into_inner();
        let routing: HashMap<String, NodeId> = serde_json::from_slice(&data)
            .map_err(|e| StorageIOError::read_snapshot(Some(meta.signature()), &e))?;

        *self.routing_table.write() = routing;
        *self.last_applied_log.write().await = meta.last_log_id;
        *self.last_membership.write().await = meta.last_membership.clone();

        let mut current = self.current_snapshot.write().await;
        *current = Some(StoredSnapshot {
            meta: meta.clone(),
            data,
        });

        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<NodeId>> {
        let current = self.current_snapshot.read().await;
        Ok(current.as_ref().map(|s| Snapshot {
            meta: s.meta.clone(),
            snapshot: Box::new(Cursor::new(s.data.clone())),
        }))
    }
}
