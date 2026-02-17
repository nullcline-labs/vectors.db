//! Standby-side replication: connects to primary, receives snapshots and
//! applies WAL frames continuously.

use super::protocol::*;
use super::ReplicationState;
use std::io;
use std::path::Path;
use tokio::io::BufWriter;
use tokio::net::TcpStream;
use vectorsdb_core::storage::wal::WalEntry;
use vectorsdb_core::storage::Database;

/// Sentinel CRC value that marks an encrypted WAL frame.
const ENCRYPTED_FRAME_SENTINEL: u32 = 0xFFFF_FFFF;

/// Connect to the primary and replicate continuously.
///
/// Reconnects with exponential backoff on failure.
pub async fn start_standby(
    primary_addr: String,
    db: Database,
    repl_state: ReplicationState,
    data_dir: String,
) {
    let mut backoff_ms: u64 = 500;
    let max_backoff_ms: u64 = 30_000;

    loop {
        tracing::info!("Connecting to primary at {}", primary_addr);
        match replicate_once(&primary_addr, &db, &repl_state, &data_dir).await {
            Ok(()) => {
                tracing::info!("Replication stream ended normally");
                backoff_ms = 500;
            }
            Err(e) => {
                tracing::warn!("Replication error: {}. Retrying in {}ms", e, backoff_ms);
            }
        }

        if !repl_state.is_standby() {
            tracing::info!("Node promoted, stopping standby replication");
            break;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
    }
}

async fn replicate_once(
    primary_addr: &str,
    db: &Database,
    repl_state: &ReplicationState,
    data_dir: &str,
) -> io::Result<()> {
    let stream = TcpStream::connect(primary_addr).await?;
    let (reader, writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    // ── Handshake ────────────────────────────────────────────────
    let handshake = Handshake {
        version: PROTOCOL_VERSION,
        node_id: "standby".into(),
    };
    write_message(&mut writer, MSG_HANDSHAKE, &encode_json(&handshake)?).await?;

    let (msg_type, payload) = read_message(&mut reader).await?;
    if msg_type != MSG_HANDSHAKE_ACK {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected handshake ack",
        ));
    }
    let ack: HandshakeAck = decode_json(&payload)?;
    if !ack.ok {
        return Err(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            "handshake rejected by primary",
        ));
    }
    tracing::info!("Connected to primary, wal_position={}", ack.wal_position);

    // ── Receive snapshots and WAL frames ─────────────────────────
    let mut snapshot_buf: Vec<u8> = Vec::new();

    loop {
        if !repl_state.is_standby() {
            tracing::info!("Promoted during replication, disconnecting");
            break;
        }

        let (msg_type, payload) = read_message(&mut reader).await?;

        match msg_type {
            MSG_SNAPSHOT_BEGIN => {
                let begin: SnapshotBegin = decode_json(&payload)?;
                tracing::info!(
                    "Receiving snapshot '{}' ({} bytes)",
                    begin.collection_name,
                    begin.total_bytes
                );
                snapshot_buf.clear();
                snapshot_buf.reserve(begin.total_bytes as usize);
                // Name stored in SnapshotEnd for verification
            }

            MSG_SNAPSHOT_CHUNK => {
                snapshot_buf.extend_from_slice(&payload);
            }

            MSG_SNAPSHOT_END => {
                let end: SnapshotEnd = decode_json(&payload)?;
                let checksum = crc32fast::hash(&snapshot_buf);
                if checksum != end.checksum {
                    tracing::error!(
                        "Snapshot '{}' checksum mismatch: expected {:#x}, got {:#x}",
                        end.collection_name,
                        end.checksum,
                        checksum
                    );
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "snapshot checksum mismatch",
                    ));
                }

                // Write snapshot to disk and load it
                let snapshot_path =
                    Path::new(data_dir).join(format!("{}.vdb", end.collection_name));
                tokio::fs::write(&snapshot_path, &snapshot_buf).await?;

                let collection = vectorsdb_core::storage::load_collection(&snapshot_path, None)
                    .map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("failed to load snapshot '{}': {}", end.collection_name, e),
                        )
                    })?;

                let mut collections = db.collections.write();
                collections.insert(end.collection_name.clone(), collection);
                drop(collections);

                tracing::info!(
                    "Loaded snapshot '{}' ({} bytes)",
                    end.collection_name,
                    snapshot_buf.len()
                );
                snapshot_buf.clear();
            }

            MSG_WAL_FRAME => {
                apply_wal_frame(db, &payload)?;
                let pos = repl_state.advance_wal_position();
                // Send ack
                let wal_ack = WalAck { wal_position: pos };
                write_message(&mut writer, MSG_WAL_ACK, &encode_json(&wal_ack)?).await?;
            }

            MSG_PING => {
                write_message(&mut writer, MSG_PONG, b"").await?;
            }

            _ => {
                tracing::warn!("Unknown message type {:#x}, ignoring", msg_type);
            }
        }
    }

    Ok(())
}

/// Deserialize WAL frame bytes and apply to the database.
///
/// WAL frames use the same on-disk format: `[u32 len BE][u32 crc BE][bincode payload]`.
/// A batch may contain multiple concatenated frames.
fn apply_wal_frame(db: &Database, framed_bytes: &[u8]) -> io::Result<()> {
    let mut offset = 0;
    while offset + 8 <= framed_bytes.len() {
        let len = u32::from_be_bytes([
            framed_bytes[offset],
            framed_bytes[offset + 1],
            framed_bytes[offset + 2],
            framed_bytes[offset + 3],
        ]) as usize;
        let stored_crc = u32::from_be_bytes([
            framed_bytes[offset + 4],
            framed_bytes[offset + 5],
            framed_bytes[offset + 6],
            framed_bytes[offset + 7],
        ]);
        offset += 8;

        if offset + len > framed_bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL frame truncated",
            ));
        }

        let data = &framed_bytes[offset..offset + len];
        offset += len;

        // Encrypted frames are passed through as-is — standby cannot decrypt
        // without the key. For now we only support plaintext replication.
        if stored_crc == ENCRYPTED_FRAME_SENTINEL {
            tracing::warn!(
                "Encrypted WAL frame received — skipping (not supported in replication yet)"
            );
            continue;
        }

        let computed_crc = crc32fast::hash(data);
        if computed_crc != stored_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL frame CRC mismatch",
            ));
        }

        let entry: WalEntry = bincode::deserialize(data)
            .map_err(|e| io::Error::other(format!("WAL deserialize error: {}", e)))?;

        db.apply_wal_entry(&entry);
    }
    Ok(())
}
