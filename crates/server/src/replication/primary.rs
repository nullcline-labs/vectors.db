//! Primary-side replication: TCP listener that streams snapshots + WAL frames
//! to connected standby nodes.

use super::protocol::*;
use super::ReplicationState;
use std::net::SocketAddr;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use vectorsdb_core::storage::{save_collection, Database};

/// Start the replication listener.  Spawns a task per standby connection.
pub async fn start_replication_listener(
    addr: SocketAddr,
    db: Database,
    repl_state: ReplicationState,
    wal_tx: broadcast::Sender<Vec<u8>>,
    data_dir: String,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Replication listener on {}", addr);

    loop {
        let (stream, peer) = listener.accept().await?;
        tracing::info!("Standby connected from {}", peer);
        let db = db.clone();
        let repl_state = repl_state.clone();
        let wal_rx = wal_tx.subscribe();
        let data_dir = data_dir.clone();
        tokio::spawn(async move {
            if let Err(e) =
                handle_standby_connection(stream, db, repl_state, wal_rx, data_dir).await
            {
                tracing::warn!("Standby {} disconnected: {}", peer, e);
            }
        });
    }
}

async fn handle_standby_connection(
    stream: TcpStream,
    db: Database,
    repl_state: ReplicationState,
    mut wal_rx: broadcast::Receiver<Vec<u8>>,
    data_dir: String,
) -> std::io::Result<()> {
    let (reader, writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    // ── Handshake ────────────────────────────────────────────────
    let (msg_type, payload) = read_message(&mut reader).await?;
    if msg_type != MSG_HANDSHAKE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "expected handshake",
        ));
    }
    let handshake: Handshake = decode_json(&payload)?;
    if handshake.version != PROTOCOL_VERSION {
        let ack = HandshakeAck {
            ok: false,
            wal_position: 0,
        };
        write_message(&mut writer, MSG_HANDSHAKE_ACK, &encode_json(&ack)?).await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("protocol version mismatch: {}", handshake.version),
        ));
    }
    tracing::info!(
        "Handshake from node '{}' v{}",
        handshake.node_id,
        handshake.version
    );

    let pos = repl_state.wal_position();
    let ack = HandshakeAck {
        ok: true,
        wal_position: pos,
    };
    write_message(&mut writer, MSG_HANDSHAKE_ACK, &encode_json(&ack)?).await?;

    // ── Snapshot transfer ────────────────────────────────────────
    send_snapshots(&mut writer, &db, &data_dir).await?;

    // ── Continuous WAL streaming ─────────────────────────────────
    let keepalive = tokio::time::Duration::from_secs(5);
    loop {
        tokio::select! {
            result = wal_rx.recv() => {
                match result {
                    Ok(framed_bytes) => {
                        write_message(&mut writer, MSG_WAL_FRAME, &framed_bytes).await?;
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("Standby lagged by {} WAL batches", n);
                        // Continue — standby will get next frames
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::info!("WAL broadcast channel closed, stopping replication");
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(keepalive) => {
                write_message(&mut writer, MSG_PING, b"").await?;
            }
        }
    }

    Ok(())
}

/// Send all collection snapshots to the standby.
async fn send_snapshots<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    db: &Database,
    data_dir: &str,
) -> std::io::Result<()> {
    let names: Vec<String> = {
        let collections = db.collections.read();
        collections.keys().cloned().collect()
    };

    for name in &names {
        let collection = match db.get_collection(name) {
            Some(c) => c,
            None => continue,
        };

        // Save snapshot to disk (reuse existing persistence), then read bytes
        save_collection(&collection, data_dir, None).map_err(|e| {
            std::io::Error::other(format!("snapshot save for '{}' failed: {}", name, e))
        })?;

        let snapshot_path = std::path::Path::new(data_dir).join(format!("{}.vdb", name));
        let snapshot_bytes = tokio::fs::read(&snapshot_path).await?;
        let checksum = crc32fast::hash(&snapshot_bytes);

        // SnapshotBegin
        let begin = SnapshotBegin {
            collection_name: name.clone(),
            total_bytes: snapshot_bytes.len() as u64,
        };
        write_message(writer, MSG_SNAPSHOT_BEGIN, &encode_json(&begin)?).await?;

        // Send chunks (64 KB each)
        const CHUNK_SIZE: usize = 64 * 1024;
        for chunk in snapshot_bytes.chunks(CHUNK_SIZE) {
            write_message(writer, MSG_SNAPSHOT_CHUNK, chunk).await?;
        }

        // SnapshotEnd
        let end = SnapshotEnd {
            collection_name: name.clone(),
            checksum,
        };
        write_message(writer, MSG_SNAPSHOT_END, &encode_json(&end)?).await?;

        tracing::info!(
            "Sent snapshot '{}' ({} bytes) to standby",
            name,
            snapshot_bytes.len()
        );
    }

    Ok(())
}
