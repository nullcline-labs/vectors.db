//! Replication wire protocol: binary-framed messages over TCP.
//!
//! Every message is `[u32 msg_type BE][u32 payload_len BE][payload]`.

use serde::{Deserialize, Serialize};
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ── Message type constants ───────────────────────────────────────────

pub const MSG_HANDSHAKE: u32 = 0x01;
pub const MSG_HANDSHAKE_ACK: u32 = 0x02;
pub const MSG_SNAPSHOT_BEGIN: u32 = 0x10;
pub const MSG_SNAPSHOT_CHUNK: u32 = 0x11;
pub const MSG_SNAPSHOT_END: u32 = 0x12;
pub const MSG_WAL_FRAME: u32 = 0x20;
pub const MSG_WAL_ACK: u32 = 0x21;
pub const MSG_PING: u32 = 0xF0;
pub const MSG_PONG: u32 = 0xF1;

/// Protocol version. Bump on breaking changes.
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum single message payload (64 MB safety limit).
const MAX_PAYLOAD_SIZE: u32 = 64 * 1024 * 1024;

// ── Typed message payloads ───────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct Handshake {
    pub version: u32,
    pub node_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeAck {
    pub ok: bool,
    pub wal_position: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotBegin {
    pub collection_name: String,
    pub total_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotEnd {
    pub collection_name: String,
    pub checksum: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WalAck {
    pub wal_position: u64,
}

// ── Encode / decode ──────────────────────────────────────────────────

/// Write a framed message: `[msg_type u32 BE][payload_len u32 BE][payload]`.
pub async fn write_message<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    msg_type: u32,
    payload: &[u8],
) -> io::Result<()> {
    let mut header = [0u8; 8];
    header[..4].copy_from_slice(&msg_type.to_be_bytes());
    header[4..8].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    writer.write_all(&header).await?;
    if !payload.is_empty() {
        writer.write_all(payload).await?;
    }
    writer.flush().await?;
    Ok(())
}

/// Read a framed message, returning `(msg_type, payload)`.
pub async fn read_message<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<(u32, Vec<u8>)> {
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).await?;
    let msg_type = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let payload_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    if payload_len > MAX_PAYLOAD_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload too large: {} bytes", payload_len),
        ));
    }
    let mut payload = vec![0u8; payload_len as usize];
    if payload_len > 0 {
        reader.read_exact(&mut payload).await?;
    }
    Ok((msg_type, payload))
}

/// Serialize a serde-compatible value to JSON bytes.
pub fn encode_json<T: Serialize>(value: &T) -> io::Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| io::Error::other(e.to_string()))
}

/// Deserialize a serde-compatible value from JSON bytes.
pub fn decode_json<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> io::Result<T> {
    serde_json::from_slice(bytes).map_err(|e| io::Error::other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_read_roundtrip() {
        let mut buf = Vec::new();
        let mut cursor = io::Cursor::new(&mut buf);
        write_message(&mut cursor, MSG_PING, b"").await.unwrap();
        let mut reader = io::Cursor::new(&buf);
        let (msg_type, payload) = read_message(&mut reader).await.unwrap();
        assert_eq!(msg_type, MSG_PING);
        assert!(payload.is_empty());
    }

    #[tokio::test]
    async fn test_write_read_with_payload() {
        let mut buf = Vec::new();
        let hs = Handshake {
            version: PROTOCOL_VERSION,
            node_id: "standby-1".into(),
        };
        let encoded = encode_json(&hs).unwrap();
        let mut cursor = io::Cursor::new(&mut buf);
        write_message(&mut cursor, MSG_HANDSHAKE, &encoded)
            .await
            .unwrap();

        let mut reader = io::Cursor::new(&buf);
        let (msg_type, payload) = read_message(&mut reader).await.unwrap();
        assert_eq!(msg_type, MSG_HANDSHAKE);
        let decoded: Handshake = decode_json(&payload).unwrap();
        assert_eq!(decoded.version, PROTOCOL_VERSION);
        assert_eq!(decoded.node_id, "standby-1");
    }

    #[tokio::test]
    async fn test_snapshot_begin_roundtrip() {
        let msg = SnapshotBegin {
            collection_name: "test_col".into(),
            total_bytes: 12345,
        };
        let encoded = encode_json(&msg).unwrap();
        let mut buf = Vec::new();
        let mut cursor = io::Cursor::new(&mut buf);
        write_message(&mut cursor, MSG_SNAPSHOT_BEGIN, &encoded)
            .await
            .unwrap();

        let mut reader = io::Cursor::new(&buf);
        let (msg_type, payload) = read_message(&mut reader).await.unwrap();
        assert_eq!(msg_type, MSG_SNAPSHOT_BEGIN);
        let decoded: SnapshotBegin = decode_json(&payload).unwrap();
        assert_eq!(decoded.collection_name, "test_col");
        assert_eq!(decoded.total_bytes, 12345);
    }

    #[tokio::test]
    async fn test_wal_frame_binary() {
        let wal_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02];
        let mut buf = Vec::new();
        let mut cursor = io::Cursor::new(&mut buf);
        write_message(&mut cursor, MSG_WAL_FRAME, &wal_data)
            .await
            .unwrap();

        let mut reader = io::Cursor::new(&buf);
        let (msg_type, payload) = read_message(&mut reader).await.unwrap();
        assert_eq!(msg_type, MSG_WAL_FRAME);
        assert_eq!(payload, wal_data);
    }

    #[test]
    fn test_encode_decode_json_handshake_ack() {
        let ack = HandshakeAck {
            ok: true,
            wal_position: 42,
        };
        let bytes = encode_json(&ack).unwrap();
        let decoded: HandshakeAck = decode_json(&bytes).unwrap();
        assert!(decoded.ok);
        assert_eq!(decoded.wal_position, 42);
    }

    #[test]
    fn test_encode_decode_json_wal_ack() {
        let ack = WalAck { wal_position: 999 };
        let bytes = encode_json(&ack).unwrap();
        let decoded: WalAck = decode_json(&bytes).unwrap();
        assert_eq!(decoded.wal_position, 999);
    }

    #[test]
    fn test_encode_decode_json_snapshot_end() {
        let end = SnapshotEnd {
            collection_name: "my_col".into(),
            checksum: 0xDEADBEEF,
        };
        let bytes = encode_json(&end).unwrap();
        let decoded: SnapshotEnd = decode_json(&bytes).unwrap();
        assert_eq!(decoded.collection_name, "my_col");
        assert_eq!(decoded.checksum, 0xDEADBEEF);
    }
}
