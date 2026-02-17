//! AES-256-GCM encryption for snapshots and WAL entries.
//!
//! Provides [`EncryptionKey`] which wraps a 256-bit key and exposes
//! authenticated encrypt/decrypt operations using random 12-byte nonces.

use aes_gcm::aead::rand_core::RngCore;
use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use std::io;
use std::path::Path;
use zeroize::Zeroize;

/// AES-256-GCM nonce size in bytes.
pub const NONCE_SIZE: usize = 12;

/// AES-256-GCM authentication tag size in bytes.
const TAG_SIZE: usize = 16;

/// Minimum ciphertext size: nonce + tag (no plaintext).
const MIN_ENCRYPTED_LEN: usize = NONCE_SIZE + TAG_SIZE;

/// A 256-bit AES-GCM encryption key.
///
/// The inner key material is zeroized on drop.
pub struct EncryptionKey {
    cipher: Aes256Gcm,
    /// Kept for zeroize-on-drop; the cipher holds a copy internally,
    /// but we ensure the raw bytes are scrubbed from memory.
    raw: ZeroizeKey,
}

/// Wrapper for 32 raw key bytes that zeroizes on drop.
struct ZeroizeKey([u8; 32]);

impl Drop for ZeroizeKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl EncryptionKey {
    /// Create an encryption key from a 64-character hex string.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let hex = hex.trim();
        if hex.len() != 64 {
            return Err(format!(
                "encryption key must be 64 hex characters (32 bytes), got {}",
                hex.len()
            ));
        }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|e| format!("invalid hex at position {}: {}", i * 2, e))?;
        }
        let key = Key::<Aes256Gcm>::from_slice(&bytes);
        let cipher = Aes256Gcm::new(key);
        Ok(Self {
            cipher,
            raw: ZeroizeKey(bytes),
        })
    }

    /// Load an encryption key from a file.
    ///
    /// The file may contain either:
    /// - 32 raw bytes, or
    /// - 64 hex characters (with optional trailing newline)
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;

        // Try raw 32 bytes first
        if data.len() == 32 {
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(&data);
            let key = Key::<Aes256Gcm>::from_slice(&bytes);
            let cipher = Aes256Gcm::new(key);
            return Ok(Self {
                cipher,
                raw: ZeroizeKey(bytes),
            });
        }

        // Try hex (with optional trailing whitespace/newline)
        let hex = String::from_utf8(data).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "key file is not valid UTF-8 or raw 32 bytes",
            )
        })?;
        Self::from_hex(&hex).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Encrypt plaintext using AES-256-GCM with a random nonce.
    ///
    /// Returns `nonce (12 bytes) || ciphertext || tag (16 bytes)`.
    pub fn encrypt(&self, plaintext: &[u8]) -> Vec<u8> {
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .expect("AES-GCM encryption should not fail");

        let mut output = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);
        output
    }

    /// Decrypt data produced by [`encrypt`](Self::encrypt).
    ///
    /// Expects `nonce (12 bytes) || ciphertext || tag (16 bytes)`.
    /// Returns the original plaintext or an error if authentication fails.
    pub fn decrypt(&self, data: &[u8]) -> io::Result<Vec<u8>> {
        if data.len() < MIN_ENCRYPTED_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "encrypted data too short: {} bytes (minimum {})",
                    data.len(),
                    MIN_ENCRYPTED_LEN
                ),
            ));
        }
        let nonce = Nonce::from_slice(&data[..NONCE_SIZE]);
        let ciphertext = &data[NONCE_SIZE..];

        self.cipher.decrypt(nonce, ciphertext).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "decryption failed: wrong key or corrupted data",
            )
        })
    }
}

// Prevent accidental debug-printing of key material
impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionKey")
            .field("algorithm", &"AES-256-GCM")
            .finish()
    }
}

// Suppress unused field warning — raw is kept solely for zeroize-on-drop
#[allow(dead_code)]
fn _assert_raw_field_used(k: &EncryptionKey) {
    let _ = &k.raw;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_hex() -> &'static str {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let plaintext = b"hello vectors.db encryption!";
        let encrypted = key.encrypt(plaintext);
        assert_ne!(&encrypted[NONCE_SIZE..], plaintext);
        let decrypted = key.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_empty_plaintext() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let encrypted = key.encrypt(b"");
        let decrypted = key.decrypt(&encrypted).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_from_hex_valid() {
        let key = EncryptionKey::from_hex(test_key_hex());
        assert!(key.is_ok());
    }

    #[test]
    fn test_from_hex_with_whitespace() {
        let hex = format!("  {}  \n", test_key_hex());
        let key = EncryptionKey::from_hex(&hex);
        assert!(key.is_ok());
    }

    #[test]
    fn test_from_hex_too_short() {
        let result = EncryptionKey::from_hex("0123456789abcdef");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("64 hex characters"));
    }

    #[test]
    fn test_from_hex_invalid_chars() {
        let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        let result = EncryptionKey::from_hex(bad);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid hex"));
    }

    #[test]
    fn test_decrypt_wrong_key_fails() {
        let key1 = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let key2 = EncryptionKey::from_hex(
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
        .unwrap();
        let encrypted = key1.encrypt(b"secret data");
        let result = key2.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_tampered_data_fails() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let mut encrypted = key.encrypt(b"important data");
        // Flip a byte in the ciphertext
        if encrypted.len() > NONCE_SIZE + 2 {
            encrypted[NONCE_SIZE + 2] ^= 0xFF;
        }
        let result = key.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_too_short_data() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let result = key.decrypt(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_hex() {
        let dir = std::env::temp_dir().join(format!("vdb_crypto_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("key.hex");
        std::fs::write(&path, test_key_hex()).unwrap();

        let key = EncryptionKey::from_file(&path).unwrap();
        let encrypted = key.encrypt(b"test");
        let decrypted = key.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, b"test");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_from_file_raw_bytes() {
        let dir = std::env::temp_dir().join(format!("vdb_crypto_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("key.bin");
        let raw: [u8; 32] = [0x42; 32];
        std::fs::write(&path, &raw).unwrap();

        let key = EncryptionKey::from_file(&path).unwrap();
        let encrypted = key.encrypt(b"test");
        let decrypted = key.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, b"test");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_each_encryption_uses_unique_nonce() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let enc1 = key.encrypt(b"same data");
        let enc2 = key.encrypt(b"same data");
        // Nonces should differ
        assert_ne!(&enc1[..NONCE_SIZE], &enc2[..NONCE_SIZE]);
        // Ciphertexts should also differ (due to different nonces)
        assert_ne!(enc1, enc2);
    }

    #[test]
    fn test_debug_does_not_leak_key() {
        let key = EncryptionKey::from_hex(test_key_hex()).unwrap();
        let debug = format!("{:?}", key);
        assert!(!debug.contains("0123456789"));
        assert!(debug.contains("AES-256-GCM"));
    }
}
