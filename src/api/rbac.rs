//! Role-Based Access Control (RBAC) for API authentication.
//!
//! Provides a three-tier permission model (`Read` < `Write` < `Admin`) where
//! each API key maps to a role. The [`required_role`](crate::api::rbac::required_role) function determines the
//! minimum role needed for a given HTTP method and path.

use serde::Deserialize;
use std::collections::HashMap;

/// Permission level for an API key.
///
/// Roles are ordered: `Read` < `Write` < `Admin`. A key with `Write` access
/// can perform all `Read` operations, and `Admin` can perform all operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Read-only access: GET requests and search queries.
    Read,
    /// Read + write access: document insertion, update, deletion.
    Write,
    /// Full access including admin endpoints (backup, rebuild, routing).
    Admin,
}

/// A single API key entry from the RBAC configuration file.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeyEntry {
    /// The bearer token string.
    pub key: String,
    /// The permission level assigned to this key.
    pub role: Role,
    /// Optional per-key rate limit in requests per second. `None` uses the global limit.
    #[serde(default)]
    pub rate_limit_rps: Option<u64>,
}

/// Resolved RBAC configuration mapping API keys to roles and per-key rate limits.
#[derive(Debug, Clone)]
pub struct RbacConfig {
    /// Map from bearer token to its assigned role.
    pub keys: HashMap<String, Role>,
    /// Map from bearer token to its per-key rate limit (requests per second).
    pub rate_limits: HashMap<String, u64>,
}

impl RbacConfig {
    /// Builds an `RbacConfig` from a list of key entries.
    pub fn from_entries(entries: Vec<ApiKeyEntry>) -> Self {
        let mut keys = HashMap::new();
        let mut rate_limits = HashMap::new();
        for e in entries {
            keys.insert(e.key.clone(), e.role);
            if let Some(rps) = e.rate_limit_rps {
                rate_limits.insert(e.key, rps);
            }
        }
        Self { keys, rate_limits }
    }

    /// Looks up the role for a bearer token. Returns `None` if the key is unknown.
    pub fn get_role(&self, token: &str) -> Option<Role> {
        self.keys.get(token).copied()
    }

    /// Returns the per-key rate limit (requests per second) if configured.
    pub fn get_rate_limit(&self, token: &str) -> Option<u64> {
        self.rate_limits.get(token).copied()
    }
}

/// Determine the minimum role required for a given method + path.
pub fn required_role(method: &str, path: &str) -> Role {
    // Admin endpoints
    if path.starts_with("/admin/") {
        return Role::Admin;
    }

    match method {
        "GET" => Role::Read,
        "POST" => {
            // Search is read-level
            if path.ends_with("/search") {
                Role::Read
            } else {
                Role::Write
            }
        }
        "PUT" | "DELETE" => Role::Write,
        _ => Role::Admin,
    }
}
