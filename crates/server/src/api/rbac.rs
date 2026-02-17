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
    /// Optional collection-level access control. If set, this key can only access
    /// collections matching these patterns. Supports exact names (`"my_col"`) and
    /// prefix globs (`"tenant_*"`). `None` means access to all collections.
    #[serde(default)]
    pub collections: Option<Vec<String>>,
}

/// Resolved RBAC configuration mapping API keys to roles and per-key rate limits.
#[derive(Debug, Clone)]
pub struct RbacConfig {
    /// Map from bearer token to its assigned role.
    pub keys: HashMap<String, Role>,
    /// Map from bearer token to its per-key rate limit (requests per second).
    pub rate_limits: HashMap<String, u64>,
    /// Map from bearer token to allowed collection patterns.
    pub collections: HashMap<String, Vec<String>>,
}

impl RbacConfig {
    /// Builds an `RbacConfig` from a list of key entries.
    pub fn from_entries(entries: Vec<ApiKeyEntry>) -> Self {
        let mut keys = HashMap::new();
        let mut rate_limits = HashMap::new();
        let mut collections = HashMap::new();
        for e in entries {
            keys.insert(e.key.clone(), e.role);
            if let Some(rps) = e.rate_limit_rps {
                rate_limits.insert(e.key.clone(), rps);
            }
            if let Some(cols) = e.collections {
                collections.insert(e.key, cols);
            }
        }
        Self {
            keys,
            rate_limits,
            collections,
        }
    }

    /// Looks up the role for a bearer token. Returns `None` if the key is unknown.
    pub fn get_role(&self, token: &str) -> Option<Role> {
        self.keys.get(token).copied()
    }

    /// Returns the per-key rate limit (requests per second) if configured.
    pub fn get_rate_limit(&self, token: &str) -> Option<u64> {
        self.rate_limits.get(token).copied()
    }

    /// Returns the collection patterns for a token, if collection-scoped.
    pub fn get_collections(&self, token: &str) -> Option<&Vec<String>> {
        self.collections.get(token)
    }
}

/// Check if a collection name matches any of the allowed patterns.
///
/// Supports exact matches (`"my_col"`) and prefix globs (`"tenant_*"`).
pub fn collection_matches(patterns: &[String], name: &str) -> bool {
    patterns.iter().any(|p| {
        if let Some(prefix) = p.strip_suffix('*') {
            name.starts_with(prefix)
        } else {
            p == name
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_matches_exact() {
        let patterns = vec!["public".to_string(), "shared".to_string()];
        assert!(collection_matches(&patterns, "public"));
        assert!(collection_matches(&patterns, "shared"));
        assert!(!collection_matches(&patterns, "private"));
        assert!(!collection_matches(&patterns, "public_extra"));
    }

    #[test]
    fn test_collection_matches_glob_prefix() {
        let patterns = vec!["tenant_*".to_string()];
        assert!(collection_matches(&patterns, "tenant_a"));
        assert!(collection_matches(&patterns, "tenant_abc"));
        assert!(collection_matches(&patterns, "tenant_"));
        assert!(!collection_matches(&patterns, "other"));
        assert!(!collection_matches(&patterns, "tenan"));
    }

    #[test]
    fn test_collection_matches_wildcard_all() {
        let patterns = vec!["*".to_string()];
        assert!(collection_matches(&patterns, "anything"));
        assert!(collection_matches(&patterns, ""));
    }

    #[test]
    fn test_collection_matches_mixed_patterns() {
        let patterns = vec!["exact_col".to_string(), "prefix_*".to_string()];
        assert!(collection_matches(&patterns, "exact_col"));
        assert!(collection_matches(&patterns, "prefix_foo"));
        assert!(!collection_matches(&patterns, "other"));
    }

    #[test]
    fn test_collection_matches_empty_patterns() {
        let patterns: Vec<String> = vec![];
        assert!(!collection_matches(&patterns, "anything"));
    }

    #[test]
    fn test_rbac_config_collections_field() {
        let config = RbacConfig::from_entries(vec![
            ApiKeyEntry {
                key: "scoped".to_string(),
                role: Role::Write,
                rate_limit_rps: None,
                collections: Some(vec!["a_*".to_string()]),
            },
            ApiKeyEntry {
                key: "unscoped".to_string(),
                role: Role::Write,
                rate_limit_rps: None,
                collections: None,
            },
        ]);
        assert_eq!(
            config.get_collections("scoped"),
            Some(&vec!["a_*".to_string()])
        );
        assert_eq!(config.get_collections("unscoped"), None);
        assert_eq!(config.get_collections("unknown"), None);
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
