//! Structured audit logging for security-sensitive operations.
//!
//! Provides [`AuditContext`] (inserted by auth middleware into request extensions)
//! and [`audit_event`] for emitting structured audit log entries with `target: "audit"`.
//! Operators can filter/route audit events via `RUST_LOG=audit=info`.

use crate::api::rbac::Role;

/// Identity and request context for audit logging.
///
/// Inserted into request extensions by [`super::auth_middleware`] for every
/// protected route. Handlers extract it via `Option<Extension<AuditContext>>`.
#[derive(Clone, Debug)]
pub struct AuditContext {
    /// Masked API key prefix (first 8 chars + "...") or "anonymous".
    pub key_prefix: String,
    /// RBAC role if resolved, `None` for legacy single-key or no-auth mode.
    pub role: Option<Role>,
    /// Client IP from `X-Forwarded-For` / `X-Real-IP` headers, or "-".
    pub client_ip: String,
}

/// Mask an API key for safe logging: first 8 chars + "...".
pub fn mask_key(token: &str) -> String {
    if token.len() <= 8 {
        "***".to_string()
    } else {
        format!("{}...", &token[..8])
    }
}

/// Extract client IP from request headers (X-Forwarded-For → X-Real-IP → "-").
pub fn extract_client_ip(req: &axum::http::Request<axum::body::Body>) -> String {
    req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(',').next().unwrap_or("-").trim().to_string())
        .or_else(|| {
            req.headers()
                .get("x-real-ip")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "-".to_string())
}

/// Emit a structured audit log entry.
///
/// All audit events use `target: "audit"` so they can be filtered independently
/// from operational logs (e.g. `RUST_LOG=audit=info`).
pub fn audit_event(ctx: &AuditContext, action: &str, resource: &str, detail: &str, outcome: &str) {
    tracing::info!(
        target: "audit",
        actor = %ctx.key_prefix,
        role = ?ctx.role,
        client_ip = %ctx.client_ip,
        action = %action,
        resource = %resource,
        detail = %detail,
        outcome = %outcome,
        "audit"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── mask_key tests ──────────────────────────────────────────────

    #[test]
    fn test_mask_key_long_token() {
        assert_eq!(mask_key("abcdefghijklmnop"), "abcdefgh...");
    }

    #[test]
    fn test_mask_key_exactly_9_chars() {
        assert_eq!(mask_key("123456789"), "12345678...");
    }

    #[test]
    fn test_mask_key_exactly_8_chars_masked() {
        // 8 chars or fewer → fully masked
        assert_eq!(mask_key("12345678"), "***");
    }

    #[test]
    fn test_mask_key_short_token() {
        assert_eq!(mask_key("abc"), "***");
    }

    #[test]
    fn test_mask_key_empty() {
        assert_eq!(mask_key(""), "***");
    }

    #[test]
    fn test_mask_key_never_reveals_full_key() {
        let key = "super-secret-api-key-12345";
        let masked = mask_key(key);
        assert!(!masked.contains("secret"));
        assert!(!masked.contains(key));
        assert!(masked.ends_with("..."));
    }

    // ── extract_client_ip tests ─────────────────────────────────────

    #[test]
    fn test_extract_ip_x_forwarded_for_single() {
        let req = axum::http::Request::builder()
            .header("x-forwarded-for", "192.168.1.1")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "192.168.1.1");
    }

    #[test]
    fn test_extract_ip_x_forwarded_for_chain() {
        // First IP in the chain is the original client
        let req = axum::http::Request::builder()
            .header("x-forwarded-for", "10.0.0.1, 172.16.0.1, 192.168.1.1")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "10.0.0.1");
    }

    #[test]
    fn test_extract_ip_x_forwarded_for_with_spaces() {
        let req = axum::http::Request::builder()
            .header("x-forwarded-for", "  10.0.0.1 , 172.16.0.1")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "10.0.0.1");
    }

    #[test]
    fn test_extract_ip_x_real_ip_fallback() {
        let req = axum::http::Request::builder()
            .header("x-real-ip", "172.16.0.5")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "172.16.0.5");
    }

    #[test]
    fn test_extract_ip_forwarded_for_takes_precedence() {
        let req = axum::http::Request::builder()
            .header("x-forwarded-for", "10.0.0.1")
            .header("x-real-ip", "172.16.0.5")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "10.0.0.1");
    }

    #[test]
    fn test_extract_ip_no_headers_returns_dash() {
        let req = axum::http::Request::builder()
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "-");
    }

    // ── AuditContext tests ──────────────────────────────────────────

    #[test]
    fn test_audit_context_clone() {
        let ctx = AuditContext {
            key_prefix: "abc12345...".to_string(),
            role: Some(Role::Write),
            client_ip: "10.0.0.1".to_string(),
        };
        let ctx2 = ctx.clone();
        assert_eq!(ctx.key_prefix, ctx2.key_prefix);
        assert_eq!(ctx.role, ctx2.role);
        assert_eq!(ctx.client_ip, ctx2.client_ip);
    }

    #[test]
    fn test_audit_context_debug_format() {
        let ctx = AuditContext {
            key_prefix: "anonymous".to_string(),
            role: None,
            client_ip: "-".to_string(),
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("anonymous"));
        assert!(debug.contains("None"));
    }

    // ── audit_event tests ───────────────────────────────────────────

    #[test]
    fn test_audit_event_does_not_panic() {
        let ctx = AuditContext {
            key_prefix: "testkey1...".to_string(),
            role: Some(Role::Admin),
            client_ip: "127.0.0.1".to_string(),
        };
        // Should not panic regardless of subscriber
        audit_event(
            &ctx,
            "create_collection",
            "test_col",
            "dimension=3",
            "success",
        );
    }

    #[test]
    fn test_audit_event_with_anonymous_context() {
        let ctx = AuditContext {
            key_prefix: "anonymous".to_string(),
            role: None,
            client_ip: "-".to_string(),
        };
        audit_event(&ctx, "search", "my_col", "mode=vector,k=10", "success");
    }

    #[test]
    fn test_audit_event_with_empty_detail() {
        let ctx = AuditContext {
            key_prefix: "abc12345...".to_string(),
            role: Some(Role::Write),
            client_ip: "10.0.0.1".to_string(),
        };
        audit_event(&ctx, "delete_collection", "test_col", "", "success");
    }

    #[test]
    fn test_audit_event_captures_correct_fields() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Default)]
        struct CapturedFields {
            action: String,
            actor: String,
            resource: String,
            outcome: String,
            target: String,
        }

        struct AuditCaptureLayer {
            captured: Arc<Mutex<Vec<CapturedFields>>>,
        }

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for AuditCaptureLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                let target = event.metadata().target().to_string();
                if target != "audit" {
                    return;
                }
                struct Visitor(CapturedFields);
                impl tracing::field::Visit for Visitor {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        let val = format!("{:?}", value);
                        match field.name() {
                            "action" => self.0.action = val,
                            "actor" => self.0.actor = val,
                            "resource" => self.0.resource = val,
                            "outcome" => self.0.outcome = val,
                            _ => {}
                        }
                    }
                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        match field.name() {
                            "action" => self.0.action = value.to_string(),
                            "actor" => self.0.actor = value.to_string(),
                            "resource" => self.0.resource = value.to_string(),
                            "outcome" => self.0.outcome = value.to_string(),
                            _ => {}
                        }
                    }
                }
                let mut visitor = Visitor(CapturedFields {
                    target,
                    ..Default::default()
                });
                event.record(&mut visitor);
                self.captured.lock().unwrap().push(visitor.0);
            }
        }

        let captured = Arc::new(Mutex::new(Vec::new()));
        let layer = AuditCaptureLayer {
            captured: captured.clone(),
        };
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            let ctx = AuditContext {
                key_prefix: "mykey123...".to_string(),
                role: Some(Role::Write),
                client_ip: "192.168.1.100".to_string(),
            };
            audit_event(
                &ctx,
                "insert_document",
                "prod_collection",
                "doc_id=abc-123",
                "success",
            );
        });

        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].target, "audit");
        assert_eq!(events[0].action, "insert_document");
        assert_eq!(events[0].actor, "mykey123...");
        assert_eq!(events[0].resource, "prod_collection");
        assert_eq!(events[0].outcome, "success");
    }

    #[test]
    fn test_audit_event_only_captures_audit_target() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt;

        struct CountLayer {
            count: Arc<Mutex<usize>>,
        }

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CountLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event.metadata().target() == "audit" {
                    *self.count.lock().unwrap() += 1;
                }
            }
        }

        let count = Arc::new(Mutex::new(0usize));
        let layer = CountLayer {
            count: count.clone(),
        };
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            // Non-audit log — should not be counted
            tracing::info!("this is a regular log");

            // Audit log — should be counted
            let ctx = AuditContext {
                key_prefix: "test...".to_string(),
                role: None,
                client_ip: "-".to_string(),
            };
            audit_event(&ctx, "test_action", "test_resource", "", "success");
        });

        assert_eq!(*count.lock().unwrap(), 1);
    }
}
