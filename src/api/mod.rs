//! REST API layer built on Axum.
//!
//! Provides HTTP handlers for collection management, document CRUD, search,
//! and admin operations. Includes middleware for authentication, RBAC, rate limiting,
//! request timeouts, body size limits, metrics collection, and request ID tracing.

/// API error types mapped to HTTP status codes.
pub mod errors;
/// HTTP request handlers and application state.
pub mod handlers;
/// Prometheus metrics recording and background collection.
pub mod metrics;
/// Request and response data transfer objects.
pub mod models;
/// Per-key token bucket rate limiter.
pub mod rate_limit;
/// Role-Based Access Control (RBAC) with Read/Write/Admin roles.
pub mod rbac;

use axum::error_handling::HandleErrorLayer;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{middleware, Router};
use errors::ApiError;
use handlers::AppState;
use std::time::{Duration, Instant};
use tower::buffer::BufferLayer;
use tower::limit::{ConcurrencyLimitLayer, RateLimitLayer};
use tower::timeout::TimeoutLayer;
use tower::ServiceBuilder;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

use crate::config;

async fn auth_middleware(
    State(state): State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, ApiError> {
    let token = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string());

    // RBAC mode
    if let Some(ref rbac) = state.rbac {
        let token = token
            .ok_or_else(|| ApiError::Unauthorized("Invalid or missing API key".to_string()))?;
        let role = rbac
            .get_role(&token)
            .ok_or_else(|| ApiError::Unauthorized("Invalid or missing API key".to_string()))?;
        let method = req.method().as_str().to_string();
        let path = req.uri().path().to_string();
        let required = rbac::required_role(&method, &path);
        if role < required {
            return Err(ApiError::Forbidden(format!(
                "Role '{:?}' insufficient, need '{:?}'",
                role, required
            )));
        }

        // Per-key rate limiting
        if let Some(rps) = rbac.get_rate_limit(&token) {
            let mut limiters = state.key_rate_limiters.lock();
            let bucket = limiters
                .entry(token)
                .or_insert_with(|| rate_limit::TokenBucket::new(rps));
            if !bucket.try_acquire() {
                return Err(ApiError::TooManyRequests(
                    "Per-key rate limit exceeded".into(),
                ));
            }
        }

        return Ok(next.run(req).await);
    }

    // Legacy single-key mode
    if let Some(ref expected_key) = state.api_key {
        use subtle::ConstantTimeEq;
        let authorized = token
            .as_deref()
            .map(|t| t.as_bytes().ct_eq(expected_key.as_bytes()).into())
            .unwrap_or(false);

        if !authorized {
            return Err(ApiError::Unauthorized(
                "Invalid or missing API key".to_string(),
            ));
        }
    }
    Ok(next.run(req).await)
}

async fn request_id_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!("request", request_id = %request_id);
    async move {
        let mut response = next.run(req).await;
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("x-request-id"),
            axum::http::HeaderValue::from_str(&request_id)
                .expect("UUID v4 is always valid ASCII for header values"),
        );
        response
    }
    .instrument(span)
    .await
}

async fn security_headers_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::HeaderName::from_static("x-content-type-options"),
        axum::http::HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("x-frame-options"),
        axum::http::HeaderValue::from_static("DENY"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("referrer-policy"),
        axum::http::HeaderValue::from_static("no-referrer"),
    );
    response
}

async fn metrics_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let start = Instant::now();
    let response = next.run(req).await;
    metrics::record_request(&method, &path, response.status().as_u16(), start.elapsed());
    response
}

use axum::extract::State;

/// Builds the Axum router with all routes and middleware layers.
///
/// The middleware stack (outermost to innermost):
/// Rate limiting → Concurrency limit → Timeout → Body limit → CORS → Compression →
/// Trace → Security headers → Request ID → Metrics → Auth.
pub fn create_router(state: AppState) -> Router {
    let protected = Router::new()
        .route(
            "/collections",
            get(handlers::list_collections).post(handlers::create_collection),
        )
        .route("/collections/:name", delete(handlers::delete_collection))
        .route(
            "/collections/:name/documents",
            post(handlers::insert_document),
        )
        .route(
            "/collections/:name/documents/batch",
            post(handlers::batch_insert_documents),
        )
        .route(
            "/collections/:name/documents/:id",
            get(handlers::get_document)
                .put(handlers::update_document)
                .delete(handlers::delete_document),
        )
        .route("/collections/:name/search", post(handlers::search))
        .route("/collections/:name/save", post(handlers::save))
        .route("/collections/:name/load", post(handlers::load))
        .route("/collections/:name/stats", get(handlers::collection_stats))
        .route(
            "/collections/:name/documents/count",
            get(handlers::document_count),
        )
        .route("/collections/:name/clear", post(handlers::clear_collection))
        .route("/admin/compact", post(handlers::compact))
        .route("/admin/rebuild/:name", post(handlers::rebuild_index))
        .route("/admin/backup", post(handlers::backup))
        .route("/admin/restore", post(handlers::restore_all))
        .route("/admin/routing", get(handlers::routing_table))
        .route("/admin/assign", post(handlers::assign_collection))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(handlers::health))
        .route("/metrics", get(handlers::metrics_endpoint))
        .merge(protected)
        .layer(middleware::from_fn(metrics_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(middleware::from_fn(security_headers_middleware))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(config::MAX_REQUEST_BODY_BYTES))
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|err: tower::BoxError| async move {
                    if err.is::<tower::timeout::error::Elapsed>() {
                        StatusCode::REQUEST_TIMEOUT
                    } else {
                        StatusCode::TOO_MANY_REQUESTS
                    }
                }))
                .layer(BufferLayer::new(1024))
                .layer(ConcurrencyLimitLayer::new(config::MAX_CONCURRENT_REQUESTS))
                .layer(RateLimitLayer::new(
                    config::RATE_LIMIT_RPS,
                    Duration::from_secs(1),
                ))
                .layer(TimeoutLayer::new(Duration::from_secs(
                    config::REQUEST_TIMEOUT_SECS,
                ))),
        )
        .with_state(state)
}
