//! API error types mapped to HTTP status codes.
//!
//! Each [`ApiError`] variant maps to a specific HTTP status code and produces
//! a JSON response body `{"error": "message"}`.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Application-level error type that implements `IntoResponse`.
///
/// Each variant maps to an HTTP status code:
/// - `NotFound` → 404
/// - `BadRequest` → 400
/// - `Unauthorized` → 401
/// - `Forbidden` → 403
/// - `Conflict` → 409
/// - `InsufficientStorage` → 507
/// - `TooManyRequests` → 429
/// - `ServiceUnavailable` → 503
/// - `Redirect` → 307 (with `Location` header)
/// - `Internal` → 500
#[derive(Debug)]
pub enum ApiError {
    /// Resource not found (404).
    NotFound(String),
    /// Invalid request parameters (400).
    BadRequest(String),
    /// Missing or invalid authentication (401).
    Unauthorized(String),
    /// Insufficient permissions (403).
    Forbidden(String),
    /// Per-key rate limit exceeded (429).
    TooManyRequests(String),
    /// Resource already exists (409).
    Conflict(String),
    /// Memory limit exceeded (507).
    InsufficientStorage(String),
    /// Service unavailable — standby node is read-only (503).
    ServiceUnavailable(String),
    /// Redirect to another node (307 with `Location` header).
    Redirect(String),
    /// Unexpected server error (500).
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Redirect(location) => match axum::http::HeaderValue::from_str(&location) {
                Ok(val) => {
                    let mut resp = (
                        StatusCode::TEMPORARY_REDIRECT,
                        axum::Json(json!({ "redirect": location })),
                    )
                        .into_response();
                    resp.headers_mut().insert(axum::http::header::LOCATION, val);
                    resp
                }
                Err(_) => {
                    let body = axum::Json(json!({ "error": "Invalid redirect location" }));
                    (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
                }
            },
            other => {
                let (status, message) = match other {
                    ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
                    ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
                    ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
                    ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, msg),
                    ApiError::TooManyRequests(msg) => (StatusCode::TOO_MANY_REQUESTS, msg),
                    ApiError::Conflict(msg) => (StatusCode::CONFLICT, msg),
                    ApiError::InsufficientStorage(msg) => (StatusCode::INSUFFICIENT_STORAGE, msg),
                    ApiError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
                    ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
                    ApiError::Redirect(_) => unreachable!(),
                };
                let body = axum::Json(json!({ "error": message }));
                (status, body).into_response()
            }
        }
    }
}
