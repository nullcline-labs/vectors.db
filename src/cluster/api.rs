//! Axum routes for Raft protocol RPCs and cluster management.
//!
//! Provides HTTP endpoints consumed by the Raft network layer for
//! vote requests, log replication, and snapshot transfer, plus management
//! endpoints for cluster initialization and membership changes.

use crate::cluster::types::{NodeId, Raft, TypeConfig};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use openraft::BasicNode;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Shared state for Raft API handlers, wrapping the Raft instance.
#[derive(Clone)]
pub struct RaftState {
    /// The openraft `Raft` instance for this node.
    pub raft: Arc<Raft>,
}

/// Builds the Axum router for Raft protocol and cluster management endpoints.
///
/// Routes:
/// - `POST /raft/vote` — Raft leader election vote RPC
/// - `POST /raft/append` — Raft log replication append entries RPC
/// - `POST /raft/snapshot` — Raft snapshot installation RPC
/// - `POST /raft/init` — Initialize a new Raft cluster
/// - `POST /raft/add-learner` — Add a learner node to the cluster
/// - `POST /raft/change-membership` — Change voter membership set
pub fn raft_router(state: RaftState) -> Router {
    Router::new()
        .route("/raft/vote", post(handle_vote))
        .route("/raft/append", post(handle_append))
        .route("/raft/snapshot", post(handle_snapshot))
        .route("/raft/init", post(handle_init))
        .route("/raft/add-learner", post(handle_add_learner))
        .route("/raft/change-membership", post(handle_change_membership))
        .with_state(state)
}

/// Error wrapper for Raft RPC handlers that returns a JSON body with 500 status.
struct RaftApiError(String);

impl IntoResponse for RaftApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": self.0})),
        )
            .into_response()
    }
}

async fn handle_vote(
    State(state): State<RaftState>,
    Json(req): Json<VoteRequest<NodeId>>,
) -> Result<Json<VoteResponse<NodeId>>, RaftApiError> {
    let resp = state
        .raft
        .vote(req)
        .await
        .map_err(|e| RaftApiError(e.to_string()))?;
    Ok(Json(resp))
}

async fn handle_append(
    State(state): State<RaftState>,
    Json(req): Json<AppendEntriesRequest<TypeConfig>>,
) -> Result<Json<AppendEntriesResponse<NodeId>>, RaftApiError> {
    let resp = state
        .raft
        .append_entries(req)
        .await
        .map_err(|e| RaftApiError(e.to_string()))?;
    Ok(Json(resp))
}

async fn handle_snapshot(
    State(state): State<RaftState>,
    Json(req): Json<InstallSnapshotRequest<TypeConfig>>,
) -> Result<Json<InstallSnapshotResponse<NodeId>>, RaftApiError> {
    let resp = state
        .raft
        .install_snapshot(req)
        .await
        .map_err(|e| RaftApiError(e.to_string()))?;
    Ok(Json(resp))
}

#[derive(Debug, Deserialize)]
struct InitRequest {
    members: BTreeMap<NodeId, String>,
}

async fn handle_init(
    State(state): State<RaftState>,
    Json(req): Json<InitRequest>,
) -> Json<serde_json::Value> {
    let members: BTreeMap<NodeId, BasicNode> = req
        .members
        .into_iter()
        .map(|(id, addr)| (id, BasicNode::new(addr)))
        .collect();
    match state.raft.initialize(members).await {
        Ok(_) => Json(serde_json::json!({"message": "Cluster initialized"})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

#[derive(Debug, Deserialize)]
struct AddLearnerRequest {
    node_id: NodeId,
    addr: String,
}

async fn handle_add_learner(
    State(state): State<RaftState>,
    Json(req): Json<AddLearnerRequest>,
) -> Json<serde_json::Value> {
    match state
        .raft
        .add_learner(req.node_id, BasicNode::new(req.addr), true)
        .await
    {
        Ok(resp) => Json(
            serde_json::json!({"message": "Learner added", "log_id": format!("{}", resp.log_id)}),
        ),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

#[derive(Debug, Deserialize)]
struct ChangeMembershipRequest {
    members: BTreeSet<NodeId>,
}

async fn handle_change_membership(
    State(state): State<RaftState>,
    Json(req): Json<ChangeMembershipRequest>,
) -> Json<serde_json::Value> {
    match state.raft.change_membership(req.members, false).await {
        Ok(resp) => Json(
            serde_json::json!({"message": "Membership changed", "log_id": format!("{}", resp.log_id)}),
        ),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}
