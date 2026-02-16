//! HTTP-based Raft RPC network transport.
//!
//! Implements openraft's `RaftNetworkFactory` and `RaftNetwork` traits using
//! reqwest HTTP client to send `AppendEntries`, `InstallSnapshot`, and `Vote`
//! RPCs to peer nodes.

use crate::cluster::types::{NodeId, TypeConfig};
use openraft::error::RaftError;
use openraft::error::{NetworkError, Unreachable};
use openraft::network::{RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use openraft::BasicNode;

/// Type alias for Raft RPC errors with default infallible application error.
pub type RPCErr<E = openraft::error::Infallible> =
    openraft::error::RPCError<NodeId, BasicNode, RaftError<NodeId, E>>;

/// Factory that creates HTTP network connections to Raft peers.
#[derive(Default)]
pub struct NetworkFactory {
    client: reqwest::Client,
}

impl NetworkFactory {
    /// Creates a new factory with a default reqwest client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

/// An HTTP connection to a single Raft peer node.
pub struct NetworkConnection {
    target_addr: String,
    client: reqwest::Client,
}

impl RaftNetworkFactory<TypeConfig> for NetworkFactory {
    type Network = NetworkConnection;

    async fn new_client(&mut self, _target: NodeId, node: &BasicNode) -> Self::Network {
        NetworkConnection {
            target_addr: node.addr.clone(),
            client: self.client.clone(),
        }
    }
}

impl RaftNetwork<TypeConfig> for NetworkConnection {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<AppendEntriesResponse<NodeId>, RPCErr> {
        let url = format!("http://{}/raft/append", self.target_addr);
        let resp = self
            .client
            .post(&url)
            .json(&rpc)
            .send()
            .await
            .map_err(|e| openraft::error::RPCError::Unreachable(Unreachable::new(&e)))?;

        let result: AppendEntriesResponse<NodeId> = resp
            .json()
            .await
            .map_err(|e| openraft::error::RPCError::Network(NetworkError::new(&e)))?;

        Ok(result)
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<InstallSnapshotResponse<NodeId>, RPCErr<openraft::error::InstallSnapshotError>>
    {
        let url = format!("http://{}/raft/snapshot", self.target_addr);
        let resp = self
            .client
            .post(&url)
            .json(&rpc)
            .send()
            .await
            .map_err(|e| openraft::error::RPCError::Unreachable(Unreachable::new(&e)))?;

        let result: InstallSnapshotResponse<NodeId> = resp
            .json()
            .await
            .map_err(|e| openraft::error::RPCError::Network(NetworkError::new(&e)))?;

        Ok(result)
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        _option: RPCOption,
    ) -> Result<VoteResponse<NodeId>, RPCErr> {
        let url = format!("http://{}/raft/vote", self.target_addr);
        let resp = self
            .client
            .post(&url)
            .json(&rpc)
            .send()
            .await
            .map_err(|e| openraft::error::RPCError::Unreachable(Unreachable::new(&e)))?;

        let result: VoteResponse<NodeId> = resp
            .json()
            .await
            .map_err(|e| openraft::error::RPCError::Network(NetworkError::new(&e)))?;

        Ok(result)
    }
}
