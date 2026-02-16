use reqwest::Client;
use std::sync::Arc;
use tempfile::TempDir;
use vectorsdb_core::storage::Database;
use vectorsdb_server::api::create_router;
use vectorsdb_server::api::handlers::AppState;
use vectorsdb_server::api::rbac::{ApiKeyEntry, RbacConfig, Role};
use vectorsdb_server::wal_async::WriteAheadLog;

async fn spawn_app_no_recorder(api_key: Option<&str>) -> (String, TempDir) {
    spawn_app_full(api_key, None, 0).await
}

async fn spawn_app_full(
    api_key: Option<&str>,
    rbac: Option<RbacConfig>,
    max_memory_bytes: usize,
) -> (String, TempDir) {
    let tmp_dir = TempDir::new().expect("Failed to create temp dir");
    let data_dir = tmp_dir.path().to_str().unwrap().to_string();

    let db = Database::new();
    let wal = Arc::new(WriteAheadLog::new(&data_dir).expect("Failed to create WAL"));

    let prometheus_handle =
        match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
            Ok(handle) => handle,
            Err(_) => metrics_exporter_prometheus::PrometheusBuilder::new()
                .build_recorder()
                .handle(),
        };

    let wal_path = std::path::PathBuf::from(&data_dir).join("wal.bin");
    let state = AppState {
        db,
        data_dir,
        wal,
        wal_path,
        api_key: api_key.map(|s| s.to_string()),
        prometheus_handle,
        max_memory_bytes,
        rbac,
        raft: None,
        node_id: None,
        routing_table: None,
        peer_addrs: None,
        start_time: std::time::Instant::now(),
        key_rate_limiters: std::sync::Arc::new(parking_lot::Mutex::new(
            std::collections::HashMap::new(),
        )),
        memory_reserved: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind");
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (base_url, tmp_dir)
}

fn client() -> Client {
    Client::new()
}

async fn create_test_collection(base_url: &str, name: &str, dimension: usize) {
    client()
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": name,
            "dimension": dimension
        }))
        .send()
        .await
        .expect("Failed to create collection");
}

async fn insert_test_document(
    base_url: &str,
    collection: &str,
    text: &str,
    embedding: Vec<f32>,
) -> reqwest::Response {
    client()
        .post(format!("{}/collections/{}/documents", base_url, collection))
        .json(&serde_json::json!({
            "text": text,
            "embedding": embedding
        }))
        .send()
        .await
        .expect("Failed to insert document")
}

async fn insert_test_document_with_metadata(
    base_url: &str,
    collection: &str,
    text: &str,
    embedding: Vec<f32>,
    metadata: serde_json::Value,
) -> reqwest::Response {
    client()
        .post(format!("{}/collections/{}/documents", base_url, collection))
        .json(&serde_json::json!({
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }))
        .send()
        .await
        .expect("Failed to insert document")
}

// ========== Existing Tests ==========

#[tokio::test]
async fn health_returns_ok() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn create_and_list_collections() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "test_col",
            "dimension": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(body.iter().any(|c| c["name"] == "test_col"));
}

#[tokio::test]
async fn create_duplicate_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "dup_col", 3).await;

    let resp = client()
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "dup_col",
            "dimension": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn delete_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "del_col", 3).await;

    let resp = client()
        .delete(format!("{}/collections/del_col", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .unwrap();
    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!body.iter().any(|c| c["name"] == "del_col"));
}

#[tokio::test]
async fn delete_nonexistent_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .delete(format!("{}/collections/no_such", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn insert_and_get_document() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "docs_col", 3).await;

    let resp =
        insert_test_document(&base_url, "docs_col", "hello world", vec![1.0, 2.0, 3.0]).await;
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap();

    let resp = client()
        .get(format!(
            "{}/collections/docs_col/documents/{}",
            base_url, id
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["text"], "hello world");
}

#[tokio::test]
async fn insert_invalid_dimension() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "dim_col", 3).await;

    let resp =
        insert_test_document(&base_url, "dim_col", "test", vec![1.0, 2.0, 3.0, 4.0, 5.0]).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn insert_nan_embedding() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "nan_col", 3).await;

    let resp = insert_test_document(&base_url, "nan_col", "test", vec![1.0, f32::NAN, 3.0]).await;
    assert_eq!(resp.status(), 422);
}

#[tokio::test]
async fn insert_empty_text() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "empty_col", 3).await;

    let resp = insert_test_document(&base_url, "empty_col", "", vec![1.0, 2.0, 3.0]).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn search_vector() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "search_col", 3).await;

    insert_test_document(&base_url, "search_col", "doc1", vec![1.0, 0.0, 0.0]).await;
    insert_test_document(&base_url, "search_col", "doc2", vec![0.0, 1.0, 0.0]).await;
    insert_test_document(&base_url, "search_col", "doc3", vec![0.9, 0.1, 0.0]).await;

    let resp = client()
        .post(format!("{}/collections/search_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["score"].as_f64().unwrap())
        .collect();
    for i in 0..scores.len() - 1 {
        assert!(
            scores[i] >= scores[i + 1],
            "Results not sorted by score: {:?}",
            scores
        );
    }
}

#[tokio::test]
async fn search_k_zero() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "kzero_col", 3).await;

    let resp = client()
        .post(format!("{}/collections/kzero_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn batch_insert() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "batch_col", 3).await;

    let documents: Vec<serde_json::Value> = (0..5)
        .map(|i| {
            serde_json::json!({
                "text": format!("doc {}", i),
                "embedding": [i as f32, 0.0, 0.0]
            })
        })
        .collect();

    let resp = client()
        .post(format!(
            "{}/collections/batch_col/documents/batch",
            base_url
        ))
        .json(&serde_json::json!({ "documents": documents }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    let ids = body["ids"].as_array().unwrap();
    assert_eq!(ids.len(), 5);
    assert_eq!(body["inserted"], 5);
}

#[tokio::test]
async fn batch_too_large() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "biglot_col", 3).await;

    let documents: Vec<serde_json::Value> = (0..1001)
        .map(|i| {
            serde_json::json!({
                "text": format!("doc {}", i),
                "embedding": [i as f32, 0.0, 0.0]
            })
        })
        .collect();

    let resp = client()
        .post(format!(
            "{}/collections/biglot_col/documents/batch",
            base_url
        ))
        .json(&serde_json::json!({ "documents": documents }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn auth_required() {
    let (base_url, _tmp) = spawn_app_no_recorder(Some("secret-key")).await;

    let resp = client()
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn auth_valid_key() {
    let (base_url, _tmp) = spawn_app_no_recorder(Some("secret-key")).await;

    let resp = client()
        .get(format!("{}/collections", base_url))
        .header("Authorization", "Bearer secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn health_no_auth() {
    let (base_url, _tmp) = spawn_app_no_recorder(Some("secret-key")).await;

    let resp = client()
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn collection_name_invalid_chars() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "a b c",
            "dimension": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn create_collection_dimension_zero() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "zero_dim",
            "dimension": 0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_request_id_header() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let request_id = resp
        .headers()
        .get("x-request-id")
        .expect("Missing X-Request-Id header")
        .to_str()
        .unwrap();

    uuid::Uuid::parse_str(request_id).expect("X-Request-Id is not a valid UUID");
}

#[tokio::test]
async fn test_pagination_offset() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "page_col", 3).await;

    for i in 0..5 {
        insert_test_document(
            &base_url,
            "page_col",
            &format!("doc {}", i),
            vec![i as f32, 0.0, 0.0],
        )
        .await;
    }

    let resp = client()
        .post(format!("{}/collections/page_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [5.0, 0.0, 0.0],
            "k": 2,
            "offset": 0
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 2);

    let resp = client()
        .post(format!("{}/collections/page_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [5.0, 0.0, 0.0],
            "k": 2,
            "offset": 2
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 2);
}

#[tokio::test]
async fn test_pagination_total() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "ptotal_col", 3).await;

    for i in 0..3 {
        insert_test_document(
            &base_url,
            "ptotal_col",
            &format!("doc {}", i),
            vec![i as f32, 0.0, 0.0],
        )
        .await;
    }

    let resp = client()
        .post(format!("{}/collections/ptotal_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 2
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["total"].as_u64().unwrap() >= 2);
    assert_eq!(body["count"], 2);
}

#[tokio::test]
async fn test_pagination_beyond_results() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "pbeyond_col", 3).await;

    insert_test_document(&base_url, "pbeyond_col", "only doc", vec![1.0, 0.0, 0.0]).await;

    let resp = client()
        .post(format!("{}/collections/pbeyond_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "offset": 100
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 0);
    assert_eq!(body["results"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_filter_eq_string() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "feq_col", 3).await;

    insert_test_document_with_metadata(
        &base_url,
        "feq_col",
        "red doc",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"color": "red"}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "feq_col",
        "blue doc",
        vec![0.9, 0.1, 0.0],
        serde_json::json!({"color": "blue"}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/feq_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "filter": { "must": [{"field": "color", "op": "eq", "value": "red"}] }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["text"], "red doc");
}

#[tokio::test]
async fn test_filter_gt_integer() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "fgt_col", 3).await;

    insert_test_document_with_metadata(
        &base_url,
        "fgt_col",
        "low",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"score": 10}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "fgt_col",
        "high",
        vec![0.9, 0.1, 0.0],
        serde_json::json!({"score": 90}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/fgt_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "filter": { "must": [{"field": "score", "op": "gt", "value": 50}] }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["text"], "high");
}

#[tokio::test]
async fn test_filter_in_operator() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "fin_col", 3).await;

    insert_test_document_with_metadata(
        &base_url,
        "fin_col",
        "cat",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"animal": "cat"}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "fin_col",
        "dog",
        vec![0.9, 0.1, 0.0],
        serde_json::json!({"animal": "dog"}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "fin_col",
        "fish",
        vec![0.8, 0.2, 0.0],
        serde_json::json!({"animal": "fish"}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/fin_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "filter": { "must": [{"field": "animal", "op": "in", "values": ["cat", "dog"]}] }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["results"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_filter_must_not() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "fmn_col", 3).await;

    insert_test_document_with_metadata(
        &base_url,
        "fmn_col",
        "active",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"status": "active"}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "fmn_col",
        "archived",
        vec![0.9, 0.1, 0.0],
        serde_json::json!({"status": "archived"}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/fmn_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "filter": { "must_not": [{"field": "status", "op": "eq", "value": "archived"}] }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["text"], "active");
}

#[tokio::test]
async fn test_filter_no_match() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "fnm_col", 3).await;

    insert_test_document_with_metadata(
        &base_url,
        "fnm_col",
        "doc1",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"type": "a"}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/fnm_col/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 10,
            "filter": { "must": [{"field": "type", "op": "eq", "value": "nonexistent"}] }
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["results"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_update_document() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "upd_col", 3).await;

    let resp =
        insert_test_document(&base_url, "upd_col", "original text", vec![1.0, 0.0, 0.0]).await;
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap().to_string();

    let resp = client()
        .put(format!("{}/collections/upd_col/documents/{}", base_url, id))
        .json(&serde_json::json!({"text": "updated text"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{}/collections/upd_col/documents/{}", base_url, id))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["text"], "updated text");
}

#[tokio::test]
async fn test_update_nonexistent() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "updne_col", 3).await;

    let fake_id = uuid::Uuid::new_v4();
    let resp = client()
        .put(format!(
            "{}/collections/updne_col/documents/{}",
            base_url, fake_id
        ))
        .json(&serde_json::json!({"text": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_update_dimension_mismatch() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "updim_col", 3).await;

    let resp = insert_test_document(&base_url, "updim_col", "test", vec![1.0, 0.0, 0.0]).await;
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap().to_string();

    let resp = client()
        .put(format!(
            "{}/collections/updim_col/documents/{}",
            base_url, id
        ))
        .json(&serde_json::json!({"embedding": [1.0, 2.0]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_collection_stats() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "stats_col", 3).await;

    insert_test_document(&base_url, "stats_col", "doc1", vec![1.0, 0.0, 0.0]).await;
    insert_test_document(&base_url, "stats_col", "doc2", vec![0.0, 1.0, 0.0]).await;

    let resp = client()
        .get(format!("{}/collections/stats_col/stats", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "stats_col");
    assert_eq!(body["document_count"], 2);
    assert_eq!(body["dimension"], 3);
    assert!(body["estimated_memory_bytes"].as_u64().unwrap() > 0);
    assert_eq!(body["deleted_count"], 0);
}

#[tokio::test]
async fn test_memory_limit_rejects_insert() {
    let (base_url, _tmp) = spawn_app_full(None, None, 1).await;
    create_test_collection(&base_url, "memlim_col", 3).await;

    insert_test_document(&base_url, "memlim_col", "doc1", vec![1.0, 0.0, 0.0]).await;

    let resp = insert_test_document(&base_url, "memlim_col", "doc2", vec![0.0, 1.0, 0.0]).await;
    assert_eq!(resp.status(), 507);
}

#[tokio::test]
async fn test_stats_deleted_count() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "sdel_col", 3).await;

    let resp = insert_test_document(&base_url, "sdel_col", "to delete", vec![1.0, 0.0, 0.0]).await;
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap();

    client()
        .delete(format!(
            "{}/collections/sdel_col/documents/{}",
            base_url, id
        ))
        .send()
        .await
        .unwrap();

    let resp = client()
        .get(format!("{}/collections/sdel_col/stats", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["deleted_count"], 1);
}

#[tokio::test]
async fn test_rebuild_index() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "reb_col", 3).await;

    let resp = insert_test_document(&base_url, "reb_col", "to delete", vec![1.0, 0.0, 0.0]).await;
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap();

    insert_test_document(&base_url, "reb_col", "keeper", vec![0.0, 1.0, 0.0]).await;

    client()
        .delete(format!("{}/collections/reb_col/documents/{}", base_url, id))
        .send()
        .await
        .unwrap();

    let resp = client()
        .post(format!("{}/admin/rebuild/reb_col", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["document_count"], 1);

    let resp = client()
        .get(format!("{}/collections/reb_col/stats", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["deleted_count"], 0);
}

#[tokio::test]
async fn test_rebuild_empty_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "rebempty_col", 3).await;

    let resp = client()
        .post(format!("{}/admin/rebuild/rebempty_col", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["document_count"], 0);
}

#[tokio::test]
async fn test_backup_and_restore() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "bkp_col", 3).await;
    insert_test_document(&base_url, "bkp_col", "backup me", vec![1.0, 0.0, 0.0]).await;

    let resp = client()
        .post(format!("{}/admin/backup", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["files"].as_array().unwrap().len() > 0);

    client()
        .delete(format!("{}/collections/bkp_col", base_url))
        .send()
        .await
        .unwrap();

    let resp = client()
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .unwrap();
    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!body.iter().any(|c| c["name"] == "bkp_col"));

    let resp = client()
        .post(format!("{}/admin/restore", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["collections_loaded"].as_u64().unwrap() >= 1);

    let resp = client()
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .unwrap();
    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(body.iter().any(|c| c["name"] == "bkp_col"));
}

#[tokio::test]
async fn test_rbac_read_only() {
    let rbac = RbacConfig::from_entries(vec![
        ApiKeyEntry {
            key: "read-key".to_string(),
            role: Role::Read,
            rate_limit_rps: None,
        },
        ApiKeyEntry {
            key: "admin-key".to_string(),
            role: Role::Admin,
            rate_limit_rps: None,
        },
    ]);
    let (base_url, _tmp) = spawn_app_full(None, Some(rbac), 0).await;

    let resp = client()
        .get(format!("{}/collections", base_url))
        .header("Authorization", "Bearer read-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .post(format!("{}/collections", base_url))
        .header("Authorization", "Bearer read-key")
        .json(&serde_json::json!({"name": "test", "dimension": 3}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test]
async fn test_rbac_write_role() {
    let rbac = RbacConfig::from_entries(vec![ApiKeyEntry {
        key: "write-key".to_string(),
        role: Role::Write,
        rate_limit_rps: None,
    }]);
    let (base_url, _tmp) = spawn_app_full(None, Some(rbac), 0).await;

    let resp = client()
        .post(format!("{}/collections", base_url))
        .header("Authorization", "Bearer write-key")
        .json(&serde_json::json!({"name": "wtest", "dimension": 3}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .post(format!("{}/admin/backup", base_url))
        .header("Authorization", "Bearer write-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test]
async fn test_rbac_admin_role() {
    let rbac = RbacConfig::from_entries(vec![ApiKeyEntry {
        key: "admin-key".to_string(),
        role: Role::Admin,
        rate_limit_rps: None,
    }]);
    let (base_url, _tmp) = spawn_app_full(None, Some(rbac), 0).await;

    let resp = client()
        .post(format!("{}/admin/backup", base_url))
        .header("Authorization", "Bearer admin-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_rbac_backward_compat() {
    let (base_url, _tmp) = spawn_app_no_recorder(Some("legacy-key")).await;

    let resp = client()
        .post(format!("{}/admin/compact", base_url))
        .header("Authorization", "Bearer legacy-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_request_body_too_large() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let large_body = "x".repeat(11 * 1024 * 1024);
    let resp = client()
        .post(format!("{}/collections", base_url))
        .header("Content-Type", "application/json")
        .body(large_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 413);
}

#[tokio::test]
async fn test_prefilter_selective() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "prefilter", 3).await;

    for i in 0..20 {
        let angle = (i as f32) * 0.3;
        insert_test_document_with_metadata(
            &base_url,
            "prefilter",
            &format!("other doc {}", i),
            vec![angle.cos(), angle.sin(), 0.1 * (i as f32 + 1.0)],
            serde_json::json!({"category": "noise"}),
        )
        .await;
    }

    insert_test_document_with_metadata(
        &base_url,
        "prefilter",
        "target alpha",
        vec![1.0, 0.0, 0.1],
        serde_json::json!({"category": "target"}),
    )
    .await;
    insert_test_document_with_metadata(
        &base_url,
        "prefilter",
        "target beta",
        vec![0.95, 0.1, 0.1],
        serde_json::json!({"category": "target"}),
    )
    .await;

    let resp = client()
        .post(format!("{}/collections/prefilter/search", base_url))
        .json(&serde_json::json!({
            "query_embedding": [1.0, 0.0, 0.0], "k": 10,
            "filter": { "must": [{"field": "category", "op": "eq", "value": "target"}] }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        2,
        "Pre-filtering should find both target docs"
    );
    for r in results {
        assert_eq!(r["metadata"]["category"], "target");
    }
}

#[tokio::test]
async fn test_standalone_mode_works() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    create_test_collection(&base_url, "standalone", 3).await;
    insert_test_document(&base_url, "standalone", "test doc", vec![1.0, 0.0, 0.0]).await;

    let resp = client()
        .post(format!("{}/collections/standalone/search", base_url))
        .json(&serde_json::json!({"query_embedding": [1.0, 0.0, 0.0], "k": 5}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_routing_table_endpoint() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!("{}/admin/routing", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["routing"].is_object());
}

#[tokio::test]
async fn test_routing_table_with_local_assignment() {
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    let tmp_dir = tempfile::TempDir::new().unwrap();
    let data_dir = tmp_dir.path().to_str().unwrap().to_string();
    let db = vectorsdb_core::storage::Database::new();
    let wal = Arc::new(WriteAheadLog::new(&data_dir).unwrap());

    let prometheus_handle =
        match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
            Ok(handle) => handle,
            Err(_) => metrics_exporter_prometheus::PrometheusBuilder::new()
                .build_recorder()
                .handle(),
        };

    let routing_table = Arc::new(RwLock::new(HashMap::new()));

    let wal_path = std::path::PathBuf::from(&data_dir).join("wal.bin");
    let state = AppState {
        db,
        data_dir,
        wal,
        wal_path,
        api_key: None,
        prometheus_handle,
        max_memory_bytes: 0,
        rbac: None,
        raft: None,
        node_id: Some(1),
        routing_table: Some(routing_table),
        peer_addrs: None,
        start_time: std::time::Instant::now(),
        key_rate_limiters: std::sync::Arc::new(parking_lot::Mutex::new(
            std::collections::HashMap::new(),
        )),
        memory_reserved: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    let app = vectorsdb_server::api::create_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let resp = client()
        .post(format!("{}/admin/assign", base_url))
        .json(&serde_json::json!({"collection": "mycol", "node_id": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{}/admin/routing", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["routing"]["mycol"], 1);
}

#[tokio::test]
async fn test_security_headers_present() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    assert_eq!(
        resp.headers().get("x-content-type-options").unwrap(),
        "nosniff"
    );
    assert_eq!(resp.headers().get("x-frame-options").unwrap(), "DENY");
    assert_eq!(
        resp.headers().get("referrer-policy").unwrap(),
        "no-referrer"
    );
}

#[tokio::test]
async fn test_metadata_too_many_keys() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "metakeys_col", 3).await;

    let mut metadata = serde_json::Map::new();
    for i in 0..65 {
        metadata.insert(format!("key_{}", i), serde_json::Value::String("v".into()));
    }

    let resp = client()
        .post(format!("{}/collections/metakeys_col/documents", base_url))
        .json(&serde_json::json!({"text": "too many keys", "embedding": [1.0, 0.0, 0.0], "metadata": metadata}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_metadata_too_large() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "metalarge_col", 3).await;

    let big_value = "x".repeat(70_000);
    let resp = client()
        .post(format!("{}/collections/metalarge_col/documents", base_url))
        .json(&serde_json::json!({"text": "large metadata", "embedding": [1.0, 0.0, 0.0], "metadata": {"big": big_value}}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_error_messages_sanitized() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .post(format!("{}/collections/nonexistent/load", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let error_msg = body["error"].as_str().unwrap_or("");
    assert!(
        !error_msg.contains('/') && !error_msg.contains("No such file"),
        "Error message leaks internal details: {}",
        error_msg
    );
}

#[tokio::test]
async fn test_create_collection_returns_config() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client().post(format!("{}/collections", base_url))
        .json(&serde_json::json!({"name": "cfg_col", "dimension": 128, "m": 32, "ef_construction": 400}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "cfg_col");
    assert_eq!(body["dimension"], 128);
    assert_eq!(body["m"], 32);
    assert_eq!(body["ef_construction"], 400);
    assert!(body["ef_search"].as_u64().unwrap() > 0);
    assert!(body["distance_metric"].as_str().is_some());
}

#[tokio::test]
async fn test_document_count_endpoint() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "cnt_col", 3).await;

    let resp = client()
        .get(format!("{}/collections/cnt_col/documents/count", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 0);

    for i in 0..3 {
        insert_test_document(
            &base_url,
            "cnt_col",
            &format!("doc {}", i),
            vec![i as f32, 0.0, 0.0],
        )
        .await;
    }

    let resp = client()
        .get(format!("{}/collections/cnt_col/documents/count", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 3);
}

#[tokio::test]
async fn test_document_count_nonexistent() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .get(format!(
            "{}/collections/no_such_col/documents/count",
            base_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_clear_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;
    create_test_collection(&base_url, "clr_col", 3).await;

    for i in 0..5 {
        insert_test_document(
            &base_url,
            "clr_col",
            &format!("doc {}", i),
            vec![i as f32, 0.0, 0.0],
        )
        .await;
    }

    let resp = client()
        .get(format!("{}/collections/clr_col/documents/count", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 5);

    let resp = client()
        .post(format!("{}/collections/clr_col/clear", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{}/collections/clr_col/documents/count", base_url))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 0);

    let resp = insert_test_document(&base_url, "clr_col", "new doc", vec![1.0, 0.0, 0.0]).await;
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_clear_nonexistent_collection() {
    let (base_url, _tmp) = spawn_app_no_recorder(None).await;

    let resp = client()
        .post(format!("{}/collections/no_such/clear", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
