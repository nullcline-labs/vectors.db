//! ANN Benchmark: GloVe-25 angular (cosine)
//! Measures Recall@10 and QPS against ground truth from ann-benchmarks.com
//!
//! Usage: cargo bench --bench ann_glove25

use std::fs::File;
use std::io::Read as _;
use std::time::Instant;
use vectors_db::hnsw::graph::{HnswConfig, HnswIndex};
use vectors_db::hnsw::search::knn_search;
use vectors_db::quantization::QuantizedVector;
// QuantizedVector still needed for insert (stores u8), but not for queries

const DATA_DIR: &str = "benchmarks/data";

/// Read a binary file with format: u32 count, u32 dim, then f32 data
fn read_fvecs(path: &str) -> (usize, usize, Vec<Vec<f32>>) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

    let mut vectors = Vec::with_capacity(count);
    let data = &buf[8..];
    for i in 0..count {
        let offset = i * dim * 4;
        let vec: Vec<f32> = (0..dim)
            .map(|j| {
                f32::from_le_bytes(data[offset + j * 4..offset + j * 4 + 4].try_into().unwrap())
            })
            .collect();
        vectors.push(vec);
    }
    (count, dim, vectors)
}

/// Read ground truth neighbors: u32 count, u32 k, then i32 neighbor IDs
fn read_ground_truth(path: &str) -> (usize, usize, Vec<Vec<i32>>) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

    let mut neighbors = Vec::with_capacity(count);
    let data = &buf[8..];
    for i in 0..count {
        let offset = i * k * 4;
        let ids: Vec<i32> = (0..k)
            .map(|j| {
                i32::from_le_bytes(data[offset + j * 4..offset + j * 4 + 4].try_into().unwrap())
            })
            .collect();
        neighbors.push(ids);
    }
    (count, k, neighbors)
}

/// Compute Recall@k: fraction of true nearest neighbors found
fn recall_at_k(predicted: &[u32], ground_truth: &[i32], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<u32> =
        ground_truth.iter().take(k).map(|&id| id as u32).collect();
    let found = predicted
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();
    found as f64 / k as f64
}

fn main() {
    println!("=== ANN Benchmark: GloVe-25 angular (cosine) ===");
    println!();

    // Load data
    print!("Loading train vectors...");
    let (train_count, dim, train_vectors) = read_fvecs(&format!("{DATA_DIR}/glove25_train.bin"));
    println!(" {train_count} vectors x {dim}d");

    print!("Loading test queries...");
    let (test_count, _, test_vectors) = read_fvecs(&format!("{DATA_DIR}/glove25_test.bin"));
    println!(" {test_count} queries");

    print!("Loading ground truth...");
    let (_, gt_k, ground_truth) = read_ground_truth(&format!("{DATA_DIR}/glove25_neighbors.bin"));
    println!(" top-{gt_k} per query");

    // === Build index ===
    println!();
    println!("--- Index Construction ---");

    let configs = vec![(
        "M=16, ef_c=200",
        HnswConfig {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            max_layers: 16,
            distance_metric: vectors_db::hnsw::DistanceMetric::Cosine,
        },
    )];

    for (label, config) in &configs {
        println!();
        println!("Config: {label}");

        let mut index = HnswIndex::new(dim, config.clone());

        // Insert all train vectors
        let t0 = Instant::now();
        for (i, vec) in train_vectors.iter().enumerate() {
            let quantized = QuantizedVector::quantize(vec);
            index.insert(i as u32, vec, quantized);
            if (i + 1) % 100_000 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                println!("  inserted {}/{train_count} ({rate:.0} vec/s)", i + 1);
            }
        }
        let build_time = t0.elapsed();
        let build_rate = train_count as f64 / build_time.as_secs_f64();
        println!(
            "  Build time: {:.2}s ({build_rate:.0} inserts/s)",
            build_time.as_secs_f64()
        );
        println!("  Index nodes: {}", index.node_count);

        // === Query benchmark at various ef_search values ===
        println!();
        println!("  ef_search | Recall@10 |    QPS    | Avg latency");
        println!("  ----------+-----------+-----------+------------");

        let ef_values = [10, 20, 40, 80, 120, 200, 400];
        let num_queries = test_count.min(10_000);
        let k = 10;

        for &ef in &ef_values {
            let old_ef = index.config.ef_search;
            index.config.ef_search = ef;

            // Warm up (10 queries)
            for q in test_vectors.iter().take(10) {
                let _ = knn_search(&index, q, k);
            }

            // Timed run
            let t0 = Instant::now();
            let mut total_recall = 0.0f64;
            for (qi, q) in test_vectors.iter().take(num_queries).enumerate() {
                let results = knn_search(&index, q, k);
                let predicted: Vec<u32> = results.iter().map(|&(_, id)| id).collect();
                total_recall += recall_at_k(&predicted, &ground_truth[qi], k);
            }
            let elapsed = t0.elapsed();

            let avg_recall = total_recall / num_queries as f64;
            let qps = num_queries as f64 / elapsed.as_secs_f64();
            let avg_latency_us = elapsed.as_micros() as f64 / num_queries as f64;

            println!(
                "  {:>9} | {:.4}    | {:>9.1} | {:.0} us",
                ef, avg_recall, qps, avg_latency_us
            );

            // Restore
            index.config.ef_search = old_ef;
        }

        // === Reference comparison ===
        println!();
        println!("  --- Reference (ann-benchmarks.com, GloVe-25 angular, k=10) ---");
        println!("  Algorithm          | Recall@10 |      QPS | Notes");
        println!("  -------------------+-----------+----------+------");
        println!("  hnsw(nmslib)       |    0.9869 |    6,452 | C++ HNSW, best balance");
        println!("  hnswlib            |    0.9622 |      574 | Standalone C++ HNSW");
        println!("  scann (Google)     |    0.9948 |    5,696 | Quantization + HNSW");
        println!("  hnsw(faiss)        |    1.0000 |      553 | Facebook HNSW");
        println!("  faiss-ivf          |    1.0000 |       79 | Exact IVF scan");
        println!("  Qdrant             |    0.9903 |    1,035 | Rust, HNSW based");
        println!("  Milvus(Knowhere)   |    0.9900 |    1,143 | Go/C++");
        println!("  Redisearch         |    1.0000 |      682 | Redis module");
        println!("  pgvector           |    0.9963 |       19 | PostgreSQL extension");
        println!("  Annoy (Spotify)    |    0.9500 |    2,000 | Tree-based, est.");
        println!("  glass              |    1.0000 |    1,130 | Graph-based");
        println!("  Vamana(diskann)    |    0.9991 |    3,161 | Microsoft DiskANN");
        println!();
        println!("  Note: Reference uses f32 vectors. vectors.db uses u8 scalar quantization,");
        println!("  so recall may be lower due to quantization loss, but memory is ~4x smaller.");
    }

    println!();
    println!("=== Benchmark complete ===");
}
