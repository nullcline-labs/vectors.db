//! BM25 Benchmark: NFCorpus from BEIR
//! Measures nDCG@10 and QPS for keyword retrieval
//!
//! Usage: cargo bench --bench bm25_nfcorpus

use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use vectors_db::bm25::inverted_index::InvertedIndex;
use vectors_db::bm25::scorer::bm25_search;

const DATA_DIR: &str = "benchmarks/data/nfcorpus";

#[derive(serde::Deserialize)]
struct CorpusDoc {
    _id: String,
    title: String,
    text: String,
}

#[derive(serde::Deserialize)]
struct Query {
    _id: String,
    text: String,
}

/// Compute DCG@k
fn dcg_at_k(relevances: &[f64], k: usize) -> f64 {
    relevances
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| {
            let rank = i as f64 + 2.0; // log2(i+2) because i is 0-indexed
            rel / rank.log2()
        })
        .sum()
}

/// Compute nDCG@k given ranked doc IDs and relevance judgments
fn ndcg_at_k(ranked_ids: &[&str], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    // Get relevances in rank order
    let relevances: Vec<f64> = ranked_ids
        .iter()
        .take(k)
        .map(|id| *qrels.get(*id).unwrap_or(&0.0))
        .collect();

    let dcg = dcg_at_k(&relevances, k);

    // Ideal DCG: sort all relevances descending
    let mut ideal_rels: Vec<f64> = qrels.values().copied().collect();
    ideal_rels.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idcg = dcg_at_k(&ideal_rels, k);

    if idcg < 1e-10 {
        0.0
    } else {
        dcg / idcg
    }
}

fn main() {
    println!("=== BM25 Benchmark: NFCorpus (BEIR) ===");
    println!();

    // Load corpus
    print!("Loading corpus...");
    let corpus_text = fs::read_to_string(format!("{DATA_DIR}/corpus.jsonl")).unwrap();
    let corpus_docs: Vec<CorpusDoc> = corpus_text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    println!(" {} documents", corpus_docs.len());

    // Load queries
    print!("Loading queries...");
    let queries_text = fs::read_to_string(format!("{DATA_DIR}/queries.jsonl")).unwrap();
    let queries: Vec<Query> = queries_text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    println!(" {} queries", queries.len());

    // Load qrels (relevance judgments)
    print!("Loading qrels...");
    let qrels_json = fs::read_to_string(format!("{DATA_DIR}/qrels_clean.json")).unwrap();
    let qrels: HashMap<String, Vec<(String, f64)>> = {
        let raw: HashMap<String, Vec<(String, i64)>> = serde_json::from_str(&qrels_json).unwrap();
        raw.into_iter()
            .map(|(qid, rels)| {
                let rels_f64: Vec<(String, f64)> = rels
                    .into_iter()
                    .map(|(cid, score)| (cid, score as f64))
                    .collect();
                (qid, rels_f64)
            })
            .collect()
    };
    println!(" {} queries with judgments", qrels.len());

    // Build inverted index
    println!();
    println!("--- Index Construction ---");

    // Map corpus _id → internal u32
    let mut id_to_str: Vec<String> = Vec::with_capacity(corpus_docs.len());
    let mut str_to_id: HashMap<String, u32> = HashMap::with_capacity(corpus_docs.len());

    let mut index = InvertedIndex::new();
    let t0 = Instant::now();
    for (i, doc) in corpus_docs.iter().enumerate() {
        let internal_id = i as u32;
        let full_text = format!("{} {}", doc.title, doc.text);
        index.add_document(internal_id, &full_text);
        id_to_str.push(doc._id.clone());
        str_to_id.insert(doc._id.clone(), internal_id);
    }
    let build_time = t0.elapsed();
    println!(
        "Build time: {:.3}s ({:.0} docs/s)",
        build_time.as_secs_f64(),
        corpus_docs.len() as f64 / build_time.as_secs_f64()
    );
    println!("Vocabulary size: {} terms", index.index.len());
    println!("Avg doc length: {:.1} tokens", index.average_doc_length());

    // === Query Benchmark ===
    println!();
    println!("--- Retrieval (BM25 Okapi, k1=1.2, b=0.75) ---");
    println!();

    let k_values = [10, 100];

    for &k in &k_values {
        // Only evaluate queries that have relevance judgments
        let eval_queries: Vec<&Query> = queries
            .iter()
            .filter(|q| qrels.contains_key(&q._id))
            .collect();

        let num_queries = eval_queries.len();

        // Warm up
        for q in eval_queries.iter().take(10) {
            let _ = bm25_search(&index, &q.text, k);
        }

        let t0 = Instant::now();
        let mut total_ndcg = 0.0f64;
        let mut queries_with_results = 0;

        for q in &eval_queries {
            let results = bm25_search(&index, &q.text, k);

            // Convert internal IDs back to string IDs
            let ranked_ids: Vec<&str> = results
                .iter()
                .filter_map(|&(internal_id, _)| {
                    id_to_str.get(internal_id as usize).map(|s| s.as_str())
                })
                .collect();

            // Get relevance map for this query
            if let Some(rels) = qrels.get(&q._id) {
                let rel_map: HashMap<String, f64> = rels.iter().cloned().collect();
                let ndcg = ndcg_at_k(&ranked_ids, &rel_map, k);
                total_ndcg += ndcg;
                queries_with_results += 1;
            }
        }

        let elapsed = t0.elapsed();
        let avg_ndcg = if queries_with_results > 0 {
            total_ndcg / queries_with_results as f64
        } else {
            0.0
        };
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        let avg_latency_us = elapsed.as_micros() as f64 / num_queries as f64;

        println!("  k={k:<4} | nDCG@{k:<4} = {avg_ndcg:.4} | QPS: {qps:.0} | Avg latency: {avg_latency_us:.0} us | Queries: {queries_with_results}");
    }

    // === Reference comparison ===
    println!();
    println!("--- Reference (BEIR Benchmark, NFCorpus test set, nDCG@10) ---");
    println!("  Implementation         | nDCG@10 | Notes");
    println!("  -----------------------+---------+------");
    println!("  BM25 Anserini (BEIR)   |  0.3218 | Lucene, k1=0.9 b=0.4, canonical baseline");
    println!("  BM25 Pyserini multi    |  0.3250 | Multi-field (title+text)");
    println!("  BM25 Elasticsearch     |  0.3428 | Best BM25-only reported");
    println!("  BM25S (stopwords+stem) |  0.3247 | Python, eager sparse scoring");
    println!("  Vespa BM25             |  0.3130 | Java");
    println!("  Manticore Search       |  0.3172 | C++");
    println!("  Weaviate BM25          |  0.2240 | Go");
    println!("  SPLADE-v2 (neural)     |  0.3475 | Learned sparse");
    println!("  DPR (neural)           |  0.1892 | Dense retrieval");
    println!();
    println!("  Note: vectors.db uses simple whitespace tokenizer (no stemming/stopwords).");
    println!("  Adding stemming+stopwords would likely push nDCG closer to 0.32+.");

    println!();
    println!("=== Benchmark complete ===");
}
