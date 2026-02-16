"""
Batch Operations: efficiently insert many documents at once.

Also demonstrates collection management and document CRUD.
"""

import random
import time

import vectorsdb

db = vectorsdb.VectorDB()
db.create_collection("bench", dimension=64)

# --- Batch insert ---

print("=== Batch Insert ===")

# Generate random documents
num_docs = 1000
documents = []
for i in range(num_docs):
    embedding = [random.gauss(0, 1) for _ in range(64)]
    documents.append({
        "text": f"Document {i}: {'important' if i % 10 == 0 else 'regular'} content",
        "embedding": embedding,
        "metadata": {
            "index": i,
            "category": ["A", "B", "C"][i % 3],
            "important": i % 10 == 0,
        },
    })

start = time.perf_counter()
ids = db.batch_insert("bench", documents)
elapsed = time.perf_counter() - start

print(f"Inserted {len(ids)} documents in {elapsed:.3f}s ({len(ids)/elapsed:.0f} docs/sec)")

info = db.collection_info("bench")
print(f"Collection: {info.document_count} docs, ~{info.estimated_memory_bytes / 1024:.1f} KB memory")

# --- Document retrieval ---

print("\n=== Document CRUD ===")

# Get a specific document
doc = db.get("bench", ids[0])
print(f"Get doc: id={doc['id'][:8]}..., text='{doc['text'][:40]}...'")

# Delete some documents
deleted_count = 0
for doc_id in ids[:10]:
    if db.delete("bench", doc_id):
        deleted_count += 1
print(f"Deleted {deleted_count} documents")

info = db.collection_info("bench")
print(f"After deletion: {info.document_count} docs")

# --- Rebuild index ---

print("\n=== Rebuild Index ===")

start = time.perf_counter()
live_count = db.rebuild("bench")
elapsed = time.perf_counter() - start
print(f"Rebuilt index: {live_count} live documents in {elapsed:.3f}s")

# --- Search within the batch ---

print("\n=== Search ===")

# Vector search
results = db.search("bench", query_embedding=documents[50]["embedding"], k=5)
print("Top 5 by vector similarity:")
for r in results:
    print(f"  {r.text[:50]} (score={r.score:.4f})")

# Filtered search: only "important" documents
results = db.search(
    "bench",
    query_embedding=documents[50]["embedding"],
    k=5,
    filter={"must": [{"field": "important", "op": "eq", "value": True}]},
)
print("\nTop 5 important docs:")
for r in results:
    print(f"  {r.text[:50]} (score={r.score:.4f})")

# --- Collection management ---

print("\n=== Collection Management ===")
print(f"Collections: {db.list_collections()}")
print(f"Total memory: {db.total_memory_bytes() / 1024:.1f} KB")

db.delete_collection("bench")
print(f"After deletion: {db.list_collections()}")

print("\nDone!")
