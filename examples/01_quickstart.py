"""
Quickstart: vectors.db in 30 seconds.

Create a database, add documents, and search — all in-process, no server needed.
"""

import vectorsdb

# 1. Create a database and a collection
db = vectorsdb.VectorDB()
db.create_collection("movies", dimension=4)

# 2. Insert some documents with embeddings
db.insert("movies", "The Matrix", [1.0, 0.0, 0.1, 0.8], metadata={"year": 1999, "genre": "sci-fi"})
db.insert("movies", "Inception", [0.9, 0.1, 0.2, 0.7], metadata={"year": 2010, "genre": "sci-fi"})
db.insert("movies", "The Godfather", [0.1, 0.9, 0.8, 0.1], metadata={"year": 1972, "genre": "drama"})
db.insert("movies", "Pulp Fiction", [0.2, 0.8, 0.7, 0.2], metadata={"year": 1994, "genre": "crime"})
db.insert("movies", "Interstellar", [0.95, 0.05, 0.15, 0.75], metadata={"year": 2014, "genre": "sci-fi"})

# 3. Search by vector similarity
print("=== Vector Search: movies like The Matrix ===")
results = db.search("movies", query_embedding=[1.0, 0.0, 0.1, 0.8], k=3)
for r in results:
    print(f"  {r.text} (score={r.score:.4f}, year={r.metadata['year']})")

# 4. Search by keyword
print("\n=== Keyword Search: 'godfather' ===")
results = db.search("movies", query_text="godfather", k=3)
for r in results:
    print(f"  {r.text} (score={r.score:.4f})")

# 5. Hybrid search (vector + keyword combined)
print("\n=== Hybrid Search: embedding + 'fiction' ===")
results = db.search(
    "movies",
    query_embedding=[0.2, 0.8, 0.7, 0.2],
    query_text="fiction",
    k=3,
    alpha=0.5,  # 50% vector, 50% keyword
)
for r in results:
    print(f"  {r.text} (score={r.score:.4f})")

print("\nDone! Total memory used:", db.total_memory_bytes(), "bytes")
