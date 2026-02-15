"""
Distance Metrics: compare cosine, euclidean, and dot product similarity.

Different metrics suit different embedding models and use cases.
"""

import vectorsdb

# Same documents, three different distance metrics
dimension = 4
docs = [
    ("North", [1.0, 0.0, 0.0, 0.0]),
    ("Northeast", [0.7, 0.7, 0.0, 0.0]),
    ("East", [0.0, 1.0, 0.0, 0.0]),
    ("South", [-1.0, 0.0, 0.0, 0.0]),
    ("Up", [0.0, 0.0, 1.0, 0.0]),
]

query = [0.9, 0.3, 0.0, 0.0]  # mostly north, slightly east

for metric in ["cosine", "euclidean", "dot_product"]:
    db = vectorsdb.VectorDB()
    db.create_collection("directions", dimension=dimension, distance_metric=metric)

    for text, embedding in docs:
        db.insert("directions", text, embedding)

    results = db.search("directions", query_embedding=query, k=5)

    print(f"=== {metric.upper()} ===")
    for r in results:
        print(f"  {r.text:12s} score={r.score:.4f}")
    print()

print("Notes:")
print("  - Cosine: measures angle between vectors (ignores magnitude)")
print("  - Euclidean: measures straight-line distance (converted to similarity)")
print("  - Dot product: combines direction and magnitude")
print("\nDone!")
