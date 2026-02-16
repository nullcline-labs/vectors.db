"""
Metadata Filtering: narrow search results using structured metadata.

Supports operators: eq, ne, gt, lt, gte, lte, in.
Filter clauses have "must" (AND) and "must_not" (exclude) lists.
"""

import vectorsdb

db = vectorsdb.VectorDB()
db.create_collection("products", dimension=3)

# Insert products with rich metadata
products = [
    {"text": "Wireless Mouse", "embedding": [1.0, 0.0, 0.0], "metadata": {"category": "electronics", "price": 29, "in_stock": True}},
    {"text": "Mechanical Keyboard", "embedding": [0.9, 0.1, 0.0], "metadata": {"category": "electronics", "price": 89, "in_stock": True}},
    {"text": "USB-C Hub", "embedding": [0.8, 0.2, 0.0], "metadata": {"category": "electronics", "price": 45, "in_stock": False}},
    {"text": "Python Cookbook", "embedding": [0.0, 1.0, 0.0], "metadata": {"category": "books", "price": 35, "in_stock": True}},
    {"text": "Rust Programming", "embedding": [0.0, 0.9, 0.1], "metadata": {"category": "books", "price": 42, "in_stock": True}},
    {"text": "Standing Desk", "embedding": [0.5, 0.0, 0.8], "metadata": {"category": "furniture", "price": 350, "in_stock": True}},
    {"text": "Monitor Arm", "embedding": [0.6, 0.0, 0.7], "metadata": {"category": "furniture", "price": 75, "in_stock": False}},
]
db.batch_insert("products", products)

query = [0.9, 0.1, 0.0]  # something electronics-like

# 1. Filter: electronics only
print("=== Electronics only ===")
results = db.search("products", query_embedding=query, k=5, filter={
    "must": [{"field": "category", "op": "eq", "value": "electronics"}]
})
for r in results:
    print(f"  {r.text} — ${r.metadata['price']} (score={r.score:.4f})")

# 2. Filter: in stock AND price under $50
print("\n=== In stock & under $50 ===")
results = db.search("products", query_embedding=query, k=5, filter={
    "must": [
        {"field": "in_stock", "op": "eq", "value": True},
        {"field": "price", "op": "lt", "value": 50},
    ]
})
for r in results:
    print(f"  {r.text} — ${r.metadata['price']} (score={r.score:.4f})")

# 3. Filter: NOT furniture
print("\n=== Exclude furniture ===")
results = db.search("products", query_embedding=query, k=5, filter={
    "must_not": [{"field": "category", "op": "eq", "value": "furniture"}]
})
for r in results:
    print(f"  {r.text} — {r.metadata['category']} (score={r.score:.4f})")

# 4. Filter: category IN [books, electronics]
print("\n=== Books or electronics (IN operator) ===")
results = db.search("products", query_embedding=query, k=5, filter={
    "must": [{"field": "category", "op": "in", "values": ["books", "electronics"]}]
})
for r in results:
    print(f"  {r.text} — {r.metadata['category']} (score={r.score:.4f})")

print("\nDone!")
