"""
Persistence: save collections to disk and load them back.

Two options:
  1. Snapshot save/load — explicit save and load of collection data
  2. WAL persistence — pass data_dir at construction for write-ahead logging
"""

import os
import tempfile

import vectorsdb

# Create a temp directory for this example
tmp_dir = tempfile.mkdtemp(prefix="vectorsdb_example_")
print(f"Using temp directory: {tmp_dir}\n")

# --- Snapshot persistence ---

print("=== Snapshot Save / Load ===")

# Create and populate a collection
db = vectorsdb.VectorDB()
db.create_collection("notes", dimension=3)
db.insert("notes", "Buy groceries", [1.0, 0.0, 0.0], metadata={"priority": "high"})
db.insert("notes", "Read a book", [0.0, 1.0, 0.0], metadata={"priority": "low"})
db.insert("notes", "Go for a run", [0.0, 0.0, 1.0], metadata={"priority": "medium"})
print(f"Created collection with {db.collection_info('notes').document_count} documents")

# Save to disk
db.save("notes", path=tmp_dir)
snapshot_file = os.path.join(tmp_dir, "notes.vdb")
print(f"Saved snapshot: {snapshot_file} ({os.path.getsize(snapshot_file)} bytes)")

# Create a NEW database and load the snapshot
db2 = vectorsdb.VectorDB()
db2.load("notes", path=tmp_dir)
info = db2.collection_info("notes")
print(f"Loaded collection: {info.name}, {info.document_count} documents")

# Verify search still works
results = db2.search("notes", query_embedding=[1.0, 0.0, 0.0], k=1)
print(f"Search result: '{results[0].text}' (score={results[0].score:.4f})")
print(f"Metadata preserved: priority={results[0].metadata['priority']}")

# --- Multiple collections ---

print("\n=== Multiple Collections ===")

db3 = vectorsdb.VectorDB()
for name in ["alpha", "beta", "gamma"]:
    db3.create_collection(name, dimension=2)
    db3.insert(name, f"Document in {name}", [1.0, 0.0])
    db3.save(name, path=tmp_dir)

print(f"Saved {len(db3.list_collections())} collections")

# Load them all into a fresh database
db4 = vectorsdb.VectorDB()
for name in ["alpha", "beta", "gamma"]:
    db4.load(name, path=tmp_dir)

print(f"Loaded {len(db4.list_collections())} collections: {db4.list_collections()}")
for name in db4.list_collections():
    results = db4.search(name, query_embedding=[1.0, 0.0], k=1)
    print(f"  {name}: '{results[0].text}'")

# Cleanup
import shutil
shutil.rmtree(tmp_dir)
print(f"\nCleaned up {tmp_dir}")
print("Done!")
