"""Tests for the vectorsdb Python bindings."""

import os
import tempfile
import uuid

import pytest

import vectorsdb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Fresh ephemeral database for each test."""
    return vectorsdb.VectorDB()


@pytest.fixture
def db_with_collection(db):
    """Database with a 3-dimensional collection pre-created."""
    db.create_collection("test", dimension=3)
    return db


@pytest.fixture
def populated_db(db_with_collection):
    """Database with a collection containing 4 documents."""
    db = db_with_collection
    db.insert("test", "cats are fluffy", [1.0, 0.0, 0.0], metadata={"category": "animals", "rating": 5})
    db.insert("test", "dogs are loyal", [0.9, 0.1, 0.0], metadata={"category": "animals", "rating": 4})
    db.insert("test", "python is great", [0.0, 1.0, 0.0], metadata={"category": "tech", "rating": 5})
    db.insert("test", "rust is fast", [0.0, 0.9, 0.1], metadata={"category": "tech", "rating": 3})
    return db


# ===========================================================================
# Collection management
# ===========================================================================


class TestCollectionManagement:
    def test_create_and_list(self, db):
        db.create_collection("col1", dimension=128)
        db.create_collection("col2", dimension=64)
        names = db.list_collections()
        assert set(names) == {"col1", "col2"}

    def test_create_duplicate_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="already exists"):
            db_with_collection.create_collection("test", dimension=3)

    def test_delete_collection(self, db_with_collection):
        assert db_with_collection.delete_collection("test") is True
        assert db_with_collection.list_collections() == []

    def test_delete_nonexistent(self, db):
        assert db.delete_collection("nope") is False

    def test_collection_info(self, db_with_collection):
        info = db_with_collection.collection_info("test")
        assert info.name == "test"
        assert info.dimension == 3
        assert info.document_count == 0
        assert info.estimated_memory_bytes >= 0

    def test_collection_info_missing_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.collection_info("missing")

    def test_create_with_custom_hnsw_params(self, db):
        db.create_collection(
            "custom", dimension=64,
            m=32, ef_construction=400, ef_search=100,
            distance_metric="euclidean",
        )
        info = db.collection_info("custom")
        assert info.name == "custom"
        assert info.dimension == 64

    def test_create_with_dot_product_metric(self, db):
        db.create_collection("dp", dimension=4, distance_metric="dot_product")
        assert "dp" in db.list_collections()

    def test_create_with_invalid_metric_raises(self, db):
        with pytest.raises(ValueError, match="unknown distance metric"):
            db.create_collection("bad", dimension=4, distance_metric="hamming")

    def test_list_empty(self, db):
        assert db.list_collections() == []


# ===========================================================================
# Document CRUD
# ===========================================================================


class TestDocumentCRUD:
    def test_insert_returns_uuid(self, db_with_collection):
        doc_id = db_with_collection.insert("test", "hello", [1.0, 0.0, 0.0])
        # Should be a valid UUID string
        parsed = uuid.UUID(doc_id)
        assert str(parsed) == doc_id

    def test_insert_and_get(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "hello world", [1.0, 0.0, 0.0],
            metadata={"key": "value"},
        )
        doc = db_with_collection.get("test", doc_id)
        assert doc["id"] == doc_id
        assert doc["text"] == "hello world"
        assert doc["metadata"] == {"key": "value"}

    def test_insert_with_custom_id(self, db_with_collection):
        custom_id = "12345678-1234-1234-1234-123456789abc"
        doc_id = db_with_collection.insert(
            "test", "custom", [0.0, 1.0, 0.0], id=custom_id,
        )
        assert doc_id == custom_id
        doc = db_with_collection.get("test", doc_id)
        assert doc["text"] == "custom"

    def test_insert_invalid_uuid_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="invalid UUID"):
            db_with_collection.insert("test", "bad", [1.0, 0.0, 0.0], id="not-a-uuid")

    def test_insert_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.insert("nope", "text", [1.0])

    def test_get_missing_document_raises(self, db_with_collection):
        fake_id = str(uuid.uuid4())
        with pytest.raises(KeyError, match="not found"):
            db_with_collection.get("test", fake_id)

    def test_get_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.get("nope", str(uuid.uuid4()))

    def test_get_invalid_uuid_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="invalid UUID"):
            db_with_collection.get("test", "garbage")

    def test_delete_document(self, db_with_collection):
        doc_id = db_with_collection.insert("test", "to delete", [1.0, 0.0, 0.0])
        assert db_with_collection.delete("test", doc_id) is True
        with pytest.raises(KeyError):
            db_with_collection.get("test", doc_id)

    def test_delete_nonexistent_document(self, db_with_collection):
        fake_id = str(uuid.uuid4())
        assert db_with_collection.delete("test", fake_id) is False

    def test_delete_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.delete("nope", str(uuid.uuid4()))

    def test_insert_no_metadata(self, db_with_collection):
        doc_id = db_with_collection.insert("test", "bare", [0.0, 0.0, 1.0])
        doc = db_with_collection.get("test", doc_id)
        assert doc["metadata"] == {}


# ===========================================================================
# Batch insert
# ===========================================================================


class TestBatchInsert:
    def test_batch_insert_returns_ids(self, db_with_collection):
        ids = db_with_collection.batch_insert("test", [
            {"text": "doc1", "embedding": [1.0, 0.0, 0.0]},
            {"text": "doc2", "embedding": [0.0, 1.0, 0.0]},
            {"text": "doc3", "embedding": [0.0, 0.0, 1.0]},
        ])
        assert len(ids) == 3
        for doc_id in ids:
            uuid.UUID(doc_id)  # validates format

    def test_batch_insert_with_metadata(self, db_with_collection):
        ids = db_with_collection.batch_insert("test", [
            {"text": "a", "embedding": [1.0, 0.0, 0.0], "metadata": {"tag": "first"}},
            {"text": "b", "embedding": [0.0, 1.0, 0.0], "metadata": {"tag": "second"}},
        ])
        doc = db_with_collection.get("test", ids[0])
        assert doc["metadata"]["tag"] == "first"

    def test_batch_insert_with_custom_ids(self, db_with_collection):
        custom_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ids = db_with_collection.batch_insert("test", [
            {"text": "custom", "embedding": [1.0, 0.0, 0.0], "id": custom_id},
        ])
        assert ids[0] == custom_id

    def test_batch_insert_missing_text_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="missing 'text'"):
            db_with_collection.batch_insert("test", [
                {"embedding": [1.0, 0.0, 0.0]},
            ])

    def test_batch_insert_missing_embedding_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="missing 'embedding'"):
            db_with_collection.batch_insert("test", [
                {"text": "no embedding"},
            ])

    def test_batch_insert_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.batch_insert("nope", [{"text": "x", "embedding": [1.0]}])

    def test_batch_insert_empty_list(self, db_with_collection):
        ids = db_with_collection.batch_insert("test", [])
        assert ids == []


# ===========================================================================
# Metadata types
# ===========================================================================


class TestMetadata:
    def test_bool_metadata_roundtrip(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "bool test", [1.0, 0.0, 0.0],
            metadata={"flag": True, "other": False},
        )
        doc = db_with_collection.get("test", doc_id)
        assert doc["metadata"]["flag"] is True
        assert doc["metadata"]["other"] is False

    def test_int_metadata_roundtrip(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "int test", [1.0, 0.0, 0.0],
            metadata={"count": 42, "negative": -7},
        )
        doc = db_with_collection.get("test", doc_id)
        assert doc["metadata"]["count"] == 42
        assert doc["metadata"]["negative"] == -7

    def test_float_metadata_roundtrip(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "float test", [1.0, 0.0, 0.0],
            metadata={"score": 3.14, "ratio": 0.5},
        )
        doc = db_with_collection.get("test", doc_id)
        assert abs(doc["metadata"]["score"] - 3.14) < 1e-6
        assert abs(doc["metadata"]["ratio"] - 0.5) < 1e-6

    def test_string_metadata_roundtrip(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "str test", [1.0, 0.0, 0.0],
            metadata={"label": "important", "empty": ""},
        )
        doc = db_with_collection.get("test", doc_id)
        assert doc["metadata"]["label"] == "important"
        assert doc["metadata"]["empty"] == ""

    def test_mixed_metadata_types(self, db_with_collection):
        doc_id = db_with_collection.insert(
            "test", "mixed", [1.0, 0.0, 0.0],
            metadata={"flag": True, "count": 10, "score": 0.95, "label": "test"},
        )
        doc = db_with_collection.get("test", doc_id)
        meta = doc["metadata"]
        assert meta["flag"] is True
        assert meta["count"] == 10
        assert abs(meta["score"] - 0.95) < 1e-6
        assert meta["label"] == "test"


# ===========================================================================
# Vector search
# ===========================================================================


class TestVectorSearch:
    def test_basic_vector_search(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].text == "cats are fluffy"
        assert results[0].score > 0.99

    def test_vector_search_ordering(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=4)
        assert len(results) == 4
        # First result should be the most similar
        assert results[0].text == "cats are fluffy"
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_vector_search_k_limits_results(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=2)
        assert len(results) == 2

    def test_vector_search_min_similarity(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[1.0, 0.0, 0.0], k=10,
            min_similarity=0.999,
        )
        # Only the exact match [1.0, 0.0, 0.0] should pass this threshold
        assert len(results) == 1
        assert results[0].text == "cats are fluffy"

    def test_vector_search_empty_collection(self, db_with_collection):
        results = db_with_collection.search("test", query_embedding=[1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.search("nope", query_embedding=[1.0, 0.0, 0.0])

    def test_search_no_query_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="at least one"):
            db_with_collection.search("test")


# ===========================================================================
# Keyword search
# ===========================================================================


class TestKeywordSearch:
    def test_basic_keyword_search(self, populated_db):
        results = populated_db.search("test", query_text="fluffy cats", k=2)
        assert len(results) >= 1
        assert results[0].text == "cats are fluffy"

    def test_keyword_search_multiple_terms(self, populated_db):
        results = populated_db.search("test", query_text="python rust", k=2)
        texts = {r.text for r in results}
        assert "python is great" in texts or "rust is fast" in texts

    def test_keyword_search_no_match(self, populated_db):
        results = populated_db.search("test", query_text="zzzznotaword", k=5)
        assert results == []


# ===========================================================================
# Hybrid search
# ===========================================================================


class TestHybridSearch:
    def test_hybrid_rrf(self, populated_db):
        results = populated_db.search(
            "test",
            query_embedding=[1.0, 0.0, 0.0],
            query_text="fluffy",
            k=2,
            fusion_method="rrf",
        )
        assert len(results) >= 1
        assert results[0].text == "cats are fluffy"

    def test_hybrid_linear(self, populated_db):
        results = populated_db.search(
            "test",
            query_embedding=[1.0, 0.0, 0.0],
            query_text="fluffy",
            k=2,
            alpha=0.5,
            fusion_method="linear",
        )
        assert len(results) >= 1
        assert results[0].text == "cats are fluffy"

    def test_hybrid_alpha_all_vector(self, populated_db):
        vec_results = populated_db.search(
            "test", query_embedding=[1.0, 0.0, 0.0], k=4,
        )
        hybrid_results = populated_db.search(
            "test",
            query_embedding=[1.0, 0.0, 0.0],
            query_text="unrelated query",
            k=4,
            alpha=1.0,
            fusion_method="linear",
        )
        # With alpha=1.0, vector results should dominate
        assert hybrid_results[0].text == vec_results[0].text


# ===========================================================================
# Filtered search
# ===========================================================================


class TestFilteredSearch:
    def test_filter_eq_string(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "category", "op": "eq", "value": "tech"}]},
        )
        for r in results:
            assert r.metadata["category"] == "tech"
        texts = {r.text for r in results}
        assert "python is great" in texts
        assert "cats are fluffy" not in texts

    def test_filter_must_not(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must_not": [{"field": "category", "op": "eq", "value": "animals"}]},
        )
        for r in results:
            assert r.metadata["category"] != "animals"

    def test_filter_gt(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "rating", "op": "gt", "value": 4}]},
        )
        for r in results:
            assert r.metadata["rating"] > 4

    def test_filter_gte(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "rating", "op": "gte", "value": 5}]},
        )
        for r in results:
            assert r.metadata["rating"] >= 5

    def test_filter_lt(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "rating", "op": "lt", "value": 4}]},
        )
        for r in results:
            assert r.metadata["rating"] < 4

    def test_filter_ne(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "category", "op": "ne", "value": "animals"}]},
        )
        for r in results:
            assert r.metadata["category"] != "animals"

    def test_filter_in(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={"must": [{"field": "rating", "op": "in", "values": [3, 5]}]},
        )
        for r in results:
            assert r.metadata["rating"] in [3, 5]

    def test_filter_combined_must_and_must_not(self, populated_db):
        results = populated_db.search(
            "test", query_embedding=[0.5, 0.5, 0.0], k=10,
            filter={
                "must": [{"field": "category", "op": "eq", "value": "tech"}],
                "must_not": [{"field": "rating", "op": "eq", "value": 3}],
            },
        )
        for r in results:
            assert r.metadata["category"] == "tech"
            assert r.metadata["rating"] != 3
        # Should only be "python is great" (tech, rating=5)
        assert len(results) == 1
        assert results[0].text == "python is great"

    def test_filter_invalid_operator_raises(self, db_with_collection):
        db_with_collection.insert("test", "x", [1.0, 0.0, 0.0], metadata={"a": 1})
        with pytest.raises(ValueError, match="unknown filter operator"):
            db_with_collection.search(
                "test", query_embedding=[1.0, 0.0, 0.0],
                filter={"must": [{"field": "a", "op": "regex", "value": ".*"}]},
            )

    def test_hybrid_filtered(self, populated_db):
        results = populated_db.search(
            "test",
            query_embedding=[1.0, 0.0, 0.0],
            query_text="fluffy",
            k=10,
            filter={"must": [{"field": "category", "op": "eq", "value": "animals"}]},
        )
        for r in results:
            assert r.metadata["category"] == "animals"


# ===========================================================================
# SearchResult properties
# ===========================================================================


class TestSearchResult:
    def test_search_result_properties(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=1)
        r = results[0]
        assert isinstance(r.id, str)
        assert isinstance(r.text, str)
        assert isinstance(r.score, float)
        assert isinstance(r.metadata, dict)

    def test_search_result_metadata_types(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=1)
        meta = results[0].metadata
        assert isinstance(meta["category"], str)
        assert isinstance(meta["rating"], int)

    def test_search_result_repr(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=1)
        r = repr(results[0])
        assert "SearchResult" in r
        assert "score=" in r
        assert "cats are fluffy" in r


# ===========================================================================
# CollectionInfo
# ===========================================================================


class TestCollectionInfo:
    def test_collection_info_after_inserts(self, populated_db):
        info = populated_db.collection_info("test")
        assert info.document_count == 4
        assert info.estimated_memory_bytes > 0

    def test_collection_info_repr(self, db_with_collection):
        info = db_with_collection.collection_info("test")
        r = repr(info)
        assert "CollectionInfo" in r
        assert "test" in r
        assert "dimension=3" in r


# ===========================================================================
# Save / Load persistence
# ===========================================================================


class TestPersistence:
    def test_save_and_load(self, db_with_collection):
        db_with_collection.insert("test", "persisted doc", [1.0, 0.0, 0.0], metadata={"key": "val"})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_with_collection.save("test", path=tmpdir)

            # Load into a fresh database
            db2 = vectorsdb.VectorDB()
            db2.load("test", path=tmpdir)

            info = db2.collection_info("test")
            assert info.document_count == 1
            assert info.dimension == 3

            results = db2.search("test", query_embedding=[1.0, 0.0, 0.0], k=1)
            assert results[0].text == "persisted doc"
            assert results[0].metadata == {"key": "val"}

    def test_save_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.save("nope", path="/tmp")

    def test_save_no_path_no_data_dir_raises(self, db_with_collection):
        with pytest.raises(ValueError, match="no path specified"):
            db_with_collection.save("test")

    def test_load_no_path_no_data_dir_raises(self, db):
        with pytest.raises(ValueError, match="no path specified"):
            db.load("test")

    def test_load_nonexistent_file_raises(self, db):
        with pytest.raises(RuntimeError, match="failed to load"):
            db.load("nope", path="/tmp/nonexistent_dir_12345")

    def test_data_dir_used_for_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("myc", dimension=2)
            db1.insert("myc", "hello", [1.0, 0.0])
            db1.save("myc")  # uses data_dir

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            db2.load("myc")  # uses data_dir
            assert db2.collection_info("myc").document_count == 1

    def test_load_replaces_existing(self, db_with_collection):
        db_with_collection.insert("test", "original", [1.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_with_collection.save("test", path=tmpdir)

            # Insert more after saving
            db_with_collection.insert("test", "extra", [0.0, 1.0, 0.0])
            assert db_with_collection.collection_info("test").document_count == 2

            # Load should replace with the saved snapshot (1 doc)
            db_with_collection.load("test", path=tmpdir)
            assert db_with_collection.collection_info("test").document_count == 1


# ===========================================================================
# Rebuild
# ===========================================================================


class TestRebuild:
    def test_rebuild_after_deletes(self, db_with_collection):
        ids = []
        for i in range(10):
            doc_id = db_with_collection.insert("test", f"doc {i}", [float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2)])
            ids.append(doc_id)

        # Delete half
        for doc_id in ids[:5]:
            db_with_collection.delete("test", doc_id)

        assert db_with_collection.collection_info("test").document_count == 5
        count = db_with_collection.rebuild("test")
        assert count == 5
        assert db_with_collection.collection_info("test").document_count == 5

    def test_rebuild_empty_collection(self, db_with_collection):
        count = db_with_collection.rebuild("test")
        assert count == 0

    def test_rebuild_missing_collection_raises(self, db):
        with pytest.raises(KeyError, match="not found"):
            db.rebuild("nope")

    def test_search_after_rebuild(self, db_with_collection):
        db_with_collection.insert("test", "kept", [1.0, 0.0, 0.0])
        doc_id = db_with_collection.insert("test", "deleted", [0.0, 1.0, 0.0])
        db_with_collection.delete("test", doc_id)
        db_with_collection.rebuild("test")

        results = db_with_collection.search("test", query_embedding=[1.0, 0.0, 0.0], k=5)
        assert len(results) == 1
        assert results[0].text == "kept"


# ===========================================================================
# Memory tracking
# ===========================================================================


class TestMemory:
    def test_total_memory_bytes_empty(self, db):
        assert db.total_memory_bytes() == 0

    def test_total_memory_grows_with_data(self, db_with_collection):
        before = db_with_collection.total_memory_bytes()
        for i in range(100):
            db_with_collection.insert("test", f"document {i}", [float(i), 0.0, 0.0])
        after = db_with_collection.total_memory_bytes()
        assert after > before


# ===========================================================================
# WAL crash recovery
# ===========================================================================


class TestWalCrashRecovery:
    def test_wal_recovery_after_crash(self):
        """Insert docs without save, drop db, reopen — data should be recovered from WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("test", dimension=3)
            db1.insert("test", "doc one", [1.0, 0.0, 0.0])
            db1.insert("test", "doc two", [0.0, 1.0, 0.0])
            db1.insert("test", "doc three", [0.0, 0.0, 1.0])
            del db1  # no save — simulate crash

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            assert db2.collection_info("test").document_count == 3
            results = db2.search("test", query_embedding=[1.0, 0.0, 0.0], k=3)
            assert len(results) == 3
            texts = {r.text for r in results}
            assert "doc one" in texts

    def test_wal_recovery_after_save(self):
        """Insert, save, insert more, crash — all data should be recovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("test", dimension=3)
            db1.insert("test", "saved doc", [1.0, 0.0, 0.0])
            db1.save("test")  # snapshot + WAL truncate

            db1.insert("test", "unsaved doc", [0.0, 1.0, 0.0])
            del db1  # crash after unsaved insert

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            assert db2.collection_info("test").document_count == 2
            results = db2.search("test", query_embedding=[0.0, 1.0, 0.0], k=1)
            assert results[0].text == "unsaved doc"

    def test_wal_truncation_on_save(self):
        """After save, WAL should be truncated (small file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("test", dimension=3)
            for i in range(20):
                db1.insert("test", f"doc {i}", [float(i), 0.0, 0.0])

            wal_path = os.path.join(tmpdir, "wal.bin")
            assert os.path.exists(wal_path)
            wal_size_before = os.path.getsize(wal_path)
            assert wal_size_before > 0

            db1.save("test")
            wal_size_after = os.path.getsize(wal_path)
            assert wal_size_after < wal_size_before

    def test_ephemeral_no_wal(self):
        """VectorDB without data_dir should not create WAL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB()
            db1.create_collection("test", dimension=3)
            db1.insert("test", "ephemeral", [1.0, 0.0, 0.0])
            # No files should be created anywhere
            assert not os.path.exists(os.path.join(tmpdir, "wal.bin"))

    def test_wal_collection_operations(self):
        """Create and delete collections are recovered from WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("col_a", dimension=3)
            db1.create_collection("col_b", dimension=5)
            db1.insert("col_a", "hello", [1.0, 0.0, 0.0])
            db1.delete_collection("col_b")
            del db1

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            names = db2.list_collections()
            assert "col_a" in names
            assert "col_b" not in names
            assert db2.collection_info("col_a").document_count == 1

    def test_wal_delete_document_recovery(self):
        """Document deletion is recovered from WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("test", dimension=3)
            doc_id = db1.insert("test", "to be deleted", [1.0, 0.0, 0.0])
            db1.insert("test", "kept", [0.0, 1.0, 0.0])
            db1.delete("test", doc_id)
            del db1

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            assert db2.collection_info("test").document_count == 1
            results = db2.search("test", query_embedding=[0.0, 1.0, 0.0], k=5)
            assert len(results) == 1
            assert results[0].text == "kept"

    def test_wal_batch_insert_recovery(self):
        """Batch insert is recovered from WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("test", dimension=3)
            docs = [
                {"text": f"batch doc {i}", "embedding": [float(i), 0.0, 0.0]}
                for i in range(5)
            ]
            db1.batch_insert("test", docs)
            del db1

            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            assert db2.collection_info("test").document_count == 5

    def test_compact_saves_all_and_truncates(self):
        """compact() saves all collections and truncates WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = vectorsdb.VectorDB(data_dir=tmpdir)
            db1.create_collection("col_a", dimension=3)
            db1.create_collection("col_b", dimension=2)
            db1.insert("col_a", "a doc", [1.0, 0.0, 0.0])
            db1.insert("col_b", "b doc", [1.0, 0.0])
            db1.compact()

            # Snapshots should exist
            assert os.path.exists(os.path.join(tmpdir, "col_a.vdb"))
            assert os.path.exists(os.path.join(tmpdir, "col_b.vdb"))

            # WAL should be truncated
            wal_path = os.path.join(tmpdir, "wal.bin")
            assert os.path.getsize(wal_path) == 0

            # Data survives reopen
            del db1
            db2 = vectorsdb.VectorDB(data_dir=tmpdir)
            assert db2.collection_info("col_a").document_count == 1
            assert db2.collection_info("col_b").document_count == 1


# ===========================================================================
# VectorDB repr
# ===========================================================================


class TestVectorDBRepr:
    def test_repr_empty(self, db):
        assert repr(db) == "VectorDB(collections=0)"

    def test_repr_with_collections(self, db):
        db.create_collection("a", dimension=2)
        db.create_collection("b", dimension=3)
        assert repr(db) == "VectorDB(collections=2)"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_insert_empty_text(self, db_with_collection):
        doc_id = db_with_collection.insert("test", "", [1.0, 0.0, 0.0])
        doc = db_with_collection.get("test", doc_id)
        assert doc["text"] == ""

    def test_search_k_zero(self, populated_db):
        results = populated_db.search("test", query_embedding=[1.0, 0.0, 0.0], k=0)
        assert results == []

    def test_large_batch(self, db_with_collection):
        docs = [
            {"text": f"doc {i}", "embedding": [float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2)]}
            for i in range(500)
        ]
        ids = db_with_collection.batch_insert("test", docs)
        assert len(ids) == 500
        assert db_with_collection.collection_info("test").document_count == 500

    def test_delete_then_get_raises(self, db_with_collection):
        doc_id = db_with_collection.insert("test", "temp", [1.0, 0.0, 0.0])
        db_with_collection.delete("test", doc_id)
        with pytest.raises(KeyError):
            db_with_collection.get("test", doc_id)

    def test_multiple_collections_independent(self, db):
        db.create_collection("a", dimension=2)
        db.create_collection("b", dimension=3)
        db.insert("a", "in a", [1.0, 0.0])
        db.insert("b", "in b", [0.0, 1.0, 0.0])

        assert db.collection_info("a").document_count == 1
        assert db.collection_info("b").document_count == 1
        assert db.collection_info("a").dimension == 2
        assert db.collection_info("b").dimension == 3
