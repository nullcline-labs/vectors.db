"""Type stubs for vectorsdb — in-memory vector database."""

from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Filter types (used by search)
# ---------------------------------------------------------------------------

class _FilterCondition(TypedDict, total=False):
    """A single filter condition.

    Required keys: ``field``, ``op``.
    Provide ``value`` for scalar operators (eq, ne, gt, lt, gte, lte)
    or ``values`` for the ``in`` operator.
    """

    field: str
    op: str  # "eq" | "ne" | "gt" | "lt" | "gte" | "lte" | "in"
    value: bool | int | float | str | None
    values: list[bool | int | float | str] | None

class _FilterClause(TypedDict, total=False):
    """Top-level filter with ``must`` (AND) and ``must_not`` (exclude) lists."""

    must: list[_FilterCondition]
    must_not: list[_FilterCondition]

# Type alias for metadata dicts
_Metadata = dict[str, bool | int | float | str]

# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class SearchResult:
    """A single search result returned by :meth:`VectorDB.search`."""

    @property
    def id(self) -> str:
        """Document UUID."""
        ...
    @property
    def text(self) -> str:
        """Document text content."""
        ...
    @property
    def score(self) -> float:
        """Relevance score (higher is more similar)."""
        ...
    @property
    def metadata(self) -> _Metadata:
        """Document metadata."""
        ...

# ---------------------------------------------------------------------------
# CollectionInfo
# ---------------------------------------------------------------------------

class CollectionInfo:
    """Information about a collection, returned by :meth:`VectorDB.collection_info`."""

    @property
    def name(self) -> str:
        """Collection name."""
        ...
    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...
    @property
    def document_count(self) -> int:
        """Number of documents in the collection."""
        ...
    @property
    def estimated_memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        ...

# ---------------------------------------------------------------------------
# VectorDB
# ---------------------------------------------------------------------------

class VectorDB:
    """In-process vector database with HNSW and BM25 indexing.

    Examples::

        db = VectorDB()                    # ephemeral (in-memory only)
        db = VectorDB(data_dir="./data")   # with WAL persistence
    """

    def __init__(self, data_dir: str | None = None) -> None:
        """Create a new database instance.

        Args:
            data_dir: Optional directory for WAL persistence and snapshots.
                      If ``None``, the database is ephemeral (in-memory only).
        """
        ...

    def create_collection(
        self,
        name: str,
        dimension: int,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
        distance_metric: str | None = None,
        store_raw_vectors: bool | None = None,
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name (must be unique).
            dimension: Embedding vector dimension.
            m: HNSW M parameter — max edges per node (default 16).
            ef_construction: HNSW build-time search width (default 200).
            ef_search: HNSW query-time search width (default 50).
            distance_metric: ``"cosine"``, ``"euclidean"``, or ``"dot_product"``
                             (default ``"cosine"``).
            store_raw_vectors: If ``True``, stores raw f32 vectors for exact
                reranking (+0.7% recall, +59% RAM). Default ``False``
                (compact mode using scalar-quantized u8 vectors only).

        Raises:
            ValueError: If the collection name already exists or the metric is unknown.
        """
        ...

    def delete_collection(self, name: str) -> bool:
        """Delete a collection by name.

        Returns:
            ``True`` if the collection existed and was deleted.
        """
        ...

    def list_collections(self) -> list[str]:
        """List all collection names."""
        ...

    def collection_info(self, name: str) -> CollectionInfo:
        """Get information about a collection.

        Raises:
            KeyError: If the collection does not exist.
        """
        ...

    def insert(
        self,
        collection: str,
        text: str,
        embedding: list[float],
        metadata: _Metadata | None = None,
        id: str | None = None,
    ) -> str:
        """Insert a single document.

        Args:
            collection: Collection name.
            text: Document text content (indexed by BM25).
            embedding: Embedding vector as a list of floats.
            metadata: Optional key-value metadata (values must be bool, int,
                      float, or str).
            id: Optional UUID string. Auto-generated if omitted.

        Returns:
            The document UUID as a string.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If the UUID is malformed or the embedding dimension
                        doesn't match.
        """
        ...

    def batch_insert(
        self,
        collection: str,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple documents at once.

        Each document dict should have keys:

        - ``"text"`` *(str, required)* — document text
        - ``"embedding"`` *(list[float], required)* — embedding vector
        - ``"metadata"`` *(dict, optional)* — key-value metadata
        - ``"id"`` *(str, optional)* — UUID string

        Args:
            collection: Collection name.
            documents: List of document dicts.

        Returns:
            List of UUID strings, one per inserted document.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If a document is missing required keys.
        """
        ...

    def get(self, collection: str, id: str) -> dict[str, Any]:
        """Get a document by UUID.

        Returns:
            A dict with keys ``"id"`` (str), ``"text"`` (str),
            and ``"metadata"`` (dict).

        Raises:
            KeyError: If the collection or document does not exist.
            ValueError: If the UUID is malformed.
        """
        ...

    def delete(self, collection: str, id: str) -> bool:
        """Delete a document by UUID.

        Returns:
            ``True`` if the document existed and was deleted.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If the UUID is malformed.
        """
        ...

    def search(
        self,
        collection: str,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        k: int = 10,
        min_similarity: float | None = None,
        alpha: float = 0.7,
        fusion_method: str = "rrf",
        filter: _FilterClause | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search a collection.

        Supports three modes depending on which query arguments are provided:

        - **Vector search**: pass ``query_embedding`` only.
        - **Keyword search**: pass ``query_text`` only.
        - **Hybrid search**: pass both ``query_embedding`` and ``query_text``.

        Args:
            collection: Collection name.
            query_embedding: Embedding vector for vector/hybrid search.
            query_text: Text query for keyword/hybrid search.
            k: Maximum number of results to return (default 10).
            min_similarity: Optional minimum score threshold.
            alpha: Hybrid search weight — ``1.0`` = all vector,
                   ``0.0`` = all keyword (default ``0.7``).
            fusion_method: ``"rrf"`` (reciprocal rank fusion) or ``"linear"``
                           (default ``"rrf"``).
            filter: Optional filter dict with ``"must"`` and/or ``"must_not"``
                    lists of conditions. Each condition has ``"field"``,
                    ``"op"`` (eq/ne/gt/lt/gte/lte/in), and ``"value"`` or
                    ``"values"``.

        Returns:
            List of :class:`SearchResult` objects, sorted by descending score.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If neither ``query_embedding`` nor ``query_text``
                        is provided.
        """
        ...

    def save(self, collection: str, path: str | None = None, encryption_key: str | None = None) -> None:
        """Save a collection snapshot to disk.

        Args:
            collection: Collection name.
            path: Directory to save to. Falls back to ``data_dir`` from
                  construction.
            encryption_key: Optional 64-character hex string for AES-256-GCM
                  encryption. When provided, the snapshot is encrypted.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If no path and no ``data_dir`` was set, or if
                  encryption_key is invalid.
            RuntimeError: If the write fails.
        """
        ...

    def load(self, name: str, path: str | None = None, encryption_key: str | None = None) -> None:
        """Load a collection snapshot from disk.

        The snapshot file is expected at ``<path>/<name>.vdb``.

        Args:
            name: Collection name (matches the filename stem).
            path: Directory to load from. Falls back to ``data_dir`` from
                  construction.
            encryption_key: Optional 64-character hex string for AES-256-GCM
                  decryption. Required if the snapshot was saved encrypted.

        Raises:
            ValueError: If no path and no ``data_dir`` was set, or if
                  encryption_key is invalid.
            RuntimeError: If the file doesn't exist, is corrupt, or
                  decryption fails (wrong key or missing key for encrypted data).
        """
        ...

    def rebuild(self, collection: str) -> int:
        """Rebuild HNSW and BM25 indices, reclaiming deleted space.

        Returns:
            Number of live documents in the rebuilt index.

        Raises:
            KeyError: If the collection does not exist.
        """
        ...

    def total_memory_bytes(self) -> int:
        """Estimate total memory usage across all collections (bytes)."""
        ...
