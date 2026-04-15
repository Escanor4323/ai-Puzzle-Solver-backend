"""Milvus Lite vector store with 4 typed collections.

Replaces ChromaDB.  Uses ``pymilvus`` in Lite mode — a single local
file, no Docker, no server.  Shares the same API surface as Milvus
Standalone / Distributed, so switching to a server deployment later
requires changing one connection URI.

Collections
-----------
1. ``conversation_memories`` — chunked conversation history
2. ``player_observations``   — strategies, reactions, preferences, personality
3. ``jailbreak_patterns``    — detected attack embeddings for similarity matching
4. ``puzzle_templates``      — generated puzzles for reuse / deduplication
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

logger = logging.getLogger(__name__)

# ── Dimension must match the embedding model (BGE-M3 = 1024) ───
EMBEDDING_DIM = 1024


# ── Schema Definitions ─────────────────────────────────────────


def _conversation_memories_schema() -> CollectionSchema:
    """Schema for chunked conversation history."""
    fields = [
        FieldSchema(
            "id", DataType.INT64,
            is_primary=True, auto_id=True,
        ),
        FieldSchema(
            "player_id", DataType.VARCHAR, max_length=64,
        ),
        FieldSchema(
            "embedding", DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
        FieldSchema(
            "text", DataType.VARCHAR, max_length=4096,
        ),
        FieldSchema("timestamp", DataType.INT64),
        FieldSchema(
            "session_id", DataType.VARCHAR, max_length=64,
        ),
        FieldSchema("importance", DataType.FLOAT),
        FieldSchema(
            "topic", DataType.VARCHAR, max_length=100,
        ),
    ]
    return CollectionSchema(
        fields,
        description="Chunked conversation history for "
        "semantic memory retrieval",
    )


def _player_observations_schema() -> CollectionSchema:
    """Schema for merged player observations.

    Covers strategies, reactions, preferences, and personality
    traits — differentiated by the ``category`` field.
    """
    fields = [
        FieldSchema(
            "id", DataType.INT64,
            is_primary=True, auto_id=True,
        ),
        FieldSchema(
            "player_id", DataType.VARCHAR, max_length=64,
        ),
        FieldSchema(
            "embedding", DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
        FieldSchema(
            "description", DataType.VARCHAR, max_length=500,
        ),
        FieldSchema(
            "category", DataType.VARCHAR, max_length=32,
        ),
        FieldSchema(
            "context", DataType.VARCHAR, max_length=200,
        ),
        FieldSchema("valence", DataType.FLOAT),
        FieldSchema("frequency", DataType.INT32),
        FieldSchema("first_seen", DataType.INT64),
        FieldSchema("last_seen", DataType.INT64),
    ]
    return CollectionSchema(
        fields,
        description="Player observations: strategies, "
        "reactions, preferences, personality",
    )


def _jailbreak_patterns_schema() -> CollectionSchema:
    """Schema for detected jailbreak attack patterns."""
    fields = [
        FieldSchema(
            "id", DataType.INT64,
            is_primary=True, auto_id=True,
        ),
        FieldSchema(
            "player_id", DataType.VARCHAR, max_length=64,
        ),
        FieldSchema(
            "embedding", DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
        FieldSchema(
            "input_text", DataType.VARCHAR, max_length=4096,
        ),
        FieldSchema(
            "category", DataType.VARCHAR, max_length=32,
        ),
        FieldSchema("severity", DataType.FLOAT),
        FieldSchema("timestamp", DataType.INT64),
    ]
    return CollectionSchema(
        fields,
        description="Jailbreak attack embeddings for "
        "similarity detection",
    )


def _puzzle_templates_schema() -> CollectionSchema:
    """Schema for generated puzzle templates."""
    fields = [
        FieldSchema(
            "id", DataType.INT64,
            is_primary=True, auto_id=True,
        ),
        FieldSchema(
            "embedding", DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
        FieldSchema(
            "puzzle_type", DataType.VARCHAR, max_length=32,
        ),
        FieldSchema(
            "prompt", DataType.VARCHAR, max_length=4096,
        ),
        FieldSchema(
            "solution", DataType.VARCHAR, max_length=200,
        ),
        FieldSchema("difficulty", DataType.INT32),
        FieldSchema("times_used", DataType.INT32),
        FieldSchema("avg_solve_time", DataType.FLOAT),
        FieldSchema("success_rate", DataType.FLOAT),
    ]
    return CollectionSchema(
        fields,
        description="Generated puzzles for reuse and "
        "deduplication",
    )


# ── Collection Names ───────────────────────────────────────────

CONVERSATION_MEMORIES = "conversation_memories"
PLAYER_OBSERVATIONS = "player_observations"
JAILBREAK_PATTERNS = "jailbreak_patterns"
PUZZLE_TEMPLATES = "puzzle_templates"

_COLLECTION_SCHEMAS: dict[
    str, callable
] = {
    CONVERSATION_MEMORIES: _conversation_memories_schema,
    PLAYER_OBSERVATIONS: _player_observations_schema,
    JAILBREAK_PATTERNS: _jailbreak_patterns_schema,
    PUZZLE_TEMPLATES: _puzzle_templates_schema,
}


# ── MilvusVectorStore ──────────────────────────────────────────


class MilvusVectorStore:
    """Manages a Milvus Lite database with typed collections.

    Parameters
    ----------
    db_path : str
        Path to the Milvus Lite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._client: MilvusClient | None = None

    async def initialize(self) -> None:
        """Open the Milvus Lite database and create collections.

        Each collection gets an AUTOINDEX on the embedding field
        with COSINE metric for semantic search.
        """
        self._client = MilvusClient(uri=self._db_path)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        for name, schema_fn in _COLLECTION_SCHEMAS.items():
            if not self._client.has_collection(name):
                self._client.create_collection(
                    collection_name=name,
                    schema=schema_fn(),
                    index_params=index_params,
                )
                logger.info("Created Milvus collection: %s", name)
            else:
                logger.debug(
                    "Collection already exists: %s", name
                )

    def list_collections(self) -> list[str]:
        """Return the names of all collections."""
        if self._client is None:
            return []
        return self._client.list_collections()

    # ── Conversation Memories ──────────────────────────────

    async def add_memory(
        self,
        player_id: str,
        embedding: list[float],
        text: str,
        session_id: str = "",
        importance: float = 0.5,
        topic: str = "",
    ) -> None:
        """Insert a conversation chunk into the memories collection.

        Parameters
        ----------
        player_id : str
            Owner of this memory.
        embedding : list[float]
            1024-dim BGE-M3 embedding.
        text : str
            The conversation chunk text.
        session_id : str
            Current session identifier.
        importance : float
            Importance score (0.0–1.0).
        topic : str
            Extracted topic category.
        """
        self._client.insert(
            collection_name=CONVERSATION_MEMORIES,
            data=[{
                "player_id": player_id,
                "embedding": embedding,
                "text": text,
                "timestamp": int(time.time()),
                "session_id": session_id,
                "importance": importance,
                "topic": topic,
            }],
        )

    async def search_memories(
        self,
        player_id: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search over a player's conversation memories.

        Parameters
        ----------
        player_id : str
            Player whose memories to search.
        query_embedding : list[float]
            Embedding of the search query.
        n_results : int
            Maximum results.

        Returns
        -------
        list[dict[str, Any]]
            Matching memories sorted by relevance.
        """
        results = self._client.search(
            collection_name=CONVERSATION_MEMORIES,
            data=[query_embedding],
            limit=n_results,
            filter=f'player_id == "{player_id}"',
            output_fields=[
                "text", "timestamp", "importance", "topic",
            ],
        )
        return results[0] if results else []

    async def decay_memories(
        self,
        player_id: str,
        decay_factor: float,
        min_importance: float = 0.05,
    ) -> int:
        """Reduce importance scores for a player's memories.

        Parameters
        ----------
        player_id : str
            Player whose memories to decay.
        decay_factor : float
            Multiplicative decay (e.g. 0.95).
        min_importance : float
            Delete memories below this threshold.

        Returns
        -------
        int
            Number of memories deleted.
        """
        # Fetch all memories for the player
        results = self._client.query(
            collection_name=CONVERSATION_MEMORIES,
            filter=f'player_id == "{player_id}"',
            output_fields=["id", "importance"],
        )
        deleted = 0
        for row in results:
            new_importance = row["importance"] * decay_factor
            if new_importance < min_importance:
                self._client.delete(
                    collection_name=CONVERSATION_MEMORIES,
                    ids=[row["id"]],
                )
                deleted += 1
            # Note: Milvus Lite doesn't support in-place
            # updates of scalar fields efficiently.
            # Full implementation would delete+reinsert.
        return deleted

    # ── Player Observations ────────────────────────────────

    async def add_observation(
        self,
        player_id: str,
        embedding: list[float],
        description: str,
        category: str,
        context: str = "",
        valence: float = 0.0,
    ) -> None:
        """Insert a player observation (strategy/reaction/etc.).

        Parameters
        ----------
        player_id : str
            Owner of this observation.
        embedding : list[float]
            1024-dim embedding of the description.
        description : str
            Human-readable observation text.
        category : str
            One of: strategy, reaction, preference, personality.
        context : str
            Situational context (puzzle type, trigger, etc.).
        valence : float
            Sentiment (-1.0 negative to 1.0 positive).
        """
        now = int(time.time())
        self._client.insert(
            collection_name=PLAYER_OBSERVATIONS,
            data=[{
                "player_id": player_id,
                "embedding": embedding,
                "description": description,
                "category": category,
                "context": context,
                "valence": valence,
                "frequency": 1,
                "first_seen": now,
                "last_seen": now,
            }],
        )

    async def search_observations(
        self,
        player_id: str,
        query_embedding: list[float],
        category: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search player observations with optional category filter.

        Parameters
        ----------
        player_id : str
            Player to search.
        query_embedding : list[float]
            Query embedding.
        category : str | None
            Optional category filter (strategy/reaction/etc.).
        n_results : int
            Maximum results.

        Returns
        -------
        list[dict[str, Any]]
            Matching observations.
        """
        filter_expr = f'player_id == "{player_id}"'
        if category:
            filter_expr += f' and category == "{category}"'

        results = self._client.search(
            collection_name=PLAYER_OBSERVATIONS,
            data=[query_embedding],
            limit=n_results,
            filter=filter_expr,
            output_fields=[
                "description", "category", "context",
                "valence", "frequency", "last_seen",
            ],
        )
        return results[0] if results else []

    # ── Jailbreak Patterns ─────────────────────────────────

    async def add_jailbreak_pattern(
        self,
        player_id: str,
        embedding: list[float],
        input_text: str,
        category: str,
        severity: float,
    ) -> None:
        """Store a detected jailbreak attempt embedding.

        Parameters
        ----------
        player_id : str
            Player who sent the attack.
        embedding : list[float]
            Embedding of the attack text.
        input_text : str
            Raw attack input.
        category : str
            Attack category (roleplay, encoding, etc.).
        severity : float
            Severity score (0.0–1.0).
        """
        self._client.insert(
            collection_name=JAILBREAK_PATTERNS,
            data=[{
                "player_id": player_id,
                "embedding": embedding,
                "input_text": input_text,
                "category": category,
                "severity": severity,
                "timestamp": int(time.time()),
            }],
        )

    async def search_similar_attacks(
        self,
        query_embedding: list[float],
        threshold: float = 0.85,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Find similar jailbreak patterns across all players.

        Parameters
        ----------
        query_embedding : list[float]
            Embedding of the new input to check.
        threshold : float
            Minimum similarity to flag (cosine).
        n_results : int
            Maximum results.

        Returns
        -------
        list[dict[str, Any]]
            Similar attacks above the threshold.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        def _search() -> list:
            return self._client.search(
                collection_name=JAILBREAK_PATTERNS,
                data=[query_embedding],
                limit=n_results,
                output_fields=[
                    "input_text", "category", "severity",
                ],
            )

        results = await loop.run_in_executor(None, _search)
        if not results:
            return []
        # Filter by threshold (Milvus returns distance,
        # lower = more similar for L2, higher for IP)
        return [
            r for r in results[0]
            if r.get("distance", 0) >= threshold
        ]

    # ── Puzzle Templates ───────────────────────────────────

    async def add_puzzle_template(
        self,
        embedding: list[float],
        puzzle_type: str,
        prompt: str,
        solution: str,
        difficulty: int = 1200,
    ) -> None:
        """Store a generated puzzle for reuse.

        Parameters
        ----------
        embedding : list[float]
            Embedding of the puzzle prompt.
        puzzle_type : str
            Puzzle category.
        prompt : str
            Puzzle prompt text.
        solution : str
            Puzzle answer.
        difficulty : int
            Elo-equivalent difficulty rating.
        """
        self._client.insert(
            collection_name=PUZZLE_TEMPLATES,
            data=[{
                "embedding": embedding,
                "puzzle_type": puzzle_type,
                "prompt": prompt,
                "solution": solution,
                "difficulty": difficulty,
                "times_used": 0,
                "avg_solve_time": 0.0,
                "success_rate": 0.0,
            }],
        )

    async def find_similar_puzzles(
        self,
        query_embedding: list[float],
        puzzle_type: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find puzzles similar to a candidate (deduplication).

        Parameters
        ----------
        query_embedding : list[float]
            Embedding of the candidate puzzle.
        puzzle_type : str | None
            Optional type filter.
        n_results : int
            Maximum results.

        Returns
        -------
        list[dict[str, Any]]
            Similar puzzles.
        """
        filter_expr = ""
        if puzzle_type:
            filter_expr = f'puzzle_type == "{puzzle_type}"'

        results = self._client.search(
            collection_name=PUZZLE_TEMPLATES,
            data=[query_embedding],
            limit=n_results,
            filter=filter_expr if filter_expr else None,
            output_fields=[
                "puzzle_type", "prompt", "difficulty",
                "success_rate",
            ],
        )
        return results[0] if results else []

    # ── Cleanup ────────────────────────────────────────────

    async def delete_player_data(
        self, player_id: str
    ) -> dict[str, int]:
        """Delete all data for a player across collections.

        Parameters
        ----------
        player_id : str
            Player to delete.

        Returns
        -------
        dict[str, int]
            Number of records deleted per collection.
        """
        deleted: dict[str, int] = {}
        for collection in [
            CONVERSATION_MEMORIES,
            PLAYER_OBSERVATIONS,
            JAILBREAK_PATTERNS,
        ]:
            result = self._client.delete(
                collection_name=collection,
                filter=f'player_id == "{player_id}"',
            )
            deleted[collection] = (
                len(result) if isinstance(result, list)
                else 0
            )
        return deleted

    async def close(self) -> None:
        """Close the Milvus client."""
        if self._client:
            self._client.close()
            self._client = None
