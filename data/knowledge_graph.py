"""NetworkX-backed player knowledge graph.

Stores player-specific facts as triples ``(subject, predicate, object)``
with temporal annotations and confidence scores.  Persists as JSON files
on disk so facts survive across sessions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """Manages per-player knowledge graphs using NetworkX."""

    def __init__(self) -> None:
        self._graphs: dict[str, nx.DiGraph] = {}

    def _get_or_create(self, player_id: str) -> nx.DiGraph:
        """Return the graph for a player, creating if needed."""
        if player_id not in self._graphs:
            self._graphs[player_id] = nx.DiGraph()
        return self._graphs[player_id]

    def load(self, player_id: str, path: str) -> None:
        """Load a player's knowledge graph from disk."""
        p = Path(path)
        if not p.is_file():
            self._graphs[player_id] = nx.DiGraph()
            return

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            g = nx.DiGraph()
            for fact in data.get("facts", []):
                subj = fact["subject"]
                obj = fact["object"]
                g.add_edge(
                    subj, obj,
                    predicate=fact["predicate"],
                    confidence=fact.get("confidence", 1.0),
                    first_seen=fact.get("first_seen", 0),
                    last_seen=fact.get("last_seen", 0),
                    recall_count=fact.get("recall_count", 0),
                )
            self._graphs[player_id] = g
            logger.debug(
                "Loaded %d facts for %s", g.number_of_edges(), player_id
            )
        except Exception as e:
            logger.warning("Failed to load KG for %s: %s", player_id, e)
            self._graphs[player_id] = nx.DiGraph()

    def save(self, player_id: str, path: str) -> None:
        """Persist a player's knowledge graph to disk."""
        g = self._graphs.get(player_id)
        if not g or g.number_of_edges() == 0:
            return

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        facts = []
        for subj, obj, data in g.edges(data=True):
            facts.append({
                "subject": subj,
                "predicate": data.get("predicate", ""),
                "object": obj,
                "confidence": data.get("confidence", 1.0),
                "first_seen": data.get("first_seen", 0),
                "last_seen": data.get("last_seen", 0),
                "recall_count": data.get("recall_count", 0),
            })

        p.write_text(
            json.dumps({"facts": facts}, indent=2),
            encoding="utf-8",
        )
        logger.debug("Saved %d facts for %s", len(facts), player_id)

    def add_fact(
        self,
        player_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a fact triple to the knowledge graph.

        If the edge already exists, update confidence and last_seen.
        """
        g = self._get_or_create(player_id)
        now = int(time.time())

        if g.has_edge(subject, obj):
            edge = g.edges[subject, obj]
            # Update existing fact
            old_conf = edge.get("confidence", 0.5)
            edge["confidence"] = min(1.0, (old_conf + confidence) / 2 + 0.1)
            edge["last_seen"] = now
            edge["recall_count"] = edge.get("recall_count", 0) + 1
        else:
            g.add_edge(
                subject, obj,
                predicate=predicate,
                confidence=confidence,
                first_seen=now,
                last_seen=now,
                recall_count=0,
            )

    def get_player_facts(
        self, player_id: str
    ) -> list[dict[str, Any]]:
        """Return all known facts about a player, sorted by confidence."""
        g = self._graphs.get(player_id)
        if not g:
            return []

        facts = []
        for subj, obj, data in g.edges(data=True):
            facts.append({
                "subject": subj,
                "predicate": data.get("predicate", ""),
                "object": obj,
                "confidence": data.get("confidence", 1.0),
                "last_seen": data.get("last_seen", 0),
                "recall_count": data.get("recall_count", 0),
            })

        # Sort by confidence descending, then by recency
        facts.sort(
            key=lambda f: (f["confidence"], f["last_seen"]),
            reverse=True,
        )
        return facts

    def remove_player_data(self, player_id: str) -> None:
        """Delete all graph data for a player."""
        self._graphs.pop(player_id, None)
