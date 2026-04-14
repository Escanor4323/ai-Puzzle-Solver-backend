"""MemGPT-inspired tiered memory management system.

Owns the complete extract → embed → store pipeline:
1. Records conversation turns into immediate memory.
2. Periodically summarises short-term memory via recursive summarisation.
3. Extracts structured facts using Qwen 3 8B (via Ollama/openai client).
4. Embeds text via BGE-M3 and writes to Milvus collections:
   - ``conversation_memories`` for semantic retrieval
   - ``player_observations`` for strategies/reactions/preferences
5. Applies Ebbinghaus forgetting-curve decay to old memories.
6. Builds RAG context for LLM prompt injection.

This module is the single owner of the "turn conversations into
searchable memory" workflow.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Any

from ai.embedding_engine import EmbeddingEngine
from ai.llm_orchestrator import LLMOrchestrator
from config import Settings
from data.knowledge_graph import KnowledgeGraphManager
from data.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages the three-tier player memory architecture.

    Dependencies are injected at construction time — no global singletons.
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_engine: EmbeddingEngine,
        llm_orchestrator: LLMOrchestrator,
        knowledge_graph: KnowledgeGraphManager,
        config: Settings,
    ) -> None:
        self._vs = vector_store
        self._embed = embedding_engine
        self._llm = llm_orchestrator
        self._kg = knowledge_graph
        self._config = config

        # Per-player session state (in-memory, not persisted)
        self._session_summaries: dict[str, str] = {}
        self._turn_buffers: dict[str, list[dict[str, str]]] = {}
        self._turn_counters: dict[str, int] = {}

    # ── RAG Context Builder ───────────────────────────────────

    async def build_context(
        self, player_id: str, current_query: str
    ) -> str:
        """Build the <player_memory> prompt section via RAG retrieval.

        Called before every LLM conversation call.  Embeds the current
        query and searches Milvus in parallel for relevant memories,
        strategies, reactions, and preferences.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        current_query : str
            The player's current message (used as the relevance query).

        Returns
        -------
        str
            Formatted context string for system prompt injection.
        """
        query_vec = self._embed.embed_text(current_query)

        # Parallel Milvus searches
        memories_task = self._vs.search_memories(
            player_id, query_vec, n_results=5
        )
        strategies_task = self._vs.search_observations(
            player_id, query_vec, category="strategy", n_results=3
        )
        reactions_task = self._vs.search_observations(
            player_id, query_vec, category="reaction", n_results=3
        )
        preferences_task = self._vs.search_observations(
            player_id, query_vec, category="preference", n_results=3
        )

        memories, strategies, reactions, preferences = (
            await asyncio.gather(
                memories_task,
                strategies_task,
                reactions_task,
                preferences_task,
            )
        )

        # Knowledge graph facts
        kg_facts = self._kg.get_player_facts(player_id) or []
        kg_facts = kg_facts[:10]

        # Session summary
        summary = self._session_summaries.get(player_id, "")

        # Assemble context
        sections: list[str] = []

        if summary:
            sections.append(f"[This session so far] {summary}")

        if kg_facts:
            fact_lines = [
                f"- {f.get('subject', '?')} "
                f"{f.get('predicate', '?')} "
                f"{f.get('object', '?')}"
                for f in kg_facts
            ]
            sections.append(
                "[Known facts]\n" + "\n".join(fact_lines)
            )

        if memories:
            mem_texts = []
            for m in memories:
                entity = m.get("entity", m)
                text = entity.get("text", "") if isinstance(entity, dict) else ""
                if text:
                    mem_texts.append(text)
            if mem_texts:
                sections.append(
                    "[Relevant past conversations]\n"
                    + "\n".join(f"- {t}" for t in mem_texts)
                )

        if strategies:
            strat_lines = []
            for s in strategies:
                entity = s.get("entity", s)
                desc = entity.get("description", "") if isinstance(entity, dict) else ""
                if desc:
                    strat_lines.append(desc)
            if strat_lines:
                sections.append(
                    "[Player strategies]\n"
                    + "\n".join(f"- {s}" for s in strat_lines)
                )

        if reactions:
            react_lines = []
            for r in reactions:
                entity = r.get("entity", r)
                desc = entity.get("description", "") if isinstance(entity, dict) else ""
                ctx = entity.get("context", "") if isinstance(entity, dict) else ""
                if desc:
                    line = f"{desc} → {ctx}" if ctx else desc
                    react_lines.append(line)
            if react_lines:
                sections.append(
                    "[What works with this player]\n"
                    + "\n".join(f"- {r}" for r in react_lines)
                )

        if preferences:
            pref_lines = []
            for p in preferences:
                entity = p.get("entity", p)
                desc = entity.get("description", "") if isinstance(entity, dict) else ""
                if desc:
                    pref_lines.append(desc)
            if pref_lines:
                sections.append(
                    "[Preferences]\n"
                    + "\n".join(f"- {p}" for p in pref_lines)
                )

        return "\n\n".join(sections) if sections else ""

    # ── Turn Recording ────────────────────────────────────────

    async def record_turn(
        self,
        player_id: str,
        role: str,
        content: str,
        session_id: str = "",
    ) -> None:
        """Record a single conversation turn.

        Buffers turns and triggers background processing when the
        buffer reaches ``MEMORY_SUMMARIZE_EVERY_N_TURNS``.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        role : str
            ``"user"`` or ``"assistant"``.
        content : str
            Message content.
        session_id : str
            Current session ID.
        """
        self._turn_buffers.setdefault(player_id, []).append({
            "role": role,
            "content": content,
        })
        self._turn_counters[player_id] = (
            self._turn_counters.get(player_id, 0) + 1
        )

        interval = self._config.SHORT_TERM_SUMMARY_INTERVAL
        if self._turn_counters[player_id] % interval == 0:
            asyncio.create_task(
                self._process_buffer(player_id, session_id)
            )

    # ── Background Processing ─────────────────────────────────

    async def _process_buffer(
        self, player_id: str, session_id: str
    ) -> None:
        """Process buffered turns: embed, extract, store, summarize.

        Triggered every N turns in the background.
        """
        buffer = self._turn_buffers.get(player_id, [])
        if not buffer:
            return

        # Take last 10 turns
        recent = buffer[-10:]
        chunk = "\n".join(
            f"{t['role']}: {t['content']}" for t in recent
        )

        turn_count = self._turn_counters.get(player_id, 0)

        try:
            # 1. Embed and store conversation memory
            embedding = self._embed.embed_text(chunk[:1990])
            await self._vs.add_memory(
                player_id=player_id,
                embedding=embedding,
                text=chunk[:1990],
                session_id=session_id,
                importance=0.8,
                topic="",
            )

            # 2. Extract facts via Ollama
            extraction = await self._llm.extract_facts_json(
                chunk, player_id
            )

            if extraction:
                # Store observations from facts
                for fact in extraction.get("facts", []):
                    subject = fact.get("subject", "player")
                    predicate = fact.get("predicate", "")
                    obj = fact.get("object", "")
                    category = fact.get("category", "preference")
                    confidence = fact.get("confidence", 0.7)

                    desc = f"{subject} {predicate} {obj}"
                    obs_embedding = self._embed.embed_text(desc)

                    await self._vs.add_observation(
                        player_id=player_id,
                        embedding=obs_embedding,
                        description=desc,
                        category=category,
                        context="",
                        valence=confidence,
                    )

                    # Also add to knowledge graph
                    self._kg.add_fact(
                        player_id,
                        subject,
                        predicate,
                        obj,
                        confidence=confidence,
                    )

                # Store strategy observations
                for strategy in extraction.get(
                    "strategy_observations", []
                ):
                    strat_embedding = self._embed.embed_text(strategy)
                    await self._vs.add_observation(
                        player_id=player_id,
                        embedding=strat_embedding,
                        description=strategy,
                        category="strategy",
                        context="",
                        valence=0.0,
                    )

            # 3. Update session summary (recursive)
            prev_summary = self._session_summaries.get(
                player_id, ""
            )
            new_summary = await self._llm.summarize_session(
                prev_summary, chunk
            )
            if new_summary:
                self._session_summaries[player_id] = new_summary

            # 4. Save knowledge graph
            self._kg.save(
                player_id,
                str(
                    self._config.KNOWLEDGE_DIR
                    / f"{player_id}.json"
                ),
            )

            logger.debug(
                "Processed buffer for %s: turns %d-%d",
                player_id,
                max(1, turn_count - 5),
                turn_count,
            )

        except Exception as e:
            logger.warning(
                "Buffer processing failed for %s: %s",
                player_id,
                e,
            )

    # ── Immediate Memory (for important events) ─────────────

    async def store_game_event(
        self,
        player_id: str,
        event_text: str,
        importance: float = 0.9,
        category: str = "reaction",
    ) -> None:
        """Immediately store an important game event as a memory.

        Bypasses the buffer — used for puzzle solves, timeouts, etc.
        These are high-importance memories that should persist.
        """
        try:
            embedding = self._embed.embed_text(event_text[:2000])
            await self._vs.add_memory(
                player_id=player_id,
                embedding=embedding,
                text=event_text[:2000],
                session_id="",
                importance=importance,
                topic="game_event",
            )
            # Also store as an observation
            await self._vs.add_observation(
                player_id=player_id,
                embedding=embedding,
                description=event_text[:500],
                category=category,
                context="game_event",
                valence=importance,
            )
            logger.debug("Stored game event for %s: %s", player_id, event_text[:80])
        except Exception as e:
            logger.warning("Failed to store game event: %s", e)

    # ── Forgetting Curve ──────────────────────────────────────

    async def apply_forgetting_curve(
        self, player_id: str
    ) -> int:
        """Decay importance scores using the Ebbinghaus formula.

        R = importance × e^(-λ_eff × days) × (1 + recall_count × 0.2)
        where λ_eff = 0.16 × (1 - importance × 0.8)

        Memories decayed below 0.1 are deleted.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        int
            Number of memories deleted.
        """
        lambda_base = self._config.FORGETTING_LAMBDA
        now = time.time()

        # Use the vector store's decay method as a simplified version
        # For fine-grained Ebbinghaus, we'd query individual memories,
        # compute per-memory decay, and upsert. The vector store's
        # decay_memories handles the basic multiplicative decay.
        try:
            # Compute a simple decay factor based on time since
            # last session (approximate 1 day = 86400s)
            decay_factor = math.exp(-lambda_base)
            deleted = await self._vs.decay_memories(
                player_id,
                decay_factor=decay_factor,
                min_importance=0.1,
            )
            logger.info(
                "Forgetting curve applied for %s: %d memories deleted",
                player_id,
                deleted,
            )
            return deleted
        except Exception as e:
            logger.warning(
                "Forgetting curve failed for %s: %s",
                player_id,
                e,
            )
            return 0

    # ── Session Lifecycle ─────────────────────────────────────

    def on_session_end(
        self, player_id: str
    ) -> str | None:
        """Finalize session. Returns summary for storage.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        str | None
            Session summary, or None if none generated.
        """
        summary = self._session_summaries.pop(player_id, None)
        self._turn_buffers.pop(player_id, None)
        self._turn_counters.pop(player_id, None)

        # Fire and forget: apply forgetting curve
        asyncio.create_task(
            self.apply_forgetting_curve(player_id)
        )

        # Save knowledge graph
        self._kg.save(
            player_id,
            str(
                self._config.KNOWLEDGE_DIR
                / f"{player_id}.json"
            ),
        )

        return summary
