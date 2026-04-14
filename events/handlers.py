"""Event handler registration for the PuzzleMind event bus.

Wires module-level handlers to the central event bus.  The handler
registrations below reflect the parallel RAG pipeline flow:

Player message arrives
  ├─→ [parallel] DeBERTa jailbreak (8ms) + DistilBERT sentiment (5ms)
  │              + Milvus similarity pre-check (10ms)
  ├─→ Rules-based intent classification (in orchestrator)
  ├─→ [parallel] Milvus: retrieve memories + observations (30ms)
  ├─→ Build system prompt with RAG context
  └─→ Claude Sonnet: stream response (800ms first token)

Background (after response):
  ├─→ Qwen 3 8B: extract facts from last 5 turns (~2s)
  ├─→ Embed + store in Milvus collections
  └─→ Apply Ebbinghaus decay to old memories
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from events.bus import EventBus

logger = logging.getLogger(__name__)


# ── Face Event Handlers ──────────────────────────────────────


async def on_face_recognized(data: dict[str, Any]) -> None:
    """Handle face recognition event.

    Starts or resumes a player session when a face is recognized.
    Updates last_seen timestamp and loads game state.

    Parameters
    ----------
    data : dict
        {"player_id": str, "player_name": str, "confidence": float}
    """
    player_id = data.get("player_id", "")
    player_name = data.get("player_name", "Unknown")
    logger.info(
        "Face recognized: %s (%s)", player_name, player_id
    )
    # In full wiring (Phase 2):
    # await session_manager.start_session(player_id)
    # await player_manager.update_last_seen(player_id)
    # await game_engine.load_or_create_state(player_id)
    # Send session:greeting via WebSocket


async def on_face_lost(data: dict[str, Any]) -> None:
    """Handle face lost event.

    Ends the current session when the player leaves the camera.

    Parameters
    ----------
    data : dict
        Empty dict or {"player_id": str}
    """
    logger.info("Face lost — ending session")
    # In full wiring (Phase 2):
    # await memory_manager.on_session_end(session_id)
    # await session_manager.end_session(session_id)
    # Send session:ended via WebSocket


async def on_face_emotion(data: dict[str, Any]) -> None:
    """Handle face emotion event.

    Records facial emotion in the emotion analyzer for composite
    emotional state tracking.

    Parameters
    ----------
    data : dict
        {"player_id": str, "emotion": str}
    """
    player_id = data.get("player_id", "")
    emotion = data.get("emotion", "neutral")
    logger.debug(
        "Face emotion for %s: %s", player_id, emotion
    )
    # In full wiring (Phase 2):
    # emotion_analyzer.record_face_emotion(player_id, emotion, 1.0)


def register_all_handlers(bus: "EventBus") -> None:
    """Register all event handlers on the bus.

    Called once during application startup after all managers
    have been initialised.

    Parameters
    ----------
    bus : EventBus
        The application event bus instance.
    """
    # ── Face Engine Events ──────────────────────────────
    bus.on("face:recognized", on_face_recognized)
    bus.on("face:lost", on_face_lost)
    bus.on("face:emotion", on_face_emotion)

    # ── Pre-processing (parallel on every message) ──────
    # bus.on("chat:message", jailbreak_detector.check_input)
    # bus.on("chat:message", emotion_analyzer.analyze_text)
    # bus.on("chat:message", jailbreak_detector.similarity_precheck)

    # ── Session Manager Events ──────────────────────────
    # (face:recognized and face:lost handlers above replace these)

    # ── LLM Orchestrator Events ─────────────────────────
    # bus.on("chat:message", llm_orchestrator.on_message)
    #   1. classify_intent() — rules-based, internal
    #   2. memory_manager.build_context() — Milvus RAG
    #   3. build_system_prompt() — inject RAG context
    #   4. stream_response() — Claude Sonnet

    # ── Memory Manager Events (background, post-response) ─
    # bus.on("llm:complete", memory_manager.record_turn)
    # bus.on("llm:complete", memory_manager.extract_and_store)
    # bus.on("session:ended", memory_manager.on_session_end)
    # bus.on("session:ended", memory_manager.apply_forgetting)

    # ── Game Engine Events ─────────────────────────────
    # bus.on("chat:message", game_engine.on_player_action)
    # bus.on("session:started", game_engine.on_session)

    # ── Puzzle Generation (GPT-4o) ─────────────────────
    # bus.on("game:needs_puzzle", llm_orchestrator.generate_puzzle_json)

    # ── Elo System Events ──────────────────────────────
    # bus.on("game:puzzle_complete", elo_system.on_complete)
