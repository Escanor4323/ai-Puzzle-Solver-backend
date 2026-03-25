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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from events.bus import EventBus


def register_all_handlers(bus: "EventBus") -> None:
    """Register all event handlers on the bus.

    Called once during application startup after all managers
    have been initialised.

    Parameters
    ----------
    bus : EventBus
        The application event bus instance.
    """
    # ── Pre-processing (parallel on every message) ──────
    # bus.on("chat:message", jailbreak_detector.check_input)
    # bus.on("chat:message", emotion_analyzer.analyze_text)
    # bus.on("chat:message", jailbreak_detector.similarity_precheck)

    # ── Face Engine Events ──────────────────────────────
    # bus.on("camera:frame", face_engine.process_frame)

    # ── Session Manager Events ──────────────────────────
    # bus.on("face:recognized", session_manager.on_recognized)
    # bus.on("face:new", session_manager.on_new_player)
    # bus.on("face:lost", session_manager.on_face_lost)

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

    pass  # Wiring happens during Phase 2
