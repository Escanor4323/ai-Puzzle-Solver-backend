"""Event handler registration for the PuzzleMind event bus.

Wires module-level handlers to the central event bus.  Each handler
responds to a specific event and delegates to the appropriate manager.
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
    # ── Face Engine Events ──────────────────────────────
    # bus.on("camera:frame", face_engine.process_frame)

    # ── Session Manager Events ──────────────────────────
    # bus.on("face:recognized", session_manager.on_recognized)
    # bus.on("face:new", session_manager.on_new_player)
    # bus.on("face:lost", session_manager.on_face_lost)

    # ── LLM Orchestrator Events ─────────────────────────
    # bus.on("chat:message", llm_orchestrator.on_message)
    # bus.on("session:started", llm_orchestrator.on_session)
    # bus.on("game:state_change", llm_orchestrator.on_state)

    # ── Memory Manager Events ──────────────────────────
    # bus.on("llm:complete", memory_manager.on_response)
    # bus.on("session:ended", memory_manager.on_session_end)

    # ── Game Engine Events ─────────────────────────────
    # bus.on("chat:message", game_engine.on_player_action)
    # bus.on("session:started", game_engine.on_session)

    # ── Elo System Events ──────────────────────────────
    # bus.on("game:puzzle_complete", elo_system.on_complete)

    # ── Jailbreak Detector Events ──────────────────────
    # bus.on("chat:message", jailbreak_detector.on_message)

    pass  # Wiring happens during Phase 2
