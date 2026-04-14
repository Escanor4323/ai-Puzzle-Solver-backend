"""PuzzleMind backend entry point.

FastAPI application with lifespan-managed startup that initializes
all subsystems in dependency order:
    0. Security checks (hardware lock, integrity, anti-tamper)
    1. Config
    2. LLM Orchestrator
    3. Embedding Engine (BGE-M3)
    4. Milvus Vector Store
    5. Knowledge Graph
    6. Emotion Analyzer (DistilBERT SST-2)
    7. Jailbreak Detector (DeBERTa + Milvus similarity)
    8. Memory Manager (Milvus RAG + Ollama extraction)
    9. Game Engine
   10. Message Pipeline (wires everything together)
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

# Suppress tokenizer parallelism warning when forking for face processing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from ws.router import (
    ConnectionManager,
    EmotionTracker,
    cancel_ai_race,
    fire_event_comment,
    route_message,
    route_message_async,
    start_ai_race_for_maze,
)

logger = logging.getLogger("puzzlemind")


def _run_security_checks() -> None:
    """Run the security check chain (hardware lock + integrity + anti-tamper).

    Called before any subsystem initialization.  Each layer exits
    silently (code 1) on failure — no error messages, no stack traces.

    In development (``HARDWARE_LOCK_ENABLED=false``), all checks are
    skipped entirely.
    """
    if not settings.HARDWARE_LOCK_ENABLED:
        logger.debug("Security checks disabled (dev mode)")
        return

    # Layer 1: Hardware fingerprint verification
    from security.hardware_lock import enforce_hardware_lock
    enforce_hardware_lock()

    # Layer 2: File integrity (SHA-256 manifest)
    from security.integrity import verify_integrity
    if not verify_integrity():
        sys.exit(1)

    # Layer 3: Anti-tamper (debugger, environment, timing)
    from security.anti_tamper import enforce_anti_tamper
    enforce_anti_tamper()

    logger.info("[PuzzleMind] Security checks passed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all subsystems at startup, clean up at shutdown."""

    # ── Step 0: Security checks ─────────────────────────────
    _run_security_checks()

    logger.info("[PuzzleMind] Starting initialization...")

    # ── Step 1: LLM Orchestrator ──────────────────────────────
    from ai.llm_orchestrator import LLMOrchestrator

    orchestrator = LLMOrchestrator(settings)
    await orchestrator.initialize()
    app.state.llm_orchestrator = orchestrator
    logger.info("[PuzzleMind] LLM Orchestrator initialized")

    # ── Step 2: Embedding Engine (BGE-M3) ─────────────────────
    from ai.embedding_engine import EmbeddingEngine

    embedding_engine = EmbeddingEngine()
    app.state.embedding_engine = embedding_engine
    logger.info("[PuzzleMind] Embedding engine loaded")

    # ── Step 3: Milvus Vector Store ───────────────────────────
    from data.vector_store import MilvusVectorStore

    vector_store = MilvusVectorStore(settings.MILVUS_DB_PATH)
    await vector_store.initialize()
    app.state.vector_store = vector_store
    logger.info("[PuzzleMind] Milvus initialized")

    # ── Step 4: Knowledge Graph ───────────────────────────────
    from data.knowledge_graph import KnowledgeGraphManager

    knowledge_graph = KnowledgeGraphManager()
    app.state.knowledge_graph = knowledge_graph
    logger.info("[PuzzleMind] Knowledge graph ready")

    # ── Step 5: Emotion Analyzer ──────────────────────────────
    from ai.emotion_analyzer import EmotionAnalyzer

    emotion_analyzer = EmotionAnalyzer()
    app.state.emotion_analyzer = emotion_analyzer
    logger.info("[PuzzleMind] Emotion analyzer ready")

    # ── Step 6: Jailbreak Detector ────────────────────────────
    from ai.jailbreak_detector import JailbreakDetector

    jailbreak_detector = JailbreakDetector(
        vector_store, embedding_engine, settings
    )
    app.state.jailbreak_detector = jailbreak_detector
    logger.info("[PuzzleMind] Jailbreak detector ready")

    # ── Step 7: Memory Manager ────────────────────────────────
    from ai.memory_manager import MemoryManager

    memory_manager = MemoryManager(
        vector_store,
        embedding_engine,
        orchestrator,
        knowledge_graph,
        settings,
    )
    app.state.memory_manager = memory_manager
    logger.info("[PuzzleMind] Memory manager ready")

    # ── Step 8: Face Processing (emotion-only, no enrollment DB) ──
    from concurrent.futures import ProcessPoolExecutor
    from ai.face_engine import FaceProcessor, FaceSessionManager, PlayerMatcher

    face_executor = ProcessPoolExecutor(max_workers=1)
    face_processor = FaceProcessor()
    face_session_manager = FaceSessionManager()
    player_matcher = PlayerMatcher()

    app.state.executor = face_executor
    app.state.face_processor = face_processor
    app.state.face_session_manager = face_session_manager
    app.state.player_matcher = player_matcher
    logger.info("[PuzzleMind] Face processing ready (emotion-only mode)")

    # ── Step 9: Game Engine ───────────────────────────────────
    from game.engine import GameEngine

    game_engine = GameEngine()
    app.state.game_engine = game_engine
    logger.info("[PuzzleMind] Game engine ready")

    # ── Step 9: Message Pipeline ──────────────────────────────
    from ai.message_pipeline import MessagePipeline

    message_pipeline = MessagePipeline(
        config=settings,
        orchestrator=orchestrator,
        game_engine=game_engine,
        jailbreak_detector=jailbreak_detector,
        emotion_analyzer=emotion_analyzer,
        memory_manager=memory_manager,
    )
    app.state.message_pipeline = message_pipeline
    logger.info("[PuzzleMind] Message pipeline ready")

    # ── Step 10: Connection Manager & Emotion Tracker ─────────
    app.state.connection_manager = ConnectionManager()
    app.state.emotion_tracker = EmotionTracker()

    logger.info("[PuzzleMind] All systems initialized")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("[PuzzleMind] Shutting down...")
    await vector_store.close()
    logger.info("[PuzzleMind] Shutdown complete")


app = FastAPI(
    title="PuzzleMind Backend",
    version="0.2.0",
    description=(
        "AI puzzle game backend with face recognition, "
        "persistent memory, and multi-provider LLM."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://localhost:5173",
        "http://127.0.0.1:1420",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Return a simple health-check response."""
    return {"status": "ok", "version": "0.2.0"}


# ── TTS (XTTS-v2 voice cloning) ───────────────────────────────────────────────
try:
    from tts.router import router as tts_router
    app.include_router(tts_router, prefix="/api/tts")
    logger.info("TTS router registered at /api/tts")
except ImportError:
    logger.warning("TTS module not available — skipping /api/tts registration")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept a WebSocket connection and route typed messages."""
    conn_mgr: ConnectionManager = app.state.connection_manager
    emo_tracker: EmotionTracker = app.state.emotion_tracker
    connection_id = f"ws_{id(websocket)}"

    await conn_mgr.connect(connection_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                envelope = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            msg_type = envelope.get("type", "")
            payload = envelope.get("payload", {})

            # ── Maze stall detection (checked before the move is processed) ──
            stall_seconds = 0.0
            if msg_type == "game:maze_move":
                maze_player_id = payload.get("player_id", "anonymous")
                stall_seconds = app.state.game_engine.check_and_record_move_time(
                    maze_player_id
                )

            # Try synchronous routing first (maze moves)
            sync_result = route_message(
                msg_type, payload,
                game_engine=app.state.game_engine,
            )
            if sync_result:
                await conn_mgr.send(connection_id, sync_result)

                import asyncio as _asyncio

                # Stall: player paused for ≥45s between moves — nudge them
                if msg_type == "game:maze_move" and stall_seconds >= 45.0:
                    _maze_st = app.state.game_engine.get_maze_state(maze_player_id)
                    _move_ct = _maze_st.move_count if _maze_st else 0
                    stall_desc = (
                        f"The player just made move #{_move_ct} after pausing "
                        f"for {int(stall_seconds)} seconds. They ARE actively "
                        f"playing and have been navigating the maze — they just "
                        f"took a long pause between this move and the last. "
                        f"Give them a playful nudge to keep the momentum going. "
                        f"Do NOT accuse them of not moving or not trying."
                    )
                    _asyncio.create_task(fire_event_comment(
                        event_description=stall_desc,
                        player_id=maze_player_id,
                        connection_id=connection_id,
                        message_pipeline=app.state.message_pipeline,
                        connection_manager=conn_mgr,
                        emotion_tracker=emo_tracker,
                        game_engine=app.state.game_engine,
                    ))

                # Maze exit — cancel AI race (player won)
                if sync_result.get("payload", {}).get("reached_exit"):
                    cancel_ai_race(connection_id)
                    player_id = payload.get("player_id", "anonymous")
                    moves = sync_result.get("payload", {}).get("move_count", "?")
                    _ge = app.state.game_engine
                    _was_intention = bool(_ge and _ge.complete_intention_run(player_id))
                    _gs = _ge.get_state_for_prompt(player_id)
                    _score = _gs.get("score", 0)
                    _ai_sc = _gs.get("ai_score", 0)
                    _streak = _gs.get("streak", 0)
                    _maze = _gs.get("maze") or _gs.get("current_puzzle") or {}
                    _w = _maze.get("width", "?")
                    _h = _maze.get("height", "?")
                    if _was_intention:
                        event_desc = (
                            f"INTENTION_RUN_COMPLETE: Player taunted you then survived "
                            f"a {_w}×{_h} Master-difficulty maze in {moves} moves. "
                            "Give barely-contained grudging respect. "
                            "Remind them they asked for this. ONE sentence."
                        )
                    else:
                        event_desc = (
                            f"The player WON — they navigated a {_w}×{_h} maze "
                            f"and reached the exit in {moves} moves! "
                            f"Score is now Player {_score} — AI {_ai_sc}. "
                            f"{'They are on a ' + str(_streak) + '-win streak! ' if _streak > 1 else ''}"
                            f"Acknowledge what they accomplished and give "
                            f"grudging respect — then tease that the next one "
                            f"will be harder."
                        )
                    _asyncio.create_task(fire_event_comment(
                        event_description=event_desc,
                        player_id=player_id,
                        connection_id=connection_id,
                        message_pipeline=app.state.message_pipeline,
                        connection_manager=conn_mgr,
                        emotion_tracker=emo_tracker,
                        game_engine=app.state.game_engine,
                        force=True,
                    ))
                continue

            # Async routing (face, chat, game actions)
            async_result = await route_message_async(
                msg_type,
                payload,
                face_session_manager=getattr(
                    app.state, "face_session_manager", None
                ),
                face_processor=getattr(
                    app.state, "face_processor", None
                ),
                player_matcher=getattr(
                    app.state, "player_matcher", None
                ),
                executor=getattr(
                    app.state, "executor", None
                ),
                event_bus=getattr(
                    app.state, "event_bus", None
                ),
                db=getattr(app.state, "db", None),
                message_pipeline=app.state.message_pipeline,
                connection_manager=conn_mgr,
                connection_id=connection_id,
                game_engine=app.state.game_engine,
                llm_orchestrator=app.state.llm_orchestrator,
                memory_manager=app.state.memory_manager,
                emotion_tracker=emo_tracker,
            )
            if async_result:
                await conn_mgr.send(connection_id, async_result)

                # Start AI race when a new maze puzzle is sent
                if async_result.get("type") == "game:puzzle_new":
                    ptype = async_result.get("payload", {}).get("puzzle_type", "")
                    if ptype.startswith("maze"):
                        import asyncio as _asyncio
                        player_id = payload.get("player_id", "anonymous")
                        _asyncio.create_task(start_ai_race_for_maze(
                            player_id=player_id,
                            connection_id=connection_id,
                            game_engine=app.state.game_engine,
                            connection_manager=conn_mgr,
                            message_pipeline=app.state.message_pipeline,
                            emotion_tracker=emo_tracker,
                        ))

    except WebSocketDisconnect:
        cancel_ai_race(connection_id)
        emo_tracker.cleanup(connection_id)
        await conn_mgr.disconnect(connection_id)


if __name__ == "__main__":
    import uvicorn

    # --verify-only: run security checks and exit
    if "--verify-only" in sys.argv:
        print("Running security verification...")
        _run_security_checks()
        print("All security checks passed.")
        sys.exit(0)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
