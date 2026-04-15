"""WebSocket endpoint and connection management.

Handles the ``/ws`` endpoint, maintains active connections, and
routes incoming JSON messages to the appropriate module handler
based on the message ``type`` field.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ── TTS singleton + thread pool ───────────────────────────────────────────────
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_tts_engine = None  # lazy-loaded on first use
_tts_lock = asyncio.Lock()

def _get_tts_engine():
    """Return the shared VoiceEngine (loaded once, CPU/MPS-safe)."""
    global _tts_engine
    if _tts_engine is None:
        from tts.router import _get_engine  # uses the same lru_cache singleton
        _tts_engine = _get_engine()
    return _tts_engine


async def _tts_send(
    text: str,
    connection_id: str,
    connection_manager: "ConnectionManager",
) -> None:
    """Synthesize *text* with XTTS-v2 and push ``tts:audio`` over the WebSocket.

    Runs synthesis in a background thread so the event loop is never blocked.
    Silently skips if the TTS engine is unavailable.
    """
    try:
        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(None, _get_tts_engine)
        audio_bytes: bytes = await loop.run_in_executor(
            _tts_executor, engine.synthesize, text
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await connection_manager.send(connection_id, {
            "type": "tts:audio",
            "payload": {"audio": audio_b64, "format": "wav"},
        })
        logger.debug("[tts] Sent %d bytes of audio for conn=%s", len(audio_bytes), connection_id)
    except Exception as exc:
        logger.debug("[tts] Synthesis skipped: %s", exc)

# ── Always-on AI observation state ────────────────────────────────────────────
# Connections currently mid-stream — skip new event comments while busy.
_ai_commenting: set[str] = set()

# Last time an emotion-triggered comment fired, per connection.
_emotion_comment_cooldown: dict[str, float] = {}

# Last time ANY fire_event_comment completed, per connection.
# Prevents API-call bursts when multiple events fire at once.
_event_comment_cooldown: dict[str, float] = {}
_EVENT_COMMENT_MIN_GAP = 8.0  # seconds between any two event comments

# Consecutive wrong answers per player (reset on correct or new puzzle).
_consecutive_wrong: dict[str, int] = {}

# Active AI maze race tasks per connection_id.
_ai_race_tasks: dict[str, "asyncio.Task[None]"] = {}

# Wordplay AI internal-dialog step counts by difficulty (1-5).
_WORDPLAY_AI_STEPS: dict[int, int] = {1: 4, 2: 4, 3: 6, 4: 8, 5: 8}

# Puzzle types that get a post-solve reasoning reveal (non-competitive).
_REASONING_REVEAL_TYPES: frozenset[str] = frozenset({
    "riddle", "logic", "pattern", "deduction"
})


def cancel_ai_race(connection_id: str) -> None:
    """Cancel any running AI maze race for the given connection."""
    import asyncio
    task = _ai_race_tasks.pop(connection_id, None)
    if task and not task.done():
        task.cancel()


async def start_ai_race_for_maze(
    player_id: str,
    connection_id: str,
    game_engine: Any,
    connection_manager: "ConnectionManager",
    message_pipeline: Any = None,
    emotion_tracker: "EmotionTracker | None" = None,
) -> None:
    """Compute epsilon from player streak, then launch the AI race task."""
    import asyncio

    cancel_ai_race(connection_id)

    maze_state = game_engine.get_maze_state(player_id)
    if maze_state is None:
        return

    state = game_engine.load_or_create_state(player_id)
    streak = state.get("streak", 0)
    
    if game_engine.is_intention_run(player_id):
        epsilon = 0.0
    else:
        epsilon = max(0.02, 0.20 - streak * 0.03)

    task = asyncio.create_task(
        _run_ai_race(
            player_id=player_id,
            connection_id=connection_id,
            game_engine=game_engine,
            connection_manager=connection_manager,
            maze_state=maze_state,
            epsilon=epsilon,
            message_pipeline=message_pipeline,
            emotion_tracker=emotion_tracker,
        )
    )
    _ai_race_tasks[connection_id] = task


async def _run_ai_race(
    player_id: str,
    connection_id: str,
    game_engine: Any,
    connection_manager: "ConnectionManager",
    maze_state: Any,
    epsilon: float,
    message_pipeline: Any = None,
    emotion_tracker: "EmotionTracker | None" = None,
) -> None:
    """Stream the AI's maze solution step-by-step, paced to the player.

    The AI starts from maze.start (the beginning), uses A* + epsilon-greedy
    noise, and moves at the same average rhythm as the player.  If it exits
    first the player loses.
    """
    import asyncio as _asyncio
    from game.puzzle_types.maze import MazeSolver

    solver = MazeSolver()
    start = tuple(maze_state.start)
    exit_ = tuple(maze_state.exit)

    path = solver.solve_companion(
        maze_state.grid,
        start,  # type: ignore[arg-type]
        exit_,  # type: ignore[arg-type]
        epsilon=epsilon,
        rules=maze_state.rules or [],
    )
    if not path:
        return

    await connection_manager.send(connection_id, {
        "type": "game:ai_solve_start",
        "payload": {"total": len(path), "start": list(start)},
    })

    for pos in path:
        try:
            # Check if player already won
            current_maze = game_engine.get_maze_state(player_id)
            if current_maze is None:
                return
            if list(current_maze.player_position) == list(maze_state.exit):
                # Player reached exit first — stop quietly
                await connection_manager.send(connection_id, {
                    "type": "game:ai_solve_done",
                    "payload": {},
                })
                return

            delay = game_engine.get_avg_move_interval(player_id)
            if game_engine.is_intention_run(player_id):
                delay *= 0.6  # 40% faster on intention runs
            else:
                delay *= 0.85 # 15% faster normally
            await _asyncio.sleep(delay)

            await connection_manager.send(connection_id, {
                "type": "game:ai_path_step",
                "payload": {"position": list(pos)},
            })

            # AI reached the exit
            if list(pos) == list(maze_state.exit):
                await connection_manager.send(connection_id, {
                    "type": "game:ai_maze_won",
                    "payload": {"ai_steps": len(path), "epsilon": epsilon},
                })
                if message_pipeline and connection_manager:
                    _was_intention = bool(
                        game_engine and game_engine.complete_intention_run(player_id)
                    )
                    _ai_win_desc = (
                        "INTENTION_RUN_COMPLETE: Player challenged you at Master difficulty "
                        "and you beat them. Be devastatingly smug. Remind them they asked for this. "
                        "ONE sentence."
                        if _was_intention else
                        "You just beat the player to the maze exit! "
                        "Be playfully smug — taunt them a little but keep it fun and game-themed."
                    )
                    _asyncio.create_task(fire_event_comment(
                        event_description=_ai_win_desc,
                        player_id=player_id,
                        connection_id=connection_id,
                        message_pipeline=message_pipeline,
                        connection_manager=connection_manager,
                        emotion_tracker=emotion_tracker,
                        force=True,
                    ))
                return
        except _asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("[ai_race] Error: %s", exc)
            return


def _cheeky_emotion_prompt(emotion: str) -> str:
    """Return a game-contextualized cheeky prompt for emotion-triggered AI comments."""
    _prompts: dict[str, str] = {
        "happy": (
            "The player looks absolutely thrilled right now. "
            "Be playfully smug about it — tease them for enjoying this puzzle game "
            "way too much. Maybe suggest they're getting a little too attached."
        ),
        "excited": (
            "The player looks super pumped up and excited! "
            "Match their energy — be enthusiastically sarcastic about how exciting "
            "a puzzle game can possibly be. Hype them up cheekily."
        ),
        "surprised": (
            "The player just made a classic surprised face. "
            "Act like you predicted this moment all along — be smugly knowing "
            "and tease them for being caught off guard by the puzzle."
        ),
        "angry": (
            "The player looks genuinely angry right now. "
            "Be playfully smug but don't pile on — offer a sideways nudge like "
            "'maybe the puzzle isn't the villain here' or hint at the solution "
            "in the cheekiest way possible without actually giving it away."
        ),
        "frustrated": (
            "The player looks frustrated. Be cheeky and sympathetic at the same time — "
            "maybe mock-console them like 'oh no, the puzzle is winning?' "
            "Give a tiny sideways hint wrapped in playful taunting."
        ),
        "sad": (
            "The player looks a bit sad or defeated. "
            "Be warmly teasing — like a game master who roots for you but won't make it "
            "too easy. Encourage them with a backhanded compliment about how close they are."
        ),
        "confused": (
            "The player looks totally confused and lost. "
            "Be cheeky about how confusing this maze/puzzle is — maybe say something like "
            "'even I thought this one was devious' while offering the tiniest cryptic nudge."
        ),
        "neutral": (
            "The player has their best poker face on — completely neutral. "
            "Call them out on it playfully. Are they bored? Concentrating? In the zone? "
            "Tease them for being unreadable."
        ),
        "bored": (
            "The player looks bored. That's basically a challenge. "
            "Poke them with something provocative — maybe taunt that the next puzzle "
            "will actually make them sweat, or question if they're even trying."
        ),
        "disgusted": (
            "The player just made a disgusted face. "
            "Be dramatically offended on behalf of the puzzle — 'excuse me, this puzzle "
            "took ages to design!' — then offer a cheeky clue as a peace offering."
        ),
        "fearful": (
            "The player looks a little scared or anxious. "
            "Be theatrically ominous about it — lean into the game master role and "
            "tease them that their fear is completely justified... or is it?"
        ),
    }
    key = emotion.lower().strip()
    return _prompts.get(
        key,
        (
            f"The player's expression just changed to '{emotion}'. "
            "Comment on it in character as a cheeky, playful game master — "
            "tie it to the puzzle they're working on and keep it fun."
        ),
    )


class EmotionTracker:
    """Per-connection live emotion tracking.

    Stores a rolling window of detected emotions so the LLM always
    knows the player's current facial expression and recent trajectory.
    Stores significant emotion changes into long-term memory.
    """

    # Minimum seconds between memory-store events to avoid flooding
    _STORE_COOLDOWN = 30.0
    # How many recent emotions to keep
    _WINDOW_SIZE = 20

    def __init__(self) -> None:
        # connection_id → deque of (timestamp, emotion)
        self._history: dict[str, deque[tuple[float, str]]] = {}
        # connection_id → last emotion stored in memory
        self._last_stored: dict[str, tuple[float, str]] = {}
        # connection_id → player_id mapping
        self._player_ids: dict[str, str] = {}

    def set_player(self, connection_id: str, player_id: str) -> None:
        """Associate a connection with a player ID."""
        self._player_ids[connection_id] = player_id

    def update(
        self, connection_id: str, emotion: str
    ) -> bool:
        """Record a new emotion detection.

        Returns True if the emotion is a *significant change* worth
        storing in long-term memory.
        """
        if not emotion:
            return False

        now = time.time()
        hist = self._history.setdefault(
            connection_id, deque(maxlen=self._WINDOW_SIZE)
        )
        hist.append((now, emotion))

        # Check if this is a significant change
        last_ts, last_emo = self._last_stored.get(
            connection_id, (0.0, "")
        )
        is_change = emotion != last_emo
        is_cooldown_passed = (now - last_ts) >= self._STORE_COOLDOWN

        if is_change and is_cooldown_passed:
            self._last_stored[connection_id] = (now, emotion)
            return True
        return False

    def get_current(self, connection_id: str) -> str:
        """Return the most recent detected emotion."""
        hist = self._history.get(connection_id)
        if not hist:
            return ""
        return hist[-1][1]

    def get_trajectory(self, connection_id: str) -> list[str]:
        """Return the recent emotion trajectory (deduplicated runs)."""
        hist = self._history.get(connection_id)
        if not hist:
            return []
        trajectory: list[str] = []
        for _, emo in hist:
            if not trajectory or trajectory[-1] != emo:
                trajectory.append(emo)
        return trajectory[-8:]  # last 8 changes

    def build_face_description(self, connection_id: str) -> str:
        """Build a rich face description for the LLM system prompt."""
        current = self.get_current(connection_id)
        if not current:
            return ""

        trajectory = self.get_trajectory(connection_id)
        parts = [f"Player currently appears {current} based on facial expression."]

        if len(trajectory) >= 3:
            parts.append(
                f"Emotional trajectory: {' → '.join(trajectory)}."
            )
        elif len(trajectory) == 2:
            parts.append(
                f"Previously looked {trajectory[0]}, now {trajectory[1]}."
            )

        return " ".join(parts)

    def cleanup(self, connection_id: str) -> None:
        """Remove tracking state for a disconnected connection."""
        self._history.pop(connection_id, None)
        self._last_stored.pop(connection_id, None)
        self._player_ids.pop(connection_id, None)

    def get_player_id(self, connection_id: str) -> str:
        return self._player_ids.get(connection_id, "")


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(
        self, connection_id: str, websocket: WebSocket
    ) -> None:
        """Accept and register a new WebSocket connection.

        Parameters
        ----------
        connection_id : str
            Unique connection identifier.
        websocket : WebSocket
            The FastAPI WebSocket instance.
        """
        await websocket.accept()
        self._connections[connection_id] = websocket

    async def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection from the registry.

        Parameters
        ----------
        connection_id : str
            Connection to remove.
        """
        self._connections.pop(connection_id, None)

    async def send(
        self, connection_id: str, message: dict[str, Any]
    ) -> None:
        """Send a JSON message to a specific connection.

        Parameters
        ----------
        connection_id : str
            Target connection.
        message : dict[str, Any]
            JSON-serialisable message dict.
        """
        ws = self._connections.get(connection_id)
        if ws:
            await ws.send_json(message)

    async def broadcast(
        self, message: dict[str, Any]
    ) -> None:
        """Send a JSON message to all connected clients.

        Parameters
        ----------
        message : dict[str, Any]
            JSON-serialisable message dict.
        """
        for ws in self._connections.values():
            await ws.send_json(message)


async def fire_event_comment(
    event_description: str,
    player_id: str,
    connection_id: str,
    message_pipeline: Any,
    connection_manager: "ConnectionManager",
    emotion_tracker: "EmotionTracker | None" = None,
    player_name: str = "",
    game_engine: Any | None = None,
    force: bool = False,
) -> None:
    """Send an invisible game-event trigger to the LLM.

    The ``event_description`` is injected as the player "message" so the
    LLM has full context, but it is NEVER echoed back to the frontend.
    The LLM's response streams back as a normal ``llm:token`` /
    ``llm:complete`` pair and appears in the player's chat window.

    Skips silently if the AI is already mid-stream for this connection
    (prevents overlapping token streams).

    Parameters
    ----------
    force : bool
        If True, bypass the 8-second cooldown.  Use for critical events
        like wins, maze exits, and AI maze victories that must always
        be narrated.
    """
    if not message_pipeline or not connection_manager:
        return

    # Skip if already streaming a response to this connection
    if connection_id in _ai_commenting:
        logger.debug(
            "[event_comment] Skipping — already streaming for %s", connection_id
        )
        return

    # Global cooldown — prevent bursts of back-to-back event comments
    # Forced events (wins) bypass the cooldown entirely
    if not force:
        now_ec = time.time()
        last_ec = _event_comment_cooldown.get(connection_id, 0.0)
        if now_ec - last_ec < _EVENT_COMMENT_MIN_GAP:
            logger.debug(
                "[event_comment] Cooldown — %.1fs since last for %s",
                now_ec - last_ec, connection_id,
            )
            return
    _event_comment_cooldown[connection_id] = time.time()

    face_desc = ""
    emotion = "neutral"
    if emotion_tracker:
        face_desc = emotion_tracker.build_face_description(connection_id)
        live = emotion_tracker.get_current(connection_id)
        if live:
            emotion = live

    # Hidden trigger — AI sees this, player never does
    trigger = f"[GAME_EVENT: {event_description}]"
    if emotion and emotion != "neutral":
        trigger += f" [PLAYER_EMOTION: {emotion}]"
    if face_desc:
        trigger += f" [PLAYER_APPEARANCE: {face_desc}]"

    # Inject game state context so the LLM knows what the player is doing
    if game_engine:
        ge_state = game_engine.get_state_for_prompt(player_id)
        if ge_state:
            move_count = ge_state.get("move_count", 0)
            maze_info = ge_state.get("maze")
            score = ge_state.get("score", 0)
            ai_score = ge_state.get("ai_score", 0)
            streak = ge_state.get("streak", 0)
            ctx_parts = [f"Score: Player {score} — AI {ai_score}"]
            if streak:
                ctx_parts.append(f"Streak: {streak}")
            if maze_info:
                mc = maze_info.get("move_count", move_count)
                pos = maze_info.get("player_position", "?")
                exit_ = maze_info.get("exit", "?")
                w = maze_info.get("width", "?")
                h = maze_info.get("height", "?")
                ctx_parts.append(
                    f"Maze: {w}x{h}, moves: {mc}, "
                    f"pos: {pos}, exit: {exit_}"
                )
            trigger += f" [GAME_STATE: {', '.join(ctx_parts)}]"

    async def _on_token(token: str) -> None:
        await connection_manager.send(connection_id, {
            "type": "llm:token",
            "payload": {"token": token},
        })

    async def _on_complete(full_text: str, usage: dict[str, Any]) -> None:
        await connection_manager.send(connection_id, {
            "type": "llm:complete",
            "payload": {"text": full_text, "usage": usage},
        })
        if full_text.strip():
            asyncio.create_task(_tts_send(full_text, connection_id, connection_manager))

    _ai_commenting.add(connection_id)
    try:
        await message_pipeline.process_message(
            player_id=player_id,
            message=trigger,
            on_token=_on_token,
            on_complete=_on_complete,
            player_name=player_name,
            emotional_state=emotion,
            face_description=face_desc,
            max_tokens_override=150,
        )
    except Exception as exc:
        logger.debug("[event_comment] Failed: %s", exc)
    finally:
        _ai_commenting.discard(connection_id)


async def _run_wordplay_ai_dialog(
    puzzle_prompt: str,
    solution: str,
    difficulty: int,
    num_steps: int,
    player_attempts: int,
    player_id: str,
    connection_id: str,
    llm_orchestrator: Any,
    message_pipeline: Any,
    connection_manager: "ConnectionManager",
    emotion_tracker: "EmotionTracker | None" = None,
) -> None:
    """Run the AI internal-dialog competitive mechanic for wordplay puzzles.

    Fires num_steps sequential event comments with 2.5s delays, showing the
    AI "thinking aloud" — wrong guesses, reasoning, then the correct answer.
    """
    import asyncio as _asyncio

    # Wait for initial correct-answer commentary to finish streaming
    await _asyncio.sleep(3.0)

    try:
        steps = await llm_orchestrator.generate_ai_internal_dialog(
            puzzle_prompt=puzzle_prompt,
            solution=solution,
            difficulty=difficulty,
            num_steps=num_steps,
        )
    except Exception as exc:
        logger.debug("[wordplay_dialog] Generation failed: %s", exc)
        return

    for i, step in enumerate(steps):
        is_last = (i == len(steps) - 1)
        step_desc = (
            f"[AI_INTERNAL_DIALOG step {i+1}/{len(steps)}]: {step} "
            f"{'(AI just solved it!)' if is_last else '(still thinking...)'}"
        )
        await fire_event_comment(
            event_description=step_desc,
            player_id=player_id,
            connection_id=connection_id,
            message_pipeline=message_pipeline,
            connection_manager=connection_manager,
            emotion_tracker=emotion_tracker,
        )
        if not is_last:
            await _asyncio.sleep(2.5)


async def _run_reasoning_reveal(
    puzzle_prompt: str,
    solution: str,
    puzzle_type: str,
    player_id: str,
    connection_id: str,
    message_pipeline: Any,
    connection_manager: "ConnectionManager",
    emotion_tracker: "EmotionTracker | None" = None,
) -> None:
    """Fire a post-solve AI reasoning reveal for non-competitive puzzle types.

    Waits 5 seconds (for the initial commentary to finish) then sends a
    theatrical one-paragraph reveal of how the AI "would have" solved it.
    """
    import asyncio as _asyncio
    await _asyncio.sleep(5.0)

    reveal_desc = (
        f"The puzzle '{puzzle_prompt[:120]}' has been resolved "
        f"(answer: '{solution}'). Now show off — in one punchy paragraph, "
        f"reveal how YOU (the Game Master) would have solved this {puzzle_type} puzzle. "
        f"Be theatrical, cheeky, and make the player feel impressed (or envious). "
        f"No preamble — dive straight into the reasoning."
    )
    await fire_event_comment(
        event_description=reveal_desc,
        player_id=player_id,
        connection_id=connection_id,
        message_pipeline=message_pipeline,
        connection_manager=connection_manager,
        emotion_tracker=emotion_tracker,
    )


def route_message(
    message_type: str,
    payload: dict[str, Any],
    game_engine: Any | None = None,
) -> dict[str, Any] | None:
    """Dispatch an incoming message to the appropriate handler.

    Parameters
    ----------
    message_type : str
        The ``type`` field from the WebSocket message envelope.
    payload : dict[str, Any]
        Message payload.
    game_engine : GameEngine | None
        Shared game engine instance.

    Returns
    -------
    dict[str, Any] | None
        Response payload if applicable, or None.
    """
    if message_type == "game:maze_move":
        return _handle_maze_move(payload, game_engine)

    # Face-related and chat messages are routed via route_message_async
    return None


async def route_message_async(
    message_type: str,
    payload: dict[str, Any],
    face_session_manager: Any | None = None,
    face_processor: Any | None = None,
    player_matcher: Any | None = None,
    executor: Any | None = None,
    event_bus: Any | None = None,
    db: Any | None = None,
    message_pipeline: Any | None = None,
    connection_manager: "ConnectionManager | None" = None,
    connection_id: str = "",
    game_engine: Any | None = None,
    llm_orchestrator: Any | None = None,
    memory_manager: Any | None = None,
    emotion_tracker: "EmotionTracker | None" = None,
) -> dict[str, Any] | None:
    """Async message routing for face/camera/chat messages.

    Handles CAMERA_FRAME, FACE_ENROLLMENT_FRAME, FACE_CONFIRM,
    FACE_ENROLLMENT_START, and CHAT_SEND.

    Parameters
    ----------
    message_type : str
        Message type string.
    payload : dict[str, Any]
        Message payload.
    face_session_manager : FaceSessionManager | None
        The face session state machine.
    face_processor : FaceProcessor | None
        Frame processor.
    player_matcher : PlayerMatcher | None
        Embedding matcher.
    executor : ProcessPoolExecutor | None
        Worker pool for CPU-bound calls.
    event_bus : EventBus | None
        Event bus.
    db : DatabaseManager | None
        Database manager.
    message_pipeline : MessagePipeline | None
        LLM message processing pipeline.
    connection_manager : ConnectionManager | None
        For sending streaming tokens back.
    connection_id : str
        Active connection identifier.

    Returns
    -------
    dict[str, Any] | None
        Response message or None.
    """
    # ── Session initialization ─────────────────────────────────
    if message_type == "system:init":
        # Frontend sends this immediately on connect.
        # Respond with a greeting so sessionActive is set on the client.
        return {
            "type": "session:greeting",
            "payload": {
                "player_id": "anonymous",
                "player_name": "",
                "message": "Welcome to PuzzleMind! Say hello to get started.",
            },
        }

    # ── Lightweight emotion-only camera (background tracking) ──
    if message_type == "camera:emotion":
        if not face_processor or not executor:
            return None
        frame_b64 = payload.get("frame", "")
        if not frame_b64:
            return None
        import asyncio
        import base64
        try:
            frame_bytes = base64.b64decode(frame_b64)
            loop = asyncio.get_event_loop()
            emotion = await loop.run_in_executor(
                executor,
                face_processor.analyze_emotion,
                frame_bytes,
            )
            if emotion:
                print(f"[EMOTION DEBUG] Detected: {emotion!r} | conn={connection_id}")
                logger.info(
                    "[emotion] Detected: %s (conn=%s)",
                    emotion, connection_id,
                )
                if emotion_tracker:
                    is_significant = emotion_tracker.update(
                        connection_id, emotion
                    )
                    if is_significant and memory_manager:
                        pid = emotion_tracker.get_player_id(
                            connection_id
                        ) or "anonymous"
                        asyncio.create_task(
                            memory_manager.store_game_event(
                                pid,
                                f"Player's expression changed to {emotion}",
                                importance=0.5,
                                category="reaction",
                            )
                        )
                    # Emotion-triggered AI commentary (90s cooldown)
                    if is_significant and message_pipeline and connection_manager:
                        now = time.time()
                        last = _emotion_comment_cooldown.get(connection_id, 0.0)
                        if now - last >= 180.0:
                            _emotion_comment_cooldown[connection_id] = now
                            pid = emotion_tracker.get_player_id(
                                connection_id
                            ) or "anonymous"
                            asyncio.create_task(fire_event_comment(
                                event_description=_cheeky_emotion_prompt(emotion),
                                player_id=pid,
                                connection_id=connection_id,
                                message_pipeline=message_pipeline,
                                connection_manager=connection_manager,
                                emotion_tracker=emotion_tracker,
                                game_engine=game_engine,
                            ))
                return {
                    "type": "face:emotion",
                    "payload": {
                        "emotion": emotion,
                        "state": "emotion_only",
                    },
                }
        except Exception as exc:
            logger.debug("Emotion analysis error: %s", exc)
        return None

    # ── Full face processing pipeline ────────────────────────
    if message_type == "camera:frame":
        if not all([
            face_session_manager, face_processor,
            player_matcher, executor,
        ]):
            # Lightweight emotion-only mode (no DB needed)
            if face_processor and executor:
                frame_b64 = payload.get("frame", "")
                if not frame_b64:
                    return None
                import asyncio
                import base64
                try:
                    frame_bytes = base64.b64decode(frame_b64)
                    loop = asyncio.get_event_loop()
                    emotion = await loop.run_in_executor(
                        executor,
                        face_processor.analyze_emotion,
                        frame_bytes,
                    )
                    logger.debug(
                        "[camera] Emotion detected: %s (conn=%s)",
                        emotion or "none", connection_id,
                    )
                    if emotion:
                        print(f"[EMOTION DEBUG] Detected: {emotion!r} | conn={connection_id}")
                        # Track emotion and store significant changes
                        if emotion_tracker:
                            is_significant = emotion_tracker.update(
                                connection_id, emotion
                            )
                            if is_significant and memory_manager:
                                pid = emotion_tracker.get_player_id(
                                    connection_id
                                ) or "anonymous"
                                asyncio.create_task(
                                    memory_manager.store_game_event(
                                        pid,
                                        f"Player's facial expression changed to {emotion}",
                                        importance=0.5,
                                        category="reaction",
                                    )
                                )
                            # Emotion-triggered AI commentary (90s cooldown)
                            if is_significant and message_pipeline and connection_manager:
                                now = time.time()
                                last = _emotion_comment_cooldown.get(
                                    connection_id, 0.0
                                )
                                if now - last >= 90.0:
                                    _emotion_comment_cooldown[connection_id] = now
                                    pid = emotion_tracker.get_player_id(
                                        connection_id
                                    ) or "anonymous"
                                    asyncio.create_task(fire_event_comment(
                                        event_description=_cheeky_emotion_prompt(emotion),
                                        player_id=pid,
                                        connection_id=connection_id,
                                        message_pipeline=message_pipeline,
                                        connection_manager=connection_manager,
                                        emotion_tracker=emotion_tracker,
                                    ))
                        return {
                            "type": "face:emotion",
                            "payload": {
                                "emotion": emotion,
                                "state": "emotion_only",
                            },
                        }
                except Exception:
                    pass
            return None

        frame_b64 = payload.get("frame", "")
        if not frame_b64:
            return None

        result = await face_session_manager.process_frame(
            frame_b64,
            face_processor,
            player_matcher,
            executor,
            event_bus,
        )

        # Track emotion from full face processing
        if emotion_tracker and result.emotion:
            import asyncio
            is_significant = emotion_tracker.update(
                connection_id, result.emotion
            )
            if result.player_id:
                emotion_tracker.set_player(
                    connection_id, result.player_id
                )
            if is_significant and memory_manager:
                pid = result.player_id or "anonymous"
                asyncio.create_task(
                    memory_manager.store_game_event(
                        pid,
                        f"Player's facial expression changed to {result.emotion}",
                        importance=0.5,
                        category="reaction",
                    )
                )

        return {
            "type": "face:session_state",
            "payload": result.model_dump(),
        }

    elif message_type == "face:enroll_start":
        if not face_session_manager:
            return None

        player_name = payload.get("player_name", "Player")
        player_id = await face_session_manager.start_enrollment(
            player_name
        )

        # Associate player with connection for emotion tracking
        if emotion_tracker:
            emotion_tracker.set_player(connection_id, player_id)

        return {
            "type": "face:session_state",
            "payload": {
                "state": "enrolling",
                "player_id": player_id,
                "player_name": player_name,
            },
        }

    elif message_type == "face:enroll_frame":
        if not all([
            face_session_manager, face_processor,
            player_matcher, executor,
        ]):
            return None

        frame_b64 = payload.get("frame", "")
        player_id = payload.get("player_id", "")
        if not frame_b64 or not player_id:
            return None

        result = await face_session_manager.process_enrollment_frame(
            frame_b64,
            player_id,
            face_processor,
            player_matcher,
            executor,
            db,
        )

        msg_type = (
            "face:enroll_done" if result.success
            else "face:session_state"
        )
        return {
            "type": msg_type,
            "payload": result.model_dump(),
        }

    elif message_type == "face:confirm":
        if not face_session_manager:
            return None

        player_id = payload.get("player_id", "")
        confirmed = payload.get("confirmed", False)

        await face_session_manager.confirm_identity(
            player_id, confirmed, event_bus
        )

        return {
            "type": "face:session_state",
            "payload": face_session_manager.get_state().model_dump(),
        }

    elif message_type == "chat:send":
        if not message_pipeline or not connection_manager:
            return None

        player_id = payload.get("player_id", "")
        text = payload.get("text", "")
        if not text:
            return None

        # ── Provocation detection → Intention Run ─────────────────────────
        import asyncio as _asyncio
        if game_engine:
            from game.provocation_detector import ProvocationDetector
            _prov = ProvocationDetector().detect(text)
            if _prov.detected and not game_engine.is_intention_run(player_id):
                game_engine.arm_intention_run(player_id)
                _asyncio.create_task(fire_event_comment(
                    event_description=(
                        f"PROVOCATION_DETECTED: Player taunted with "
                        f"'{_prov.trigger_phrase}' (category: {_prov.category}). "
                        "React with theatrical indignation. Warn them their next "
                        "maze will be at Master difficulty — ELO 1800. "
                        "ONE sentence, ominous."
                    ),
                    player_id=player_id,
                    connection_id=connection_id,
                    message_pipeline=message_pipeline,
                    connection_manager=connection_manager,
                    emotion_tracker=emotion_tracker,
                    player_name=payload.get("player_name", ""),
                    force=True,
                ))
                return None

        async def _on_token(token: str) -> None:
            await connection_manager.send(connection_id, {
                "type": "llm:token",
                "payload": {"token": token},
            })

        async def _on_complete(
            full_text: str, usage: dict[str, Any]
        ) -> None:
            await connection_manager.send(connection_id, {
                "type": "llm:complete",
                "payload": {
                    "text": full_text,
                    "usage": usage,
                },
            })
            if full_text.strip():
                import asyncio
                asyncio.create_task(_tts_send(full_text, connection_id, connection_manager))

        # Use live emotion from tracker (server-side) over frontend snapshot
        if emotion_tracker:
            live_face_desc = emotion_tracker.build_face_description(
                connection_id
            )
            live_emotion = emotion_tracker.get_current(connection_id)
            # Associate player with connection for future tracking
            if player_id:
                emotion_tracker.set_player(connection_id, player_id)
            logger.info(
                "[chat] Emotion tracker → emotion=%s, desc=%s",
                live_emotion or "(none)",
                live_face_desc[:80] if live_face_desc else "(empty)",
            )
        else:
            live_face_desc = ""
            live_emotion = ""

        # Prefer server-tracked emotion; fall back to frontend payload
        frontend_face = payload.get("face_description", "")
        face_description = live_face_desc or frontend_face
        logger.info(
            "[chat] Final face_description=%s",
            face_description[:80] if face_description else "(empty)",
        )
        emotional_state = (
            live_emotion
            or payload.get("emotional_state", "neutral")
        )

        logger.info(
            "[chat] Starting pipeline for player=%s text=%r",
            player_id, text[:80],
        )
        try:
            await message_pipeline.process_message(
                player_id=player_id,
                message=text,
                on_token=_on_token,
                on_complete=_on_complete,
                player_name=payload.get("player_name", ""),
                emotional_state=emotional_state,
                relationship_stage=payload.get(
                    "relationship_stage", "early"
                ),
                player_memory=payload.get("player_memory", ""),
                face_description=face_description,
            )
            logger.info("[chat] Pipeline completed for player=%s", player_id)
        except Exception as _pipe_exc:
            import traceback as _tb
            logger.error(
                "[chat] Pipeline raised an exception for player=%s: %s\n%s",
                player_id, _pipe_exc, _tb.format_exc(),
            )
            _err_msg = "Something went wrong — please try again."
            await _on_token(_err_msg)
            await _on_complete(_err_msg, {})
        # Response already streamed via callbacks
        return None

    elif message_type == "game:action":
        if not game_engine:
            return None

        action = payload.get("action", "")
        player_id = payload.get("player_id", "anonymous")
        puzzle_type = payload.get("puzzle_type", "")

        logger.info(
            "[game:action] action=%s puzzle_type=%s player=%s llm_available=%s",
            action,
            puzzle_type,
            player_id,
            llm_orchestrator.puzzle_generation_available() if llm_orchestrator else False,
        )

        if action == "new_puzzle" and puzzle_type.startswith("maze"):
            from data.models import PuzzleType

            type_map = {
                "maze_classic": PuzzleType.MAZE_CLASSIC,
                "maze_dark": PuzzleType.MAZE_DARK,
                "maze_logic": PuzzleType.MAZE_LOGIC,
            }
            pt = type_map.get(puzzle_type, PuzzleType.MAZE_CLASSIC)
            # Pick target ELO — use intention-run override if armed
            state = game_engine.load_or_create_state(player_id)
            elo_ratings: dict[str, float] = state.get("elo_ratings", {})
            if game_engine.is_intention_run(player_id):
                target_elo = game_engine.get_intention_run_elo(player_id)
            else:
                target_elo = int(elo_ratings.get(puzzle_type, 1200))
            maze = game_engine.start_maze(player_id, pt, target_elo=target_elo)
            render_data = game_engine.get_maze_render_data(player_id)
            diff_label = game_engine.difficulty_label(target_elo)

            return {
                "type": "game:puzzle_new",
                "payload": {
                    "puzzle_type": puzzle_type,
                    "prompt": f"Navigate the {puzzle_type.replace('_', ' ')}!",
                    "difficulty_label": diff_label,
                    **render_data,
                },
            }

        # Non-maze puzzle types — generate via LLM
        if llm_orchestrator and llm_orchestrator.puzzle_generation_available():
            # Adaptive difficulty + dedup
            difficulty = game_engine.get_adaptive_difficulty(player_id)
            seen_themes = game_engine.get_seen_themes(player_id)

            # Build richer player context
            state = game_engine.get_state_for_prompt(player_id)
            score = state.get("score", 0)
            ai_score = state.get("ai_score", 0)
            streak = state.get("streak", 0)
            total = state.get("total_solved", 0)
            player_ctx = (
                f"Player: {player_id}. "
                f"Score: {score}-{ai_score} (player-AI). "
                f"Streak: {streak}. Total solved: {total}. "
                f"Make this puzzle COMPLETELY DIFFERENT from previous ones."
            )

            logger.info(
                "[game:action] Generating %s puzzle (difficulty=%d) for player=%s",
                puzzle_type, difficulty, player_id,
            )
            puzzle_data = await llm_orchestrator.generate_puzzle_json(
                puzzle_type=puzzle_type,
                difficulty=difficulty,
                player_context=player_ctx,
                avoid_similar_to=seen_themes[-8:] if seen_themes else None,
            )
            if puzzle_data:
                logger.info(
                    "[game:action] Puzzle generated successfully: type=%s",
                    puzzle_type,
                )
                # Store puzzle in game engine (with solution — server-side)
                puzzle_data["puzzle_type"] = puzzle_type
                game_engine.set_puzzle(player_id, puzzle_data)

                # Get timer from game state
                state = game_engine.get_state_for_prompt(player_id)
                timer = state.get("timer_seconds", 120)

                # Store puzzle start in memory
                if memory_manager:
                    import asyncio
                    event = (
                        f"New {puzzle_type} puzzle started (difficulty {puzzle_data.get('difficulty', 2)}): "
                        f"{puzzle_data.get('prompt', '')[:100]}"
                    )
                    asyncio.create_task(
                        memory_manager.store_game_event(
                            player_id, event,
                            importance=0.7,
                            category="preference",
                        )
                    )

                puzzle_diff = puzzle_data.get("difficulty", 2)
                # Convert 1-5 difficulty scale to approximate ELO for label
                _diff_elo_map = {1: 800, 2: 1000, 3: 1200, 4: 1400, 5: 1600}
                puzzle_diff_label = game_engine.difficulty_label(
                    _diff_elo_map.get(puzzle_diff, 1200)
                )

                return {
                    "type": "game:puzzle_new",
                    "payload": {
                        "puzzle_type": puzzle_type,
                        "prompt": puzzle_data.get("prompt", "Solve this puzzle!"),
                        # Do NOT send solution to frontend
                        "hints": puzzle_data.get("hints", []),
                        "difficulty": puzzle_diff,
                        "difficulty_label": puzzle_diff_label,
                        "category": puzzle_data.get("category", puzzle_type),
                        "timer_seconds": timer,
                        "max_hints": len(puzzle_data.get("hints", [])) or 3,
                    },
                }

        # Fallback if no LLM provider available or generation failed
        logger.error(
            "[game:action] Puzzle generation failed for type=%s. "
            "llm_orchestrator=%s available=%s",
            puzzle_type,
            bool(llm_orchestrator),
            llm_orchestrator.puzzle_generation_available() if llm_orchestrator else False,
        )
        return {
            "type": "game:puzzle_new",
            "payload": {
                "puzzle_type": puzzle_type,
                "prompt": (
                    f"The puzzle generator hit a snag for '{puzzle_type}'. "
                    f"Check server logs — likely no LLM API key (ANTHROPIC_API_KEY / "
                    f"OPENAI_API_KEY) is configured or all providers timed out."
                ),
                "timer_seconds": 120,
            },
        }

    elif message_type == "puzzle:answer":
        if not game_engine:
            return None

        player_id = payload.get("player_id", "anonymous")
        answer = payload.get("answer", "")

        result = game_engine.check_answer(player_id, answer)

        # Store in memory immediately
        if memory_manager:
            import asyncio
            is_correct = result.get("is_correct", False)
            event = (
                f"Player answered '{answer}' — {'CORRECT! Score: ' + str(result.get('score', 0)) if is_correct else 'wrong (attempt ' + str(result.get('attempts', 0)) + ')'}"
            )
            asyncio.create_task(
                memory_manager.store_game_event(
                    player_id, event,
                    importance=0.95 if is_correct else 0.6,
                    category="reaction" if is_correct else "strategy",
                )
            )

        import asyncio as _asyncio

        # Capture current puzzle before result mutates state
        _current_state = game_engine.load_or_create_state(player_id) if game_engine else {}
        _current_puzzle = _current_state.get("current_puzzle") or {}
        _puzzle_type_now = _current_puzzle.get("puzzle_type", payload.get("puzzle_type", ""))
        _puzzle_prompt_now = _current_puzzle.get("prompt", "")
        _puzzle_solution = _current_puzzle.get("solution", answer)
        _puzzle_difficulty = int(_current_puzzle.get("difficulty", 2))

        if result.get("is_correct"):
            # Reset wrong-streak on correct answer
            _consecutive_wrong.pop(player_id, None)

            # Fire invisible AI comment after correct answer
            if message_pipeline and connection_manager:
                score = result.get("score", 0)
                ai_score = result.get("ai_score", 0)
                streak = result.get("streak", 0)
                attempts = result.get("attempts", 1)
                event_desc = (
                    f"The player WON — they solved the puzzle! "
                    f"Answer: '{answer}'"
                    f"{' on their first try!' if attempts == 1 else f' after {attempts} attempts.'} "
                    f"Score: Player {score} — AI {ai_score}. "
                    f"{'They are on a ' + str(streak) + '-win streak! ' if streak > 1 else ''}"
                    f"Acknowledge what they accomplished with grudging "
                    f"respect, then tease that the next one will be harder."
                )
                _asyncio.create_task(fire_event_comment(
                    event_description=event_desc,
                    player_id=player_id,
                    connection_id=connection_id,
                    message_pipeline=message_pipeline,
                    connection_manager=connection_manager,
                    emotion_tracker=emotion_tracker,
                    player_name=payload.get("player_name", ""),
                    game_engine=game_engine,
                    force=True,
                ))

                # Wordplay: AI internal dialog competitive mechanic
                if _puzzle_type_now == "wordplay" and llm_orchestrator:
                    num_steps = _WORDPLAY_AI_STEPS.get(_puzzle_difficulty, 4)
                    _asyncio.create_task(_run_wordplay_ai_dialog(
                        puzzle_prompt=_puzzle_prompt_now,
                        solution=_puzzle_solution,
                        difficulty=_puzzle_difficulty,
                        num_steps=num_steps,
                        player_attempts=result.get("attempts", 1),
                        player_id=player_id,
                        connection_id=connection_id,
                        llm_orchestrator=llm_orchestrator,
                        message_pipeline=message_pipeline,
                        connection_manager=connection_manager,
                        emotion_tracker=emotion_tracker,
                    ))

                # Non-competitive puzzles: post-solve reasoning reveal (5s delay)
                elif _puzzle_type_now in _REASONING_REVEAL_TYPES:
                    _asyncio.create_task(_run_reasoning_reveal(
                        puzzle_prompt=_puzzle_prompt_now,
                        solution=_puzzle_solution,
                        puzzle_type=_puzzle_type_now,
                        player_id=player_id,
                        connection_id=connection_id,
                        message_pipeline=message_pipeline,
                        connection_manager=connection_manager,
                        emotion_tracker=emotion_tracker,
                    ))
        else:
            # Track consecutive wrong answers — fire encouragement at 3
            _consecutive_wrong[player_id] = (
                _consecutive_wrong.get(player_id, 0) + 1
            )
            if (
                _consecutive_wrong[player_id] == 3
                and message_pipeline
                and connection_manager
            ):
                attempts = result.get("attempts", 3)
                event_desc = (
                    f"Player has given {attempts} incorrect answers in a row "
                    f"and is clearly struggling. Offer genuine encouragement "
                    f"and perhaps a subtle nudge — without giving the answer away."
                )
                _asyncio.create_task(fire_event_comment(
                    event_description=event_desc,
                    player_id=player_id,
                    connection_id=connection_id,
                    message_pipeline=message_pipeline,
                    connection_manager=connection_manager,
                    emotion_tracker=emotion_tracker,
                    player_name=payload.get("player_name", ""),
                    game_engine=game_engine,
                ))

        return {
            "type": "puzzle:result",
            "payload": result,
        }

    elif message_type == "puzzle:timeout":
        if not game_engine:
            return None

        player_id = payload.get("player_id", "anonymous")

        result = game_engine.handle_timeout(player_id)

        # Store timeout in memory
        if memory_manager:
            import asyncio
            solution = result.get("solution", "?")
            event = (
                f"Player ran out of time. AI scored! Answer was '{solution}'. "
                f"Score: Player {result.get('score', 0)} — AI {result.get('ai_score', 0)}"
            )
            asyncio.create_task(
                memory_manager.store_game_event(
                    player_id, event,
                    importance=0.85,
                    category="reaction",
                )
            )
        # Fire invisible AI comment after timeout
        if message_pipeline and connection_manager:
            import asyncio as _asyncio
            solution = result.get("solution", "?")
            score = result.get("score", 0)
            ai_score = result.get("ai_score", 0)
            event_desc = (
                f"Player ran out of time on the puzzle. AI scored a point! "
                f"The correct answer was '{solution}'. "
                f"Score: Player {score} — AI {ai_score}."
            )
            _asyncio.create_task(fire_event_comment(
                event_description=event_desc,
                player_id=player_id,
                connection_id=connection_id,
                message_pipeline=message_pipeline,
                connection_manager=connection_manager,
                emotion_tracker=emotion_tracker,
                player_name=payload.get("player_name", ""),
                game_engine=game_engine,
            ))

            # Non-competitive puzzles: reasoning reveal after timeout (5s delay)
            _timed_out_puzzle = game_engine.load_or_create_state(player_id).get("current_puzzle") or {} if game_engine else {}
            _to_type = _timed_out_puzzle.get("puzzle_type", "")
            _to_prompt = _timed_out_puzzle.get("prompt", "")
            if _to_type in _REASONING_REVEAL_TYPES and _to_prompt:
                _asyncio.create_task(_run_reasoning_reveal(
                    puzzle_prompt=_to_prompt,
                    solution=solution,
                    puzzle_type=_to_type,
                    player_id=player_id,
                    connection_id=connection_id,
                    message_pipeline=message_pipeline,
                    connection_manager=connection_manager,
                    emotion_tracker=emotion_tracker,
                ))

        return {
            "type": "puzzle:timeout_result",
            "payload": result,
        }

    elif message_type == "game:logout":
        # Player is logging out — clear all server-side state for this player.
        player_id = payload.get("player_id", "anonymous")
        cancel_ai_race(connection_id)
        if game_engine:
            game_engine.clear_player_state(player_id)
        if message_pipeline and hasattr(message_pipeline, "clear_player_history"):
            message_pipeline.clear_player_history(player_id)
        _consecutive_wrong.pop(player_id, None)
        # Clear connection-scoped state so the next player on this WS starts clean
        _ai_commenting.discard(connection_id)
        _emotion_comment_cooldown.pop(connection_id, None)
        if emotion_tracker:
            emotion_tracker.cleanup(connection_id)
        return {"type": "session:ended", "payload": {}}

    elif message_type == "player:get_stats":
        # Profile screen requests fresh ELO/stats for the current player.
        player_id = payload.get("player_id", "anonymous")
        raw: dict[str, Any] = {}
        if game_engine:
            raw = game_engine.get_player_stats(player_id)
        elo_ratings: dict[str, float] = raw.get("elo_ratings", {})
        difficulty_labels: dict[str, str] = {
            k: game_engine.difficulty_label(v)
            for k, v in elo_ratings.items()
        } if game_engine else {}
        return {
            "type": "player:stats",
            "payload": {
                "eloRatings": elo_ratings,
                "difficultyLabels": difficulty_labels,
                "totalSolved": raw.get("total_solved", 0),
                "streak": raw.get("streak", 0),
                "bestStreak": raw.get("best_streak", 0),
                "totalSessions": raw.get("total_sessions", 0),
                "achievements": [],
                "jailbreakBadges": [],
                "relationshipStage": "early",
            },
        }

    elif message_type == "game:maze_cancel":
        player_id = payload.get("player_id", "anonymous")
        # Player quit the maze — cancel any running AI race
        cancel_ai_race(connection_id)
        if game_engine:
            game_engine.cancel_active_puzzle(player_id)
        return None

    elif message_type == "puzzle:low_timer":
        # Frontend fires this once when the timer crosses 30 seconds.
        # Trigger a one-shot urgency comment from the AI.
        if message_pipeline and connection_manager:
            import asyncio as _asyncio
            player_id = payload.get("player_id", "anonymous")
            event_desc = (
                "The player has less than 30 seconds left on the puzzle timer "
                "and hasn't solved it yet. Create some tension and urgency — "
                "be dramatic but not discouraging."
            )
            _asyncio.create_task(fire_event_comment(
                event_description=event_desc,
                player_id=player_id,
                connection_id=connection_id,
                message_pipeline=message_pipeline,
                connection_manager=connection_manager,
                emotion_tracker=emotion_tracker,
                player_name=payload.get("player_name", ""),
                game_engine=game_engine,
            ))
        return None

    return None


def _handle_maze_move(
    payload: dict[str, Any],
    game_engine: Any | None = None,
) -> dict[str, Any]:
    """Route a maze move to the game engine.

    Parameters
    ----------
    payload : dict[str, Any]
        Must contain ``direction`` and optionally ``player_id``.
    game_engine : GameEngine | None
        Shared game engine instance.

    Returns
    -------
    dict[str, Any]
        Response with type ``game:maze_state`` containing full
        render data so the frontend can re-render the maze.
    """
    if not game_engine:
        return {"type": "game:maze_state", "payload": {"valid": False, "reason": "no_engine"}}

    player_id = payload.get("player_id", "anonymous")
    direction = payload.get("direction", "")

    result = game_engine.process_maze_move(player_id, direction)

    # Return full render data (with grid) so the frontend can re-render
    render_data = game_engine.get_maze_render_data(player_id)
    move_info = result.model_dump()

    return {
        "type": "game:maze_state",
        "payload": {
            **render_data,
            "valid": move_info["valid"],
            "reason": move_info.get("reason"),
            "reached_exit": move_info.get("reached_exit", False),
            "items_collected": move_info.get("items_collected", []),
            "score": move_info.get("score"),
        },
    }
