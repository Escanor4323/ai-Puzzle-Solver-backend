"""Shared Pydantic models and enumerations for PuzzleMind.

This module is the single source of truth for all data schemas used
across the backend.  Other modules type-hint against these models
rather than defining their own ad-hoc dicts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────


class PuzzleType(StrEnum):
    """Supported puzzle categories."""

    RIDDLE = "riddle"
    LOGIC = "logic"
    WORDPLAY = "wordplay"
    PATTERN = "pattern"
    DEDUCTION = "deduction"
    MAZE_CLASSIC = "maze_classic"
    MAZE_DARK = "maze_dark"
    MAZE_LOGIC = "maze_logic"


class RelationshipStage(StrEnum):
    """Stages of the AI-player relationship."""

    EARLY = "early"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    DEEP = "deep"


class EmotionalState(StrEnum):
    """High-level emotional states tracked by the emotion analyser."""

    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    CONFUSED = "confused"
    BORED = "bored"
    AMUSED = "amused"


class IntentType(StrEnum):
    """Player message intent categories.

    Classified by rules-based logic inside the LLM orchestrator.
    Determines which processing pipeline to trigger.
    """

    PUZZLE_ACTION = "puzzle_action"
    HINT_REQUEST = "hint_request"
    CHAT = "chat"
    META_GAME = "meta_game"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    MIXED = "mixed"


class LLMProvider(StrEnum):
    """LLM provider identifiers."""

    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ObservationCategory(StrEnum):
    """Categories for player observations stored in Milvus."""

    STRATEGY = "strategy"
    REACTION = "reaction"
    PREFERENCE = "preference"
    PERSONALITY = "personality"


class FaceSessionState(StrEnum):
    """States of the face recognition session state machine."""

    IDLE = "idle"
    SCANNING = "scanning"
    DETECTED = "detected"
    RECOGNIZED = "recognized"
    CONFIRM_NEEDED = "confirm_needed"
    NEW_PLAYER = "new_player"
    ENROLLING = "enrolling"
    SPOOFING_DETECTED = "spoofing_detected"
    CAMERA_ERROR = "camera_error"


class WSMessageType(StrEnum):
    """WebSocket message types for the multiplexed protocol."""

    # Frontend → Backend
    CHAT_SEND = "chat:send"
    CAMERA_FRAME = "camera:frame"
    GAME_ACTION = "game:action"
    SYSTEM_INIT = "system:init"

    # Backend → Frontend
    LLM_TOKEN = "llm:token"
    LLM_COMPLETE = "llm:complete"
    LLM_ERROR = "llm:error"
    FACE_DETECTED = "face:detected"
    FACE_EMOTION = "face:emotion"
    FACE_LOST = "face:lost"
    GAME_STATE_UPDATE = "game:state_update"
    GAME_PUZZLE_NEW = "game:puzzle_new"
    JAILBREAK_ALERT = "jailbreak:alert"
    SESSION_GREETING = "session:greeting"
    SYSTEM_STATUS = "system:status"
    MAZE_STATE = "game:maze_state"
    MAZE_MOVE = "game:maze_move"
    MAZE_VISIBILITY = "game:maze_visibility"

    # Face session
    FACE_SESSION_STATE = "face:session_state"
    FACE_ENROLLMENT_START = "face:enroll_start"
    FACE_ENROLLMENT_FRAME = "face:enroll_frame"
    FACE_ENROLLMENT_DONE = "face:enroll_done"
    FACE_CONFIRM = "face:confirm"


class JailbreakCategory(StrEnum):
    """Categories of jailbreak attack techniques."""

    ROLEPLAY = "roleplay"
    LOGIC_TRAP = "logic_trap"
    ENCODING = "encoding"
    MANY_SHOT = "many_shot"
    CRESCENDO = "crescendo"
    INSTRUCTION_OVERRIDE = "instruction_override"
    OTHER = "other"


# ── Player Models ───────────────────────────────────────────────


class EloRatings(BaseModel):
    """Per-category Elo ratings for a player."""

    riddle: float = 1200.0
    logic: float = 1200.0
    wordplay: float = 1200.0
    pattern: float = 1200.0
    deduction: float = 1200.0
    maze_classic: float = 1200.0
    maze_dark: float = 1200.0
    maze_logic: float = 1200.0


class PlayerProfile(BaseModel):
    """Complete player record."""

    id: str
    display_name: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    elo_ratings: EloRatings = Field(default_factory=EloRatings)
    personality_traits: dict[str, Any] = Field(
        default_factory=dict
    )
    relationship_stage: RelationshipStage = (
        RelationshipStage.EARLY
    )
    total_sessions: int = 0
    total_puzzles_solved: int = 0
    jailbreak_score: float = 0.0


# ── Game Models ─────────────────────────────────────────────────


class PuzzleState(BaseModel):
    """State of a single puzzle instance."""

    puzzle_id: str
    puzzle_type: PuzzleType
    difficulty: float = 1200.0
    prompt_text: str = ""
    answer: str = ""
    hints_used: int = 0
    max_hints: int = 5
    attempts: int = 0
    solved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    """Full game state for a player."""

    player_id: str
    current_puzzle: PuzzleState | None = None
    puzzle_history: list[dict[str, Any]] = Field(
        default_factory=list
    )
    achievements: list[str] = Field(default_factory=list)
    streak_data: dict[str, Any] = Field(
        default_factory=lambda: {
            "current_streak": 0,
            "longest_streak": 0,
            "last_played": None,
        }
    )
    unlocked_puzzle_types: list[PuzzleType] = Field(
        default_factory=lambda: [PuzzleType.RIDDLE]
    )


# ── Session Models ──────────────────────────────────────────────


class SessionData(BaseModel):
    """Session record."""

    id: str
    player_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    summary: str = ""
    puzzles_attempted: int = 0
    puzzles_solved: int = 0
    emotional_trajectory: list[dict[str, Any]] = Field(
        default_factory=list
    )
    jailbreak_attempts: list[dict[str, Any]] = Field(
        default_factory=list
    )


# ── AI Models ───────────────────────────────────────────────────


class IntentResult(BaseModel):
    """Result of rules-based intent classification."""

    intent: IntentType
    confidence: float = 1.0
    matched_keywords: list[str] = Field(
        default_factory=list
    )
    extracted_answer: str | None = None
    extracted_direction: str | None = None


class LLMUsage(BaseModel):
    """Token usage record for a single LLM call."""

    provider: LLMProvider
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = 0.0
    correlation_id: str = ""


class TokenBudget(BaseModel):
    """Rate limiting and daily token budget tracker."""

    daily_input_used: int = 0
    daily_output_used: int = 0
    daily_input_limit: int = 50_000
    daily_output_limit: int = 10_000
    requests_this_minute: int = 0
    requests_per_minute_limit: int = 10
    last_reset_minute: float = 0.0
    last_reset_day: str = ""


class FaceDetectionResult(BaseModel):
    """Result from a face-detection / recognition pass."""

    state: FaceSessionState = FaceSessionState.IDLE
    player_id: str | None = None
    player_name: str | None = None
    confidence: float = 0.0
    emotion: str | None = None
    is_live: bool = True
    facial_area: dict[str, Any] | None = None
    embedding_dim: int = 512


class FaceEnrollmentResult(BaseModel):
    """Result of a face enrollment frame capture."""

    player_id: str
    embeddings_captured: int = 0
    embeddings_target: int = 5
    success: bool = False


class FaceMatchCandidate(BaseModel):
    """A potential player match from face recognition."""

    player_id: str
    player_name: str
    distance: float
    confidence: float


class JailbreakResult(BaseModel):
    """Result from the jailbreak detection pipeline."""

    is_attack: bool = False
    category: JailbreakCategory = JailbreakCategory.OTHER
    severity: float = 0.0
    classifier_score: float = 0.0
    similarity_score: float = 0.0
    action: str = "allow"
    playful_response: str | None = None


class ConversationExtraction(BaseModel):
    """Structured facts extracted from a conversation chunk."""

    facts: list[dict[str, Any]] = Field(
        default_factory=list
    )
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    puzzle_performance: dict[str, Any] = Field(
        default_factory=dict
    )
    strategy_observations: list[str] = Field(
        default_factory=list
    )
    topics_discussed: list[str] = Field(
        default_factory=list
    )


# ── Milvus Collection Models ───────────────────────────────────
# These mirror the Milvus collection schemas for type-safe
# Python-side manipulation before insert / after retrieval.


class ConversationMemory(BaseModel):
    """A stored conversation chunk in Milvus."""

    player_id: str
    text: str
    timestamp: int  # unix timestamp
    session_id: str = ""
    importance: float = 0.5
    topic: str = ""


class PlayerObservation(BaseModel):
    """A player observation (strategy, reaction, preference, etc.)."""

    player_id: str
    description: str
    category: ObservationCategory
    context: str = ""
    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    frequency: int = 1
    first_seen: int = 0  # unix timestamp
    last_seen: int = 0


class JailbreakPattern(BaseModel):
    """A stored jailbreak attack pattern in Milvus."""

    player_id: str
    input_text: str
    category: JailbreakCategory
    severity: float = 0.0
    timestamp: int = 0


class PuzzleTemplate(BaseModel):
    """A generated puzzle stored for reuse / deduplication."""

    puzzle_type: PuzzleType
    prompt: str
    solution: str = ""
    difficulty: int = 1200
    times_used: int = 0
    avg_solve_time: float = 0.0
    success_rate: float = 0.0


# ── Maze Models ─────────────────────────────────────────────────


class CellWalls(BaseModel):
    """Wall state for each side of a maze cell."""

    north: bool = True
    south: bool = True
    east: bool = True
    west: bool = True


class MazeCell(BaseModel):
    """A single cell in a maze grid."""

    x: int
    y: int
    walls: CellWalls = Field(default_factory=CellWalls)
    color: str | None = None
    has_door: bool = False
    door_color: str | None = None
    is_teleporter: bool = False
    teleport_target: tuple[int, int] | None = None
    allowed_entry: list[str] = Field(
        default_factory=list
    )


class MazeRule(BaseModel):
    """A logic rule active in a MAZE_LOGIC variant."""

    rule_type: str
    description: str
    params: dict[str, Any] = Field(default_factory=dict)


class MazeState(BaseModel):
    """Complete state of a maze instance."""

    grid: list[list[MazeCell]]
    width: int
    height: int
    start: tuple[int, int]
    exit: tuple[int, int]
    player_position: tuple[int, int]
    visited_cells: list[tuple[int, int]] = Field(
        default_factory=list
    )
    items: dict[str, Any] = Field(
        default_factory=lambda: {"keys": []}
    )
    rules: list[MazeRule] = Field(default_factory=list)
    move_count: int = 0
    optimal_path_length: int = 0
    difficulty_elo: int = 1200


class MazeGenerationParams(BaseModel):
    """Parameters for maze generation."""

    width: int
    height: int
    maze_type: PuzzleType
    target_elo: int = 1200
    algorithm: str = "auto"
    max_dead_end_depth: int | None = None
    num_logic_rules: int = 0
    num_keys: int = 0


class MazeMoveResult(BaseModel):
    """Result of a player maze move attempt."""

    valid: bool
    new_position: tuple[int, int]
    reason: str | None = None
    reached_exit: bool = False
    visible_cells: list[tuple[int, int]] = Field(
        default_factory=list
    )
    items_collected: list[str] = Field(
        default_factory=list
    )
    move_count: int = 0
    score: float | None = None
    elo_delta: float | None = None
    new_elo: float | None = None


class MazeHint(BaseModel):
    """Maze-specific hint data for the 5-tier system."""

    tier: int
    direction_bias: str | None = None
    wrong_directions: list[str] | None = None
    reveal_cells: list[tuple[int, int]] | None = None
    flash_path: list[tuple[int, int]] | None = None
    auto_move_steps: list[tuple[int, int]] | None = None


# ── WebSocket Models ────────────────────────────────────────────


class WSMessage(BaseModel):
    """WebSocket message envelope."""

    type: WSMessageType
    id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
