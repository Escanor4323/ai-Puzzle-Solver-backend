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
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    MIXED = "mixed"


class ObservationCategory(StrEnum):
    """Categories for player observations stored in Milvus."""

    STRATEGY = "strategy"
    REACTION = "reaction"
    PREFERENCE = "preference"
    PERSONALITY = "personality"


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


class FaceDetectionResult(BaseModel):
    """Result from a face-detection / recognition pass."""

    detected: bool = False
    player_id: str | None = None
    confidence: float = 0.0
    is_new_player: bool = False
    emotion: str | None = None
    anti_spoofing_pass: bool = True


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


# ── WebSocket Models ────────────────────────────────────────────


class WSMessage(BaseModel):
    """WebSocket message envelope."""

    type: WSMessageType
    id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
