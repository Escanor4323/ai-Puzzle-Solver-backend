"""WebSocket message protocol definitions.

Defines the JSON envelope format for all messages between the
Svelte frontend and the FastAPI backend, and the enumeration of
valid message types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class WSMessageType(StrEnum):
    """All valid WebSocket message types.

    Frontend → Backend:
        CHAT_SEND, CAMERA_FRAME, GAME_ACTION, SYSTEM_INIT

    Backend → Frontend:
        LLM_TOKEN, LLM_COMPLETE, LLM_ERROR, FACE_DETECTED,
        FACE_EMOTION, FACE_LOST, GAME_STATE_UPDATE,
        GAME_PUZZLE_NEW, JAILBREAK_ALERT, SESSION_GREETING,
        SYSTEM_STATUS
    """

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


@dataclass
class WSMessage:
    """JSON message envelope for the multiplexed WebSocket.

    All messages between the frontend and backend follow this
    envelope structure.
    """

    type: WSMessageType
    id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    correlation_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "type": self.type.value,
            "id": self.id,
            "timestamp": self.timestamp,
            "correlationId": self.correlation_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WSMessage:
        """Deserialise from a JSON-compatible dict."""
        return cls(
            type=WSMessageType(data.get("type", "")),
            id=data.get("id", ""),
            timestamp=data.get(
                "timestamp", datetime.now().isoformat()
            ),
            correlation_id=data.get("correlationId", ""),
            payload=data.get("payload", {}),
        )
