"""FastAPI router for the TTS endpoint (edge-tts).

Endpoint
--------
POST /api/tts/synthesize
    Body:  {"text": "<string, 1-500 chars>"}
    Returns: audio/mpeg (MP3)

Security
--------
- Only ``text`` is accepted from callers — no reference audio path.
- Input length validated by Pydantic before reaching the engine.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])


# ── Engine singleton (loaded once per process) ────────────────────────────────


@lru_cache(maxsize=1)
def _get_engine():  # noqa: ANN201
    """Return the shared VoiceEngine (constructed once)."""
    from tts.voice_engine import VoiceEngine

    return VoiceEngine()


# ── Request model ─────────────────────────────────────────────────────────────


class SynthesizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to synthesize (1–500 characters).",
    )
    # Note: no reference_audio field — intentional security boundary.


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/synthesize",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "MP3 audio of the synthesized speech.",
        },
        422: {"description": "Validation error (empty text or too long)."},
        503: {"description": "TTS engine not available."},
    },
)
async def synthesize(request: SynthesizeRequest) -> Response:
    """Synthesize *text* in the Caine voice and return MP3 bytes."""
    try:
        engine = _get_engine()
    except Exception as exc:
        logger.error("TTS engine failed to load: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="TTS engine is unavailable. Check server logs.",
        ) from exc

    try:
        audio = engine.synthesize(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Synthesis failed.") from exc

    return Response(content=audio, media_type="audio/mpeg")


@router.get("/health")
async def health() -> dict[str, str]:
    """Quick liveness check — does not instantiate the engine."""
    return {"status": "ok", "engine": "edge-tts"}
