"""Edge-TTS VoiceEngine — fast neural TTS for the Caine GM persona.

Uses Microsoft Edge's neural TTS service via the ``edge-tts`` package.
Typical synthesis: 200-400 ms, no GPU, no model download.

Usage::

    from tts.voice_engine import VoiceEngine
    engine = VoiceEngine()
    audio_bytes = engine.synthesize("Welcome to the show!")  # returns MP3
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_TEXT_LEN = 500
# Theatrical, authoritative voice — best fit for the Caine GM persona.
_VOICE = "en-US-ChristopherNeural"

# Module-level import so tests can patch "tts.voice_engine.edge_tts".
try:
    import edge_tts  # type: ignore[import-untyped]
except ImportError:
    edge_tts = None  # type: ignore[assignment]


class VoiceEngine:
    """Edge-TTS wrapper for the Caine GM voice.

    Instantiate once at app startup; call ``synthesize()`` repeatedly.

    Parameters
    ----------
    reference_wav:
        Accepted but unused — kept for API compatibility with existing callers.
    """

    def __init__(self, reference_wav: Path | None = None) -> None:
        if edge_tts is None:
            raise ImportError(
                "edge-tts is required.\n"
                "Install with:  pip install edge-tts"
            )
        logger.info("VoiceEngine ready (edge-tts voice=%s)", _VOICE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """Convert *text* to MP3 audio bytes via Edge neural TTS.

        Runs the async Edge-TTS call inside a fresh event loop so this method
        is safe to call from a ThreadPoolExecutor worker thread.

        Returns
        -------
        bytes
            Raw MP3 audio data (decodable by Web Audio API ``decodeAudioData``).

        Raises
        ------
        ValueError
            If *text* is empty or exceeds 500 characters.
        RuntimeError
            If synthesis produces no audio.
        """
        text = text.strip()
        if not text:
            raise ValueError("text must be a non-empty string")
        if len(text) > _MAX_TEXT_LEN:
            raise ValueError(
                f"text exceeds maximum length of {_MAX_TEXT_LEN} characters "
                f"(got {len(text)})"
            )

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._synthesize_async(text))
        finally:
            loop.close()

    async def _synthesize_async(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice=_VOICE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio = buf.getvalue()
        if not audio:
            raise RuntimeError("Edge TTS produced no audio")
        return audio
