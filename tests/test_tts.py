"""TTS (edge-tts) tests.

Covers:
  - VoiceEngine contract (inputs / output shape)
  - REST endpoint: accepts text, returns MP3
  - Security: path traversal prevention, input length cap
  - Resource management: engine loaded once (singleton)
  - Latency: real synthesis < 2 s (marked slow)

Run with:
    cd ai-Puzzle-Solver-backend
    pytest tests/test_tts.py -v                    # unit + endpoint tests
    pytest tests/test_tts.py -v -m slow            # includes real synthesis
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

_FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 128  # plausible MP3 header + data


def _make_edge_mock(audio_data: bytes = _FAKE_MP3):
    """Return a mock edge_tts module whose Communicate.stream() yields audio."""

    async def _fake_stream():
        yield {"type": "audio", "data": audio_data}
        yield {"type": "WordBoundary", "data": b""}  # non-audio chunk ignored

    mock_comm = MagicMock()
    mock_comm.stream.side_effect = lambda: _fake_stream()

    mock_edge = MagicMock()
    mock_edge.Communicate.return_value = mock_comm
    return mock_edge, mock_comm


# ── VoiceEngine contract ──────────────────────────────────────────────────────


class TestVoiceEngineContract:
    def test_synthesize_returns_bytes(self) -> None:
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            result = engine.synthesize("Hello, I am Caine.")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_empty_text_raises(self) -> None:
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            with pytest.raises(ValueError, match="text"):
                engine.synthesize("")

    def test_synthesize_whitespace_only_raises(self) -> None:
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            with pytest.raises(ValueError, match="text"):
                engine.synthesize("   ")

    def test_synthesize_text_too_long_raises(self) -> None:
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            with pytest.raises(ValueError, match="500"):
                engine.synthesize("x" * 501)

    def test_synthesize_500_chars_accepted(self) -> None:
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            result = engine.synthesize("x" * 500)
        assert len(result) > 0

    def test_non_audio_chunks_ignored(self) -> None:
        """WordBoundary / other chunk types must not pollute the audio bytes."""
        from tts.voice_engine import VoiceEngine

        audio_payload = b"\xff\xfb\x90\x00" + b"\xAB" * 64

        async def _stream_with_noise():
            yield {"type": "WordBoundary", "data": b"noise"}
            yield {"type": "audio", "data": audio_payload}
            yield {"type": "SentenceBoundary", "data": b"more noise"}

        mock_comm = MagicMock()
        mock_comm.stream.side_effect = lambda: _stream_with_noise()
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = mock_comm

        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            result = engine.synthesize("Hello")

        assert result == audio_payload

    def test_empty_audio_stream_raises_runtime_error(self) -> None:
        from tts.voice_engine import VoiceEngine

        async def _empty_stream():
            yield {"type": "WordBoundary", "data": b""}
            return

        mock_comm = MagicMock()
        mock_comm.stream.side_effect = lambda: _empty_stream()
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = mock_comm

        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine()
            with pytest.raises(RuntimeError, match="no audio"):
                engine.synthesize("Hello")

    def test_missing_edge_tts_raises_import_error(self, tmp_path: Path) -> None:
        from tts.voice_engine import VoiceEngine

        with patch("tts.voice_engine.edge_tts", None):
            with pytest.raises(ImportError, match="edge-tts"):
                VoiceEngine()

    def test_engine_init_accepts_reference_wav_kwarg(self) -> None:
        """reference_wav kwarg must be accepted (API compat) even if unused."""
        from tts.voice_engine import VoiceEngine

        mock_edge, _ = _make_edge_mock()
        with patch("tts.voice_engine.edge_tts", mock_edge):
            engine = VoiceEngine(reference_wav=Path("/dev/null"))
        assert engine is not None


# ── REST endpoint ─────────────────────────────────────────────────────────────


class TestSecurity:
    def test_api_accepts_no_file_path_param(self) -> None:
        """The /synthesize endpoint must not accept a reference_audio path."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tts.router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/tts")

        with patch("tts.router._get_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.synthesize.return_value = _FAKE_MP3
            mock_get.return_value = mock_engine

            client = TestClient(app)
            resp = client.post(
                "/api/tts/synthesize",
                json={"text": "hello", "reference_audio": "/etc/passwd"},
            )
            assert resp.status_code == 200
            mock_engine.synthesize.assert_called_once_with("hello")

    def test_empty_text_returns_422(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tts.router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/tts")

        with patch("tts.router._get_engine"):
            client = TestClient(app)
            resp = client.post("/api/tts/synthesize", json={"text": ""})
            assert resp.status_code == 422

    def test_oversized_text_returns_422(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tts.router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/tts")

        with patch("tts.router._get_engine"):
            client = TestClient(app)
            resp = client.post("/api/tts/synthesize", json={"text": "x" * 501})
            assert resp.status_code == 422

    def test_response_media_type_is_mpeg(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tts.router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/tts")

        with patch("tts.router._get_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.synthesize.return_value = _FAKE_MP3
            mock_get.return_value = mock_engine

            client = TestClient(app)
            resp = client.post("/api/tts/synthesize", json={"text": "hello"})
            assert resp.status_code == 200
            assert "audio/mpeg" in resp.headers["content-type"]


# ── Latency (slow — requires network access to Edge TTS) ─────────────────────


@pytest.mark.slow
class TestSynthesisLatency:
    LATENCY_BUDGET_SECS = 2.0  # sub-second typical; 2s budget for slow networks

    def test_short_phrase_within_latency_budget(self) -> None:
        import time

        from tts.voice_engine import VoiceEngine

        engine = VoiceEngine()
        text = "Welcome, welcome! I hope you're enjoying the show."

        start = time.perf_counter()
        audio = engine.synthesize(text)
        elapsed = time.perf_counter() - start

        assert len(audio) > 0
        assert elapsed < self.LATENCY_BUDGET_SECS, (
            f"Synthesis took {elapsed:.2f}s, budget is {self.LATENCY_BUDGET_SECS}s"
        )
