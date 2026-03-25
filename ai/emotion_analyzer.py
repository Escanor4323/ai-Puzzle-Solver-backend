"""Dual-source emotional tone analysis.

Text emotion:  **DistilBERT SST-2** via ONNX Runtime (<5ms per message).
  A small dedicated sentiment model — more accurate than keyword
  heuristics, faster than an API call.  Runs on every player message
  alongside the jailbreak check.

Face emotion:  **DeepFace built-in** (~30ms, included with GhostFaceNet).
  Comes free with the face recognition pipeline — no additional model.

The two signals are combined into a composite emotional state that
drives dynamic tone guidance in the LLM system prompt.
"""

from __future__ import annotations

from typing import Any


class EmotionAnalyzer:
    """Detects and tracks player emotional state from text and face.

    Attributes
    ----------
    _text_classifier : onnxruntime.InferenceSession | None
        DistilBERT SST-2 ONNX session for text sentiment.
    _trajectory : dict[str, list]
        Per-player emotional trajectory (last N data points).
    """

    def __init__(self) -> None:
        self._text_classifier = None
        self._tokenizer = None
        self._trajectory: dict[str, list[dict]] = {}

    async def initialize(self) -> None:
        """Load the DistilBERT SST-2 ONNX model and tokenizer.

        The model is loaded once and kept in memory for the
        process lifetime (~67MB RAM).
        """
        pass

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyse a player message for sentiment via DistilBERT.

        Parameters
        ----------
        text : str
            Player input text.

        Returns
        -------
        dict[str, Any]
            ``{"label": "POSITIVE"|"NEGATIVE", "score": float,
            "emotional_state": EmotionalState}``
        """
        pass

    def record_face_emotion(
        self, player_id: str, emotion: str, confidence: float
    ) -> None:
        """Record a facial-expression emotion from DeepFace.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        emotion : str
            Detected emotion label from DeepFace.
        confidence : float
            Detection confidence (0.0–1.0).
        """
        pass

    def get_current_state(
        self, player_id: str
    ) -> dict[str, Any]:
        """Return the composite emotional state for a player.

        Combines the most recent text sentiment and face emotion
        into a single assessment.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        dict[str, Any]
            Current emotional assessment.
        """
        pass

    def get_trajectory(
        self, player_id: str, last_n: int = 5
    ) -> list[dict[str, Any]]:
        """Return emotional trajectory over the last N interactions.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        last_n : int
            Number of recent data points.

        Returns
        -------
        list[dict[str, Any]]
            Chronological emotional state snapshots.
        """
        return self._trajectory.get(player_id, [])[-last_n:]
