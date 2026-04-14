"""Dual-source emotional tone analysis.

Text emotion:  **DistilBERT SST-2** via transformers pipeline (<10ms).
  SST-2 outputs POSITIVE/NEGATIVE with a score.  We enhance with keyword
  detection to map to the 6-state EmotionalState enum.

Face emotion:  **DeepFace built-in** (~30ms, included with GhostFaceNet).
  Comes free with the face recognition pipeline — no additional model.

The two signals are combined into a composite emotional state (70% text,
30% face) that drives dynamic tone guidance in the LLM system prompt.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from data.models import EmotionalState

logger = logging.getLogger(__name__)


# ── Keyword Sets for SST-2 Enhancement ────────────────────────

_FRUSTRATION_KEYWORDS = frozenset({
    "ugh", "stuck", "hate", "impossible", "give up",
    "stupid", "annoying", "frustrated", "can't",
    "this sucks", "no way", "ridiculous",
})

_EXCITEMENT_KEYWORDS = frozenset({
    "omg", "yes", "got it", "wow", "amazing",
    "awesome", "brilliant", "finally", "yesss",
    "let's go", "incredible",
})

_CONFUSION_KEYWORDS = frozenset({
    "what", "huh", "confused", "don't understand",
    "what do you mean", "makes no sense", "lost",
    "how does", "i don't get it",
})

_BOREDOM_KEYWORDS = frozenset({
    "meh", "whatever", "boring", "ok", "k", "sure",
    "fine", "next", "skip",
})

_AMUSEMENT_KEYWORDS = frozenset({
    "lol", "haha", "lmao", "funny", "hilarious",
    "that's good", "nice one", "joke",
})

# DeepFace emotion → EmotionalState mapping
_FACE_EMOTION_MAP: dict[str, EmotionalState] = {
    "happy": EmotionalState.EXCITED,
    "sad": EmotionalState.FRUSTRATED,
    "angry": EmotionalState.FRUSTRATED,
    "surprise": EmotionalState.EXCITED,
    "fear": EmotionalState.CONFUSED,
    "disgust": EmotionalState.FRUSTRATED,
    "neutral": EmotionalState.NEUTRAL,
}


class EmotionAnalyzer:
    """Detects and tracks player emotional state from text and face.

    Lazy-loads DistilBERT SST-2 on first ``analyze_text`` call.
    """

    def __init__(self) -> None:
        self._text_classifier = None
        self._text_history: dict[str, deque] = {}
        self._face_history: dict[str, deque] = {}

    def _load_text_classifier(self) -> None:
        """Lazy load DistilBERT SST-2 for text sentiment."""
        from transformers import pipeline

        logger.info("Loading emotion text classifier (DistilBERT SST-2)...")
        self._text_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
            device="cpu",
        )
        logger.info("Emotion text classifier loaded")

    # ── Text Analysis ─────────────────────────────────────────

    def analyze_text(
        self, text: str, player_id: str
    ) -> EmotionalState:
        """Analyse a player message for emotional state.

        Combines DistilBERT SST-2 sentiment with keyword detection
        to produce a 6-state classification.

        Parameters
        ----------
        text : str
            Player input text.
        player_id : str
            Player identifier for history tracking.

        Returns
        -------
        EmotionalState
            Detected emotional state.
        """
        if not self._text_classifier:
            self._load_text_classifier()

        # Run DistilBERT
        result = self._text_classifier(text)[0]
        label: str = result["label"]  # "POSITIVE" or "NEGATIVE"
        score: float = result["score"]  # confidence 0.0-1.0

        # Keyword enhancement
        text_lower = text.lower()

        has_frustration = any(
            kw in text_lower for kw in _FRUSTRATION_KEYWORDS
        )
        has_excitement = any(
            kw in text_lower for kw in _EXCITEMENT_KEYWORDS
        )
        has_confusion = any(
            kw in text_lower for kw in _CONFUSION_KEYWORDS
        )
        has_boredom = any(
            kw in text_lower for kw in _BOREDOM_KEYWORDS
        )
        has_amusement = any(
            kw in text_lower for kw in _AMUSEMENT_KEYWORDS
        )
        exclamation_count = text.count("!")

        # Combine model output + keywords
        state: EmotionalState

        if label == "NEGATIVE":
            if has_frustration:
                state = EmotionalState.FRUSTRATED
            elif has_confusion:
                state = EmotionalState.CONFUSED
            elif len(text.strip()) < 5 or has_boredom:
                state = EmotionalState.BORED
            elif score > 0.9:
                state = EmotionalState.FRUSTRATED
            else:
                state = EmotionalState.NEUTRAL
        else:  # POSITIVE
            if has_excitement or exclamation_count >= 2:
                state = EmotionalState.EXCITED
            elif has_amusement:
                state = EmotionalState.AMUSED
            elif score > 0.95:
                state = EmotionalState.EXCITED
            else:
                state = EmotionalState.NEUTRAL

        # Record in rolling history
        history = self._text_history.setdefault(
            player_id, deque(maxlen=20)
        )
        history.append({
            "state": state.value,
            "label": label,
            "score": score,
            "source": "text",
        })

        return state

    # ── Face Emotion ──────────────────────────────────────────

    def record_face_emotion(
        self, player_id: str, emotion: str, confidence: float = 1.0
    ) -> None:
        """Record a facial-expression emotion from DeepFace.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        emotion : str
            Detected emotion label from DeepFace
            (happy, sad, angry, surprise, fear, disgust, neutral).
        confidence : float
            Detection confidence (0.0-1.0).
        """
        mapped = _FACE_EMOTION_MAP.get(
            emotion.lower(), EmotionalState.NEUTRAL
        )

        history = self._face_history.setdefault(
            player_id, deque(maxlen=20)
        )
        history.append({
            "state": mapped.value,
            "raw_emotion": emotion,
            "confidence": confidence,
            "source": "face",
        })

    # ── Composite State ───────────────────────────────────────

    def get_current_state(
        self, player_id: str
    ) -> EmotionalState:
        """Return the composite emotional state for a player.

        Combines text (70% weight) and face (30% weight) histories.
        Recent readings are weighted more heavily.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        EmotionalState
            Current composite emotional state.
        """
        text_history = self._text_history.get(player_id, deque())
        face_history = self._face_history.get(player_id, deque())

        if not text_history and not face_history:
            return EmotionalState.NEUTRAL

        # Count weighted votes from recent entries
        votes: dict[str, float] = {}

        # Text history (70% weight)
        for i, entry in enumerate(text_history):
            # More recent = higher weight
            weight = 0.7 * (i + 1) / max(len(text_history), 1)
            state = entry["state"]
            votes[state] = votes.get(state, 0.0) + weight

        # Face history (30% weight)
        for i, entry in enumerate(face_history):
            weight = (
                0.3
                * (i + 1)
                / max(len(face_history), 1)
                * entry.get("confidence", 1.0)
            )
            state = entry["state"]
            votes[state] = votes.get(state, 0.0) + weight

        if not votes:
            return EmotionalState.NEUTRAL

        # Return the state with the highest weighted vote
        winner = max(votes, key=lambda k: votes[k])
        try:
            return EmotionalState(winner)
        except ValueError:
            return EmotionalState.NEUTRAL

    # ── Trajectory ────────────────────────────────────────────

    def get_trajectory(
        self, player_id: str, last_n: int = 10
    ) -> list[str]:
        """Return emotional trajectory as a list of state names.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        last_n : int
            Number of recent data points.

        Returns
        -------
        list[str]
            Chronological emotional state names.
        """
        text_history = self._text_history.get(
            player_id, deque()
        )
        return [
            entry["state"]
            for entry in list(text_history)[-last_n:]
        ]
