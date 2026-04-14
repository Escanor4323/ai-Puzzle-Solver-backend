"""Lightweight provocation detection for PuzzleMind chat messages.

Detects player taunts that should trigger an "Intention Run" — a single
maze run at maximum difficulty (ELO 1800) as a response to the challenge.

Design choices:
- Pure keyword/regex: zero LLM cost, <1ms latency, deterministic.
- Word-boundary anchors prevent false positives from partial matches.
- Case-insensitive matching after punctuation strip.
- Conservative keyword list — expand based on player feedback over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProvocationResult:
    """Result of a provocation detection check."""
    detected: bool
    trigger_phrase: str   # matched phrase (empty string if not detected)
    category: str         # "boast" | "challenge" | "dismissal" | ""


# ── Compiled pattern groups ────────────────────────────────────────────────

_BOAST = re.compile(
    r"\b("
    r"too easy|is that all|pathetic|boring|can'?t beat me|"
    r"child'?s play|yawn|weak|laughable|joke of a maze|"
    r"embarrassing|not even trying|effortless|a joke|trivial"
    r")\b",
    re.IGNORECASE,
)

_CHALLENGE = re.compile(
    r"\b("
    r"bring it(?: on)?|show me what you'?ve? got|do your worst|"
    r"try harder|step it up|give me a real(?: maze| challenge)?|"
    r"you can'?t stop me|i dare you|make it harder|"
    r"is that your best|that'?s all you got|got anything harder"
    r")\b",
    re.IGNORECASE,
)

_DISMISSAL = re.compile(
    r"\b("
    r"lame|whatever|not impressed|that'?s it\??|seriously\??|"
    r"how disappointing|how boring|so easy|way too easy|"
    r"wake me up|falling asleep|put me to sleep|snooze"
    r")\b",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("boast", _BOAST),
    ("challenge", _CHALLENGE),
    ("dismissal", _DISMISSAL),
]

# Strip punctuation that doesn't appear inside words. Keep apostrophes so
# contractions like "can't" remain intact for word-boundary matching.
_STRIP_RE = re.compile(r"[!?.,\"]+")


class ProvocationDetector:
    """Stateless provocation detector.

    Usage::

        detector = ProvocationDetector()
        result = detector.detect("is that all you've got?")
        if result.detected:
            # arm intention run
    """

    def detect(self, message: str) -> ProvocationResult:
        """Check whether *message* contains a provocation phrase.

        Parameters
        ----------
        message:
            Raw player chat message.

        Returns
        -------
        ProvocationResult
            ``detected=True`` with the matched phrase and category if a
            provocation is found; otherwise ``detected=False``.
        """
        # Normalise: strip leading/trailing whitespace, collapse runs of
        # spaces, remove punctuation that could break word-boundary anchors.
        normalised = _STRIP_RE.sub(" ", message.strip())

        for category, pattern in _PATTERNS:
            match = pattern.search(normalised)
            if match:
                return ProvocationResult(
                    detected=True,
                    trigger_phrase=match.group(0).lower(),
                    category=category,
                )

        return ProvocationResult(detected=False, trigger_phrase="", category="")
