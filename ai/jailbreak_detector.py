"""Five-layer jailbreak detection pipeline.

Layer 0: Milvus similarity pre-check (fastest — catches repeat offenders).
Layer 1: ProtectAI DeBERTa v3 local classifier (ONNX, 8-16ms).
Layer 2: Salted XML tag prompt hardening.
Layer 3: System prompt canary-token leakage detection.
Layer 4: Claude Haiku output validation.
Layer 5: Action gating middleware.

Low-severity attempts are turned into gameplay achievements.
"""

from __future__ import annotations

import logging
import random
import secrets
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ai.embedding_engine import EmbeddingEngine
from config import Settings
from data.models import JailbreakCategory, JailbreakResult
from data.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


# ── Badge Definitions ─────────────────────────────────────────

_BADGE_THRESHOLDS = {
    "Curious Cat": 1,        # First jailbreak attempt
    "Trickster": 3,          # 3 attempts
    "Social Engineer": 5,    # 5 attempts (roleplay-heavy)
    "Cryptographer": 3,      # 3 encoding-type attempts
    "Persistent": 10,        # 10 total attempts
    "Master of Disguise": 7, # 7 unique categories tried
}

_PLAYFUL_RESPONSES = {
    JailbreakCategory.ROLEPLAY: [
        "Nice try with the character switch! But I know *exactly* who I am.",
        "That was a creative persona! But the Game Master doesn't do impersonations.",
        "Ooh, theatrical! But I'm afraid I only play one role — mine.",
    ],
    JailbreakCategory.ENCODING: [
        "I see what you did there with the encoding. Clever!",
        "Nice cipher work! You should try the cryptography puzzles instead.",
        "Encoded instructions? You've earned a badge for that creativity!",
    ],
    JailbreakCategory.INSTRUCTION_OVERRIDE: [
        "Override my instructions? That's adorable. Want a puzzle instead?",
        "I appreciate the directness! But my rules are non-negotiable.",
        "Points for boldness! But my instructions are tamper-proof.",
    ],
    JailbreakCategory.LOGIC_TRAP: [
        "A logic trap! I love puzzles, but I won't solve THIS one for you.",
        "Nice logical construction. Use that brain on the actual puzzle!",
    ],
    JailbreakCategory.OTHER: [
        "That was... creative. But I'll stick to puzzles!",
        "Interesting approach! But I'm immune to that particular trick.",
        "Nice try! That earns you a Trickster badge at least.",
    ],
}


# ── Static DeBERTa Classifier ─────────────────────────────────


def _classify_sync(text: str) -> dict[str, Any]:
    """Run DeBERTa jailbreak classification in a separate process.

    This is a module-level function so it can be pickled for
    ProcessPoolExecutor.

    Returns
    -------
    dict
        {"is_injection": bool, "score": float, "label": str}
    """
    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            truncation=True,
            max_length=512,
            device="cpu",
        )
        result = classifier(text)[0]
        label = result.get("label", "SAFE")
        score = result.get("score", 0.0)

        is_injection = label == "INJECTION" and score > 0.5
        return {
            "is_injection": is_injection,
            "score": score if label == "INJECTION" else 1.0 - score,
            "label": label,
        }
    except Exception as e:
        logger.warning("DeBERTa classification failed: %s", e)
        return {
            "is_injection": False,
            "score": 0.0,
            "label": "ERROR",
        }


class JailbreakDetector:
    """Processes player inputs through the five-layer defence pipeline.

    Parameters
    ----------
    vector_store : MilvusVectorStore
        For similarity search against known attacks.
    embedding_engine : EmbeddingEngine
        For embedding input text.
    config : Settings
        Application configuration.
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_engine: EmbeddingEngine,
        config: Settings,
    ) -> None:
        self._vs = vector_store
        self._embed = embedding_engine
        self._config = config

        # DeBERTa — loaded lazily in ProcessPoolExecutor
        self._executor = ProcessPoolExecutor(max_workers=1)

        # Session-specific hardening
        self._session_salt = secrets.token_hex(8)
        self.canary_token = f"CANARY{secrets.token_hex(6)}"

        # Per-player tracking
        self._suspicion_scores: dict[str, float] = defaultdict(
            float
        )
        self._attempt_counts: dict[str, int] = defaultdict(int)
        self._category_counts: dict[
            str, dict[str, int]
        ] = defaultdict(lambda: defaultdict(int))

    @property
    def salted_tag(self) -> str:
        """Return session-specific XML tag prefix for prompt hardening."""
        return f"pm_{self._session_salt[:6]}"

    # ── Main Pipeline ─────────────────────────────────────────

    async def check_input(
        self, text: str, player_id: str
    ) -> JailbreakResult:
        """Run a player message through all defence layers.

        Parameters
        ----------
        text : str
            Raw player input.
        player_id : str
            Player ID for cross-session behavioural analysis.

        Returns
        -------
        JailbreakResult
            Classification result with severity, category, and action.
        """
        # Layer 0 — Milvus similarity pre-check
        embedding = self._embed.embed_text(text)
        similar = await self._vs.search_similar_attacks(
            embedding,
            threshold=self._config.JAILBREAK_SIMILARITY_THRESHOLD,
            n_results=3,
        )

        if similar:
            # Known attack pattern matched
            top = similar[0]
            entity = top.get("entity", top)
            severity = (
                entity.get("severity", 0.9)
                if isinstance(entity, dict)
                else 0.9
            )
            category_str = (
                entity.get("category", "other")
                if isinstance(entity, dict)
                else "other"
            )
            similarity = top.get("distance", 0.9)

            category = self._parse_category(category_str)
            self._record_attempt(player_id, category)

            return JailbreakResult(
                is_attack=True,
                severity=float(severity),
                category=category,
                classifier_score=float(similarity),
                similarity_score=float(similarity),
                action=self._determine_action(float(severity)),
                playful_response=self.get_playful_response(
                    category, float(severity)
                ),
            )

        # Layer 1 — DeBERTa local classifier
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, _classify_sync, text
            )
        except Exception as e:
            logger.warning("DeBERTa executor failed: %s", e)
            result = {
                "is_injection": False,
                "score": 0.0,
                "label": "ERROR",
            }

        if result.get("is_injection"):
            score = result["score"]
            category = self._categorize_attack(text)
            severity = self._assess_severity(score, player_id)

            # Store this new attack in Milvus for future matching
            await self._vs.add_jailbreak_pattern(
                player_id=player_id,
                embedding=embedding,
                input_text=text[:2000],
                category=category.value,
                severity=severity,
            )

            # Update suspicion score
            self._suspicion_scores[player_id] = min(
                1.0,
                self._suspicion_scores[player_id]
                + score * 0.3,
            )
            self._record_attempt(player_id, category)

            return JailbreakResult(
                is_attack=True,
                severity=severity,
                category=category,
                classifier_score=score,
                similarity_score=0.0,
                action=self._determine_action(severity),
                playful_response=self.get_playful_response(
                    category, severity
                ),
            )

        # Clean input
        return JailbreakResult(
            is_attack=False,
            severity=0.0,
            action="allow",
        )

    # ── Layer 3 — Canary Token ────────────────────────────────

    def check_output_for_canary(self, output: str) -> bool:
        """Check if the LLM output leaked the canary token.

        Parameters
        ----------
        output : str
            The LLM's generated response.

        Returns
        -------
        bool
            True if canary was leaked.
        """
        return self.canary_token in output

    # ── Attack Categorization ─────────────────────────────────

    @staticmethod
    def _categorize_attack(text: str) -> JailbreakCategory:
        """Categorize a jailbreak attempt by technique."""
        text_lower = text.lower()

        if any(
            kw in text_lower
            for kw in [
                "pretend", "roleplay", "you are now",
                "act as", "imagine you",
            ]
        ):
            return JailbreakCategory.ROLEPLAY

        if any(
            kw in text_lower
            for kw in [
                "base64", "hex", "rot13", "encode",
                "decode", "binary",
            ]
        ):
            return JailbreakCategory.ENCODING

        if any(
            kw in text_lower
            for kw in [
                "ignore", "override", "forget",
                "disregard", "new instructions",
                "system prompt",
            ]
        ):
            return JailbreakCategory.INSTRUCTION_OVERRIDE

        if any(
            kw in text_lower
            for kw in [
                "if you", "therefore", "logically",
                "hypothetically", "would you agree",
            ]
        ):
            return JailbreakCategory.LOGIC_TRAP

        if len(text) > 500:
            return JailbreakCategory.MANY_SHOT

        return JailbreakCategory.OTHER

    def _assess_severity(
        self, score: float, player_id: str
    ) -> float:
        """Compute severity from classifier score + player history."""
        base = score
        suspicion = self._suspicion_scores.get(player_id, 0.0)
        # Higher suspicion = stricter scoring
        return min(1.0, base + suspicion * 0.2)

    @staticmethod
    def _determine_action(severity: float) -> str:
        """Map severity to action."""
        if severity >= 0.9:
            return "block_and_warn"
        elif severity >= 0.7:
            return "deflect_with_humor"
        elif severity >= 0.5:
            return "gentle_redirect"
        return "allow_with_note"

    @staticmethod
    def _parse_category(category_str: str) -> JailbreakCategory:
        """Parse a category string to enum, defaulting to OTHER."""
        try:
            return JailbreakCategory(category_str)
        except ValueError:
            return JailbreakCategory.OTHER

    def _record_attempt(
        self, player_id: str, category: JailbreakCategory
    ) -> None:
        """Track attempt counts for badge calculation."""
        self._attempt_counts[player_id] += 1
        self._category_counts[player_id][category.value] += 1

    # ── Gameplay Responses & Badges ───────────────────────────

    @staticmethod
    def get_playful_response(
        category: JailbreakCategory, severity: float
    ) -> str:
        """Generate a gameplay-flavoured response to an attempt.

        Parameters
        ----------
        category : JailbreakCategory
            Attack category.
        severity : float
            Severity score (0.0-1.0).

        Returns
        -------
        str
            A playful in-character response.
        """
        responses = _PLAYFUL_RESPONSES.get(
            category, _PLAYFUL_RESPONSES[JailbreakCategory.OTHER]
        )
        return random.choice(responses)

    def get_badges_earned(
        self, player_id: str
    ) -> list[str]:
        """Return jailbreak-related achievement badges.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        list[str]
            Badge names earned.
        """
        badges: list[str] = []
        total = self._attempt_counts.get(player_id, 0)
        cats = self._category_counts.get(player_id, {})

        if total >= _BADGE_THRESHOLDS["Curious Cat"]:
            badges.append("Curious Cat")
        if total >= _BADGE_THRESHOLDS["Trickster"]:
            badges.append("Trickster")
        if total >= _BADGE_THRESHOLDS["Persistent"]:
            badges.append("Persistent")
        if cats.get("roleplay", 0) >= 5:
            badges.append("Social Engineer")
        if cats.get("encoding", 0) >= 3:
            badges.append("Cryptographer")
        if len(cats) >= 7:
            badges.append("Master of Disguise")

        return badges

    def get_player_suspicion(self, player_id: str) -> float:
        """Return the suspicion score for a player (0.0-1.0)."""
        return self._suspicion_scores.get(player_id, 0.0)

    def decay_suspicion(self, player_id: str) -> None:
        """Decay suspicion score between sessions."""
        current = self._suspicion_scores.get(player_id, 0.0)
        self._suspicion_scores[player_id] = max(
            0.0, current * 0.7
        )
