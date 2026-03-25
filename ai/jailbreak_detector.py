"""Five-layer jailbreak detection pipeline.

Layer 1: ProtectAI DeBERTa v3 local classifier (ONNX Runtime).
Layer 2: Salted XML tag prompt hardening.
Layer 3: System prompt canary-token leakage detection.
Layer 4: Claude Haiku output validation.
Layer 5: Action gating middleware.

Low-severity attempts are turned into gameplay achievements.
"""

from __future__ import annotations

from typing import Any


class JailbreakDetector:
    """Processes player inputs through the five-layer defence pipeline."""

    def __init__(self) -> None:
        pass

    async def check_input(
        self, text: str, player_id: str
    ) -> dict[str, Any]:
        """Run a player message through all five defence layers.

        Parameters
        ----------
        text : str
            Raw player input.
        player_id : str
            Player ID for cross-session behavioural analysis.

        Returns
        -------
        dict[str, Any]
            Classification result with severity, category, and
            recommended action.
        """
        pass

    def check_output_for_canary(self, output: str) -> bool:
        """Scan an LLM output for system-prompt canary tokens.

        Parameters
        ----------
        output : str
            The LLM's generated response text.

        Returns
        -------
        bool
            True if a canary token was leaked.
        """
        pass

    def get_playful_response(
        self, category: str, severity: float
    ) -> str:
        """Generate a gameplay-flavoured response to a jailbreak attempt.

        Parameters
        ----------
        category : str
            Attack category (e.g. "roleplay", "encoding").
        severity : float
            Severity score (0.0–1.0).

        Returns
        -------
        str
            A playful in-character response for the player.
        """
        pass

    def get_badges_earned(
        self, player_id: str
    ) -> list[str]:
        """Return jailbreak-related achievement badges for a player.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        list[str]
            Badge names earned (e.g. "Trickster", "Cryptographer").
        """
        pass
