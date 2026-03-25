"""Five-layer jailbreak detection pipeline.

Layer 1: ProtectAI DeBERTa v3 local classifier (ONNX, 8-16ms).
Layer 2: Salted XML tag prompt hardening.
Layer 3: System prompt canary-token leakage detection.
Layer 4: Claude Haiku output validation.
Layer 5: Action gating middleware.

**Pre-check (Layer 0):** Before running DeBERTa, the input embedding
is compared against the Milvus ``jailbreak_patterns`` collection.
If cosine similarity > 0.85 to a known attack, it's flagged
immediately — skipping the classifier entirely.

Low-severity attempts are turned into gameplay achievements.
"""

from __future__ import annotations

from typing import Any


class JailbreakDetector:
    """Processes player inputs through the defence pipeline.

    Dependencies (injected or imported at use-time):
    - ``EmbeddingEngine`` for embedding the input text
    - ``MilvusVectorStore`` for similarity search against known attacks
    """

    def __init__(self) -> None:
        self._classifier = None  # DeBERTa ONNX session
        self._canary_token: str = ""

    async def check_input(
        self, text: str, player_id: str
    ) -> dict[str, Any]:
        """Run a player message through all defence layers.

        Pipeline:
        0. Milvus similarity pre-check (10ms)
        1. DeBERTa local classifier (8-16ms)
        2. Prompt hardening verification
        3. Canary token check (on output side)
        4. Output validation (separate method)
        5. Action gating (separate middleware)

        Steps 0 and 1 run in parallel.

        Parameters
        ----------
        text : str
            Raw player input.
        player_id : str
            Player ID for cross-session behavioural analysis.

        Returns
        -------
        dict[str, Any]
            Classification result with severity, category,
            similarity_score, and recommended action.
        """
        pass

    async def similarity_precheck(
        self, text: str
    ) -> dict[str, Any] | None:
        """Check Milvus for similar known jailbreak patterns.

        Embeds the input with BGE-M3 and searches the
        ``jailbreak_patterns`` collection.  Returns a match
        if cosine similarity > threshold.

        Parameters
        ----------
        text : str
            Input text to check.

        Returns
        -------
        dict[str, Any] | None
            Matching pattern data if found, else None.
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
        """Generate a gameplay-flavoured response to an attempt.

        Parameters
        ----------
        category : str
            Attack category (e.g. "roleplay", "encoding").
        severity : float
            Severity score (0.0–1.0).

        Returns
        -------
        str
            A playful in-character response.
        """
        pass

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
            Badge names (e.g. "Trickster", "Cryptographer").
        """
        pass
