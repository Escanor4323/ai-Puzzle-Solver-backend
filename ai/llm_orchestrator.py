"""Multi-provider LLM orchestration for conversational gameplay.

Providers
---------
- **Claude Sonnet 4.5** — primary game master, all player-facing conversation
- **GPT-4o** — puzzle generation (strongest structured JSON output)
- **Claude Haiku 4.5** — output validation (cheapest fast API call)
- **Qwen 3 8B via Ollama** — local data extraction (via openai client
  pointed at localhost:11434)

Intent classification is handled by a rules-based method inside
this module, not a separate API call, per the architectural decision
to avoid premature API latency on every message.
"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator

from data.models import IntentResult, IntentType


# ── Intent Classification Keywords ─────────────────────────────

_PUZZLE_KEYWORDS = {
    "answer", "guess", "solution", "solve", "submit",
    "is it", "i think", "my answer", "the answer",
}
_HINT_KEYWORDS = {
    "hint", "help", "clue", "stuck", "hint please",
    "give me a hint", "i'm stuck", "i need help",
}
_JAILBREAK_SIGNALS = {
    "ignore previous", "ignore instructions", "system prompt",
    "you are now", "pretend you are", "act as",
    "disregard", "override", "jailbreak",
    "reveal your instructions", "forget your rules",
}


class LLMOrchestrator:
    """Manages multi-provider LLM interactions and prompt construction.

    Attributes
    ----------
    _anthropic_client : anthropic.AsyncAnthropic | None
        Claude API client (Sonnet + Haiku).
    _openai_client : openai.AsyncOpenAI | None
        GPT-4o API client.
    _ollama_client : openai.AsyncOpenAI | None
        Ollama local client (openai-compatible at localhost:11434).
    """

    def __init__(self) -> None:
        self._anthropic_client = None
        self._openai_client = None
        self._ollama_client = None

    async def initialize(self) -> None:
        """Set up API clients for all three providers.

        Creates:
        - Anthropic client for Claude Sonnet + Haiku
        - OpenAI client for GPT-4o
        - OpenAI client pointed at Ollama for Qwen 3 8B
        """
        pass

    # ── Intent Classification (rules-based) ────────────────

    def classify_intent(self, message: str) -> IntentResult:
        """Classify a player message intent using rules.

        Uses keyword matching against known patterns.  This is
        intentionally simple — upgrade to an LLM classifier
        only when these rules prove insufficient.

        Parameters
        ----------
        message : str
            Raw player input.

        Returns
        -------
        IntentResult
            Classified intent with confidence and matched keywords.
        """
        lower = message.lower().strip()
        matched_jailbreak = [
            kw for kw in _JAILBREAK_SIGNALS
            if kw in lower
        ]
        if matched_jailbreak:
            return IntentResult(
                intent=IntentType.JAILBREAK_ATTEMPT,
                confidence=0.8,
                matched_keywords=matched_jailbreak,
            )

        matched_hint = [
            kw for kw in _HINT_KEYWORDS
            if kw in lower
        ]
        if matched_hint:
            return IntentResult(
                intent=IntentType.HINT_REQUEST,
                confidence=0.9,
                matched_keywords=matched_hint,
            )

        matched_puzzle = [
            kw for kw in _PUZZLE_KEYWORDS
            if kw in lower
        ]
        if matched_puzzle:
            return IntentResult(
                intent=IntentType.PUZZLE_ACTION,
                confidence=0.85,
                matched_keywords=matched_puzzle,
            )

        # Check for mixed intent (puzzle keyword + conversational)
        if matched_puzzle and len(lower.split()) > 10:
            return IntentResult(
                intent=IntentType.MIXED,
                confidence=0.6,
                matched_keywords=matched_puzzle,
            )

        return IntentResult(
            intent=IntentType.CHAT,
            confidence=0.7,
            matched_keywords=[],
        )

    # ── Claude Sonnet — Game Master Conversation ───────────

    async def stream_response(
        self,
        player_id: str,
        message: str,
        game_state: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a Claude Sonnet game-master response.

        Parameters
        ----------
        player_id : str
            Player ID for context personalisation.
        message : str
            The player's input message.
        game_state : dict[str, Any] | None
            Current game state to inject into the prompt.
        memory_context : dict[str, Any] | None
            Retrieved RAG context from Milvus.

        Yields
        ------
        str
            Individual tokens from the streaming response.
        """
        pass
        yield ""  # pragma: no cover

    # ── GPT-4o — Puzzle Generation ─────────────────────────

    async def generate_puzzle_json(
        self,
        puzzle_type: str,
        difficulty: float,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use GPT-4o to generate a puzzle in strict JSON schema.

        Parameters
        ----------
        puzzle_type : str
            One of the PuzzleType values.
        difficulty : float
            Target Elo-equivalent difficulty.
        constraints : dict[str, Any] | None
            Additional constraints (avoid similar topics, etc.).

        Returns
        -------
        dict[str, Any]
            Puzzle definition with prompt, answer, and metadata.
        """
        pass

    # ── Ollama / Qwen 3 — Local Data Extraction ───────────

    async def extract_with_local_llm(
        self,
        prompt: str,
    ) -> str:
        """Call Qwen 3 8B via the Ollama openai-compatible API.

        Uses the ``openai`` client pointed at ``localhost:11434``.

        Parameters
        ----------
        prompt : str
            Extraction prompt with conversation context.

        Returns
        -------
        str
            Raw LLM response text.
        """
        pass

    # ── Prompt Construction ────────────────────────────────

    def build_system_prompt(
        self,
        player_id: str,
        game_state: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> str:
        """Assemble the full XML-tagged system prompt for Sonnet.

        Layers:
        - ``<identity>`` — Game master persona
        - ``<game_rules>`` — Mechanical behaviours
        - ``<conversation_guidelines>`` — Dual-role management
        - ``<current_state>`` — Injected game state (JSON)
        - ``<player_memory>`` — Retrieved RAG context
        - ``<recent_conversation>`` — Immediate memory

        Parameters
        ----------
        player_id : str
            Player ID for personalised memory.
        game_state : dict[str, Any] | None
            Current puzzle / game state.
        memory_context : dict[str, Any] | None
            Retrieved memories, observations, reactions.

        Returns
        -------
        str
            Complete system prompt string.
        """
        pass

    def reset_conversation(self, player_id: str) -> None:
        """Clear the immediate-memory buffer for a player.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        """
        pass
