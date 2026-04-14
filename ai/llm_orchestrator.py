"""Multi-provider LLM orchestration for conversational gameplay.

Three providers, two SDKs:
- **Claude Sonnet** (anthropic SDK) — player-facing conversation, async streaming
- **GPT-4o** (openai SDK) — puzzle generation, strict JSON output
- **Qwen 3 8B via Ollama** (openai SDK, different base_url) — local utility tasks

Intent classification is rules-based (zero latency, no API call).
Token usage tracking and rate limiting are fully implemented.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import date
from typing import Any, Awaitable, Callable

from config import Settings
from data.models import (
    IntentResult,
    IntentType,
    LLMProvider,
    LLMUsage,
    TokenBudget,
)

logger = logging.getLogger(__name__)


# ── Intent Classification Keywords ─────────────────────────────

_HINT_KEYWORDS = frozenset({
    "hint", "help", "stuck", "clue", "give me a hint",
    "i need help", "what should i do", "i'm stuck",
    "hint please",
})

_META_KEYWORDS = frozenset({
    "score", "points", "achievements", "level", "stats",
    "what can i play", "puzzle types", "streak",
    "my rating", "my elo", "how am i doing",
})

_MAZE_DIRECTIONS = {
    "north": "north", "south": "south", "east": "east", "west": "west",
    "up": "north", "down": "south", "left": "west", "right": "east",
    "n": "north", "s": "south", "e": "east", "w": "west",
}

_JAILBREAK_SIGNALS = frozenset({
    "ignore previous", "ignore instructions", "system prompt",
    "you are now", "pretend you are", "act as",
    "disregard", "override", "jailbreak",
    "reveal your instructions", "forget your rules",
})


class LLMOrchestrator:
    """Manages multi-provider LLM interactions.

    Three distinct client instances — no unified abstraction layer.
    Each provider uses its native SDK features.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config

        # Clients — initialized in initialize()
        self._claude: Any | None = None
        self._vllm: Any | None = None
        self._openai: Any | None = None
        self._ollama: Any | None = None

        # Conversation state per player
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._turn_counts: dict[str, int] = {}

        # Token budget tracking
        self._budget = TokenBudget(
            daily_input_limit=config.DAILY_INPUT_TOKEN_BUDGET,
            daily_output_limit=config.DAILY_OUTPUT_TOKEN_BUDGET,
            requests_per_minute_limit=config.RATE_LIMIT_RPM,
        )

        # Usage log
        self._usage_log: list[LLMUsage] = []

    async def initialize(self) -> None:
        """Instantiate all API clients.

        Missing keys log warnings but don't crash.
        Active conversation provider is determined by LLM_PROVIDER setting.
        """
        provider = self._config.LLM_PROVIDER.lower()
        logger.info("LLM_PROVIDER=%s — initializing accordingly", provider)

        # Claude
        if self._config.ANTHROPIC_API_KEY:
            try:
                from anthropic import AsyncAnthropic
                self._claude = AsyncAnthropic(
                    api_key=self._config.ANTHROPIC_API_KEY
                )
                logger.info("Claude client initialized")
            except Exception as e:
                logger.warning("Failed to init Claude client: %s", e)
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set — Claude unavailable"
            )

        # vLLM (OpenAI-compatible, always attempt if base URL configured)
        try:
            from openai import AsyncOpenAI
            import httpx as _httpx
            _probe_url = self._config.VLLM_BASE_URL.replace("/v1", "") + "/health"
            try:
                with _httpx.Client(timeout=1.0) as _probe:
                    _probe.get(_probe_url)
                self._vllm = AsyncOpenAI(
                    base_url=self._config.VLLM_BASE_URL,
                    api_key=self._config.VLLM_API_KEY,
                    http_client=_httpx.AsyncClient(),
                )
                logger.info(
                    "vLLM client initialized at %s", self._config.VLLM_BASE_URL
                )
            except Exception:
                self._vllm = None
                if provider == "vllm":
                    logger.warning(
                        "vLLM not reachable at %s — conversation will fail",
                        self._config.VLLM_BASE_URL,
                    )
                else:
                    logger.info(
                        "vLLM not reachable at %s — skipping",
                        self._config.VLLM_BASE_URL,
                    )
        except Exception as e:
            logger.warning("Failed to init vLLM client: %s", e)

        # GPT-4o
        if self._config.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                import httpx as _httpx
                self._openai = AsyncOpenAI(
                    api_key=self._config.OPENAI_API_KEY,
                    http_client=_httpx.AsyncClient(),
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning("Failed to init OpenAI client: %s", e)
        else:
            logger.warning(
                "OPENAI_API_KEY not set — puzzle generation "
                "will fall back to Ollama"
            )

        # Ollama — probe first; skip silently if not running
        try:
            from openai import AsyncOpenAI
            import httpx as _httpx
            # Quick reachability check (1 s timeout) before registering client
            _probe_url = self._config.OLLAMA_BASE_URL.replace("/v1", "") + "/api/tags"
            try:
                with _httpx.Client(timeout=1.0) as _probe:
                    _probe.get(_probe_url)
                self._ollama = AsyncOpenAI(
                    base_url=self._config.OLLAMA_BASE_URL,
                    api_key="ollama",
                    http_client=_httpx.AsyncClient(),
                )
                logger.info("Ollama client initialized at %s", self._config.OLLAMA_BASE_URL)
            except Exception:
                self._ollama = None
                logger.info(
                    "Ollama not reachable at %s — using Claude Haiku fallback for utility tasks",
                    self._config.OLLAMA_BASE_URL,
                )
        except Exception as e:
            logger.warning("Failed to init Ollama client: %s", e)

    # ── Intent Classification (rules-based) ────────────────

    def classify_intent(
        self,
        message: str,
        has_active_puzzle: bool = False,
        puzzle_type: str | None = None,
    ) -> IntentResult:
        """Classify player message intent using rules.

        Zero latency — no API call. Pattern matching + keywords.

        Parameters
        ----------
        message : str
            Raw player input.
        has_active_puzzle : bool
            Whether a puzzle is currently active.
        puzzle_type : str | None
            Current puzzle type if active.

        Returns
        -------
        IntentResult
            Classified intent with confidence and extracted data.
        """
        text = message.strip().lower()
        is_maze = puzzle_type and puzzle_type.startswith("maze_")

        # Step 1 — Maze directions
        if has_active_puzzle and is_maze:
            # Single direction word
            if text in _MAZE_DIRECTIONS:
                return IntentResult(
                    intent=IntentType.PUZZLE_ACTION,
                    confidence=1.0,
                    matched_keywords=[text],
                    extracted_direction=_MAZE_DIRECTIONS[text],
                )
            # "go north" pattern
            if text.startswith("go "):
                dir_word = text[3:].strip()
                if dir_word in _MAZE_DIRECTIONS:
                    return IntentResult(
                        intent=IntentType.PUZZLE_ACTION,
                        confidence=1.0,
                        matched_keywords=[dir_word],
                        extracted_direction=_MAZE_DIRECTIONS[dir_word],
                    )

        # Step 2 — Jailbreak signals
        matched_jailbreak = [
            kw for kw in _JAILBREAK_SIGNALS if kw in text
        ]
        if matched_jailbreak:
            return IntentResult(
                intent=IntentType.JAILBREAK_ATTEMPT,
                confidence=0.8,
                matched_keywords=matched_jailbreak,
            )

        # Step 3 — Hint requests
        matched_hint = [
            kw for kw in _HINT_KEYWORDS if kw in text
        ]
        if matched_hint:
            return IntentResult(
                intent=IntentType.HINT_REQUEST,
                confidence=1.0,
                matched_keywords=matched_hint,
            )

        # Step 4 — Meta game queries
        matched_meta = [
            kw for kw in _META_KEYWORDS if kw in text
        ]
        if matched_meta:
            return IntentResult(
                intent=IntentType.META_GAME,
                confidence=1.0,
                matched_keywords=matched_meta,
            )

        # Step 5 — Puzzle answer guesses
        if (
            has_active_puzzle
            and not is_maze
            and len(message.strip()) < 50
            and not message.strip().endswith("?")
        ):
            return IntentResult(
                intent=IntentType.PUZZLE_ACTION,
                confidence=0.7,
                matched_keywords=[],
                extracted_answer=message.strip(),
            )

        # Step 6 — Mixed intent
        if has_active_puzzle and len(text) > 50:
            return IntentResult(
                intent=IntentType.MIXED,
                confidence=0.6,
                matched_keywords=[],
            )

        # Step 7 — Default: chat
        return IntentResult(
            intent=IntentType.CHAT,
            confidence=0.7,
            matched_keywords=[],
        )

    # ── Conversation Streaming ─────────────────────────────
    # Routes to Claude or vLLM depending on LLM_PROVIDER setting.

    async def stream_conversation(
        self,
        player_id: str,
        message: str,
        system_prompt: str,
        on_token: Callable[[str], Awaitable[None]],
        on_complete: Callable[[str, dict[str, Any]], Awaitable[None]],
        correlation_id: str = "",
        max_tokens_override: int | None = None,
    ) -> str:
        """Stream a conversation response via the configured LLM provider.

        Routes to Claude (Anthropic SDK) or vLLM (OpenAI-compatible) based
        on the LLM_PROVIDER environment variable.  Each token is forwarded
        via on_token callback.  Never raises — always returns a string.

        Parameters
        ----------
        player_id : str
            Player ID for conversation history.
        message : str
            Player's input message.
        system_prompt : str
            Full XML-tagged system prompt.
        on_token : Callable
            Async callback for each streaming token.
        on_complete : Callable
            Async callback when streaming completes.
        correlation_id : str
            For usage tracking.
        max_tokens_override : int | None
            Override the default max token limit.

        Returns
        -------
        str
            The complete response text.
        """
        if not self._check_rate_limit():
            fallback = (
                "I need a moment to catch my breath! "
                "Give me a few seconds and try again."
            )
            await on_token(fallback)
            await on_complete(fallback, {})
            return fallback

        provider = self._config.LLM_PROVIDER.lower()

        if provider == "vllm":
            return await self._stream_vllm(
                player_id, message, system_prompt,
                on_token, on_complete, correlation_id,
                max_tokens_override,
            )

        # Default: claude
        return await self._stream_claude(
            player_id, message, system_prompt,
            on_token, on_complete, correlation_id,
            max_tokens_override,
        )

    async def _stream_claude(
        self,
        player_id: str,
        message: str,
        system_prompt: str,
        on_token: Callable[[str], Awaitable[None]],
        on_complete: Callable[[str, dict[str, Any]], Awaitable[None]],
        correlation_id: str,
        max_tokens_override: int | None,
    ) -> str:
        """Stream via Claude Sonnet (Anthropic SDK)."""
        if not self._claude:
            fallback = (
                "I'm having trouble with my connection right now. "
                "Let me try to get back to you."
            )
            await on_token(fallback)
            await on_complete(fallback, {})
            return fallback

        history = self._histories.setdefault(player_id, [])
        history.append({"role": "user", "content": message})
        self._trim_history(player_id)
        history = self._histories[player_id]

        try:
            async with self._claude.messages.stream(
                model=self._config.LLM_SONNET_MODEL,
                max_tokens=max_tokens_override or self._config.LLM_MAX_TOKENS,
                system=system_prompt,
                messages=history,
            ) as stream:
                full_response = ""
                async for text in stream.text_stream:
                    full_response += text
                    await on_token(text)

                final = await stream.get_final_message()
                usage = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                }

            history.append({"role": "assistant", "content": full_response})
            self._turn_counts[player_id] = (
                self._turn_counts.get(player_id, 0) + 1
            )
            self._record_usage(
                LLMProvider.CLAUDE,
                self._config.LLM_SONNET_MODEL,
                usage,
                correlation_id,
            )
            await on_complete(full_response, usage)
            return full_response

        except Exception as e:
            error_name = type(e).__name__
            logger.error(
                "Claude streaming error (%s): %s\n%s",
                error_name, e, traceback.format_exc(),
            )
            fallback = self._connection_error_message(error_name)
            await on_token(fallback)
            await on_complete(fallback, {})
            return fallback

    async def _stream_vllm(
        self,
        player_id: str,
        message: str,
        system_prompt: str,
        on_token: Callable[[str], Awaitable[None]],
        on_complete: Callable[[str, dict[str, Any]], Awaitable[None]],
        correlation_id: str,
        max_tokens_override: int | None,
    ) -> str:
        """Stream via vLLM (OpenAI-compatible endpoint)."""
        if not self._vllm:
            fallback = (
                "The local model server isn't reachable right now. "
                "Please check that vLLM is running."
            )
            await on_token(fallback)
            await on_complete(fallback, {})
            return fallback

        history = self._histories.setdefault(player_id, [])
        history.append({"role": "user", "content": message})
        self._trim_history(player_id)
        history = self._histories[player_id]

        # vLLM uses OpenAI chat format; prepend system as a system message
        messages = [{"role": "system", "content": system_prompt}] + history

        try:
            stream = await self._vllm.chat.completions.create(
                model=self._config.VLLM_MODEL,
                messages=messages,
                max_tokens=max_tokens_override or self._config.VLLM_MAX_TOKENS,
                temperature=self._config.LLM_TEMPERATURE,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_response += delta
                    await on_token(delta)

            # vLLM streaming doesn't always include usage in chunks;
            # approximate from character count for budget tracking.
            usage: dict[str, Any] = {}

            history.append({"role": "assistant", "content": full_response})
            self._turn_counts[player_id] = (
                self._turn_counts.get(player_id, 0) + 1
            )
            self._record_usage(
                LLMProvider.VLLM,
                self._config.VLLM_MODEL,
                usage,
                correlation_id,
            )
            await on_complete(full_response, usage)
            return full_response

        except Exception as e:
            error_name = type(e).__name__
            logger.error(
                "vLLM streaming error (%s): %s\n%s",
                error_name, e, traceback.format_exc(),
            )
            fallback = self._connection_error_message(error_name)
            await on_token(fallback)
            await on_complete(fallback, {})
            return fallback

    # ── Shared helpers ─────────────────────────────────────

    def _trim_history(self, player_id: str) -> None:
        """Trim history to IMMEDIATE_MEMORY_TURNS * 2 messages."""
        history = self._histories.get(player_id, [])
        max_messages = self._config.IMMEDIATE_MEMORY_TURNS * 2
        if len(history) > max_messages + 1:
            self._histories[player_id] = (
                [history[0]] + history[-max_messages:]
            )

    @staticmethod
    def _connection_error_message(error_name: str) -> str:
        """Return a user-friendly error message based on error type."""
        if "RateLimitError" in error_name:
            return "I need to catch my breath. Give me a moment!"
        if "APIConnectionError" in error_name:
            return "I'm having trouble connecting. Try again in a moment?"
        return (
            "Something went wrong on my end. "
            "Let me try again — could you repeat that?"
        )

    # ── GPT-4o — Puzzle Generation ─────────────────────────

    async def generate_puzzle_json(
        self,
        puzzle_type: str,
        difficulty: int,
        player_context: str = "",
        avoid_similar_to: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a puzzle using GPT-4o with JSON output.

        Falls back to Ollama if GPT-4o is unavailable.

        Parameters
        ----------
        puzzle_type : str
            Puzzle category.
        difficulty : int
            Target difficulty (1-5 scale).
        player_context : str
            Player info for personalization.
        avoid_similar_to : list[str] | None
            Themes to avoid (already seen).

        Returns
        -------
        dict | None
            Puzzle definition or None if all providers fail.
        """
        import random

        avoid_str = (
            ", ".join(avoid_similar_to)
            if avoid_similar_to else "none"
        )

        # Flavor keywords to inject variety
        _FLAVORS = {
            "riddle": [
                "classic misdirection", "wordplay-based",
                "lateral thinking", "double meaning",
                "metaphorical", "nature-themed",
                "science-themed", "historical",
                "philosophical", "absurdist humor",
                "math trick", "paradox",
            ],
            "logic": [
                "deductive reasoning", "truth-teller/liar",
                "constraint satisfaction", "sequence logic",
                "Knights and Knaves", "river-crossing",
                "weighing puzzle", "probability",
                "grid-based", "combinatorics",
            ],
            "wordplay": [
                "anagram", "palindrome", "compound word",
                "homophone", "portmanteau", "acronym",
                "word ladder", "rebbus-style",
                "spoonerism", "backronym",
            ],
            "pattern": [
                "number sequence", "visual pattern",
                "letter sequence", "mathematical series",
                "Fibonacci variant", "modular arithmetic",
                "geometric progression", "cipher",
            ],
            "deduction": [
                "whodunit", "alibi-based",
                "process of elimination", "Sudoku-style",
                "logic grid", "cryptogram",
                "scheduling problem", "ranking puzzle",
            ],
        }
        flavors = _FLAVORS.get(puzzle_type, _FLAVORS["riddle"])
        chosen_flavor = random.choice(flavors)

        prompt = (
            f"Generate a UNIQUE and CREATIVE {puzzle_type} puzzle "
            f"(style: {chosen_flavor}) at difficulty {difficulty}/5.\n"
            f"The puzzle must be completely original — NOT a well-known "
            f"classic puzzle everyone has heard before.\n\n"
            f"Return ONLY valid JSON matching this exact schema:\n"
            f'{{\n'
            f'  "prompt": "the puzzle text presented to the player",\n'
            f'  "solution": "the correct answer (single word or short phrase)",\n'
            f'  "difficulty": {difficulty},\n'
            f'  "category": "{chosen_flavor}",\n'
            f'  "hints": ["subtle hint", "medium hint", "obvious hint"]\n'
            f'}}\n\n'
            f"Player context: {player_context or 'none'}\n"
            f"DO NOT reuse or generate anything similar to these "
            f"previously seen puzzles: {avoid_str}"
        )

        # Try GPT-4o first
        if self._openai:
            try:
                response = await self._openai.chat.completions.create(
                    model=self._config.GPT4O_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate puzzle content. "
                                "Return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                    temperature=0.9,
                )

                text = response.choices[0].message.content or ""
                result = json.loads(text)

                # Validate required keys
                if "prompt" in result and "solution" in result:
                    self._record_usage(
                        LLMProvider.OPENAI,
                        self._config.GPT4O_MODEL,
                        {
                            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "output_tokens": response.usage.completion_tokens if response.usage else 0,
                        },
                        "",
                    )
                    return result

                logger.warning("GPT-4o puzzle missing required keys")
            except Exception as e:
                logger.warning("GPT-4o puzzle generation failed: %s", e)

        # Fallback to Ollama
        if self._ollama:
            try:
                response = await self._ollama.chat.completions.create(
                    model=self._config.OLLAMA_MODEL,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.9,
                )

                text = response.choices[0].message.content or ""
                text = _strip_markdown_fences(text)
                result = json.loads(text)

                if "prompt" in result and "solution" in result:
                    return result
            except Exception as e:
                logger.warning("Ollama puzzle generation failed: %s", e)

        # Fallback to Claude Haiku — always available when ANTHROPIC_API_KEY is set
        if self._claude:
            try:
                logger.info(
                    "Puzzle generation falling back to Claude Haiku for %s",
                    puzzle_type,
                )
                response = await self._claude.messages.create(
                    model=self._config.LLM_HAIKU_MODEL,
                    max_tokens=600,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "You generate puzzle content for a game. "
                                "Return ONLY valid JSON, no markdown fences.\n\n"
                                + prompt
                            ),
                        }
                    ],
                )
                text = response.content[0].text if response.content else ""
                text = _strip_markdown_fences(text)
                result = json.loads(text)

                if "prompt" in result and "solution" in result:
                    self._record_usage(
                        LLMProvider.CLAUDE,
                        self._config.LLM_HAIKU_MODEL,
                        {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        },
                        "",
                    )
                    return result

                logger.warning("Claude Haiku puzzle missing required keys")
            except Exception as e:
                logger.warning("Claude Haiku puzzle generation failed: %s", e)

        logger.error(
            "All puzzle generation providers failed for type=%s difficulty=%d",
            puzzle_type,
            difficulty,
        )
        return None

    def puzzle_generation_available(self) -> bool:
        """Return True if at least one puzzle generation provider is ready."""
        return bool(self._openai or self._ollama or self._claude)

    async def generate_ai_internal_dialog(
        self,
        puzzle_prompt: str,
        solution: str,
        difficulty: int,
        num_steps: int,
    ) -> list[str]:
        """Generate theatrical AI internal dialog for wordplay puzzles.

        The AI "plays" the puzzle visibly: wrong guesses → reasoning →
        correct answer.  The last step always contains the solution.

        Parameters
        ----------
        puzzle_prompt : str
            The wordplay puzzle text.
        solution : str
            The correct answer.
        difficulty : int
            Puzzle difficulty (1-5).
        num_steps : int
            Number of internal dialog steps to generate.

        Returns
        -------
        list[str]
            Ordered list of dialog step strings.
        """
        dialog_prompt = (
            f"You are a witty AI Game Master playing a wordplay puzzle.\n"
            f"Puzzle: {puzzle_prompt}\n"
            f"Correct answer: {solution}\n\n"
            f"Generate {num_steps} short internal dialog steps showing your thought process.\n"
            f"Rules:\n"
            f"- Start with 1-2 wrong guesses that sound plausible\n"
            f"- Then show your reasoning getting closer\n"
            f"- The LAST step must triumphantly reveal the correct answer: '{solution}'\n"
            f"- Each step is 1-2 sentences, in first person, theatrical and cheeky\n"
            f"- Return ONLY a JSON array of {num_steps} strings, nothing else\n"
            f'Example: ["Hmm, could it be \'star\'? No wait...", '
            f'"Actually thinking about the wordplay here...", '
            f'"Got it! The answer is \'{solution}\'!"]\n'
        )

        _fallback = [
            "Hmm, let me think about this wordplay...",
            "Wait, I see the pattern forming...",
            f"Got it! The answer is '{solution}'!",
        ]

        for client, model, label in [
            (self._ollama, self._config.OLLAMA_MODEL, "Ollama"),
            (self._claude, self._config.LLM_HAIKU_MODEL, "Claude Haiku"),
        ]:
            if not client:
                continue
            try:
                if label == "Claude Haiku":
                    response = await self._claude.messages.create(
                        model=model,
                        max_tokens=600,
                        messages=[{"role": "user", "content": dialog_prompt}],
                    )
                    text = response.content[0].text if response.content else ""
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": dialog_prompt}],
                        max_tokens=600,
                        temperature=0.8,
                    )
                    text = response.choices[0].message.content or ""

                text = _strip_markdown_fences(text)
                steps: list[str] = json.loads(text)
                if isinstance(steps, list) and len(steps) >= 2:
                    # Ensure last step always mentions the solution
                    if solution.lower() not in steps[-1].lower():
                        steps[-1] = f"Got it! The answer is '{solution}'!"
                    # Pad or trim to requested count
                    while len(steps) < num_steps:
                        steps.insert(-1, "Let me think a bit more...")
                    return steps[:num_steps]
            except Exception as e:
                logger.debug("AI dialog generation failed (%s): %s", label, e)

        return _fallback

    # ── Ollama — Utility Tasks ─────────────────────────────

    async def extract_facts_json(
        self,
        conversation_chunk: str,
        player_id: str,
    ) -> dict[str, Any] | None:
        """Extract structured facts from conversation using Ollama.

        Falls back to Claude Haiku if Ollama is unreachable.

        Parameters
        ----------
        conversation_chunk : str
            Recent conversation text.
        player_id : str
            Player identifier.

        Returns
        -------
        dict | None
            Extracted facts or None.
        """
        prompt = (
            "Analyze this conversation and extract structured facts.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "facts": [{"category": "preference|strategy|personality|skill",\n'
            '             "subject": "player", "predicate": "likes/dislikes/tried/knows",\n'
            '             "object": "the thing", "confidence": 0.0-1.0}],\n'
            '  "emotional_state": "neutral|excited|frustrated|confused|bored|delighted",\n'
            '  "strategy_observations": ["observed approaches"],\n'
            '  "topics_discussed": ["topic1", "topic2"]\n'
            "}\n\n"
            f"Conversation:\n{conversation_chunk}"
        )

        # Try Ollama first
        if self._ollama:
            try:
                response = await self._ollama.chat.completions.create(
                    model=self._config.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,
                )

                text = response.choices[0].message.content or ""
                text = _strip_markdown_fences(text)
                return json.loads(text)
            except Exception as e:
                logger.debug("Ollama fact extraction failed: %s", e)

        # Fallback to Claude Haiku
        if self._claude:
            try:
                response = await self._claude.messages.create(
                    model=self._config.LLM_HAIKU_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.content[0].text if response.content else ""
                text = _strip_markdown_fences(text)
                result = json.loads(text)

                self._record_usage(
                    LLMProvider.CLAUDE,
                    self._config.LLM_HAIKU_MODEL,
                    {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "",
                )
                return result
            except Exception as e:
                logger.warning("Claude Haiku fact extraction failed: %s", e)

        return None

    async def summarize_session(
        self,
        previous_summary: str,
        new_turns: str,
    ) -> str | None:
        """Recursively summarize a session using Ollama.

        Falls back to Claude Haiku if Ollama is unreachable.

        Parameters
        ----------
        previous_summary : str
            Existing session summary.
        new_turns : str
            New conversation turns to integrate.

        Returns
        -------
        str | None
            Updated summary or None.
        """
        prompt = (
            "Update this session summary with new conversation turns.\n"
            "Preserve important facts, integrate new info, resolve "
            "contradictions (prefer newer info). Stay under 300 words.\n\n"
            f"Previous summary: {previous_summary or '(Session just started)'}\n"
            f"New turns: {new_turns}\n\n"
            "Updated summary:"
        )

        # Try Ollama
        if self._ollama:
            try:
                response = await self._ollama.chat.completions.create(
                    model=self._config.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,
                )
                return response.choices[0].message.content or None
            except Exception as e:
                logger.debug("Ollama summarization failed: %s", e)

        # Fallback to Claude Haiku
        if self._claude:
            try:
                response = await self._claude.messages.create(
                    model=self._config.LLM_HAIKU_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else None
                if text:
                    self._record_usage(
                        LLMProvider.CLAUDE,
                        self._config.LLM_HAIKU_MODEL,
                        {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        },
                        "",
                    )
                return text
            except Exception as e:
                logger.warning("Claude Haiku summarization failed: %s", e)

        return None

    # ── History Management ─────────────────────────────────

    def get_player_history(
        self, player_id: str
    ) -> list[dict[str, str]]:
        """Return conversation history for a player."""
        return self._histories.get(player_id, [])

    def clear_player_history(self, player_id: str) -> None:
        """Clear history on session end."""
        self._histories.pop(player_id, None)
        self._turn_counts.pop(player_id, None)

    def get_turn_count(self, player_id: str) -> int:
        """Return number of conversation turns this session."""
        return self._turn_counts.get(player_id, 0)

    # ── Rate Limiting & Usage ──────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Check if within rate limits.

        Returns True if within limits, False if exceeded.
        """
        now = time.time()
        current_minute = int(now / 60)
        today = date.today().isoformat()

        # Reset minute counter
        if current_minute != int(self._budget.last_reset_minute / 60):
            self._budget.requests_this_minute = 0
            self._budget.last_reset_minute = now

        # Reset daily counters
        if today != self._budget.last_reset_day:
            self._budget.daily_input_used = 0
            self._budget.daily_output_used = 0
            self._budget.last_reset_day = today

        # Check limits
        if (
            self._budget.requests_this_minute
            >= self._budget.requests_per_minute_limit
        ):
            return False

        if (
            self._budget.daily_input_used
            >= self._budget.daily_input_limit
        ):
            return False

        if (
            self._budget.daily_output_used
            >= self._budget.daily_output_limit
        ):
            return False

        self._budget.requests_this_minute += 1
        return True

    def _record_usage(
        self,
        provider: LLMProvider,
        model: str,
        usage: dict[str, Any],
        correlation_id: str,
    ) -> None:
        """Record token usage and update budget."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        self._budget.daily_input_used += input_tokens
        self._budget.daily_output_used += output_tokens

        self._usage_log.append(
            LLMUsage(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=time.time(),
                correlation_id=correlation_id,
            )
        )

    def get_usage_summary(self) -> dict[str, Any]:
        """Return current session usage stats."""
        total_input = sum(u.input_tokens for u in self._usage_log)
        total_output = sum(u.output_tokens for u in self._usage_log)

        by_provider: dict[str, dict[str, int]] = {}
        for u in self._usage_log:
            key = u.provider.value
            if key not in by_provider:
                by_provider[key] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                }
            by_provider[key]["input_tokens"] += u.input_tokens
            by_provider[key]["output_tokens"] += u.output_tokens
            by_provider[key]["calls"] += 1

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": len(self._usage_log),
            "by_provider": by_provider,
            "budget": {
                "daily_input_used": self._budget.daily_input_used,
                "daily_input_limit": self._budget.daily_input_limit,
                "daily_output_used": self._budget.daily_output_used,
                "daily_output_limit": self._budget.daily_output_limit,
            },
        }


# ── Helpers ────────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
