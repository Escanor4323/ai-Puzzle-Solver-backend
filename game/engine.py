"""Core game state machine.

Owns the authoritative game state for each player.  The LLM never
tracks state — it only narrates around the deterministic mechanics
managed here.  Handles puzzle lifecycle (load, check, advance),
answer validation, and state serialisation for prompt injection.
"""

from __future__ import annotations

from typing import Any


class GameEngine:
    """Deterministic game-state manager."""

    def __init__(self) -> None:
        pass

    def load_or_create_state(
        self, player_id: str
    ) -> dict[str, Any]:
        """Load existing game state or initialise a new one.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        dict[str, Any]
            Current game state.
        """
        pass

    def check_answer(
        self, player_id: str, answer: str
    ) -> dict[str, Any]:
        """Validate a player's answer against the current puzzle.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        answer : str
            The player's submitted answer.

        Returns
        -------
        dict[str, Any]
            Result with is_correct, feedback, and updated state.
        """
        pass

    def request_hint(
        self, player_id: str
    ) -> dict[str, Any]:
        """Advance the hint tier for the current puzzle.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        dict[str, Any]
            Hint tier info and instruction text.
        """
        pass

    def set_puzzle(
        self, player_id: str, puzzle: dict[str, Any]
    ) -> None:
        """Set a new puzzle as the active puzzle for a player.

        Parameters
        ----------
        player_id : str
            Unique player identifier.
        puzzle : dict[str, Any]
            Puzzle definition from the generator.
        """
        pass

    def get_state_for_prompt(
        self, player_id: str
    ) -> dict[str, Any]:
        """Return a compact representation of game state for LLM context.

        Parameters
        ----------
        player_id : str
            Unique player identifier.

        Returns
        -------
        dict[str, Any]
            Minimal game-state dict safe to inject into prompts.
        """
        pass
