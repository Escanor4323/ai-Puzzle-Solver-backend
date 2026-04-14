"""Elo rating system for adaptive puzzle difficulty.

Maintains separate Elo ratings per player per puzzle category (logic,
wordplay, pattern, spatial, lateral thinking).  Each puzzle attempt is
a "match" between the player's rating and the puzzle's difficulty.
Targets a 65-75% success rate to maintain flow state.
"""

from __future__ import annotations

import math


class EloSystem:
    """Manages Elo ratings for players and puzzles."""

    # Streak scalar caps: streak 1→1.0x, 3→1.15x, 5→1.30x, 7+→1.50x
    _STREAK_BREAKPOINTS: tuple[tuple[int, float], ...] = (
        (7, 1.50),
        (5, 1.30),
        (3, 1.15),
        (1, 1.00),
    )

    # Maze speed bonus: avg seconds-per-move → multiplier
    # Below 1.5 s/move = blazing fast (+50%), above 5 s = no bonus
    _SPEED_BREAKPOINTS: tuple[tuple[float, float], ...] = (
        (1.5, 1.50),
        (2.5, 1.25),
        (3.5, 1.10),
        (5.0, 1.00),
    )

    def __init__(
        self,
        initial_rating: float = 1200.0,
        k_factor_new: float = 48.0,
        k_factor_stable: float = 16.0,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor_new = k_factor_new
        self.k_factor_stable = k_factor_stable

    def expected_score(
        self, player_rating: float, puzzle_rating: float
    ) -> float:
        """Compute the expected probability of the player winning.

        Parameters
        ----------
        player_rating : float
            Player's Elo rating for this category.
        puzzle_rating : float
            Puzzle's difficulty rating.

        Returns
        -------
        float
            Expected score E(θ, β) between 0 and 1.
        """
        return 1.0 / (1.0 + math.pow(10.0, (puzzle_rating - player_rating) / 400.0))

    def update_ratings(
        self,
        player_rating: float,
        puzzle_rating: float,
        won: bool,
        games_played: int = 0,
    ) -> tuple[float, float]:
        """Update both ratings after a puzzle attempt.

        Parameters
        ----------
        player_rating : float
            Current player Elo.
        puzzle_rating : float
            Current puzzle difficulty Elo.
        won : bool
            True if the player solved the puzzle.
        games_played : int
            Total games for K-factor decay.

        Returns
        -------
        tuple[float, float]
            (new_player_rating, new_puzzle_rating).
        """
        k_player = self.k_factor_new if games_played < 30 else self.k_factor_stable
        k_puzzle = self.k_factor_stable

        expected = self.expected_score(player_rating, puzzle_rating)
        actual = 1.0 if won else 0.0

        new_player = player_rating + k_player * (actual - expected)
        new_puzzle = puzzle_rating + k_puzzle * (expected - actual)

        return new_player, new_puzzle

    def select_difficulty(
        self,
        player_rating: float,
        available_ratings: list[float],
    ) -> float:
        """Choose a puzzle difficulty that targets 65-75% success.

        Parameters
        ----------
        player_rating : float
            Player's current Elo for the category.
        available_ratings : list[float]
            Pool of available puzzle difficulty ratings.

        Returns
        -------
        float
            Selected puzzle difficulty rating.
        """
        # Target difficulty where expected win rate ≈ 70%
        # E = 0.70  →  puzzle_rating = player_rating - 400*log10(0.70/0.30) ≈ player - 145
        target = player_rating - 145.0
        if not available_ratings:
            return target
        return min(available_ratings, key=lambda r: abs(r - target))

    # ── Maze-specific ELO helpers ─────────────────────────────────────────

    def streak_scalar(self, win_streak: int) -> float:
        """Return a multiplier for the player's ELO gain based on win streak.

        The scalar rewards consistency: a player on a hot streak earns
        progressively more ELO per win, capped at 1.5× at 7+ consecutive wins.

        Streak  Multiplier
        ──────  ──────────
        < 1        1.00×
        1–2        1.00×
        3–4        1.15×
        5–6        1.30×
        7+         1.50×
        """
        for threshold, multiplier in self._STREAK_BREAKPOINTS:
            if win_streak >= threshold:
                return multiplier
        return 1.0

    def speed_multiplier(self, avg_seconds_per_move: float) -> float:
        """Return a ELO bonus multiplier based on average time per move.

        Faster navigation signals higher maze mastery and earns bonus ELO
        on top of the base delta.  Slow solves (>5 s/move) get no bonus.

        Avg s/move  Multiplier
        ──────────  ──────────
        ≤ 1.5 s       1.50×   (blazing fast)
        ≤ 2.5 s       1.25×   (fast)
        ≤ 3.5 s       1.10×   (normal)
        ≤ 5.0 s       1.00×   (slow — no bonus)
        > 5.0 s       1.00×
        """
        for threshold, multiplier in self._SPEED_BREAKPOINTS:
            if avg_seconds_per_move <= threshold:
                return multiplier
        return 1.0

    def compute_maze_elo_delta(
        self,
        player_rating: float,
        maze_difficulty_elo: int,
        won: bool,
        games_played: int = 0,
        win_streak: int = 0,
        avg_seconds_per_move: float | None = None,
    ) -> float:
        """Compute the ELO delta for a maze completion with bonuses.

        On a win the raw delta is amplified by:
          1. speed_multiplier  — reward fast navigation
          2. streak_scalar     — reward hot streaks

        On a loss the raw (negative) delta is returned unchanged so
        players are always penalised fairly for losing.

        Parameters
        ----------
        player_rating : float
        maze_difficulty_elo : int
            Computed difficulty of this specific maze instance.
        won : bool
        games_played : int
            Total games played (drives K-factor decay).
        win_streak : int
            Current consecutive wins before this maze.
        avg_seconds_per_move : float | None
            Mean seconds between valid moves.  None means no speed bonus.

        Returns
        -------
        float
            Signed ELO delta to apply to the player's maze rating.
        """
        new_player, _ = self.update_ratings(
            player_rating,
            float(maze_difficulty_elo),
            won,
            games_played,
        )
        delta = new_player - player_rating

        if won and delta > 0:
            speed_mult = (
                self.speed_multiplier(avg_seconds_per_move)
                if avg_seconds_per_move is not None
                else 1.0
            )
            streak_mult = self.streak_scalar(win_streak)
            delta *= speed_mult * streak_mult

        return delta

    @staticmethod
    def get_maze_generation_params(
        player_elo: int,
    ) -> dict[str, object]:
        """Map player Elo to maze generation parameters.

        Delegates to the canonical ``elo_to_size`` brackets defined
        in ``game.engine.GameEngine.start_maze``.

        Parameters
        ----------
        player_elo : int
            Player's Elo rating for the maze category.

        Returns
        -------
        dict
            Parameters compatible with MazeGenerationParams.
        """
        # Single source of truth — matches GameEngine.start_maze
        elo_to_size = {
            (0, 1000): (7, 7, 1, 0, 0),       # Easy
            (1000, 1200): (10, 10, 4, 1, 0),   # Medium
            (1200, 1400): (14, 14, 6, 2, 1),   # Hard
            (1400, 1600): (20, 20, None, 3, 2), # Expert
            (1600, 9999): (25, 25, None, 5, 3), # Master
        }
        width, height, max_dead, num_rules, num_keys = (
            10, 10, 4, 0, 0,
        )
        for (lo, hi), (w, h, md, nr, nk) in elo_to_size.items():
            if lo <= player_elo < hi:
                width, height, max_dead, num_rules, num_keys = (
                    w, h, md, nr, nk,
                )
                break

        return {
            "width": width,
            "height": height,
            "max_dead_end_depth": max_dead,
            "num_logic_rules": num_rules,
            "num_keys": num_keys,
            "target_elo": player_elo,
        }

