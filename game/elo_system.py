"""Elo rating system for adaptive puzzle difficulty.

Maintains separate Elo ratings per player per puzzle category (logic,
wordplay, pattern, spatial, lateral thinking).  Each puzzle attempt is
a "match" between the player's rating and the puzzle's difficulty.
Targets a 65-75% success rate to maintain flow state.
"""

from __future__ import annotations


class EloSystem:
    """Manages Elo ratings for players and puzzles."""

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
        pass

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
        pass

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
        pass
