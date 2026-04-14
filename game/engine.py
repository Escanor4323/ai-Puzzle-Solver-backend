"""Core game state machine.

Owns the authoritative game state for each player.  The LLM never
tracks state — it only narrates around the deterministic mechanics
managed here.  Handles puzzle lifecycle (load, check, advance),
answer validation, and state serialisation for prompt injection.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from data.models import (
    MazeGenerationParams,
    MazeHint,
    MazeMoveResult,
    MazeState,
    PuzzleType,
)
from game.elo_system import EloSystem
from game.puzzle_types.maze import (
    MazeGenerator,
    MazeSolver,
    VisibilityEngine,
)


class GameEngine:
    """Deterministic game-state manager."""

    # Max number of recent puzzle themes to remember for dedup
    _SEEN_PUZZLE_HISTORY = 20
    # Fields that survive logout / server restart
    _PERSISTENT_KEYS = (
        "elo_ratings", "total_solved", "best_streak", "score",
        "ai_score", "total_sessions",
    )

    _logger = logging.getLogger("puzzlemind.engine")

    def __init__(self, persist_dir: str | Path = "data/players") -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._maze_states: dict[str, MazeState] = {}
        self._maze_generator = MazeGenerator()
        self._maze_solver = MazeSolver()
        self._visibility_engine = VisibilityEngine()
        self._elo_system = EloSystem()
        # Track seen puzzle themes per player for deduplication
        self._seen_puzzles: dict[str, list[str]] = {}
        # Track last maze move time per player for stall detection
        self._last_move_times: dict[str, float] = {}
        # Rolling window of recent move intervals for AI race pacing
        self._move_intervals: dict[str, deque] = {}
        # Maze start wall-clock time for speed bonus computation
        self._maze_start_times: dict[str, float] = {}
        # Intention Run state — armed when player provokes, consumed by start_maze
        self._intention_run: dict[str, bool] = {}
        # Snapshot of ELO ratings taken when arming an intention run (for restore)
        self._saved_elo: dict[str, dict[str, float]] = {}
        # Persistent profile dir
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def load_or_create_state(
        self, player_id: str
    ) -> dict[str, Any]:
        """Load existing game state or initialise a new one.

        On first access, tries to restore persistent fields from disk
        (ELO ratings, total_solved, best_streak, etc.).
        """
        if player_id not in self._states:
            profile = self._load_profile(player_id)
            self._states[player_id] = {
                "player_id": player_id,
                "score": profile.get("score", 0),
                "ai_score": profile.get("ai_score", 0),
                "streak": 0,  # streak resets per session
                "best_streak": profile.get("best_streak", 0),
                "total_solved": profile.get("total_solved", 0),
                "total_sessions": profile.get("total_sessions", 0) + 1,
                "elo_ratings": profile.get("elo_ratings", {}),
                "has_active_puzzle": False,
                "puzzle_type": None,
                "current_puzzle": None,
                "attempts": 0,
                "hints_given": 0,
                "max_hints": 3,
                "timer_seconds": 120,
                "puzzle_start_time": None,
            }
            # Bump session count and save
            self._save_profile(player_id)
        return self._states[player_id]

    def set_puzzle(
        self, player_id: str, puzzle: dict[str, Any]
    ) -> None:
        """Set a new puzzle as the active puzzle for a player."""
        state = self.load_or_create_state(player_id)
        state["has_active_puzzle"] = True
        state["puzzle_type"] = puzzle.get("puzzle_type", "riddle")
        state["current_puzzle"] = puzzle
        state["attempts"] = 0
        state["hints_given"] = 0
        state["max_hints"] = len(puzzle.get("hints", [])) or 3
        state["puzzle_start_time"] = time.time()

        # Timer based on difficulty
        difficulty = puzzle.get("difficulty", 2)
        timer_map = {1: 180, 2: 120, 3: 90, 4: 60, 5: 45}
        state["timer_seconds"] = timer_map.get(difficulty, 120)

        # Record puzzle theme for deduplication
        prompt_text = puzzle.get("prompt", "")
        category = puzzle.get("category", "")
        solution = puzzle.get("solution", "")
        theme = f"{category}: {prompt_text[:60]}"
        if solution:
            theme += f" (answer: {solution})"
        seen = self._seen_puzzles.setdefault(player_id, [])
        seen.append(theme)
        # Keep only the last N
        if len(seen) > self._SEEN_PUZZLE_HISTORY:
            self._seen_puzzles[player_id] = seen[
                -self._SEEN_PUZZLE_HISTORY:
            ]

    def check_and_record_move_time(self, player_id: str) -> float:
        """Record the current time as the player's latest maze move.

        Returns the number of seconds elapsed since the previous move,
        or 0.0 on the first move.  Used for stall detection and AI pacing.
        """
        now = time.time()
        elapsed = now - self._last_move_times.get(player_id, now)
        self._last_move_times[player_id] = now
        # Track rolling window of move intervals (ignore stalls >30s)
        if 0.1 < elapsed < 30.0:
            window = self._move_intervals.setdefault(
                player_id, deque(maxlen=10)
            )
            window.append(elapsed)
        return elapsed

    def get_avg_move_interval(self, player_id: str) -> float:
        """Return rolling average move interval in seconds.

        Used to pace the AI race — the AI moves at the same rhythm as
        the player.  Defaults to 2.0s before any moves are recorded.
        Clamped to [0.1, 3.0] so the AI never feels frozen or instant.
        """
        window = self._move_intervals.get(player_id)
        if not window:
            return 2.0
        avg = sum(window) / len(window)
        return max(0.1, min(3.0, avg))

    def reset_move_intervals(self, player_id: str) -> None:
        """Clear move timing history when a new maze starts."""
        self._move_intervals.pop(player_id, None)
        self._last_move_times.pop(player_id, None)

    def get_seen_themes(self, player_id: str) -> list[str]:
        """Return recently seen puzzle themes for deduplication."""
        return self._seen_puzzles.get(player_id, [])

    def get_adaptive_difficulty(self, player_id: str) -> int:
        """Compute difficulty (1–5) based on player performance.

        - Winning streak → harder puzzles
        - Losing streak / timeouts → easier puzzles
        - Defaults to 2
        """
        state = self._states.get(player_id)
        if not state:
            return 2

        score = state.get("score", 0)
        ai_score = state.get("ai_score", 0)
        streak = state.get("streak", 0)
        total = state.get("total_solved", 0)

        # Base difficulty from win ratio
        if total >= 3:
            win_rate = score / max(score + ai_score, 1)
            if win_rate > 0.75:
                base = 4
            elif win_rate > 0.55:
                base = 3
            elif win_rate < 0.3:
                base = 1
            else:
                base = 2
        else:
            base = 2

        # Streak modifier
        if streak >= 3:
            base = min(5, base + 1)
        elif streak <= -2:
            base = max(1, base - 1)

        return base

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Strip answer prefixes and trailing punctuation for comparison."""
        import re
        s = text.strip().lower()
        for prefix in (
            "the answer is ", "answer: ", "answer is ",
            "it is ", "its ", "it's ", "the number is ",
        ):
            if s.startswith(prefix):
                s = s[len(prefix):]
        # Remove trailing punctuation
        s = s.rstrip(".,;:!?").strip()
        return s

    def check_answer(
        self, player_id: str, answer: str
    ) -> dict[str, Any]:
        """Validate a player's answer against the current puzzle."""
        import re
        state = self._states.get(player_id)
        if not state or not state.get("current_puzzle"):
            return {
                "is_correct": False,
                "feedback": "No active puzzle.",
                "attempts": 0,
            }

        puzzle = state["current_puzzle"]
        raw_solution = puzzle.get("solution", "").strip().lower()
        raw_guess = answer.strip().lower()
        state["attempts"] = state.get("attempts", 0) + 1

        # Exact match first
        is_correct = raw_guess == raw_solution

        # Normalized match (strips prefixes/punctuation)
        if not is_correct:
            norm_guess = self._normalize_answer(raw_guess)
            norm_solution = self._normalize_answer(raw_solution)
            is_correct = norm_guess == norm_solution

        # Numeric fuzzy match: last number in guess == last number in solution
        if not is_correct:
            nums_guess = re.findall(r'\d+(?:\.\d+)?', raw_guess)
            nums_solution = re.findall(r'\d+(?:\.\d+)?', raw_solution)
            if nums_guess and nums_solution and nums_guess[-1] == nums_solution[-1]:
                is_correct = True

        solution = raw_solution
        guess = raw_guess
        if is_correct:
            state["score"] = state.get("score", 0) + 1
            state["streak"] = state.get("streak", 0) + 1
            state["total_solved"] = state.get("total_solved", 0) + 1
            best = state.get("best_streak", 0)
            if state["streak"] > best:
                state["best_streak"] = state["streak"]
            state["has_active_puzzle"] = False
            self._save_profile(player_id)
        else:
            # Check partial match
            pass

        return {
            "is_correct": is_correct,
            "attempts": state["attempts"],
            "score": state["score"],
            "ai_score": state.get("ai_score", 0),
            "streak": state["streak"],
            "solved": is_correct,
        }

    def handle_timeout(
        self, player_id: str
    ) -> dict[str, Any]:
        """Handle when the player runs out of time — AI gets a point."""
        state = self._states.get(player_id)
        if not state or not state.get("current_puzzle"):
            return {"ai_scored": False}

        puzzle = state["current_puzzle"]
        state["ai_score"] = state.get("ai_score", 0) + 1
        state["streak"] = 0
        state["has_active_puzzle"] = False
        self._save_profile(player_id)

        return {
            "ai_scored": True,
            "ai_score": state["ai_score"],
            "score": state["score"],
            "solution": puzzle.get("solution", "???"),
            "streak": 0,
        }

    def request_hint(
        self, player_id: str
    ) -> dict[str, Any]:
        """Advance the hint tier for the current puzzle."""
        state = self._states.get(player_id)
        if not state or not state.get("current_puzzle"):
            return {"tier": 1, "hint_text": ""}

        puzzle = state["current_puzzle"]
        hints = puzzle.get("hints", [])
        given = state.get("hints_given", 0)

        if given >= len(hints):
            return {
                "tier": given + 1,
                "hint_text": "No more hints available!",
                "exhausted": True,
            }

        hint_text = hints[given]
        state["hints_given"] = given + 1

        return {
            "tier": given + 1,
            "hint_text": hint_text,
            "exhausted": (given + 1) >= len(hints),
        }

    def get_state_for_prompt(
        self, player_id: str
    ) -> dict[str, Any]:
        """Return a compact representation of game state for LLM context."""
        state = self._states.get(player_id)
        if not state:
            # Check if there's an active maze
            maze = self._maze_states.get(player_id)
            if maze:
                return {
                    "player_id": player_id,
                    "has_active_puzzle": True,
                    "puzzle_type": "maze_classic",
                    "score": 0,
                    "ai_score": 0,
                    "streak": 0,
                    "move_count": maze.move_count,
                    "current_puzzle": {
                        "type": "maze",
                        "width": maze.width,
                        "height": maze.height,
                        "player_position": maze.player_position,
                        "exit": maze.exit,
                        "move_count": maze.move_count,
                    },
                }
            return {}

        # Build compact state (exclude solution from puzzle data)
        result = {
            "player_id": player_id,
            "has_active_puzzle": state.get("has_active_puzzle", False),
            "puzzle_type": state.get("puzzle_type"),
            "score": state.get("score", 0),
            "ai_score": state.get("ai_score", 0),
            "streak": state.get("streak", 0),
            "attempts": state.get("attempts", 0),
            "hints_given": state.get("hints_given", 0),
            "max_hints": state.get("max_hints", 3),
            "timer_seconds": state.get("timer_seconds", 120),
        }

        puzzle = state.get("current_puzzle")
        if puzzle:
            # Include puzzle info but NEVER the solution
            result["current_puzzle"] = {
                "type": puzzle.get("puzzle_type", ""),
                "prompt": puzzle.get("prompt", ""),
                "difficulty": puzzle.get("difficulty", 2),
                "category": puzzle.get("category", ""),
                "hints_available": len(puzzle.get("hints", [])),
                "hints_given": state.get("hints_given", 0),
            }

        # Include maze state if active
        maze = self._maze_states.get(player_id)
        if maze:
            result["maze"] = {
                "width": maze.width,
                "height": maze.height,
                "player_position": maze.player_position,
                "exit": maze.exit,
                "move_count": maze.move_count,
            }

        return result

    # ── Maze Methods ───────────────────────────────────────────

    @staticmethod
    def difficulty_label(elo: float) -> str:
        """Map an ELO rating to a human-readable difficulty label."""
        if elo >= 1600:
            return "Master"
        if elo >= 1400:
            return "Expert"
        if elo >= 1200:
            return "Hard"
        if elo >= 1000:
            return "Medium"
        return "Easy"

    def start_maze(
        self,
        player_id: str,
        maze_type: PuzzleType,
        target_elo: int = 1200,
    ) -> MazeState:
        """Generate and start a new maze for a player."""
        # Each bracket: (width, height, max_dead_end_depth, num_rules, num_keys)
        elo_to_size = {
            (0, 1000): (7, 7, 1, 0, 0),       # Easy   — tiny, no mechanics
            (1000, 1200): (10, 10, 4, 1, 0),   # Medium — moderate corridors
            (1200, 1400): (14, 14, 6, 2, 1),   # Hard   — larger + one key
            (1400, 1600): (20, 20, None, 3, 2), # Expert — open + multi-key
            (1600, 9999): (25, 25, None, 5, 3), # Master — large + many rules
        }

        width, height, max_dead, num_rules, num_keys = (
            10, 10, 4, 0, 0,
        )
        for (lo, hi), (w, h, md, nr, nk) in elo_to_size.items():
            if lo <= target_elo < hi:
                width, height, max_dead, num_rules, num_keys = (
                    w, h, md, nr, nk,
                )
                break

        params = MazeGenerationParams(
            width=width,
            height=height,
            maze_type=maze_type,
            target_elo=target_elo,
            max_dead_end_depth=max_dead,
            num_logic_rules=num_rules if maze_type == PuzzleType.MAZE_LOGIC else 0,
            num_keys=num_keys if maze_type == PuzzleType.MAZE_LOGIC else 0,
        )

        maze = self._maze_generator.generate(params)
        self._maze_states[player_id] = maze
        self._maze_start_times[player_id] = time.monotonic()
        self.reset_move_intervals(player_id)

        # Also set game state
        state = self.load_or_create_state(player_id)
        state["has_active_puzzle"] = True
        state["puzzle_type"] = maze_type.value if hasattr(maze_type, 'value') else str(maze_type)

        return maze

    def process_maze_move(
        self, player_id: str, direction: str
    ) -> MazeMoveResult:
        """Process a player's maze movement."""
        maze = self._maze_states.get(player_id)
        if maze is None:
            return MazeMoveResult(
                valid=False,
                new_position=(0, 0),
                reason="no_maze",
            )

        x, y = maze.player_position
        cell = maze.grid[x][y]

        deltas = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
        }
        if direction not in deltas:
            return MazeMoveResult(
                valid=False,
                new_position=maze.player_position,
                reason="invalid_direction",
                move_count=maze.move_count,
            )

        dx, dy = deltas[direction]
        nx, ny = x + dx, y + dy

        if getattr(cell.walls, direction):
            return MazeMoveResult(
                valid=False,
                new_position=maze.player_position,
                reason="wall",
                move_count=maze.move_count,
            )

        if not (0 <= nx < maze.width and 0 <= ny < maze.height):
            return MazeMoveResult(
                valid=False,
                new_position=maze.player_position,
                reason="wall",
                move_count=maze.move_count,
            )

        target_cell = maze.grid[nx][ny]

        if target_cell.has_door and target_cell.door_color:
            if target_cell.door_color not in maze.items.get("keys", []):
                return MazeMoveResult(
                    valid=False,
                    new_position=maze.player_position,
                    reason="locked_door",
                    move_count=maze.move_count,
                )

        if target_cell.allowed_entry:
            if direction not in target_cell.allowed_entry:
                return MazeMoveResult(
                    valid=False,
                    new_position=maze.player_position,
                    reason="one_way",
                    move_count=maze.move_count,
                )

        maze.player_position = (nx, ny)
        maze.move_count += 1
        if (nx, ny) not in [tuple(v) for v in maze.visited_cells]:
            maze.visited_cells.append((nx, ny))

        items_collected: list[str] = []
        if target_cell.color and target_cell.has_door is False:
            for rule in maze.rules:
                if (
                    rule.rule_type == "locked_door"
                    and rule.params.get("color") == target_cell.color
                ):
                    keys = maze.items.get("keys", [])
                    if target_cell.color not in keys:
                        keys.append(target_cell.color)
                        maze.items["keys"] = keys
                        items_collected.append(target_cell.color)
                    break

        if target_cell.has_door and target_cell.door_color:
            keys = maze.items.get("keys", [])
            if target_cell.door_color in keys:
                keys.remove(target_cell.door_color)
                maze.items["keys"] = keys
                target_cell.has_door = False

        if target_cell.is_teleporter and target_cell.teleport_target:
            tp = target_cell.teleport_target
            maze.player_position = tp
            if tp not in [tuple(v) for v in maze.visited_cells]:
                maze.visited_cells.append(tp)

        # Record move timestamp for speed bonus computation
        now = time.monotonic()
        self._last_move_times[player_id] = now

        reached_exit = maze.player_position == maze.exit
        score = None
        elo_delta: float | None = None
        if reached_exit:
            score = self.compute_maze_score(maze)
            state = self._states.get(player_id)
            if state:
                state["has_active_puzzle"] = False
                state["score"] = state.get("score", 0) + 1
                win_streak = state.get("streak", 0)
                state["streak"] = win_streak + 1
                best = state.get("best_streak", 0)
                if state["streak"] > best:
                    state["best_streak"] = state["streak"]
                state["total_solved"] = state.get("total_solved", 0) + 1

                # ── Maze ELO update with speed + streak bonuses ──────────
                maze_start = self._maze_start_times.get(player_id)
                avg_spm: float | None = None
                if maze_start is not None and maze.move_count > 0:
                    elapsed = now - maze_start
                    avg_spm = elapsed / maze.move_count

                maze_type_key = state.get("puzzle_type", "maze_classic") or "maze_classic"
                elo_key = maze_type_key  # e.g. "maze_classic", "maze_dark", "maze_logic"
                elo_ratings: dict[str, float] = state.setdefault("elo_ratings", {})
                current_elo = elo_ratings.get(elo_key, 1200.0)
                games_played = state.get("total_solved", 1)

                elo_delta = self._elo_system.compute_maze_elo_delta(
                    player_rating=current_elo,
                    maze_difficulty_elo=maze.difficulty_elo,
                    won=True,
                    games_played=games_played,
                    win_streak=win_streak,
                    avg_seconds_per_move=avg_spm,
                )
                elo_ratings[elo_key] = round(current_elo + elo_delta, 1)
                self._save_profile(player_id)
                # Clean up timing data
                self._maze_start_times.pop(player_id, None)

        visible = self._visibility_engine.get_visible_cells(
            maze.grid,
            maze.player_position,
            maze.visited_cells,
        )

        return MazeMoveResult(
            valid=True,
            new_position=maze.player_position,
            reached_exit=reached_exit,
            visible_cells=visible,
            items_collected=items_collected,
            move_count=maze.move_count,
            score=score,
        )

    def get_maze_hint(self, player_id: str) -> MazeHint:
        """Get a maze-specific hint based on current tier."""
        maze = self._maze_states.get(player_id)
        if maze is None:
            return MazeHint(tier=1)

        tier = min(getattr(maze, "_hints_given", 0) + 1, 5)

        path = self._maze_solver.solve(
            maze.grid,
            maze.player_position,
            maze.exit,
            rules=maze.rules,
        )

        hint = MazeHint(tier=tier)

        if path and len(path) > 1:
            next_pos = path[1]
            dx = next_pos[0] - maze.player_position[0]
            dy = next_pos[1] - maze.player_position[1]
            dir_map = {
                (0, -1): "north",
                (0, 1): "south",
                (1, 0): "east",
                (-1, 0): "west",
            }
            next_dir = dir_map.get((dx, dy), "north")

            if tier == 1:
                hint.direction_bias = next_dir
            elif tier == 2:
                wrong = [d for d in ["north", "south", "east", "west"] if d != next_dir]
                hint.wrong_directions = wrong
            elif tier == 3:
                hint.reveal_cells = path[1:4]
            elif tier == 4:
                hint.flash_path = path[1:]
            elif tier == 5:
                hint.auto_move_steps = path[1:6]

        if not hasattr(maze, "_hints_given"):
            object.__setattr__(maze, "_hints_given", 0)
        object.__setattr__(
            maze, "_hints_given",
            getattr(maze, "_hints_given", 0) + 1,
        )

        return hint

    def clear_player_state(self, player_id: str) -> None:
        """Remove transient per-player state on logout.

        Persistent fields (ELO, total_solved, best_streak) are saved
        to disk before the in-memory state is dropped.
        """
        self._save_profile(player_id)  # persist before clearing
        self._states.pop(player_id, None)
        self._maze_states.pop(player_id, None)
        self._maze_start_times.pop(player_id, None)
        self._last_move_times.pop(player_id, None)
        self._move_intervals.pop(player_id, None)
        self._seen_puzzles.pop(player_id, None)
        self._intention_run.pop(player_id, None)
        self._saved_elo.pop(player_id, None)

    def cancel_active_puzzle(self, player_id: str) -> None:
        """Cancel the current puzzle and reset the active state without scoring."""
        state = self._states.get(player_id)
        if state:
            state["has_active_puzzle"] = False
            state["current_puzzle"] = None
        self._maze_states.pop(player_id, None)
        self._maze_start_times.pop(player_id, None)
        self.reset_move_intervals(player_id)

    # ── Intention Run ──────────────────────────────────────────────────────

    _INTENTION_RUN_ELO = 1800
    """ELO used for the challenge maze when an intention run is armed."""

    def arm_intention_run(self, player_id: str) -> None:
        """Arm an intention run for the player.

        Snapshots all current ELO ratings so they can be restored after
        the run.  Has no effect if a run is already armed (one at a time).
        """
        if self._intention_run.get(player_id):
            return  # already armed — don't re-arm mid-run
        state = self._states.get(player_id, {})
        self._saved_elo[player_id] = dict(state.get("elo_ratings", {}))
        self._intention_run[player_id] = True

    def is_intention_run(self, player_id: str) -> bool:
        """Return True if an intention run is currently armed."""
        return bool(self._intention_run.get(player_id))

    def get_intention_run_elo(self, player_id: str) -> int:
        """Return the challenge ELO (1800) if armed, else 0."""
        return self._INTENTION_RUN_ELO if self._intention_run.get(player_id) else 0

    def complete_intention_run(self, player_id: str) -> bool:
        """Finalise and disarm an intention run.

        Restores the player's ELO ratings from the pre-run snapshot,
        discarding any ELO delta accumulated during the challenge maze.

        Returns
        -------
        bool
            True if a run was active and has been completed; False otherwise.
        """
        if not self._intention_run.get(player_id):
            return False
        saved = self._saved_elo.get(player_id, {})
        state = self._states.get(player_id)
        if state is not None and saved:
            state["elo_ratings"] = dict(saved)
        self._intention_run.pop(player_id, None)
        self._saved_elo.pop(player_id, None)
        return True

    # ── Profile Persistence ────────────────────────────────────

    def _profile_path(self, player_id: str) -> Path:
        """Return the filesystem path for a player profile JSON."""
        safe_id = player_id.replace("/", "_").replace("..", "_")
        return self._persist_dir / f"{safe_id}.json"

    def _load_profile(self, player_id: str) -> dict[str, Any]:
        """Load persistent player profile from disk, or return empty dict."""
        path = self._profile_path(player_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._logger.info(
                    "Loaded profile for %s from %s", player_id, path,
                )
                return data
            except (json.JSONDecodeError, OSError) as exc:
                self._logger.warning(
                    "Failed to load profile %s: %s", path, exc,
                )
        return {}

    def _save_profile(self, player_id: str) -> None:
        """Persist only the durable fields to disk."""
        state = self._states.get(player_id)
        if not state:
            return
        profile: dict[str, Any] = {
            k: state[k] for k in self._PERSISTENT_KEYS if k in state
        }
        profile["player_id"] = player_id
        path = self._profile_path(player_id)
        try:
            path.write_text(
                json.dumps(profile, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            self._logger.warning(
                "Failed to save profile %s: %s", path, exc,
            )

    def get_player_stats(self, player_id: str) -> dict[str, Any]:
        """Return player stats for the profile/settings screen."""
        state = self.load_or_create_state(player_id)
        return {
            "elo_ratings": state.get("elo_ratings", {}),
            "total_solved": state.get("total_solved", 0),
            "total_sessions": state.get("total_sessions", 0),
            "streak": state.get("streak", 0),
            "best_streak": state.get("best_streak", 0),
            "score": state.get("score", 0),
            "ai_score": state.get("ai_score", 0),
        }

    def get_maze_state(self, player_id: str) -> "MazeState | None":
        """Return the raw MazeState for a player (used by AI solver)."""
        return self._maze_states.get(player_id)

    def get_maze_render_data(self, player_id: str) -> dict[str, Any]:
        """Return maze data formatted for the frontend renderer."""
        maze = self._maze_states.get(player_id)
        if maze is None:
            return {}

        data: dict[str, Any] = {
            "width": maze.width,
            "height": maze.height,
            "start": maze.start,
            "exit": maze.exit,
            "player_position": maze.player_position,
            "move_count": maze.move_count,
            "items": maze.items,
        }

        visible = self._visibility_engine.get_visible_cells(
            maze.grid,
            maze.player_position,
            maze.visited_cells,
        )

        grid_data: list[list[dict[str, Any] | None]] = []
        for x in range(maze.width):
            row: list[dict[str, Any] | None] = []
            for y in range(maze.height):
                cell = maze.grid[x][y]
                cell_dict: dict[str, Any] = {
                    "x": cell.x,
                    "y": cell.y,
                    "walls": {
                        "north": cell.walls.north,
                        "south": cell.walls.south,
                        "east": cell.walls.east,
                        "west": cell.walls.west,
                    },
                    "color": cell.color,
                    "has_door": cell.has_door,
                    "door_color": cell.door_color,
                    "is_teleporter": cell.is_teleporter,
                }
                row.append(cell_dict)
            grid_data.append(row)

        data["grid"] = grid_data
        data["visible_cells"] = visible
        data["rules"] = [r.model_dump() for r in maze.rules]

        return data

    @staticmethod
    def compute_maze_score(maze_state: MazeState) -> float:
        """Compute maze completion score."""
        if maze_state.move_count == 0:
            return 1.0
        return min(
            1.0,
            maze_state.optimal_path_length / maze_state.move_count,
        )
