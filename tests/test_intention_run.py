"""Integration tests for the Intention Run mechanic.

Tests the full lifecycle:
  provocation detected → arm → ELO override → maze start → complete → ELO restored

Run with:
    cd ai-Puzzle-Solver-backend
    pytest tests/test_intention_run.py -v
"""

import pytest

from game.engine import GameEngine
from game.provocation_detector import ProvocationDetector


@pytest.fixture
def engine() -> GameEngine:
    return GameEngine()


@pytest.fixture
def detector() -> ProvocationDetector:
    return ProvocationDetector()


# ── Arm / state checks ────────────────────────────────────────────────────────


class TestArmIntentionRun:
    def test_arm_sets_flag(self, engine: GameEngine) -> None:
        engine.arm_intention_run("p1")
        assert engine.is_intention_run("p1")

    def test_arm_is_idempotent(self, engine: GameEngine) -> None:
        """Calling arm twice should not corrupt the saved ELO snapshot."""
        state = engine.load_or_create_state("p1")
        state["elo_ratings"] = {"maze_classic": 1350.0}

        engine.arm_intention_run("p1")
        # Second arm should be a no-op
        engine.arm_intention_run("p1")
        assert engine.is_intention_run("p1")

    def test_not_armed_by_default(self, engine: GameEngine) -> None:
        assert not engine.is_intention_run("unknown_player")

    def test_get_intention_run_elo_when_armed(self, engine: GameEngine) -> None:
        engine.arm_intention_run("p1")
        assert engine.get_intention_run_elo("p1") == engine._INTENTION_RUN_ELO

    def test_get_intention_run_elo_when_not_armed(self, engine: GameEngine) -> None:
        assert engine.get_intention_run_elo("p1") == 0


# ── ELO snapshot and restoration ─────────────────────────────────────────────


class TestEloSnapshot:
    def test_elo_restored_after_complete(self, engine: GameEngine) -> None:
        """Player's real ELO must be restored when the intention run ends."""
        state = engine.load_or_create_state("p1")
        state["elo_ratings"] = {"maze_classic": 1250.0, "riddle": 1100.0}

        engine.arm_intention_run("p1")
        # Simulate ELO drift during the run (engine internals might update it)
        state["elo_ratings"]["maze_classic"] = 1800.0

        engine.complete_intention_run("p1")

        restored = state["elo_ratings"]
        assert restored["maze_classic"] == pytest.approx(1250.0)
        assert restored["riddle"] == pytest.approx(1100.0)

    def test_complete_clears_flag(self, engine: GameEngine) -> None:
        engine.arm_intention_run("p1")
        engine.complete_intention_run("p1")
        assert not engine.is_intention_run("p1")

    def test_complete_returns_true_when_was_armed(self, engine: GameEngine) -> None:
        engine.arm_intention_run("p1")
        assert engine.complete_intention_run("p1") is True

    def test_complete_returns_false_when_not_armed(self, engine: GameEngine) -> None:
        assert engine.complete_intention_run("p1") is False


# ── clear_player_state cleans up intention run ────────────────────────────────


class TestClearPlayerState:
    def test_clear_removes_intention_run(self, engine: GameEngine) -> None:
        engine.arm_intention_run("p1")
        engine.clear_player_state("p1")
        assert not engine.is_intention_run("p1")
        assert engine.complete_intention_run("p1") is False


# ── Provocation → arm integration ─────────────────────────────────────────────


class TestProvocationToArm:
    def test_provocation_triggers_arm(
        self, engine: GameEngine, detector: ProvocationDetector
    ) -> None:
        """Simulates the router flow: detect → arm."""
        msg = "is that all you've got?"
        result = detector.detect(msg)
        assert result.detected

        # Simulate router action on detection
        if result.detected and not engine.is_intention_run("p1"):
            engine.arm_intention_run("p1")

        assert engine.is_intention_run("p1")

    def test_no_double_arm_on_repeated_taunts(
        self, engine: GameEngine, detector: ProvocationDetector
    ) -> None:
        """Second taunt while run is already armed must not overwrite the ELO snapshot."""
        state = engine.load_or_create_state("p1")
        state["elo_ratings"] = {"maze_classic": 1300.0}

        # First taunt
        r1 = detector.detect("bring it on")
        if r1.detected and not engine.is_intention_run("p1"):
            engine.arm_intention_run("p1")

        assert engine.is_intention_run("p1")

        # Simulate ELO drifting during run
        state["elo_ratings"]["maze_classic"] = 1800.0

        # Second taunt — should be ignored by the guard
        r2 = detector.detect("too easy")
        if r2.detected and not engine.is_intention_run("p1"):
            engine.arm_intention_run("p1")

        # Complete run — snapshot from FIRST arm must be restored
        engine.complete_intention_run("p1")
        assert state["elo_ratings"]["maze_classic"] == pytest.approx(1300.0)
