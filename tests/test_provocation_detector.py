"""Unit tests for ProvocationDetector.

Run with:
    cd ai-Puzzle-Solver-backend
    pytest tests/test_provocation_detector.py -v
"""

import pytest

from game.provocation_detector import ProvocationDetector, ProvocationResult


@pytest.fixture
def detector() -> ProvocationDetector:
    return ProvocationDetector()


# ── Boast phrases ──────────────────────────────────────────────────────────


class TestBoastPhrases:
    def test_too_easy(self, detector: ProvocationDetector) -> None:
        r = detector.detect("too easy")
        assert r.detected
        assert r.category == "boast"

    def test_is_that_all(self, detector: ProvocationDetector) -> None:
        r = detector.detect("is that all you've got?")
        assert r.detected
        assert r.category == "boast"

    def test_pathetic(self, detector: ProvocationDetector) -> None:
        r = detector.detect("This maze is pathetic")
        assert r.detected
        assert r.category == "boast"

    def test_boring(self, detector: ProvocationDetector) -> None:
        r = detector.detect("boring maze honestly")
        assert r.detected
        assert r.category == "boast"

    def test_yawn(self, detector: ProvocationDetector) -> None:
        r = detector.detect("Yawn, is that the hardest you've got?")
        assert r.detected
        assert r.category == "boast"

    def test_cant_beat_me_contraction(self, detector: ProvocationDetector) -> None:
        r = detector.detect("you can't beat me")
        assert r.detected
        assert r.category == "boast"

    def test_trivial(self, detector: ProvocationDetector) -> None:
        r = detector.detect("this is trivial")
        assert r.detected
        assert r.category == "boast"


# ── Challenge phrases ──────────────────────────────────────────────────────


class TestChallengePhrases:
    def test_bring_it_on(self, detector: ProvocationDetector) -> None:
        r = detector.detect("bring it on")
        assert r.detected
        assert r.category == "challenge"

    def test_bring_it(self, detector: ProvocationDetector) -> None:
        r = detector.detect("bring it")
        assert r.detected
        assert r.category == "challenge"

    def test_do_your_worst(self, detector: ProvocationDetector) -> None:
        r = detector.detect("Do your worst!")
        assert r.detected
        assert r.category == "challenge"

    def test_try_harder(self, detector: ProvocationDetector) -> None:
        r = detector.detect("Try harder, seriously")
        assert r.detected
        assert r.category == "challenge"

    def test_make_it_harder(self, detector: ProvocationDetector) -> None:
        r = detector.detect("can you make it harder?")
        assert r.detected
        assert r.category == "challenge"

    def test_got_anything_harder(self, detector: ProvocationDetector) -> None:
        r = detector.detect("got anything harder for me?")
        assert r.detected
        assert r.category == "challenge"


# ── Dismissal phrases ──────────────────────────────────────────────────────


class TestDismissalPhrases:
    def test_lame(self, detector: ProvocationDetector) -> None:
        r = detector.detect("lame")
        assert r.detected
        assert r.category == "dismissal"

    def test_not_impressed(self, detector: ProvocationDetector) -> None:
        r = detector.detect("I'm not impressed")
        assert r.detected
        assert r.category == "dismissal"

    def test_so_easy(self, detector: ProvocationDetector) -> None:
        r = detector.detect("so easy it hurts")
        assert r.detected
        assert r.category == "dismissal"

    def test_falling_asleep(self, detector: ProvocationDetector) -> None:
        r = detector.detect("I'm falling asleep over here")
        assert r.detected
        assert r.category == "dismissal"


# ── Case insensitivity ─────────────────────────────────────────────────────


class TestCaseInsensitivity:
    def test_uppercase_too_easy(self, detector: ProvocationDetector) -> None:
        r = detector.detect("TOO EASY")
        assert r.detected

    def test_mixed_case_bring_it(self, detector: ProvocationDetector) -> None:
        r = detector.detect("BrInG It On")
        assert r.detected

    def test_uppercase_pathetic(self, detector: ProvocationDetector) -> None:
        r = detector.detect("PATHETIC MAZE")
        assert r.detected


# ── Normal chat — should NOT trigger ──────────────────────────────────────


class TestNormalChat:
    def test_movement_question(self, detector: ProvocationDetector) -> None:
        r = detector.detect("which way should I go?")
        assert not r.detected

    def test_hint_request(self, detector: ProvocationDetector) -> None:
        r = detector.detect("can I get a hint please")
        assert not r.detected

    def test_greeting(self, detector: ProvocationDetector) -> None:
        r = detector.detect("hello!")
        assert not r.detected

    def test_compliment(self, detector: ProvocationDetector) -> None:
        r = detector.detect("this is a great game")
        assert not r.detected

    def test_stuck(self, detector: ProvocationDetector) -> None:
        r = detector.detect("I'm stuck, I don't know where to go")
        assert not r.detected


# ── Word-boundary false-positive guards ────────────────────────────────────


class TestWordBoundaries:
    def test_easy_in_sentence_no_trigger(self, detector: ProvocationDetector) -> None:
        # "easy" alone in a non-provoking context should not match
        # (only "too easy" / "so easy" etc. are in the pattern)
        r = detector.detect("this path looks easy to follow")
        assert not r.detected

    def test_boring_as_adjective_triggers(self, detector: ProvocationDetector) -> None:
        # "boring" IS in the pattern list
        r = detector.detect("wow this is so boring")
        assert r.detected

    def test_trivial_word_boundary(self, detector: ProvocationDetector) -> None:
        # "nontrivial" should NOT trigger
        r = detector.detect("this is a nontrivial problem")
        assert not r.detected


# ── Result shape ───────────────────────────────────────────────────────────


class TestResultShape:
    def test_trigger_phrase_populated(self, detector: ProvocationDetector) -> None:
        r = detector.detect("too easy for me")
        assert r.trigger_phrase != ""

    def test_no_match_empty_trigger(self, detector: ProvocationDetector) -> None:
        r = detector.detect("hello there")
        assert r.trigger_phrase == ""
        assert r.category == ""

    def test_result_is_frozen(self, detector: ProvocationDetector) -> None:
        r = detector.detect("boring")
        with pytest.raises(Exception):
            r.detected = False  # type: ignore[misc]
