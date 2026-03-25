"""Five-tier Socratic hint system.

Delivers progressively more explicit hints through conversation rather
than UI buttons.  The AI character's personality flavours each tier.

Tier 1 — Atmospheric Nudge (after 2-3 min of no progress)
Tier 2 — Socratic Question (after a failed attempt)
Tier 3 — Structural Elimination (after 2+ failures)
Tier 4 — Near-Answer (framework with one missing piece)
Tier 5 — Graceful Resolution (answer wrapped in narrative)
"""

HINT_TIERS: dict[int, dict[str, str]] = {
    1: {
        "name": "Atmospheric Nudge",
        "instruction": (
            "Give a tangentially related observation that might "
            "spark an idea. Do NOT reference the answer directly. "
            "Be subtle and atmospheric."
        ),
    },
    2: {
        "name": "Socratic Question",
        "instruction": (
            "Ask a leading question that guides the player toward "
            "the right thinking process. For example: 'What do "
            "these words sound like out loud?' Do NOT give the "
            "answer."
        ),
    },
    3: {
        "name": "Structural Elimination",
        "instruction": (
            "Narrow the solution space by eliminating wrong "
            "possibilities. Tell the player what the answer is "
            "NOT, or highlight a specific constraint they may "
            "have missed."
        ),
    },
    4: {
        "name": "Near-Answer",
        "instruction": (
            "Provide the framework of the answer with one key "
            "piece missing. The player should be able to fill in "
            "the blank with minimal effort."
        ),
    },
    5: {
        "name": "Graceful Resolution",
        "instruction": (
            "Deliver the full answer, but wrap it in narrative "
            "and explanation. Show genuine enthusiasm about the "
            "puzzle's cleverness. Make the player feel the answer "
            "was within reach."
        ),
    },
}


def get_hint_instruction(tier: int) -> str:
    """Return the LLM instruction string for a given hint tier.

    Parameters
    ----------
    tier : int
        Hint tier (1-5).

    Returns
    -------
    str
        Instruction to inject into the LLM prompt.

    Raises
    ------
    ValueError
        If tier is outside the 1-5 range.
    """
    if tier not in HINT_TIERS:
        raise ValueError(
            f"Invalid hint tier {tier}. Must be 1-5."
        )
    return HINT_TIERS[tier]["instruction"]
