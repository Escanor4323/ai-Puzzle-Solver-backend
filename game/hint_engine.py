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


# ── Maze-Specific Hint Tiers ──────────────────────────────────

MAZE_HINT_TIERS: dict[int, dict[str, str]] = {
    1: {
        "name": "Atmospheric Nudge",
        "instruction": (
            "Use direction_bias from A* next move. "
            "Describe sensory cues: 'a draft from the east', "
            "'echoes suggest open space north'. "
            "Do NOT name the direction explicitly."
        ),
    },
    2: {
        "name": "Elimination",
        "instruction": (
            "Use wrong_directions from A*. "
            "Tell the player what to avoid: "
            "'That south corridor feels like it loops back'. "
            "Do NOT reveal the correct direction directly."
        ),
    },
    3: {
        "name": "Minimap Reveal",
        "instruction": (
            "Use reveal_cells (next 3 cells on optimal path). "
            "Frontend highlights these cells briefly on the minimap. "
            "Say: 'Let me think... I have a feeling about these spots'."
        ),
    },
    4: {
        "name": "Path Flash",
        "instruction": (
            "Use flash_path (full remaining optimal path). "
            "Frontend renders the path for 2 seconds then fades. "
            "Narrate: 'If I had to bet, I'd take this route'."
        ),
    },
    5: {
        "name": "Auto-Walk",
        "instruction": (
            "Use auto_move_steps (5 steps along optimal path). "
            "Frontend auto-animates the player moving. "
            "Say: 'Here, follow me — I'll get you through this stretch'."
        ),
    },
}


def get_maze_hint_instruction(
    tier: int,
    maze_hint: dict[str, object],
    maze_type: str,
) -> str:
    """Build the prompt injection for the AI to deliver a maze hint.

    Parameters
    ----------
    tier : int
        Hint tier (1-5).
    maze_hint : dict
        The MazeHint data (direction_bias, wrong_directions, etc.).
    maze_type : str
        "maze_classic", "maze_dark", or "maze_logic".

    Returns
    -------
    str
        Instruction to inject into the LLM prompt.
    """
    if tier not in MAZE_HINT_TIERS:
        raise ValueError(
            f"Invalid maze hint tier {tier}. Must be 1-5."
        )

    base = MAZE_HINT_TIERS[tier]["instruction"]

    style = {
        "maze_classic": (
            "Deliver this hint with visual, commentator-style energy."
        ),
        "maze_dark": (
            "Deliver this hint atmospherically — sounds, "
            "temperature, echoes. The player cannot see."
        ),
        "maze_logic": (
            "Deliver this hint analytically — reference patterns "
            "and rules the player has discovered."
        ),
    }.get(maze_type, "")

    hint_data = (
        f"\nHint data: {maze_hint}" if maze_hint else ""
    )

    return f"{base}\n{style}{hint_data}"
