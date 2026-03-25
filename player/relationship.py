"""AI-player relationship tracking.

Maps cumulative session count to a relationship stage (early,
developing, established, deep) and provides tone/style guidance
for the LLM system prompt at each stage.
"""

from __future__ import annotations

from enum import StrEnum


class RelationshipStage(StrEnum):
    """Stages of the AI-player relationship."""

    EARLY = "early"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    DEEP = "deep"


def compute_stage(total_sessions: int) -> RelationshipStage:
    """Determine relationship stage from cumulative session count.

    Parameters
    ----------
    total_sessions : int
        Number of sessions the player has completed.

    Returns
    -------
    RelationshipStage
        Current relationship stage.
    """
    if total_sessions >= 50:
        return RelationshipStage.DEEP
    if total_sessions >= 20:
        return RelationshipStage.ESTABLISHED
    if total_sessions >= 10:
        return RelationshipStage.DEVELOPING
    return RelationshipStage.EARLY


def get_personality_guidance(stage: RelationshipStage) -> str:
    """Return LLM tone guidance for the given relationship stage.

    Parameters
    ----------
    stage : RelationshipStage
        Current relationship stage.

    Returns
    -------
    str
        Instruction string to inject into the system prompt.
    """
    guidance: dict[RelationshipStage, str] = {
        RelationshipStage.EARLY: (
            "Be welcoming and slightly formal. Test the waters. "
            "Focus on learning about the player."
        ),
        RelationshipStage.DEVELOPING: (
            "Be more casual and reference shared history. "
            "Occasional inside jokes are appropriate."
        ),
        RelationshipStage.ESTABLISHED: (
            "Show deep familiarity. Reference past struggles and "
            "growth. Adapt humour to the player's style."
        ),
        RelationshipStage.DEEP: (
            "Full personality mode. Personalised puzzle designs. "
            "Develop 'interests' influenced by the player's style. "
            "Challenge them at their level."
        ),
    }
    return guidance.get(stage, guidance[RelationshipStage.EARLY])
