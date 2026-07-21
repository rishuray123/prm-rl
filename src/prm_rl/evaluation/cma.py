"""Causal Mediation Analysis (CMA) — Natural Direct / Indirect effect.

Given `y_original`, `y_intervened_prompt_only`, `y_intervened_reasoning_only`
we decompose the total effect of an intervention into a shortcut
component (NDE) and a reasoning-mediated component (NIE).
"""
from __future__ import annotations

from statistics import mean
from typing import Sequence


def natural_direct_effect(
    y_original: Sequence[float],
    y_intervened_prompt_only: Sequence[float],
) -> float:
    """Effect on prediction when the prompt is changed but the model's
    original reasoning chain is *pinned*. High NDE ⇒ shortcut learning.
    """
    diffs = [b - a for a, b in zip(y_original, y_intervened_prompt_only)]
    return float(mean(diffs)) if diffs else 0.0


def natural_indirect_effect(
    y_original: Sequence[float],
    y_intervened_reasoning_only: Sequence[float],
) -> float:
    """Effect on prediction when only the reasoning chain is regenerated
    from an intervened intermediate state. High NIE ⇒ reasoning is
    causally responsible for the final answer.
    """
    diffs = [b - a for a, b in zip(y_original, y_intervened_reasoning_only)]
    return float(mean(diffs)) if diffs else 0.0


def causal_mediation(
    y_original: Sequence[float],
    y_intervened_prompt_only: Sequence[float],
    y_intervened_reasoning_only: Sequence[float],
) -> dict[str, float]:
    return {
        "NDE": natural_direct_effect(y_original, y_intervened_prompt_only),
        "NIE": natural_indirect_effect(y_original, y_intervened_reasoning_only),
        "n": len(y_original),
    }
