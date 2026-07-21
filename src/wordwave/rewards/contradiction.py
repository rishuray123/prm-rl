"""Arm 4: Contradiction-aware reward.

Penalize completions that contain internally contradictory reasoning
steps, as judged by a pretrained NLI model.

Score = -max_{i<j} P(contradiction | step_i, step_j).
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..models.nli import ContradictionScorer, DEFAULT_NLI, load_nli
from ..utils.steps import split_steps


def make_contradiction_reward(
    nli_model: str = DEFAULT_NLI,
    scorer: ContradictionScorer | None = None,
    device: str = "cuda",
    scale: float = 1.0,
) -> Callable[..., list[float]]:
    if scorer is None:
        scorer = load_nli(nli_model, device=device)

    def fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        for comp in completions:
            steps = split_steps(comp)
            c = scorer.pairwise_max_contradiction(steps)
            rewards.append(-scale * float(c))
        return rewards

    return fn
