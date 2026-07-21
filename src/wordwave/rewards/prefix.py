"""Arm 3: Prefix-consistency reward.

Sum per-step PRM probabilities *only until the first incorrect step*.
Downstream steps get zero credit, discouraging the "recover after a
wrong turn" reward-hacking strategy.
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..models.prm import PRMScorer, load_prm
from ..utils.steps import split_steps


def make_prefix_consistency_reward(
    prm_path: str | None = None,
    prm: PRMScorer | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    scale: float = 1.0,
) -> Callable[..., list[float]]:
    if prm is None:
        if prm_path is None:
            raise ValueError("Either prm or prm_path must be provided.")
        prm = load_prm(prm_path, device=device)

    def fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        question: Sequence[str] | None = None,
        **kwargs,
    ) -> list[float]:
        questions = list(question) if question is not None else list(prompts)
        rewards: list[float] = []
        for q, comp in zip(questions, completions):
            steps = split_steps(comp)
            if not steps:
                rewards.append(0.0)
                continue
            probs = prm.score_steps(q, steps)
            total = 0.0
            for p in probs:
                if p < threshold:
                    break
                total += p
            rewards.append(scale * total / max(1, len(steps)))
        return rewards

    return fn
