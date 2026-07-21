"""Arm 2: Naive process reward.

Score each step with the PRM and sum (or mean) the per-step probabilities.
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..models.prm import PRMScorer, load_prm
from ..utils.steps import split_steps


def make_naive_process_reward(
    prm_path: str | None = None,
    prm: PRMScorer | None = None,
    device: str = "cuda",
    aggregation: str = "mean",
    scale: float = 1.0,
) -> Callable[..., list[float]]:
    """Build a naive process-reward function.

    Either provide an already-instantiated `prm` or a `prm_path` to load.
    """
    if prm is None:
        if prm_path is None:
            raise ValueError("Either prm or prm_path must be provided.")
        prm = load_prm(prm_path, device=device)

    if aggregation not in {"sum", "mean"}:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

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
            agg = sum(probs) if aggregation == "sum" else sum(probs) / len(probs)
            rewards.append(scale * float(agg))
        return rewards

    return fn
