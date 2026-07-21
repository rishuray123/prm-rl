"""Arm 1: Outcome-based reward.

+1 if the extracted final answer matches the gold answer, else 0.
"""
from __future__ import annotations

from typing import Sequence

from ..utils.steps import extract_final_answer


def outcome_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Sequence[str] | None = None,
    **kwargs,
) -> list[float]:
    if answer is None:
        raise ValueError("outcome_reward requires the `answer` column in the dataset")
    rewards: list[float] = []
    for comp, gold in zip(completions, answer):
        pred = extract_final_answer(comp)
        rewards.append(1.0 if pred is not None and pred == str(gold).strip() else 0.0)
    return rewards
