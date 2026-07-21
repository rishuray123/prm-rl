"""Arm 6: Hybrid process + outcome reward with outcome-conditioned centering
(PROGRS-style).

The intuition: for each *outcome bucket* (correct / incorrect final
answer), we center the process reward around the group mean before
combining with the outcome bonus. This dampens "goal drift" — the
policy is not rewarded for producing high-PRM but wrong reasoning
relative to the average high-PRM-and-wrong trajectory.

    r_i = alpha * outcome_i + beta * (process_i - mean(process | outcome_i))
"""
from __future__ import annotations

from statistics import mean
from typing import Callable, Sequence

from ..models.prm import PRMScorer, load_prm
from ..utils.steps import extract_final_answer, split_steps


def make_hybrid_reward(
    prm_path: str | None = None,
    prm: PRMScorer | None = None,
    device: str = "cuda",
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Callable[..., list[float]]:
    if prm is None:
        if prm_path is None:
            raise ValueError("Either prm or prm_path must be provided.")
        prm = load_prm(prm_path, device=device)

    def fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        answer: Sequence[str] | None = None,
        question: Sequence[str] | None = None,
        **kwargs,
    ) -> list[float]:
        if answer is None:
            raise ValueError("hybrid reward requires the `answer` column")
        questions = list(question) if question is not None else list(prompts)
        outcomes: list[float] = []
        processes: list[float] = []
        for q, comp, gold in zip(questions, completions, answer):
            pred = extract_final_answer(comp)
            outcomes.append(1.0 if pred is not None and pred == str(gold).strip() else 0.0)
            steps = split_steps(comp)
            probs = prm.score_steps(q, steps) if steps else []
            processes.append(sum(probs) / len(probs) if probs else 0.0)

        by_outcome: dict[float, list[float]] = {0.0: [], 1.0: []}
        for o, p in zip(outcomes, processes):
            by_outcome[o].append(p)
        means = {k: (mean(v) if v else 0.0) for k, v in by_outcome.items()}

        rewards: list[float] = []
        for o, p in zip(outcomes, processes):
            centered = p - means[o]
            rewards.append(alpha * o + beta * centered)
        return rewards

    return fn
