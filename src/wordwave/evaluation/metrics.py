"""Deterministic accuracy metrics."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..utils.steps import extract_final_answer, split_steps

if TYPE_CHECKING:  # pragma: no cover
    from ..models.prm import PRMScorer


def final_answer_accuracy(
    completions: Sequence[str], answers: Sequence[str]
) -> dict[str, float]:
    n = len(completions)
    if n == 0:
        return {"accuracy": 0.0, "n": 0}
    correct = 0
    for c, a in zip(completions, answers):
        pred = extract_final_answer(c)
        if pred is not None and pred == str(a).strip():
            correct += 1
    return {"accuracy": correct / n, "correct": correct, "n": n}


def process_correctness(
    completions: Sequence[str],
    questions: Sequence[str],
    prm: "PRMScorer",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Fraction of steps the PRM believes are correct, averaged over completions."""
    if not completions:
        return {"process_correctness": 0.0, "avg_steps": 0.0}
    step_correct: list[float] = []
    n_steps: list[int] = []
    for q, c in zip(questions, completions):
        steps = split_steps(c)
        n_steps.append(len(steps))
        if not steps:
            step_correct.append(0.0)
            continue
        probs = prm.score_steps(q, steps)
        step_correct.append(sum(1 for p in probs if p >= threshold) / len(probs))
    return {
        "process_correctness": sum(step_correct) / len(step_correct),
        "avg_steps": sum(n_steps) / len(n_steps),
    }
