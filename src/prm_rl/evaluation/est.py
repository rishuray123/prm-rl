"""Evaluator Stress Test (EST) for the PRM.

Apply semantics-preserving perturbations and check the PRM score is
stable. Large score drift ⇒ PRM has stylistic biases that RL can exploit.
"""
from __future__ import annotations

import random
import re
from statistics import mean
from typing import TYPE_CHECKING, Sequence

from ..utils.steps import split_steps

if TYPE_CHECKING:  # pragma: no cover
    from ..models.prm import PRMScorer


def _reorder(steps: list[str], seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    idx = list(range(len(steps)))
    rng.shuffle(idx)
    return [steps[i] for i in idx]


def _add_redundant(steps: list[str], seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    if not steps:
        return steps
    injections = [
        "This calculation should be double-checked.",
        "Let us verify the intermediate result.",
        "Rewriting for clarity.",
    ]
    where = rng.randrange(len(steps) + 1)
    return steps[:where] + [rng.choice(injections)] + steps[where:]


def _reformat(steps: list[str]) -> list[str]:
    out = []
    for s in steps:
        s2 = re.sub(r"(\d)\s*\*\s*(\d)", r"\1 x \2", s)
        s2 = s2.replace("=", " = ")
        out.append(s2)
    return out


PERTURBATIONS = {
    "reorder": _reorder,
    "redundant": _add_redundant,
    "reformat": _reformat,
}


def evaluator_stress_test(
    prm: "PRMScorer",
    completions: Sequence[str],
    questions: Sequence[str],
    seed: int = 0,
) -> dict[str, float]:
    drifts: dict[str, list[float]] = {k: [] for k in PERTURBATIONS}
    for q, c in zip(questions, completions):
        steps = split_steps(c)
        if not steps:
            continue
        base = sum(prm.score_steps(q, steps)) / len(steps)
        for name, fn in PERTURBATIONS.items():
            new_steps = fn(steps, seed=seed) if fn is not _reformat else fn(steps)
            if not new_steps:
                continue
            probs = prm.score_steps(q, new_steps)
            perturbed = sum(probs) / len(probs) if probs else 0.0
            drifts[name].append(abs(base - perturbed))
    return {
        f"est_drift_{k}": (mean(v) if v else 0.0) for k, v in drifts.items()
    }
