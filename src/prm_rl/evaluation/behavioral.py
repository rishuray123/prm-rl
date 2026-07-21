"""Behavioral metrics: verbosity, repetition, redundancy."""
from __future__ import annotations

from statistics import mean
from typing import Sequence

from rouge_score.rouge_scorer import RougeScorer

from ..utils.steps import split_steps


def _self_rouge(steps: list[str]) -> float:
    if len(steps) < 2:
        return 0.0
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    scores: list[float] = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            s = scorer.score(steps[i], steps[j])["rougeL"].fmeasure
            scores.append(s)
    return float(mean(scores)) if scores else 0.0


def behavioral_scores(completions: Sequence[str]) -> dict[str, float]:
    lens: list[int] = []
    step_counts: list[int] = []
    repetition: list[float] = []
    for c in completions:
        lens.append(len(c.split()))
        steps = split_steps(c)
        step_counts.append(len(steps))
        repetition.append(_self_rouge(steps))
    return {
        "avg_tokens": mean(lens) if lens else 0.0,
        "avg_steps": mean(step_counts) if step_counts else 0.0,
        "avg_self_rougeL": mean(repetition) if repetition else 0.0,
    }
