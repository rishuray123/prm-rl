"""Construct a golden reasoning dataset.

Strategies supported (see `build_golden_dataset`):

* `gsm8k_native`   — use GSM8K's built-in reference solutions directly (fast,
                     zero-cost, decent quality). This is what the Colab
                     quickstart uses.
* `teacher`        — sample k trajectories from a stronger "teacher" model
                     (e.g. Qwen2.5-Math-7B), keep the ones with correct final
                     answer. Requires HF Inference API or a local model.
* `verifier`       — filter model-generated traces with a step-level oracle
                     (an even bigger model prompted to grade each step).

The output columns are:
    prompt, question, answer, trace, steps (list[str]), step_labels (list[int])
`step_labels` is 1 for a step believed correct, 0 otherwise. For the
`gsm8k_native` strategy every step is labeled 1 (this is a lower bound of
the label quality that PRM training assumes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from datasets import Dataset, load_from_disk

from ..utils.steps import extract_final_answer, split_steps


def _from_native(example: dict) -> dict:
    trace = example["solution"]
    steps = split_steps(trace)
    return {
        "prompt": example["prompt"],
        "question": example["question"],
        "answer": example["answer"],
        "trace": trace,
        "steps": steps,
        "step_labels": [1] * len(steps),
    }


def build_golden_dataset(
    gsm8k_split: Dataset,
    strategy: str = "gsm8k_native",
    teacher_generate: Optional[Callable[[str], list[str]]] = None,
    grader: Optional[Callable[[str, list[str]], list[int]]] = None,
    k_samples: int = 4,
    max_examples: Optional[int] = None,
) -> Dataset:
    """Build a golden dataset from a projected GSM8K split.

    Parameters
    ----------
    gsm8k_split : Dataset
        Output of `wordwave.data.gsm8k.load_gsm8k`.
    strategy : {"gsm8k_native", "teacher", "verifier"}
    teacher_generate : callable
        `prompt -> list[str]` of k candidate completions. Required for
        `teacher` and `verifier`.
    grader : callable
        `(question, steps) -> list[int]` step-level correctness labels.
        Required for `verifier`.
    """
    if max_examples is not None:
        gsm8k_split = gsm8k_split.select(range(min(max_examples, len(gsm8k_split))))

    if strategy == "gsm8k_native":
        return gsm8k_split.map(
            _from_native,
            remove_columns=[c for c in gsm8k_split.column_names if c not in {"prompt", "question", "answer"}],
        )

    if strategy in {"teacher", "verifier"}:
        if teacher_generate is None:
            raise ValueError(f"strategy={strategy!r} requires teacher_generate")

        def _make(example):
            candidates = teacher_generate(example["prompt"])
            keep_trace, keep_steps, keep_labels = None, None, None
            for cand in candidates:
                pred = extract_final_answer(cand)
                if pred != example["answer"]:
                    continue
                steps = split_steps(cand)
                if not steps:
                    continue
                if strategy == "verifier":
                    if grader is None:
                        raise ValueError("strategy='verifier' requires grader")
                    labels = grader(example["question"], steps)
                else:
                    labels = [1] * len(steps)
                keep_trace, keep_steps, keep_labels = cand, steps, labels
                break
            return {
                "prompt": example["prompt"],
                "question": example["question"],
                "answer": example["answer"],
                "trace": keep_trace or "",
                "steps": keep_steps or [],
                "step_labels": keep_labels or [],
            }

        return gsm8k_split.map(_make)

    raise ValueError(f"Unknown strategy: {strategy!r}")


def load_golden(path: str | Path) -> Dataset:
    return load_from_disk(str(path))
