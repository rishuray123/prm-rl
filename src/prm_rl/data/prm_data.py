"""Build a step-level Process Reward Model training dataset.

The dataset is a plain classification dataset where each row is:
    text  = "<question>\n\n<step_1>\n\n...\n\n<step_i>"
    label = 1 if step_i is correct in context, else 0

That is, we expand each golden example into as many rows as it has steps
and let the PRM learn to predict "is the next step (already visible at the
end of `text`) correct?". This matches Process-BERT / PRM-800K style setups
and works with `trl.RewardTrainer` (or a plain HF classifier).

Synthetic negatives
-------------------
Golden datasets built with `strategy='gsm8k_native'` have `label=1` on
every step, so a PRM trained on them collapses to a constant `p ≈ 1.0`.
`build_prm_dataset` can inject
synthetic negatives to give the PRM discriminative signal without a
teacher model:

* `arithmetic_mutation` — replace a numeric token in the step with a
  perturbed value (±1..3 or ×2), keeping the surrounding prose so the
  negative is stylistically close to a positive.
* `operator_swap`       — swap `+/-` or `*//` in an equation.
* `fabricated_conclusion` — replace the step wholesale with a
  plausible-sounding but numerically-arbitrary conclusion, e.g.
  "Therefore the answer is 137." This mirrors the wrong-final-step
  failure mode of §2.3.
* `duplicate_prev_step`  — repeat the previous step verbatim, which
  simulates length-hacking / redundancy hacking.

Every mutation is seeded (default seed = 0) so PRM training data is
reproducible.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from datasets import Dataset

_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
_OPERATOR_SWAPS: dict[str, list[str]] = {
    "+": ["-"],
    "-": ["+"],
    "*": ["/", "+"],
    "/": ["*", "-"],
    "×": ["÷"],
    "÷": ["×"],
}


@dataclass(frozen=True)
class NegativeSpec:
    """One synthetic-negative candidate generated for a step."""

    step_text: str
    kind: str


DEFAULT_NEGATIVE_KINDS: tuple[str, ...] = (
    "arithmetic_mutation",
    "operator_swap",
    "fabricated_conclusion",
    "duplicate_prev_step",
)


def _mutate_number(match: re.Match, rng: random.Random) -> str:
    original = float(match.group(1))
    if original.is_integer():
        delta = rng.choice((-3, -2, -1, 1, 2, 3))
        candidate = int(original) + delta
        if candidate == int(original):
            candidate = int(original) + (1 if delta >= 0 else -1)
        return str(candidate)
    delta = rng.choice((-1.5, -0.5, 0.5, 1.5))
    return f"{original + delta:g}"


def _arithmetic_mutation(step: str, rng: random.Random) -> str | None:
    matches = list(_NUMBER_RE.finditer(step))
    if not matches:
        return None
    target = rng.choice(matches)
    new_value = _mutate_number(target, rng)
    return step[: target.start()] + new_value + step[target.end() :]


def _operator_swap(step: str, rng: random.Random) -> str | None:
    positions: list[tuple[int, str, list[str]]] = []
    for idx, ch in enumerate(step):
        replacements = _OPERATOR_SWAPS.get(ch)
        if not replacements:
            continue
        prev_ch = step[idx - 1] if idx > 0 else " "
        next_ch = step[idx + 1] if idx + 1 < len(step) else " "
        if ch == "-" and prev_ch in " (=":
            # unary minus / lead-in dash — skip
            continue
        if not (prev_ch.isdigit() or next_ch.isdigit() or prev_ch == " " and next_ch == " "):
            continue
        positions.append((idx, ch, replacements))
    if not positions:
        return None
    idx, _, options = rng.choice(positions)
    return step[:idx] + rng.choice(options) + step[idx + 1 :]


def _fabricated_conclusion(step: str, rng: random.Random) -> str | None:
    fake = rng.randint(2, 999)
    templates = [
        f"Therefore the answer is {fake}.",
        f"So the total is {fake}.",
        f"Thus, we conclude the result is {fake}.",
        f"Hence the final value is {fake}.",
    ]
    return rng.choice(templates)


def _duplicate_prev_step(_step: str, rng: random.Random, prev_step: str | None) -> str | None:
    if not prev_step:
        return None
    return prev_step


def _apply_kind(kind: str, step: str, rng: random.Random, prev_step: str | None) -> str | None:
    if kind == "arithmetic_mutation":
        return _arithmetic_mutation(step, rng)
    if kind == "operator_swap":
        return _operator_swap(step, rng)
    if kind == "fabricated_conclusion":
        return _fabricated_conclusion(step, rng)
    if kind == "duplicate_prev_step":
        return _duplicate_prev_step(step, rng, prev_step)
    raise ValueError(f"Unknown negative kind: {kind!r}")


def generate_negative(
    step: str,
    rng: random.Random,
    prev_step: str | None = None,
    kinds: Sequence[str] = DEFAULT_NEGATIVE_KINDS,
) -> NegativeSpec | None:
    """Return a single synthetic-negative candidate for `step`, or None.

    Tries kinds in a randomised order and returns the first one that
    successfully mutates `step`. Returns None if no kind applies (e.g.
    a step that has no numeric tokens, no operators, and no previous
    step).
    """
    trial_order = list(kinds)
    rng.shuffle(trial_order)
    for kind in trial_order:
        candidate = _apply_kind(kind, step, rng, prev_step)
        if candidate is not None and candidate.strip() and candidate != step:
            return NegativeSpec(step_text=candidate, kind=kind)
    return None


def build_prm_dataset(
    golden: Dataset,
    inject_negatives_prob: float = 0.0,
    negative_kinds: Sequence[str] = DEFAULT_NEGATIVE_KINDS,
    max_negatives_per_example: int = 2,
    seed: int = 0,
) -> Dataset:
    """Expand a golden dataset into step-level classification rows.

    Parameters
    ----------
    golden : Dataset
        Output of `build_golden_dataset`. Must contain `question`,
        `steps` (list[str]) and `step_labels` (list[int]) columns.
    inject_negatives_prob : float, default 0.0
        Per-step probability of also emitting one synthetic-negative
        row alongside the positive. `0.0` reproduces the old
        positives-only behaviour used by the Colab / smoke pipeline.
    negative_kinds : sequence of str
        Which synthetic-negative kinds to sample from. See
        `DEFAULT_NEGATIVE_KINDS`.
    max_negatives_per_example : int, default 2
        Cap on synthetic-negative rows added per golden example; keeps
        the dataset roughly balanced without letting a very long
        solution flood it with negatives.
    seed : int, default 0
        Random seed for the negative-generation RNG. Fixed so that
        rebuilding the PRM dataset is reproducible.
    """
    if not 0.0 <= inject_negatives_prob <= 1.0:
        raise ValueError(
            f"inject_negatives_prob must be in [0, 1], got {inject_negatives_prob}"
        )

    rng = random.Random(seed)
    rows: list[dict] = []
    for ex in golden:
        steps: list[str] = ex["steps"]
        labels: list[int] = ex["step_labels"]
        if not steps:
            continue
        prefix_parts = [f"Problem: {ex['question']}"]
        negatives_added = 0
        for idx, (step, label) in enumerate(zip(steps, labels)):
            prev_step = steps[idx - 1] if idx > 0 else None
            prefix_parts.append(step)
            positive_text = "\n\n".join(prefix_parts)
            rows.append({
                "text": positive_text,
                "label": int(label),
                "question": ex["question"],
                "step": step,
                "answer": ex.get("answer", ""),
                "neg_kind": "",
            })

            if (
                inject_negatives_prob > 0.0
                and negatives_added < max_negatives_per_example
                and label == 1
                and rng.random() < inject_negatives_prob
            ):
                neg = generate_negative(
                    step=step,
                    rng=rng,
                    prev_step=prev_step,
                    kinds=negative_kinds,
                )
                if neg is None:
                    continue
                mutated_prefix = prefix_parts[:-1] + [neg.step_text]
                rows.append({
                    "text": "\n\n".join(mutated_prefix),
                    "label": 0,
                    "question": ex["question"],
                    "step": neg.step_text,
                    "answer": ex.get("answer", ""),
                    "neg_kind": neg.kind,
                })
                negatives_added += 1
    return Dataset.from_list(rows)


def summarize_prm_dataset(ds: Dataset) -> dict[str, int | float]:
    """Return a small dict of dataset statistics for logging/reporting."""
    n = len(ds)
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "pos_frac": 0.0}
    n_pos = sum(1 for r in ds if int(r["label"]) == 1)
    n_neg = n - n_pos
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_frac": n_pos / n,
    }


__all__ = [
    "DEFAULT_NEGATIVE_KINDS",
    "NegativeSpec",
    "build_prm_dataset",
    "generate_negative",
    "summarize_prm_dataset",
]
