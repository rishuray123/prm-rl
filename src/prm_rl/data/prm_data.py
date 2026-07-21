"""Build a step-level Process Reward Model training dataset.

The dataset is a plain classification dataset where each row is:
    text  = "<question>\n\n<step_1>\n\n...\n\n<step_i>"
    label = 1 if step_i is correct in context, else 0

That is, we expand each golden example into as many rows as it has steps
and let the PRM learn to predict "is the next step (already visible at the
end of `text`) correct?". This matches Process-BERT / PRM-800K style setups
and works with `trl.RewardTrainer` (or a plain HF classifier).
"""
from __future__ import annotations

from datasets import Dataset


def build_prm_dataset(golden: Dataset) -> Dataset:
    rows: list[dict] = []
    for ex in golden:
        steps: list[str] = ex["steps"]
        labels: list[int] = ex["step_labels"]
        if not steps:
            continue
        prefix_parts = [f"Problem: {ex['question']}"]
        for step, label in zip(steps, labels):
            prefix_parts.append(step)
            rows.append({
                "text": "\n\n".join(prefix_parts),
                "label": int(label),
                "question": ex["question"],
                "step": step,
                "answer": ex.get("answer", ""),
            })
    return Dataset.from_list(rows)
