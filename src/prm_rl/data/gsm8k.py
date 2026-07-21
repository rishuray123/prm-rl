"""GSM8K loader + prompt formatting.

Uses the HuggingFace `gsm8k` dataset. All rows are given a canonical
`prompt` (user question wrapped in a chat template friendly instruction) and
a canonical `answer` (the numeric gold answer, e.g. `"18"`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from ..utils.steps import gsm8k_gold_answer

PROMPT_TEMPLATE = (
    "Solve the following math problem step-by-step. "
    "Show your reasoning, one step per line separated by a blank line. "
    "End with a line of the form `#### <final answer>` where "
    "`<final answer>` is a plain number.\n\n"
    "Problem: {question}"
)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def _project(example: dict) -> dict:
    prompt = format_prompt(example["question"])
    gold = gsm8k_gold_answer(example["answer"]) or ""
    return {
        "prompt": prompt,
        "question": example["question"],
        "solution": example["answer"],
        "answer": gold,
    }


def load_gsm8k(
    split: str = "train",
    subset: str = "main",
    cache_dir: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 0,
) -> Dataset:
    """Load a projected GSM8K split.

    Columns after projection: `prompt, question, solution, answer`.
    """
    ds = load_dataset("gsm8k", subset, split=split, cache_dir=cache_dir)
    ds = ds.map(_project, remove_columns=[c for c in ds.column_names if c not in {"question", "answer"}])
    ds = ds.filter(lambda ex: ex["answer"] != "")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def save_local(dsd: DatasetDict, path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(path))
