"""Step-level parsing utilities for math reasoning traces.

We keep this deliberately simple: reasoning is split on blank lines, and if
the model uses GSM8K's `#### <answer>` convention the final answer is
extracted from there.
"""
from __future__ import annotations

import re
from typing import Sequence

_ANSWER_RE = re.compile(r"####\s*([-+]?\$?\s*[-+]?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")


def split_steps(text: str) -> list[str]:
    """Split a reasoning trace into steps.

    Priority order:
    1. `#### answer` marker: everything before is treated as steps.
    2. Blank-line separated blocks (`\n\n`).
    3. Newline separated lines.
    """
    if not text:
        return []
    body = text
    m = _ANSWER_RE.search(text)
    if m:
        body = text[: m.start()].rstrip()

    parts: list[str]
    if "\n\n" in body:
        parts = [p.strip() for p in body.split("\n\n")]
    else:
        parts = [p.strip() for p in body.splitlines()]
    return [p for p in parts if p]


def extract_final_answer(text: str) -> str | None:
    """Extract the final answer from a completion.

    Accepts either the GSM8K `#### N` marker or, as a fallback, the last
    number in the completion. Returns a canonical string (no `$`, commas
    stripped, trailing `.0` removed) or None.
    """
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        nums = _NUMBER_RE.findall(text)
        if not nums:
            return None
        candidate = nums[-1]
    return canonicalize_number(candidate)


def canonicalize_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def gsm8k_gold_answer(answer_field: str) -> str | None:
    """Extract the gold answer from a GSM8K `answer` field.

    GSM8K stores the gold as `... reasoning ... #### 42`.
    """
    return extract_final_answer(answer_field)


def steps_are_correct(steps: Sequence[str], gold_final: str) -> list[bool]:
    """Cheap heuristic labeler used only for smoke tests / bootstrapping.

    Marks a step as "correct" if it does not contradict the gold final by
    asserting a different final answer. Real PRM labels should come from
    an oracle model or human annotation (see `data/golden.py`).
    """
    labels: list[bool] = []
    for step in steps:
        m = _ANSWER_RE.search(step)
        if m:
            labels.append(canonicalize_number(m.group(1)) == gold_final)
        else:
            labels.append(True)
    return labels
