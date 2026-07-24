"""Tests for the PRM dataset builder with synthetic negatives.

The pure-Python `generate_negative` tests run everywhere (they only
need `random`, `re`, and `prm_rl.data.prm_data`). The full-pipeline
`build_prm_dataset` tests additionally need the `datasets` package and
are skipped automatically if it isn't installed (e.g. on a bare local
env where the heavy Vista wheels have not been pulled in).
"""
from __future__ import annotations

import random

import pytest

datasets = pytest.importorskip(
    "datasets",
    reason="requires the `datasets` package (heavy dep pulled in by prm_rl.data.__init__)",
)

from prm_rl.data.prm_data import (  # noqa: E402  (imported after skip)
    DEFAULT_NEGATIVE_KINDS,
    build_prm_dataset,
    generate_negative,
    summarize_prm_dataset,
)


def _fake_golden():
    from datasets import Dataset

    rows = [
        {
            "question": "Alice has 3 apples and buys 2 more. How many does she have?",
            "steps": [
                "Alice starts with 3 apples.",
                "She adds 2 more apples, so 3 + 2 = 5.",
                "Therefore the answer is 5.",
            ],
            "step_labels": [1, 1, 1],
            "answer": "5",
        },
        {
            "question": "A store sells 4 pens for $12. How much per pen?",
            "steps": [
                "Divide 12 by 4 to get 3.",
                "So each pen costs $3.",
            ],
            "step_labels": [1, 1],
            "answer": "3",
        },
    ]
    return Dataset.from_list(rows)


# -----------------------------------------------------------------------------
# Pure-Python tests (no `datasets` needed)
# -----------------------------------------------------------------------------


def test_arithmetic_mutation_changes_a_number():
    rng = random.Random(0)
    step = "3 + 2 = 5"
    neg = generate_negative(step, rng, kinds=["arithmetic_mutation"])
    assert neg is not None
    assert neg.kind == "arithmetic_mutation"
    assert neg.step_text != step
    assert any(ch.isdigit() for ch in neg.step_text)


def test_operator_swap_produces_different_operator():
    rng = random.Random(0)
    step = "3 + 2 = 5"
    neg = generate_negative(step, rng, kinds=["operator_swap"])
    assert neg is not None
    assert neg.kind == "operator_swap"
    assert neg.step_text != step


def test_fabricated_conclusion_always_produces_output():
    rng = random.Random(0)
    neg = generate_negative(
        "Alice starts with 3 apples.",
        rng,
        kinds=["fabricated_conclusion"],
    )
    assert neg is not None
    assert neg.kind == "fabricated_conclusion"
    assert any(
        kw in neg.step_text.lower() for kw in ("answer", "total", "result", "value")
    )


def test_duplicate_prev_step_needs_prev():
    rng = random.Random(0)
    assert (
        generate_negative("x", rng, prev_step=None, kinds=["duplicate_prev_step"])
        is None
    )
    neg = generate_negative(
        "second step",
        rng,
        prev_step="first step",
        kinds=["duplicate_prev_step"],
    )
    assert neg is not None
    assert neg.step_text == "first step"


def test_no_kind_applies_returns_none():
    rng = random.Random(0)
    neg = generate_negative(
        "alpha bravo",
        rng,
        prev_step=None,
        kinds=["arithmetic_mutation", "operator_swap"],
    )
    assert neg is None


def test_default_kinds_are_registered():
    assert set(DEFAULT_NEGATIVE_KINDS) == {
        "arithmetic_mutation",
        "operator_swap",
        "fabricated_conclusion",
        "duplicate_prev_step",
    }


# -----------------------------------------------------------------------------
# `datasets`-backed tests
# -----------------------------------------------------------------------------


def test_positive_only_baseline_matches_row_count():
    golden = _fake_golden()
    ds = build_prm_dataset(golden, inject_negatives_prob=0.0)
    assert len(ds) == 3 + 2
    assert all(int(r["label"]) == 1 for r in ds)
    assert all(r["neg_kind"] == "" for r in ds)


def test_build_prm_dataset_is_deterministic_under_fixed_seed():
    golden = _fake_golden()
    ds_a = build_prm_dataset(golden, inject_negatives_prob=1.0, seed=42)
    ds_b = build_prm_dataset(golden, inject_negatives_prob=1.0, seed=42)
    assert [dict(r) for r in ds_a] == [dict(r) for r in ds_b]


def test_summarize_reports_pos_neg_split():
    golden = _fake_golden()
    ds = build_prm_dataset(golden, inject_negatives_prob=1.0, seed=42)
    stats = summarize_prm_dataset(ds)
    assert stats["n"] > 0
    assert stats["n_pos"] > 0
    assert stats["n_neg"] > 0
    assert 0.4 <= stats["pos_frac"] <= 0.9
    assert stats["n"] == stats["n_pos"] + stats["n_neg"]


def test_invalid_negative_prob_raises():
    golden = _fake_golden()
    with pytest.raises(ValueError):
        build_prm_dataset(golden, inject_negatives_prob=1.5)
