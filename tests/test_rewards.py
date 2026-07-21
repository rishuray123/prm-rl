"""Tests for the model-free reward functions.

Model-backed rewards (naive_process, prefix_consistency, contradiction,
hybrid) are exercised in `test_rewards_integration.py` behind a
`PRMRL_RUN_HEAVY` env-flag so CI stays fast.
"""
from prm_rl.rewards.counterfactual import make_counterfactual_reward
from prm_rl.rewards.outcome import outcome_reward


def test_outcome_reward_correct_and_wrong():
    prompts = ["ignored"] * 2
    completions = [
        "reasoning...\n\n#### 42",
        "reasoning...\n\n#### 100",
    ]
    r = outcome_reward(prompts, completions, answer=["42", "42"])
    assert r == [1.0, 0.0]


def test_outcome_reward_missing_answer_column():
    import pytest
    with pytest.raises(ValueError):
        outcome_reward(["p"], ["c"], answer=None)


def test_counterfactual_reward_grounding_fraction():
    fn = make_counterfactual_reward(scale=1.0)
    prompts = ["dummy"]
    questions = ["Alice has 3 apples and 5 oranges."]
    completions = ["She has 3 apples. So the answer is 3."]
    r = fn(prompts, completions, question=questions)
    # 3 out of {3,5} numeric tokens referenced -> 0.5
    assert r == [0.5]


def test_counterfactual_reward_no_numbers_in_question():
    fn = make_counterfactual_reward()
    r = fn(["p"], ["c"], question=["No numbers here"])
    assert r == [0.0]
