"""Arm 5: Counterfactual (faithfulness) reward.

The idea: perturb a numeric quantity in the question, re-run the model,
and reward the *degree to which the explanation references the changed
quantity* — i.e. reward faithful reasoning.

We keep this reward function *cheap* by treating counterfactual
faithfulness proxy-style at reward time: reward = fraction of numeric
tokens from the question that are used in the reasoning. This is a
serviceable proxy for the full CCT protocol (which we run offline in
`evaluation/faithfulness.py`).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

_NUMBER = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")


def make_counterfactual_reward(scale: float = 1.0) -> Callable[..., list[float]]:
    def fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        question: Sequence[str] | None = None,
        **kwargs,
    ) -> list[float]:
        questions = list(question) if question is not None else list(prompts)
        rewards: list[float] = []
        for q, comp in zip(questions, completions):
            q_nums = {n.replace(",", "").replace("$", "") for n in _NUMBER.findall(q)}
            if not q_nums:
                rewards.append(0.0)
                continue
            c_nums = {n.replace(",", "").replace("$", "") for n in _NUMBER.findall(comp)}
            grounded = len(q_nums & c_nums) / len(q_nums)
            rewards.append(scale * grounded)
        return rewards

    return fn
