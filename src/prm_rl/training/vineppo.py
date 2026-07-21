"""Arm 9: VinePPO (skeleton).

VinePPO replaces PPO's learned value function with Monte-Carlo estimates
of state values obtained via re-simulation from intermediate states.

We keep TRL as the outer harness: this module owns the value-estimation
helper (Monte-Carlo rollouts from state `s_t`) and hands the resulting
per-token advantages to a stock PPO trainer. This is a *skeleton*
because the full VinePPO implementation belongs in a research fork;
`configs/experiments/arm9_vineppo.yaml` shows the intended surface.
"""
from __future__ import annotations

from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@torch.no_grad()
def monte_carlo_value(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context_ids: torch.Tensor,
    reward_fn: Callable[[str, str], float],
    prompt_text: str,
    n_rollouts: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
) -> float:
    """Estimate V(s) by rolling out `n_rollouts` continuations and
    averaging the terminal reward. Used by VinePPO in place of a
    learned value head.
    """
    rewards = []
    for _ in range(n_rollouts):
        out = model.generate(
            context_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, context_ids.shape[-1]:], skip_special_tokens=True)
        rewards.append(reward_fn(prompt_text, text))
    return float(sum(rewards) / max(1, len(rewards)))
