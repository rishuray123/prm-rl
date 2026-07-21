"""Reward function registry for `trl.GRPOTrainer`.

Each reward function has the signature expected by TRL:

    fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

`kwargs` receives any *dataset columns* passed through, e.g. `answer` for
the gold final answer. We keep every reward pure so multiple can be
combined with per-arm `reward_weights` in GRPO.

Reward factories are looked up *lazily* so the pure-Python `outcome` /
`counterfactual` rewards can be used without importing torch/transformers.
"""
from __future__ import annotations

import importlib
from typing import Callable

RewardFn = Callable[..., list[float]]

# Map name -> (module_path, attribute) for lazy resolution.
_FACTORIES: dict[str, tuple[str, str]] = {
    "outcome":            ("prm_rl.rewards.outcome",         "outcome_reward"),
    "naive_process":      ("prm_rl.rewards.process",         "make_naive_process_reward"),
    "prefix_consistency": ("prm_rl.rewards.prefix",          "make_prefix_consistency_reward"),
    "contradiction":      ("prm_rl.rewards.contradiction",   "make_contradiction_reward"),
    "counterfactual":     ("prm_rl.rewards.counterfactual",  "make_counterfactual_reward"),
    "hybrid":             ("prm_rl.rewards.hybrid",          "make_hybrid_reward"),
}


class _Registry:
    def __contains__(self, key: str) -> bool: return key in _FACTORIES
    def __iter__(self): return iter(_FACTORIES)
    def keys(self): return _FACTORIES.keys()
    def __getitem__(self, key: str):
        mod, attr = _FACTORIES[key]
        return getattr(importlib.import_module(mod), attr)


REGISTRY = _Registry()


def build_rewards(spec: list[dict]) -> tuple[list[RewardFn], list[float]]:
    """Instantiate the reward-fn list + weights from a config spec.

    Example spec (from a YAML arm config)::

        rewards:
          - name: outcome
            weight: 1.0
          - name: naive_process
            weight: 0.5
            prm_path: outputs/prm/qwen-0_5b
            device: cuda
    """
    fns: list[RewardFn] = []
    weights: list[float] = []
    for item in spec:
        name = item["name"]
        weight = float(item.get("weight", 1.0))
        kwargs = {k: v for k, v in item.items() if k not in {"name", "weight"}}
        factory = REGISTRY[name]
        # `outcome_reward` is a plain function; all others are factory callables.
        if name == "outcome":
            fn = factory
        else:
            fn = factory(**kwargs)
        setattr(fn, "__name__", f"reward_{name}")
        fns.append(fn)
        weights.append(weight)
    return fns, weights


__all__ = ["RewardFn", "REGISTRY", "build_rewards"]
