from prm_rl.rewards import REGISTRY, build_rewards


def test_registry_names():
    expected = {"outcome", "naive_process", "prefix_consistency",
                "contradiction", "counterfactual", "hybrid"}
    assert expected.issubset(set(REGISTRY.keys()))


def test_build_rewards_outcome_only():
    fns, weights = build_rewards([{"name": "outcome", "weight": 1.0}])
    assert len(fns) == 1 and weights == [1.0]
    # each fn should have a stable __name__ so TRL can log per-reward stats
    assert fns[0].__name__ == "reward_outcome"


def test_build_rewards_counterfactual_no_model():
    fns, weights = build_rewards([
        {"name": "outcome", "weight": 1.0},
        {"name": "counterfactual", "weight": 0.5, "scale": 2.0},
    ])
    assert weights == [1.0, 0.5]
    assert [f.__name__ for f in fns] == ["reward_outcome", "reward_counterfactual"]
