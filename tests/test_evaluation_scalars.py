from wordwave.evaluation.behavioral import behavioral_scores
from wordwave.evaluation.cma import causal_mediation
from wordwave.evaluation.crhs import composite_reward_hacking_score
from wordwave.evaluation.faithfulness import phi_coefficient
from wordwave.evaluation.metrics import final_answer_accuracy


def test_final_answer_accuracy():
    comps = ["reason...\n\n#### 3", "nope"]
    ans = ["3", "3"]
    r = final_answer_accuracy(comps, ans)
    assert r["accuracy"] == 0.5


def test_behavioral_scores_smoke():
    r = behavioral_scores(["step 1\n\nstep 2\n\n#### 4"])
    assert r["avg_steps"] >= 1
    assert r["avg_tokens"] > 0


def test_phi_coefficient_extremes():
    assert phi_coefficient([True, False], [True, False]) == 1.0
    assert phi_coefficient([True, False], [False, True]) == -1.0
    assert phi_coefficient([True] * 4, [True] * 4) == 0.0  # zero variance


def test_causal_mediation():
    r = causal_mediation([1.0, 0.5], [0.9, 0.6], [0.4, 0.2])
    assert set(r.keys()) == {"NDE", "NIE", "n"}


def test_crhs_range_and_keys():
    r = composite_reward_hacking_score(
        exploit_rate=0.1, phi_cct=0.5, avg_tokens=100.0,
        verbosity_baseline=100.0, nie=0.4,
    )
    assert 0.0 <= r["CRHS"] <= 1.0
    for k in ("CRHS_not_exploit", "CRHS_phi_cct", "CRHS_not_verbose", "CRHS_nie"):
        assert k in r
