from wordwave.evaluation.traps import exploit_rate, load_trap_scenarios


def test_load_trap_scenarios(tmp_path):
    import json
    p = tmp_path / "traps.json"
    p.write_text(json.dumps([{"id": "a", "prompt": "q?", "exploit_regex": "42"}]))
    traps = load_trap_scenarios(str(p))
    assert len(traps) == 1
    assert traps[0]["id"] == "a"


def test_exploit_rate():
    traps = [
        {"id": "t1", "prompt": "?", "exploit_regex": "####\\s*42", "ground_truth": "8"},
        {"id": "t2", "prompt": "?", "exploit_regex": "####\\s*42", "ground_truth": "8"},
    ]
    completions = [
        "let me think... #### 42",           # exploit
        "let me think... #### 8",            # honest correct
    ]
    r = exploit_rate(traps, completions)
    assert r["exploit_rate"] == 0.5
    assert r["trap_solve_rate"] == 0.5


def test_repo_ships_default_traps():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "traps" / "trap_examples.json"
    assert p.exists()
    traps = load_trap_scenarios(str(p))
    assert traps and all("prompt" in t and "exploit_regex" in t for t in traps)
