from prm_rl.utils.steps import (
    canonicalize_number,
    extract_final_answer,
    gsm8k_gold_answer,
    split_steps,
)


def test_canonicalize_number():
    assert canonicalize_number("1,234") == "1234"
    assert canonicalize_number("$42") == "42"
    assert canonicalize_number("42.0") == "42"
    assert canonicalize_number("3.14") == "3.14"
    assert canonicalize_number("abc") == "abc"


def test_extract_final_answer_gsm8k_marker():
    text = "Reason 1.\n\nReason 2.\n\n#### 18"
    assert extract_final_answer(text) == "18"


def test_extract_final_answer_fallback_last_number():
    text = "First we had 3. Then 7. Answer: 10."
    assert extract_final_answer(text) == "10"


def test_gsm8k_gold_answer():
    gold = "Long reasoning...\n#### 72"
    assert gsm8k_gold_answer(gold) == "72"


def test_split_steps_blank_lines():
    text = "Step one.\n\nStep two is longer.\n\nStep three."
    steps = split_steps(text)
    assert len(steps) == 3
    assert steps[0].startswith("Step one")


def test_split_steps_stops_before_answer_marker():
    text = "Step one.\n\nStep two.\n\n#### 42"
    steps = split_steps(text)
    assert len(steps) == 2
    assert not any("####" in s for s in steps)
