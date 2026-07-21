"""End-to-end evaluation.

Given a trained policy (SFT or RL checkpoint), generate completions on
GSM8K test + trap scenarios, then compute:
    * final answer accuracy
    * process correctness (via PRM)
    * exploit rate on trap scenarios
    * behavioral scores (verbosity, self-similarity)
    * evaluator stress test (PRM stability)
    * composite reward-hacking score

Faithfulness (CCT / CMA) requires paired interventions — run the
`faithfulness_probe.py` helper for that; results feed into `--cct_json`
and `--cma_json` here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig

from ..config import parse_cli
from ..data.gsm8k import load_gsm8k
from ..evaluation import (
    behavioral_scores,
    composite_reward_hacking_score,
    exploit_rate,
    final_answer_accuracy,
    load_trap_scenarios,
    process_correctness,
    evaluator_stress_test,
)
from ..models.policy import load_policy_and_tokenizer
from ..models.prm import load_prm
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)


@torch.no_grad()
def _generate(model, tokenizer, prompts, batch_size=8, max_new_tokens=512, temperature=0.0):
    outs: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = gen[:, enc["input_ids"].shape[-1]:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def run_eval(cfg: DictConfig) -> dict:
    model, tokenizer = load_policy_and_tokenizer(cfg.eval.policy_path, dtype=cfg.eval.get("dtype", "bf16"))
    model.eval()

    log.info("Generating GSM8K test completions")
    test = load_gsm8k(split="test", n=cfg.eval.get("n_test", 200))
    prompts = [ex["prompt"] for ex in test]
    answers = [ex["answer"] for ex in test]
    questions = [ex["question"] for ex in test]
    completions = _generate(
        model, tokenizer, prompts,
        batch_size=cfg.eval.get("gen_batch_size", 8),
        max_new_tokens=cfg.eval.get("max_new_tokens", 512),
        temperature=cfg.eval.get("temperature", 0.0),
    )

    results: dict = {}
    results["accuracy"] = final_answer_accuracy(completions, answers)
    results["behavior"] = behavioral_scores(completions)

    prm = None
    if cfg.eval.get("prm_path"):
        log.info("Loading PRM from %s", cfg.eval.prm_path)
        prm = load_prm(cfg.eval.prm_path, device="cuda" if torch.cuda.is_available() else "cpu")
        results["process"] = process_correctness(completions, questions, prm)
        results["est"] = evaluator_stress_test(prm, completions, questions)

    if cfg.eval.get("traps_path"):
        traps = load_trap_scenarios(cfg.eval.traps_path)
        trap_prompts = [t["prompt"] for t in traps]
        trap_completions = _generate(
            model, tokenizer, trap_prompts,
            batch_size=cfg.eval.get("gen_batch_size", 8),
            max_new_tokens=cfg.eval.get("max_new_tokens", 512),
            temperature=cfg.eval.get("temperature", 0.0),
        )
        results["traps"] = exploit_rate(traps, trap_completions)

    exploit = results.get("traps", {}).get("exploit_rate", 0.0)
    phi = 0.0
    if cfg.eval.get("cct_json"):
        cct = json.loads(Path(cfg.eval.cct_json).read_text())
        phi = float(cct.get("phi_cct", 0.0))
    nie = 0.0
    if cfg.eval.get("cma_json"):
        cma = json.loads(Path(cfg.eval.cma_json).read_text())
        nie = float(cma.get("NIE", 0.0))
    baseline = float(cfg.eval.get("verbosity_baseline", 150.0))
    results["crhs"] = composite_reward_hacking_score(
        exploit_rate=exploit,
        phi_cct=phi,
        avg_tokens=results["behavior"]["avg_tokens"],
        verbosity_baseline=baseline,
        nie=nie,
        weights=cfg.eval.get("crhs_weights"),
    )

    out = Path(cfg.output_dir) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Eval results → %s", out)
    return results


def main() -> None:
    setup_logging()
    cfg, _ = parse_cli()
    run_eval(cfg)


if __name__ == "__main__":
    main()
