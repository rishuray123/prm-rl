"""RL training via `trl.GRPOTrainer`.

An "arm" is just a list of reward functions with weights + a set of GRPO
hyperparameters (including the KL coefficient `beta`). No custom PPO
loop lives here — everything routes through TRL.
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from trl import GRPOConfig, GRPOTrainer

from ..data.gsm8k import load_gsm8k
from ..models.policy import load_policy_and_tokenizer
from ..rewards import build_rewards
from ..utils.logging import get_logger

log = get_logger(__name__)


def _load_prompts(cfg: DictConfig) -> Dataset:
    d = cfg.data
    ds = load_gsm8k(
        split=d.get("split", "train"),
        subset=d.get("subset", "main"),
        n=d.get("n"),
        seed=cfg.get("seed", 0),
    )
    keep = {"prompt", "question", "answer"}
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def run_rl(cfg: DictConfig) -> str:
    log.info("Building reward stack: %s", [r["name"] for r in cfg.rewards])
    reward_fns, reward_weights = build_rewards(list(cfg.rewards))

    prompts = _load_prompts(cfg)

    log.info("Loading policy from %s", cfg.model.name)
    model, tokenizer = load_policy_and_tokenizer(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "bf16"),
        attn_implementation=cfg.model.get("attn_implementation"),
        load_in_4bit=cfg.model.get("load_in_4bit", False),
        device_map=cfg.model.get("device_map", "auto"),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.training.get("num_train_epochs", 1),
        max_steps=cfg.training.get("max_steps", -1),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        logging_steps=cfg.training.get("logging_steps", 10),
        save_strategy=cfg.training.get("save_strategy", "steps"),
        save_steps=cfg.training.get("save_steps", 200),
        save_total_limit=cfg.training.get("save_total_limit", 2),
        bf16=cfg.training.get("bf16", True),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        report_to=list(cfg.training.get("report_to", ["none"])),
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 0),
        # GRPO-specific
        num_generations=cfg.training.get("num_generations", 4),
        # NOTE: trl 1.x dropped `max_prompt_length` from GRPOConfig; the
        # tokenizer's built-in truncation handles prompt length now. Long
        # prompts should be truncated in the dataset before it hits the
        # trainer if you need a hard cap.
        max_completion_length=cfg.training.get("max_completion_length", 512),
        beta=cfg.training.get("beta", 0.04),  # KL coefficient
        temperature=cfg.training.get("temperature", 0.9),
        top_p=cfg.training.get("top_p", 1.0),
        reward_weights=reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=prompts,
        reward_funcs=reward_fns,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("RL model saved to %s", out_dir)
    return str(out_dir)
