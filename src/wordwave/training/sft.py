"""SFT on the golden dataset via `trl.SFTTrainer`.

Formats each example as::

    <prompt>\n\n<trace>\n\n#### <answer>

and lets `SFTTrainer` handle tokenization / packing / logging.
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ..data.golden import load_golden
from ..models.policy import load_policy_and_tokenizer
from ..utils.logging import get_logger

log = get_logger(__name__)


def _format_example(ex: dict) -> str:
    trace = ex["trace"].rstrip()
    return f"{ex['prompt']}\n\n{trace}"


def _build_hf_dataset(ds: Dataset) -> Dataset:
    return ds.map(lambda ex: {"text": _format_example(ex)}, remove_columns=ds.column_names)


def run_sft(cfg: DictConfig) -> str:
    log.info("Loading golden dataset from %s", cfg.data.golden_path)
    golden = load_golden(cfg.data.golden_path)
    train = _build_hf_dataset(golden)

    log.info("Loading policy %s", cfg.model.name)
    model, tokenizer = load_policy_and_tokenizer(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "bf16"),
        attn_implementation=cfg.model.get("attn_implementation"),
        load_in_4bit=cfg.model.get("load_in_4bit", False),
        device_map=cfg.model.get("device_map", "auto"),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        logging_steps=cfg.training.get("logging_steps", 10),
        save_strategy=cfg.training.get("save_strategy", "epoch"),
        save_total_limit=cfg.training.get("save_total_limit", 2),
        bf16=cfg.training.get("bf16", True),
        max_seq_length=cfg.training.get("max_seq_length", 1024),
        packing=cfg.training.get("packing", False),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        report_to=list(cfg.training.get("report_to", ["none"])),
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 0),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("SFT model saved to %s", out_dir)
    return str(out_dir)
