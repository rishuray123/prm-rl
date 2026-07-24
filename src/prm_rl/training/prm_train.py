"""Train the Process Reward Model.

We frame the PRM as a binary sequence classifier trained on the
step-level dataset produced by `prm_rl.data.build_prm_dataset` — this
is a stock HF `Trainer` setup, no custom RL loop required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import Dataset
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..data.golden import load_golden
from ..data.prm_data import (
    DEFAULT_NEGATIVE_KINDS,
    build_prm_dataset,
    summarize_prm_dataset,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


def _tokenize(tokenizer, max_length: int):
    def _fn(batch):
        enc = tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )
        enc["labels"] = batch["label"]
        return enc

    return _fn


def _compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def run_prm(cfg: DictConfig) -> str:
    log.info("Loading golden dataset from %s", cfg.data.golden_path)
    golden = load_golden(cfg.data.golden_path)
    inject_prob = float(cfg.data.get("inject_negatives_prob", 0.0))
    max_negs = int(cfg.data.get("max_negatives_per_example", 2))
    neg_kinds = list(cfg.data.get("negative_kinds", DEFAULT_NEGATIVE_KINDS))
    neg_seed = int(cfg.data.get("negatives_seed", cfg.get("seed", 0)))
    prm_ds: Dataset = build_prm_dataset(
        golden,
        inject_negatives_prob=inject_prob,
        negative_kinds=neg_kinds,
        max_negatives_per_example=max_negs,
        seed=neg_seed,
    )
    stats = summarize_prm_dataset(prm_ds)
    log.info(
        "Built PRM dataset: n=%d (pos=%d, neg=%d, pos_frac=%.3f)"
        " [inject_negatives_prob=%.2f]",
        stats["n"], stats["n_pos"], stats["n_neg"], stats["pos_frac"], inject_prob,
    )

    prm_ds = prm_ds.train_test_split(test_size=cfg.data.get("val_size", 0.05), seed=cfg.get("seed", 0))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The training text is "Problem: ... step_1 ... step_i-1 step_i"
    # with the label-carrying step at the TAIL. HF tokenizers default
    # to truncation_side='right', which drops the tail when a sequence
    # exceeds max_length — making positive/negative pairs (which
    # differ only at the tail) look identical to the model. Force
    # left-truncation so we always keep the discriminative step.
    # See docs/knowledge-base.md §6.1.
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, num_labels=2)

    tok_fn = _tokenize(
        tokenizer,
        cfg.training.get("max_length", cfg.training.get("max_seq_length", 1024)),
    )
    train = prm_ds["train"].map(tok_fn, batched=True, remove_columns=prm_ds["train"].column_names)
    val = prm_ds["test"].map(tok_fn, batched=True, remove_columns=prm_ds["test"].column_names)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.get("per_device_eval_batch_size", 32),
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.01),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.06),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=cfg.training.get("logging_steps", 20),
        report_to=list(cfg.training.get("report_to", ["none"])),
        bf16=cfg.training.get("bf16", True),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        seed=cfg.get("seed", 0),
        run_name=cfg.get("run_name"),
    )
    # transformers>=5 renamed `tokenizer` -> `processing_class` on Trainer.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("PRM saved to %s", out_dir)
    return str(out_dir)
