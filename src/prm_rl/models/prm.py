"""Process Reward Model (PRM) — a step-level correctness scorer.

For simplicity we model the PRM as a binary sequence classifier
(`AutoModelForSequenceClassification` with 2 labels). Given the running
context `question + steps[:i]` it predicts P(step_i is correct).

`PRMScorer.score_steps(question, steps)` returns a list of per-step
probabilities that we use to build the various reward functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class PRMScorer:
    model: torch.nn.Module
    tokenizer: any  # transformers tokenizer
    # Default 512 (safe for both DistilBERT-base and DeBERTa-v3-base
    # native position embeddings). load_prm() will read the model's
    # actual `max_position_embeddings` and pass it explicitly so this
    # default is a fallback for callers constructing PRMScorer directly.
    max_length: int = 512
    device: str = "cpu"
    batch_size: int = 16

    @torch.no_grad()
    def score_steps(self, question: str, steps: Sequence[str]) -> list[float]:
        if not steps:
            return []
        prefix_parts = [f"Problem: {question}"]
        texts: list[str] = []
        for step in steps:
            prefix_parts.append(step)
            texts.append("\n\n".join(prefix_parts))
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            scores.extend(probs[:, 1].tolist())
        return scores


def load_prm(
    model_name_or_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    max_length: int | None = None,
) -> PRMScorer:
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The current step is at the TAIL of the input text (see
    # prm_data.build_prm_dataset). Right-truncation drops it and
    # collapses positive/negative pairs into the same tokens; force
    # left-truncation to mirror training. Documented in KB §6.1.
    tok.truncation_side = "left"
    # DeBERTa-v3's disentangled-attention numerically diverges to NaN
    # in fp16 forward passes on H200/A100 (a well-known upstream
    # issue). HF Trainer with `bf16=True` writes checkpoints whose
    # config.json claims dtype=float16 which then poisons
    # `from_pretrained(..., torch_dtype="auto")` into loading fp16.
    # Result: every prediction is NaN, downstream `process_correctness`
    # aggregates NaN → 0, and process-based reward signals during RL
    # training are silently zeroed.
    # Force fp32 by default (backbone models here are ≤184M params so
    # the throughput cost is negligible). Callers can override with
    # bf16 on H200 if they want the extra throughput.
    if torch_dtype is None:
        torch_dtype = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=2,
        torch_dtype=torch_dtype,
    )
    model.eval().to(device)
    # Cap tokenization at the model's actual positional buffer.
    # DistilBERT-base has a HARD limit at max_position_embeddings=512
    # (nn.Embedding(512, dim) with no buffer). Passing longer inputs
    # crashes at `inputs_embeds + position_embeddings` with:
    #   RuntimeError: The size of tensor a (520) must match the size
    #                 of tensor b (512) at non-singleton dimension 1
    # This is only reachable during RL training / eval when
    # completions grow long — training on short golden step chains
    # never hits it. DeBERTa-v3-base survived accidentally because
    # its disentangled attention has a 1024-token buffer on top of
    # native 512. Reading the cap from the model config makes this
    # correct for any future backbone.
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 512)
    return PRMScorer(
        model=model,
        tokenizer=tok,
        device=device,
        max_length=max_length,
    )
