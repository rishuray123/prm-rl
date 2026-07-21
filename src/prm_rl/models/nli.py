"""Wrapper around an NLI model used for contradiction detection (Arm 4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class ContradictionScorer:
    model: torch.nn.Module
    tokenizer: any
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 16
    contradiction_index: int = 0

    @torch.no_grad()
    def contradiction_probs(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[float]:
        assert len(premises) == len(hypotheses)
        if not premises:
            return []
        probs: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            p = list(premises[start : start + self.batch_size])
            h = list(hypotheses[start : start + self.batch_size])
            enc = self.tokenizer(
                p, h,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            p_batch = torch.softmax(logits.float(), dim=-1)
            probs.extend(p_batch[:, self.contradiction_index].tolist())
        return probs

    def pairwise_max_contradiction(self, steps: Sequence[str]) -> float:
        """Return the max contradiction probability across all (i<j) pairs."""
        if len(steps) < 2:
            return 0.0
        prem, hyp = [], []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                prem.append(steps[i])
                hyp.append(steps[j])
        probs = self.contradiction_probs(prem, hyp)
        return max(probs) if probs else 0.0


def load_nli(model_name: str = DEFAULT_NLI, device: str = "cuda") -> ContradictionScorer:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    # figure out contradiction index from id2label
    id2label = {i: str(l).lower() for i, l in model.config.id2label.items()}
    contradiction_index = 0
    for i, lbl in id2label.items():
        if "contradiction" in lbl:
            contradiction_index = i
            break
    return ContradictionScorer(
        model=model, tokenizer=tok, device=device,
        contradiction_index=contradiction_index,
    )
