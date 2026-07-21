"""Load a causal-LM policy + tokenizer with sane defaults."""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name, None)


def load_policy_and_tokenizer(
    model_name: str,
    *,
    dtype: str | None = "bf16",
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str | None = "auto",
    **kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant_kwargs: dict[str, Any] = {}
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
            )
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("bitsandbytes required for quantized load") from e

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_resolve_dtype(dtype),
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **quant_kwargs,
        **kwargs,
    )
    return model, tok
