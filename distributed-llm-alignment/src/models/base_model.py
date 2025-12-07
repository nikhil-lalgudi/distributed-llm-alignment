"""Helpers for loading causal language models used across training stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def load_causal_lm(
    model_name_or_path: str,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch_dtype or (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_flash_attention and hasattr(model, "config"):
        setattr(model.config, "use_cache", False)

    return ModelBundle(model=model, tokenizer=tokenizer)


def freeze_except_lora(model: PreTrainedModel, lora_layers: str | None = None) -> None:
    """Freeze base weights while keeping optional adapter modules trainable."""
    for name, param in model.named_parameters():
        trainable = lora_layers and lora_layers in name
        param.requires_grad = bool(trainable)


def count_trainable_params(model: PreTrainedModel) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }
