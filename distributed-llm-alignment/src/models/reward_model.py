"""Reward model architecture for preference learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

Pooling = Literal["last_token", "mean"]


@dataclass
class RewardArtifacts:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def build_reward_model(
    base_model_name_or_path: str,
    pooling: Pooling = "last_token",
    dropout: float = 0.1,
) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    head = RewardModel(backbone, pooling=pooling, dropout=dropout)
    return head, tokenizer


class RewardModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, pooling: Pooling = "last_token", dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        hidden = backbone.config.hidden_size
        self.scorer = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = (
            self._last_token(hidden_states, attention_mask)
            if self.pooling == "last_token"
            else self._mean_pool(hidden_states, attention_mask)
        )
        return self.scorer(pooled).squeeze(-1)

    @staticmethod
    def _last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.size(0)), lengths]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked = hidden_states * attention_mask.unsqueeze(-1)
        return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)


def pairwise_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
