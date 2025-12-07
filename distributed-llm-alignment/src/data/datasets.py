"""Dataset utilities for instruction tuning, preference learning, and distillation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class Sample:
    prompt: str
    response: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    reward: Optional[float] = None


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class InstructionDataset(Dataset):
    """Tokenizes prompt/response pairs for SFT."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        mask_prompt: bool = True,
        path: str | Path | None = None,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if records is None and path is None:
            raise ValueError("Provide either records or path for InstructionDataset")
        self.records = records if records is not None else read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.eos = tokenizer.eos_token or tokenizer.pad_token or "</s>"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        prompt = item["prompt"].strip()
        response = item["response"].strip()
        text = f"{prompt}\n\n{response}{self.eos}"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        labels = input_ids.clone()
        if self.mask_prompt:
            prompt_ids = self.tokenizer(
                f"{prompt}\n\n",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
            prompt_len = prompt_ids.shape[0]
            labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return pad_batch(batch, self.tokenizer.pad_token_id or 0)


class PreferenceDataset(Dataset):
    """Encodes (prompt, chosen, rejected) triples for DPO/RLHF."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        path: str | Path | None = None,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if records is None and path is None:
            raise ValueError("Provide either records or path for PreferenceDataset")
        self.records = records if records is not None else read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or tokenizer.pad_token or "</s>"

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        text = f"{prompt}\n\n{response}{self.eos}"
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        prompt = item["prompt"].strip()
        chosen = item["chosen"].strip()
        rejected = item["rejected"].strip()
        chosen_tok = self._tokenize(prompt, chosen)
        rejected_tok = self._tokenize(prompt, rejected)
        return {
            "chosen_input_ids": chosen_tok["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tok["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tok["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tok["attention_mask"].squeeze(0),
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "chosen": pad_batch(
                [
                    {
                        "input_ids": item["chosen_input_ids"],
                        "attention_mask": item["chosen_attention_mask"],
                    }
                    for item in batch
                ],
                self.tokenizer.pad_token_id or 0,
            ),
            "rejected": pad_batch(
                [
                    {
                        "input_ids": item["rejected_input_ids"],
                        "attention_mask": item["rejected_attention_mask"],
                    }
                    for item in batch
                ],
                self.tokenizer.pad_token_id or 0,
            ),
        }


class TeacherRolloutDataset(Dataset):
    """Student distillation dataset comprised of teacher responses."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or tokenizer.pad_token or "</s>"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        prompt = item["prompt"].strip()
        response = item["teacher_response"].strip()
        reward = float(item.get("reward", 1.0))
        text = f"{prompt}\n\n{response}{self.eos}"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }

    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_tensors = pad_batch(batch, self.tokenizer.pad_token_id or 0)
        rewards = torch.stack([item["reward"] for item in batch], dim=0)
        batch_tensors["reward"] = rewards
        return batch_tensors


class EvalPromptDataset(Dataset):
    """Plain prompt list used during evaluation."""

    def __init__(self, path: str | Path) -> None:
        self.records = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def pad_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, List[torch.Tensor]] = {key: [] for key in keys}
    for example in batch:
        for key in keys:
            collated[key].append(example[key])

    padded: Dict[str, torch.Tensor] = {}
    for key, tensors in collated.items():
        pad_value = -100 if key == "labels" else 0
        if key in {"input_ids", "attention_mask"}:
            pad_value = pad_token_id if key == "input_ids" else 0
        padded[key] = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=pad_value,
        )
    return padded


def _apply_template(template: Optional[str], record: Dict[str, Any]) -> str:
    if not template:
        return record
    return template.format(**record)


def load_instruction_records(cfg: Dict[str, Any], split: str = "train") -> List[Dict[str, Any]]:
    source = cfg.get("source", "local")
    limit = cfg.get("limit")
    records: List[Dict[str, Any]]
    if source == "hf":
        split_name = cfg.get(f"{split}_split") or cfg.get("split", split)
        ds = load_dataset(
            cfg["hf_path"],
            cfg.get("hf_name"),
            split=split_name,
            streaming=False,
        )
        cols = cfg.get("columns", {})
        prompt_key = cols.get("prompt", "prompt")
        response_key = cols.get("response", "response")
        template = cfg.get("template")
        records = []
        for row in ds:
            prompt_val = row[prompt_key]
            response_val = row[response_key]
            if template:
                prompt_val = template.format(**row)
            records.append({"prompt": prompt_val, "response": response_val})
    else:
        path = cfg.get(f"{split}_path") or cfg.get("path")
        records = read_jsonl(path)
    if limit:
        records = records[: int(limit)]
    return records


def load_preference_records(cfg: Dict[str, Any], split: str = "train") -> List[Dict[str, Any]]:
    source = cfg.get("source", "local")
    limit = cfg.get("limit")
    if source == "hf":
        split_name = cfg.get(f"{split}_split") or cfg.get("split", split)
        ds = load_dataset(
            cfg["hf_path"],
            cfg.get("hf_name"),
            split=split_name,
            streaming=False,
        )
        cols = cfg.get("columns", {})
        prompt_key = cols.get("prompt", "prompt")
        chosen_key = cols.get("chosen", "chosen")
        rejected_key = cols.get("rejected", "rejected")
        template = cfg.get("template")
        records = []
        for row in ds:
            prompt_val = row[prompt_key]
            if template:
                prompt_val = template.format(**row)
            records.append(
                {
                    "prompt": prompt_val,
                    "chosen": row[chosen_key],
                    "rejected": row[rejected_key],
                }
            )
    else:
        path = cfg.get(f"{split}_path") or cfg.get("path") or cfg.get("preference_path")
        records = read_jsonl(path)
    if limit:
        records = records[: int(limit)]
    return records


def build_instruction_dataset(cfg: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, split: str = "train") -> InstructionDataset:
    records = load_instruction_records(cfg, split=split)
    return InstructionDataset(
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", cfg.get("max_seq_length", 2048)),
        mask_prompt=cfg.get("mask_prompt", True),
        records=records,
    )


def build_preference_dataset(cfg: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, split: str = "train") -> PreferenceDataset:
    records = load_preference_records(cfg, split=split)
    return PreferenceDataset(
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", cfg.get("max_seq_length", 1024)),
        records=records,
    )
