"""Latency benchmarking for teacher vs student models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from src.training.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure autoregressive latency")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def measure_model(
    model_path: str,
    batch_sizes: List[int],
    seq_lengths: List[int],
    warmup_steps: int,
    measure_steps: int,
) -> List[Dict[str, float]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    measurements: List[Dict[str, float]] = []
    for batch in batch_sizes:
        for seq in seq_lengths:
            input_ids = torch.randint(0, vocab_size - 1, (batch, seq), device=device)
            attention = torch.ones_like(input_ids)
            with torch.no_grad():
                for _ in range(warmup_steps):
                    model(input_ids=input_ids, attention_mask=attention)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(measure_steps):
                    model(input_ids=input_ids, attention_mask=attention)
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            tokens = batch * seq * measure_steps
            measurements.append(
                {
                    "batch_size": batch,
                    "seq_length": seq,
                    "tokens_per_second": tokens / duration,
                    "latency_ms": (duration / measure_steps) * 1000,
                }
            )
    return measurements


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    latency_cfg = config["latency"]
    results: Dict[str, List[Dict[str, float]]] = {}
    for model_name, model_path in config["models"].items():
        results[model_name] = measure_model(
            model_path,
            latency_cfg["batch_sizes"],
            latency_cfg["seq_lengths"],
            latency_cfg.get("warmup_steps", 3),
            latency_cfg.get("measure_steps", 10),
        )

    output_path = Path(config["logging"].get("output_path", "logs/eval/results.json")).with_name("latency.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        json.dump(results, writer, indent=2)


if __name__ == "__main__":
    main()
