"""Generate on-policy rollouts from aligned teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.datasets import read_jsonl
from src.models.reward_model import build_reward_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample teacher responses")
    parser.add_argument("--teacher", required=True, help="Path or HF id for teacher model")
    parser.add_argument("--prompts", required=True, help="JSONL prompts file")
    parser.add_argument("--output", required=True, help="Destination JSONL")
    parser.add_argument("--reward_model", default=None, help="Optional reward model checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    reward_model = None
    reward_tokenizer = None
    if args.reward_model:
        reward_model, reward_tokenizer = build_reward_model(args.reward_model)
        ckpt_dir = Path(args.reward_model)
        if ckpt_dir.is_dir():
            weight_file = ckpt_dir / "pytorch_model.bin"
            if weight_file.exists():
                state_dict = torch.load(weight_file, map_location="cpu")
                reward_model.load_state_dict(state_dict, strict=False)
        reward_model.to(device)
        reward_model.eval()

    prompts = [item["prompt"] for item in read_jsonl(args.prompts)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.output).open("w", encoding="utf-8") as writer:
        for batch_prompts in tqdm(chunk_list(prompts, args.batch_size), desc="Generating"):
            encodings = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encodings,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                )
            prompt_lengths = encodings["attention_mask"].sum(dim=1)
            responses = []
            rewards = []
            for gen, prompt, prompt_len in zip(generated, batch_prompts, prompt_lengths):
                response = tokenizer.decode(gen[int(prompt_len) :], skip_special_tokens=True)
                responses.append(response)
                reward_value = None
                if reward_model and reward_tokenizer:
                    fused = reward_tokenizer(
                        f"{prompt}\n\n{response}",
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=reward_tokenizer.model_max_length,
                    ).to(device)
                    with torch.no_grad():
                        score = reward_model(
                            input_ids=fused["input_ids"],
                            attention_mask=fused["attention_mask"],
                        )
                    reward_value = float(score.item())
                record = {
                    "prompt": prompt,
                    "teacher_response": response,
                }
                if reward_value is not None:
                    record["reward"] = reward_value
                writer.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
