"""Lightweight PPO-style RLHF loop built on top of Accelerate."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.datasets import read_jsonl
from src.models.base_model import load_causal_lm
from src.models.reward_model import build_reward_model
from src.training.utils import (
    RunningLoss,
    get_accelerator,
    load_config,
    log_rank_zero,
    prepare_output_dirs,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF PPO loop")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_prompts(path: str | Path) -> List[str]:
    samples = read_jsonl(path)
    return [record["prompt"] for record in samples]


def sequence_logprob(model, input_ids, attention_mask) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    seq_logprob = (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return seq_logprob


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 0))
    accelerator = get_accelerator(config)

    model_cfg: Dict = config["model"]
    policy_bundle = load_causal_lm(model_cfg["policy_model_name_or_path"], gradient_checkpointing=True)
    ref_bundle = load_causal_lm(model_cfg["reference_model_name_or_path"], gradient_checkpointing=False)
    reward_cfg = config.get("reward_model", {})
    reward_base = reward_cfg.get("base_model_name_or_path", model_cfg["policy_model_name_or_path"])
    reward_model, reward_tokenizer = build_reward_model(
        reward_base,
        pooling="last_token",
        dropout=0.1,
    )
    reward_checkpoint = reward_cfg.get("path")
    if reward_checkpoint:
        ckpt_file = Path(reward_checkpoint)
        if ckpt_file.is_dir():
            safetensor = ckpt_file / "pytorch_model.bin"
            if safetensor.exists():
                state_dict = torch.load(safetensor, map_location="cpu")
                reward_model.load_state_dict(state_dict, strict=False)

    prompts = load_prompts(config["sampling"]["prompt_path"])
    generation_params = config["ppo"].get("generation_params", {"max_new_tokens": 256})

    policy_model = policy_bundle.model
    ref_model = ref_bundle.model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config["ppo"].get("learning_rate", 1e-6),
        betas=(0.9, 0.95),
    )

    policy_model, optimizer = accelerator.prepare(policy_model, optimizer)
    ref_model = accelerator.prepare_model(ref_model)
    reward_model = accelerator.prepare_model(reward_model)

    prepare_output_dirs(config["logging"]["output_dir"], config["logging"].get("log_dir", "logs/rlhf"))

    steps = config["ppo"].get("steps", 1024)
    batch_size = config["ppo"].get("batch_size", 64)
    kl_coef = config["ppo"].get("kl_coef", 0.1)
    running_loss = RunningLoss()

    for step in range(steps):
        batch_prompts = random.sample(prompts, k=min(batch_size, len(prompts)))
        tokenized = policy_bundle.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_cfg.get("max_seq_length", 1024),
        ).to(accelerator.device)

        with torch.no_grad():
            generated = accelerator.unwrap_model(policy_model).generate(**tokenized, **generation_params)
        attention_mask = (generated != policy_bundle.tokenizer.pad_token_id).long()

        policy_logp = sequence_logprob(policy_model, generated, attention_mask)
        with torch.no_grad():
            ref_logp = sequence_logprob(ref_model, generated, attention_mask)

        prompt_lengths = tokenized["attention_mask"].sum(dim=1)
        responses = []
        for gen, length in zip(generated, prompt_lengths):
            gen_response = gen[int(length) :]
            responses.append(policy_bundle.tokenizer.decode(gen_response, skip_special_tokens=True))
        fused_text = [f"{p}\n\n{r}" for p, r in zip(batch_prompts, responses)]
        reward_inputs = reward_tokenizer(
            fused_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_cfg.get("max_seq_length", 1024),
        ).to(accelerator.device)
        reward_scores = reward_model(
            input_ids=reward_inputs["input_ids"],
            attention_mask=reward_inputs["attention_mask"],
        )

        kl = policy_logp - ref_logp
        rewards = reward_scores - kl_coef * kl
        advantages = rewards - rewards.mean()

        loss = -(advantages.detach() * policy_logp).mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        running_loss.update(loss.detach().float().item())
        if (step + 1) % 10 == 0 and accelerator.is_main_process:
            accelerator.log({"train/loss": running_loss.average, "train/kl": kl.mean().item()}, step=step + 1)
            running_loss = RunningLoss()

    accelerator.wait_for_everyone()
    accelerator.save_state(config["logging"]["output_dir"])
    log_rank_zero(accelerator, "RLHF PPO loop complete")


if __name__ == "__main__":
    main()
