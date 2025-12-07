"""Direct Preference Optimization training."""
from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.datasets import PreferenceDataset
from src.models.base_model import load_causal_lm
from src.training.utils import (
    RunningLoss,
    get_accelerator,
    load_config,
    log_rank_zero,
    prepare_output_dirs,
    save_accelerator_state,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def compute_logprobs(model, input_ids, attention_mask) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:].clone()
    mask = attention_mask[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    token_logps = (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return token_logps


def dpo_loss(policy_pos, policy_neg, ref_pos, ref_neg, beta: float) -> torch.Tensor:
    advantages = beta * ((policy_pos - policy_neg) - (ref_pos - ref_neg))
    return -torch.nn.functional.logsigmoid(advantages).mean()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 0))
    accelerator = get_accelerator(config)

    model_cfg: Dict = config["model"]
    policy_bundle = load_causal_lm(model_cfg["policy_model_name_or_path"], gradient_checkpointing=True)
    ref_bundle = load_causal_lm(model_cfg["reference_model_name_or_path"], gradient_checkpointing=False)
    ref_bundle.model.eval()
    for param in ref_bundle.model.parameters():
        param.requires_grad = False

    dataset = PreferenceDataset(
        config["data"]["preference_path"],
        policy_bundle.tokenizer,
        max_length=model_cfg.get("max_seq_length", 1024),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["optimization"]["micro_batch_size"],
        shuffle=True,
        collate_fn=dataset.collate,
    )

    optimizer = torch.optim.AdamW(
        policy_bundle.model.parameters(),
        lr=config["optimization"]["learning_rate"],
        weight_decay=0.01,
    )

    policy_bundle.model, optimizer, dataloader = accelerator.prepare(
        policy_bundle.model, optimizer, dataloader
    )
    ref_model = accelerator.prepare_model(ref_bundle.model)

    beta = model_cfg.get("beta", 0.1)
    total_steps = config["optimization"]["max_train_steps"]
    log_every = config["logging"].get("log_every_steps", 10)
    eval_every = config["logging"].get("eval_every_steps", 100)
    save_every = config["logging"].get("save_every_steps", 200)
    output_dir = config["logging"]["output_dir"]
    prepare_output_dirs(output_dir)

    running = RunningLoss()
    global_step = 0

    for epoch in range(10_000):
        for batch in dataloader:
            with accelerator.accumulate(policy_bundle.model):
                chosen = batch["chosen"]
                rejected = batch["rejected"]
                pol_pos = compute_logprobs(policy_bundle.model, chosen["input_ids"], chosen["attention_mask"])
                pol_neg = compute_logprobs(policy_bundle.model, rejected["input_ids"], rejected["attention_mask"])
                ref_pos = compute_logprobs(ref_model, chosen["input_ids"], chosen["attention_mask"])
                ref_neg = compute_logprobs(ref_model, rejected["input_ids"], rejected["attention_mask"])
                loss = dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            running.update(loss.detach().float().item())
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                accelerator.log({"train/loss": running.average}, step=global_step)
                running = RunningLoss()

            if global_step % eval_every == 0 and accelerator.is_main_process:
                preference_rate = (pol_pos > pol_neg).float().mean().item()
                accelerator.log({"train/preference_rate": preference_rate}, step=global_step)

            if global_step % save_every == 0:
                save_accelerator_state(accelerator, output_dir, f"step_{global_step}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_accelerator_state(accelerator, output_dir, "final")
    log_rank_zero(accelerator, "DPO training complete")


if __name__ == "__main__":
    main()
