"""Reward model training script."""
from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.datasets import PreferenceDataset
from src.models.reward_model import build_reward_model, pairwise_loss
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
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def evaluate(model, dataloader, accelerator) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            chosen_scores = model(
                input_ids=batch["chosen"]["input_ids"],
                attention_mask=batch["chosen"]["attention_mask"],
            )
            rejected_scores = model(
                input_ids=batch["rejected"]["input_ids"],
                attention_mask=batch["rejected"]["attention_mask"],
            )
            loss = pairwise_loss(chosen_scores, rejected_scores)
            losses.append(accelerator.gather(loss.repeat(1)).mean().item())
            correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.size(0)
    model.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 0))
    accelerator = get_accelerator(config)

    model_cfg: Dict = config["model"]
    reward_model, tokenizer = build_reward_model(
        model_cfg["base_model_name_or_path"],
        pooling=model_cfg.get("pooling", "last_token"),
        dropout=model_cfg.get("dropout", 0.1),
    )

    dataset = PreferenceDataset(
        config["data"]["train_path"],
        tokenizer,
        max_length=model_cfg.get("max_seq_length", 1024),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["optimization"]["micro_batch_size"],
        shuffle=True,
        collate_fn=dataset.collate,
    )

    eval_loader = None
    if config["data"].get("eval_path"):
        eval_dataset = PreferenceDataset(
            config["data"]["eval_path"],
            tokenizer,
            max_length=model_cfg.get("max_seq_length", 1024),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["optimization"]["micro_batch_size"],
            shuffle=False,
            collate_fn=eval_dataset.collate,
        )

    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=config["optimization"]["learning_rate"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
    )

    if eval_loader:
        reward_model, optimizer, dataloader, eval_loader = accelerator.prepare(
            reward_model, optimizer, dataloader, eval_loader
        )
    else:
        reward_model, optimizer, dataloader = accelerator.prepare(
            reward_model, optimizer, dataloader
        )

    total_steps = config["optimization"]["max_train_steps"]
    log_every = config["logging"].get("log_every_steps", 10)
    eval_every = config["logging"].get("eval_every_steps", 100)
    save_every = config["logging"].get("save_every_steps", 200)
    output_dir = config["logging"]["output_dir"]
    log_dir = config["logging"].get("log_dir", "logs/reward")
    prepare_output_dirs(output_dir, log_dir)

    running = RunningLoss()
    global_step = 0

    for epoch in range(10_000):
        for batch in dataloader:
            with accelerator.accumulate(reward_model):
                chosen_scores = reward_model(
                    input_ids=batch["chosen"]["input_ids"],
                    attention_mask=batch["chosen"]["attention_mask"],
                )
                rejected_scores = reward_model(
                    input_ids=batch["rejected"]["input_ids"],
                    attention_mask=batch["rejected"]["attention_mask"],
                )
                loss = pairwise_loss(chosen_scores, rejected_scores)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            running.update(loss.detach().float().item())
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                accelerator.log({"train/loss": running.average}, step=global_step)
                running = RunningLoss()

            if eval_loader and global_step % eval_every == 0:
                metrics = evaluate(reward_model, eval_loader, accelerator)
                accelerator.log({"eval/loss": metrics["loss"], "eval/acc": metrics["accuracy"]}, step=global_step)

            if global_step % save_every == 0:
                save_accelerator_state(accelerator, output_dir, f"step_{global_step}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_accelerator_state(accelerator, output_dir, "final")
    log_rank_zero(accelerator, "Finished reward training")


if __name__ == "__main__":
    main()
