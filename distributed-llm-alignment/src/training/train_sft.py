"""Supervised fine-tuning entrypoint."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.data.datasets import InstructionDataset
from src.models.base_model import count_trainable_params, load_causal_lm
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
    parser = argparse.ArgumentParser(description="Distributed SFT training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def evaluate_loss(model, dataloader, accelerator) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = accelerator.gather(outputs.loss.repeat(batch["input_ids"].size(0))).mean()
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    accelerator = get_accelerator(config)
    log_rank_zero(accelerator, f"Loaded config from {args.config}")

    model_cfg: Dict = config["model"]
    bundle = load_causal_lm(
        model_cfg["model_name_or_path"],
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        use_flash_attention=model_cfg.get("use_flash_attention", False),
    )

    train_dataset = InstructionDataset(
        config["data"]["train_path"],
        tokenizer=bundle.tokenizer,
        max_length=model_cfg.get("max_seq_length", 2048),
        mask_prompt=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["optimization"]["micro_batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate,
    )

    eval_loader = None
    if config["data"].get("eval_path"):
        eval_dataset = InstructionDataset(
            config["data"]["eval_path"],
            tokenizer=bundle.tokenizer,
            max_length=model_cfg.get("max_seq_length", 2048),
            mask_prompt=True,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["optimization"]["micro_batch_size"],
            shuffle=False,
            collate_fn=eval_dataset.collate,
        )

    model = bundle.model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimization"]["learning_rate"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
        betas=(0.9, 0.95),
    )

    if eval_loader:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            model, optimizer, train_loader, eval_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    scheduler = get_scheduler(
        name=config["optimization"].get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=config["optimization"].get("warmup_steps", 0),
        num_training_steps=config["optimization"].get("max_train_steps"),
    )

    total_steps = config["optimization"]["max_train_steps"]
    log_every = config["logging"].get("log_every_steps", 20)
    save_every = config["logging"].get("save_every_steps", 500)
    eval_every = config["logging"].get("eval_every_steps", 200)
    output_dir = config["logging"]["output_dir"]
    log_dir = config["logging"].get("log_dir", "logs/sft")
    prepare_output_dirs(output_dir, log_dir)

    log_rank_zero(
        accelerator,
        f"Trainable parameters: {count_trainable_params(model)}",
    )

    progress_bar = range(total_steps)
    running_loss = RunningLoss()
    global_step = 0

    for epoch in range(10_000):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss.update(loss.detach().float().item())
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                accelerator.log({"train/loss": running_loss.average}, step=global_step)
                running_loss = RunningLoss()

            if eval_loader and global_step % eval_every == 0:
                eval_loss = evaluate_loss(model, eval_loader, accelerator)
                accelerator.log({"eval/loss": eval_loss}, step=global_step)

            if global_step % save_every == 0:
                tag = f"step_{global_step}"
                save_accelerator_state(accelerator, output_dir, tag)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_accelerator_state(accelerator, output_dir, "final")
    log_rank_zero(accelerator, "Training complete")


if __name__ == "__main__":
    main()
