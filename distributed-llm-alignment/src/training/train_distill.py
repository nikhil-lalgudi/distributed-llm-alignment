"""On-policy distillation script."""
from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.datasets import TeacherRolloutDataset
from src.models.base_model import load_causal_lm
from src.training.utils import (
    RunningLoss,
    get_accelerator,
    load_config,
    log_rank_zero,
    get_distributed_sampler,
    maybe_clip_gradients,
    prepare_output_dirs,
    save_accelerator_state,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill aligned teacher into student")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 0))
    accelerator = get_accelerator(config)

    model_cfg: Dict = config["model"]
    student_bundle = load_causal_lm(
        model_cfg["student_model_name_or_path"],
        gradient_checkpointing=True,
        use_flash_attention=model_cfg.get("use_flash_attention", False),
    )
    dataset = TeacherRolloutDataset(
        config["data"]["teacher_samples_path"],
        student_bundle.tokenizer,
        max_length=model_cfg.get("max_seq_length", 2048),
    )
    train_sampler = get_distributed_sampler(dataset, accelerator, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config["optimization"]["micro_batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=dataset.collate,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        student_bundle.model.parameters(),
        lr=config["optimization"]["learning_rate"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
    )

    student_bundle.model, optimizer, dataloader = accelerator.prepare(
        student_bundle.model, optimizer, dataloader
    )

    total_steps = config["optimization"]["max_train_steps"]
    log_every = config["logging"].get("log_every_steps", 20)
    save_every = config["logging"].get("save_every_steps", 400)
    eval_every = config["logging"].get("eval_every_steps", 200)
    output_dir = config["logging"]["output_dir"]
    prepare_output_dirs(output_dir)

    running = RunningLoss()
    global_step = 0
    eff_batch = (
        config["optimization"]["micro_batch_size"]
        * accelerator.num_processes
        * config.get("hardware", {}).get("gradient_accumulation_steps", 1)
    )
    target_batch = config["optimization"].get("total_batch_size", eff_batch)
    log_rank_zero(
        accelerator,
        f"Effective global batch size: {eff_batch} (target {target_batch})",
    )

    max_grad_norm = config["optimization"].get("max_grad_norm", 1.0)
    for epoch in range(10_000):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for batch in dataloader:
            reward_values = batch.pop("reward")
            with accelerator.accumulate(student_bundle.model):
                outputs = student_bundle.model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                maybe_clip_gradients(accelerator, student_bundle.model, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            running.update(loss.detach().float().item())
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                accelerator.log(
                    {
                        "train/loss": running.average,
                        "train/reward_mean": reward_values.mean().item(),
                    },
                    step=global_step,
                )
                running = RunningLoss()

            if global_step % save_every == 0:
                save_accelerator_state(accelerator, output_dir, f"step_{global_step}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_accelerator_state(accelerator, output_dir, "final")
    log_rank_zero(accelerator, "Distillation complete")


if __name__ == "__main__":
    main()
