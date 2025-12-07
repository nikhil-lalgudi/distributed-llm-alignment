"""On-policy distillation script."""
from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
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
    distill_cfg: Dict = config.get("distill", {})
    student_bundle = load_causal_lm(
        model_cfg["student_model_name_or_path"],
        gradient_checkpointing=True,
        use_flash_attention=model_cfg.get("use_flash_attention", False),
    )
    teacher_bundles: List = []
    if distill_cfg.get("use_kl", False) and distill_cfg.get("on_policy", False):
        teacher_paths = distill_cfg.get("teacher_model_names_or_paths") or []
        if not teacher_paths:
            single = distill_cfg.get("teacher_model_name_or_path") or model_cfg.get("teacher_path")
            if single:
                teacher_paths = [single]
        if not teacher_paths:
            raise ValueError("KL distillation requested but no teacher model path provided")
        for tpath in teacher_paths:
            tb = load_causal_lm(
                tpath,
                gradient_checkpointing=False,
                use_flash_attention=False,
            )
            tb.model.eval()
            teacher_bundles.append(tb)
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

    if teacher_bundles:
        models_to_prepare = [student_bundle.model] + [tb.model for tb in teacher_bundles] + [optimizer, dataloader]
        prepared = accelerator.prepare(*models_to_prepare)
        student_bundle.model = prepared[0]
        for i, tb in enumerate(teacher_bundles):
            tb.model = prepared[i + 1]
        optimizer = prepared[-2]
        dataloader = prepared[-1]
    else:
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
    use_kl = distill_cfg.get("use_kl", False)
    on_policy = distill_cfg.get("on_policy", False)

    for epoch in range(10_000):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for batch in dataloader:
            reward_values = batch.pop("reward")
            with accelerator.accumulate(student_bundle.model):
                if use_kl and on_policy and teacher_bundles:
                    labels = batch.pop("labels", None)
                    attention_mask = batch.get("attention_mask")
                    student_outputs = student_bundle.model(**batch)
                    student_logits = student_outputs.logits
                    student_log_probs = F.log_softmax(student_logits, dim=-1)
                    with torch.no_grad():
                        teacher_probs_list = []
                        for tb in teacher_bundles:
                            t_out = tb.model(**batch)
                            t_logits = t_out.logits
                            teacher_probs_list.append(F.softmax(t_logits, dim=-1))
                        teacher_probs = torch.stack(teacher_probs_list, dim=0).mean(dim=0)
                    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)
                    if attention_mask is not None:
                        loss = (kl_per_token * attention_mask).sum() / attention_mask.sum()
                    else:
                        loss = kl_per_token.mean()
                else:
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
