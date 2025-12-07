# Distributed On-Policy Distillation for Preference-Aligned LLMs

End-to-end research scaffold for training, aligning, distilling, and evaluating large language models across multi-GPU clusters. The project follows six phases that mirror modern preference optimization workflows: SFT → Reward Modeling → DPO/RLHF → On-Policy Distillation → Evaluation → Packaging.

## Highlights
- Compatible with `accelerate`, `DeepSpeed`, or native FSDP launches.
- Clear separation between configs, launch scripts, data loaders, and training loops.
- Support for both DPO and PPO style RLHF, plus reward-weighted on-policy distillation.
- Built-in evaluation hooks for alignment quality and latency/efficiency metrics.

## Repository Layout
```
distributed-llm-alignment/
├── config/                 # YAML configs for every phase
├── data/                   # Place raw/preprocessed corpora here
├── scripts/                # Cluster-friendly launchers
├── src/                    # Python package with modular components
├── checkpoints/            # Model checkpoints per stage
├── logs/                   # Experiment logs / metrics
├── requirements.txt        # Core Python dependencies
└── README.md               # Project documentation
```

## Quick Start
1. **Environment**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Cluster configuration**: edit `config/*.yaml` to reflect your hardware, dataset paths, and preferred backend.
3. **Launch phases** (example with `accelerate`):
   ```bash
   bash scripts/launch_sft.sh
   bash scripts/launch_reward.sh
   bash scripts/launch_dpo.sh
   bash scripts/launch_distill.sh
   bash scripts/launch_eval.sh
   ```

## Phase Overview
1. **SFT**: Fine-tune base model on curated instruction data via `src/training/train_sft.py`.
2. **Reward Model**: Train scalar preference model using pairwise data via `src/training/train_reward.py`.
3. **Policy Alignment**: Run DPO (`train_dpo.py`) or PPO (`train_rlhf.py`) starting from SFT weights.
4. **Distillation**: Generate on-policy traces from the teacher and transfer knowledge into a smaller student via `train_distill.py`.
5. **Evaluation**: Compare models on alignment/safety metrics and latency using `src/eval/eval_alignment.py` and `src/eval/eval_latency.py`.
6. **Packaging**: Collect metrics, plots, and qualitative samples for reports/portfolio.

## Phase-by-Phase Commands
```bash
# Phase 1: Supervised Fine-Tuning
bash scripts/launch_sft.sh config/sft_config.yaml
# or use presets: config/sft_alpaca.yaml, config/sft_ultrachat.yaml

# Phase 2: Reward Modeling
bash scripts/launch_reward.sh config/reward_config.yaml
# or use HF preset: config/reward_hh.yaml

# Phase 3a: DPO Alignment
bash scripts/launch_dpo.sh config/dpo_config.yaml
# or HF preset: config/dpo_hh.yaml

# Phase 3b (optional): PPO RLHF Loop
bash scripts/launch_rlhf.sh config/rlhf_config.yaml

# Phase 4: Teacher Rollouts + Distillation
python -m src.training.generate_teacher_data --teacher checkpoints/dpo/latest --prompts data/processed/sft_eval.jsonl --output data/processed/teacher_rollouts.jsonl
bash scripts/launch_distill.sh config/distill_config.yaml
# (multi-teacher / KL on-policy) bash scripts/launch_distill_multi.sh config/distill_config.yaml

# Phase 5: Evaluation
bash scripts/launch_eval.sh config/eval_config.yaml
```

## Data Preparation
- **Raw data ingestion**: drop any third-party datasets into `data/raw/`. Use ad-hoc scripts or notebooks to normalize into the JSONL formats listed below and store under `data/processed/`.
- **Tokenization sanity**: run a quick `python -m src.training.generate_teacher_data ... --max_new_tokens 1` to ensure tokenizers agree on prompt formats before kicking off long runs.
- **Sharding for scale**: the dataset helpers in `src/data/datasets.py` operate on JSONL files; for very large corpora, pre-shard into multiple files and concatenate via symlinks or `datasets` streaming.

### Data Source Presets
Ready-made YAML presets live in `config/data_sources/`:
- `sft_alpaca.yaml` (HF: `yahma/alpaca-cleaned`)
- `sft_ultrachat.yaml` (HF: `HuggingFaceH4/ultrachat_200k`)
- `pref_hh_rlhf.yaml` (HF: `Anthropic/hh-rlhf`)
- `pref_shp.yaml` (HF: `stanfordnlp/SHP`)

Use them by swapping the `data:` block in your phase config (e.g., `config/sft_alpaca.yaml`, `config/reward_hh.yaml`, `config/dpo_hh.yaml`). Each preset supports `limit` to cap records and `eval_split` for validation.

## Standard JSONL Schemas
| Stage | Required Keys |
| --- | --- |
| SFT | `{"prompt": str, "response": str}` |
| Reward/DPO | `{"prompt": str, "chosen": str, "rejected": str}` |
| RLHF prompts | `{"prompt": str}` |
| Teacher rollouts | `{"prompt": str, "teacher_response": str, "reward": float}` |

## Evaluation Artifacts
- `logs/eval/results.json`: aggregate alignment metrics for every model × benchmark pair.
- `logs/eval/summary.md`: Markdown table ready for reports or README excerpts.
- `logs/eval/latency.json`: throughput + latency comparisons for the same checkpoints (produced by `eval_latency.py`).
- To splice samples into blog posts, run `python -m src.training.generate_teacher_data ... --max_new_tokens 128` against each checkpoint and collate the JSONL outputs.

## Troubleshooting Checklist
- Validate cluster visibility via `accelerate env` before launches.
- Double-check `config/*` paths whenever moving checkpoints between machines.
- If you change the tokenizer or base model mid-project, regenerate processed datasets to avoid EOS/id mismatches.
- For large models, prefer `hardware.deepspeed_config` (Zero-3 provided) and set `mixed_precision: bf16` on A100/H100. Tune `gradient_accumulation_steps` so `micro_batch_size * world_size * grad_accum = total_batch_size`.

## Backends
- **DeepSpeed Zero-3**: enabled by setting `hardware.deepspeed_config: config/deepspeed_zero3.json` (already in defaults).
- **FSDP**: set `hardware.fsdp` in configs, e.g.:
   ```yaml
   hardware:
      mixed_precision: bf16
      gradient_accumulation_steps: 32
      fsdp:
         sharding_strategy: 1   # FULL_SHARD
         offload_params: false
         auto_wrap_policy: size
         min_num_params: 1_000_000
   ```

## Ablation Fragments (drop-in YAML overlays)
- `config/ablations/low_lr.yaml`: smaller LR + longer warmup.
- `config/ablations/high_beta_dpo.yaml`: sharper DPO beta.
- `config/ablations/clip_grad_0_5.yaml`: tighter gradient clipping.
Apply by merging keys into your base config (or via Hydra-style overlays if you adopt Hydra/OmegaConf).

## Dataset Presets
- SFT: `config/data_sources/sft_alpaca.yaml`, `config/data_sources/sft_ultrachat.yaml`.
- Preference: `config/data_sources/pref_hh_rlhf.yaml`, `config/data_sources/pref_shp.yaml`.
- RLHF prompts: `config/data_sources/rlhf_prompts_hh.yaml`.
Swap the `data:` or `sampling:` blocks in your configs with these presets to train on real HF datasets without manual preprocessing.

## Distillation Modes
- **Cross-entropy (default)**: `distill.use_kl: false` uses teacher responses as labels.
- **On-policy KL (teacher ensemble optional)**: set `distill.use_kl: true`, `distill.on_policy: true`, and specify one or more teachers via `teacher_model_name_or_path` or `teacher_model_names_or_paths` (probs are averaged). Use `scripts/launch_distill_multi.sh` to run with these settings.

## Data Expectations
- **Instruction SFT**: JSONL entries with `{"prompt": str, "response": str}`.
- **Preference Data**: JSONL entries with `{"prompt": str, "chosen": str, "rejected": str}`.
- **Teacher Samples**: JSONL entries with `{"prompt": str, "teacher_response": str, "reward": float}` generated via `src/training/generate_teacher_data.py`.

## Monitoring & Logging
- Uses `wandb` hooks when available; otherwise falls back to JSON + tqdm logging under `logs/`.
- Checkpointing policy is controlled via each config file (steps/epochs, keep-last-N, etc.).

## Extending
- Swap in alternative backends (FSDP, ZeRO) by editing the corresponding config blocks.
- Plug in new evaluation benchmarks by adding loaders to `src/eval/eval_alignment.py`.
- Add ablations by composing additional Hydra/OMEGACONF config fragments (if desired).

## References
- DPO: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems (NeurIPS 2023). arXiv:2305.18290.
- RLHF: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., … & Christiano, P. (2022). Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS 2022). arXiv:2203.02155.
- On-policy distillation: Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Agarwal, R., Vieillard, N., & others (2024). On-Policy Distillation of Language Models. In International Conference on Learning Representations (ICLR 2024). arXiv:2306.13649.
