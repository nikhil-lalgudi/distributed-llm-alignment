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

# Phase 2: Reward Modeling
bash scripts/launch_reward.sh config/reward_config.yaml

# Phase 3a: DPO Alignment
bash scripts/launch_dpo.sh config/dpo_config.yaml

# Phase 3b (optional): PPO RLHF Loop
bash scripts/launch_rlhf.sh config/rlhf_config.yaml

# Phase 4: Teacher Rollouts + Distillation
python -m src.training.generate_teacher_data --teacher checkpoints/dpo/latest --prompts data/processed/sft_eval.jsonl --output data/processed/teacher_rollouts.jsonl
bash scripts/launch_distill.sh config/distill_config.yaml

# Phase 5: Evaluation
bash scripts/launch_eval.sh config/eval_config.yaml
```

## Data Preparation
- **Raw data ingestion**: drop any third-party datasets into `data/raw/`. Use ad-hoc scripts or notebooks to normalize into the JSONL formats listed below and store under `data/processed/`.
- **Tokenization sanity**: run a quick `python -m src.training.generate_teacher_data ... --max_new_tokens 1` to ensure tokenizers agree on prompt formats before kicking off long runs.
- **Sharding for scale**: the dataset helpers in `src/data/datasets.py` operate on JSONL files; for very large corpora, pre-shard into multiple files and concatenate via symlinks or `datasets` streaming.

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

## Status Checklist
- [x] Project skeleton and documentation
- [x] Configuration templates and launch scripts
- [x] Modular data/model/training code paths
- [ ] Integrate real datasets + run training (user action)

## References
- DPO: Rafailov et al. 2023
- RLHF: Ouyang et al. 2022
- On-policy distillation: Hinton et al. 2015, OpenAI alignment blogs
