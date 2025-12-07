#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/dpo_config.yaml}
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file config/accelerate_config.yaml \
  src/training/train_dpo.py --config "$CONFIG"
