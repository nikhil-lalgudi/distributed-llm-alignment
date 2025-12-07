#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/sft_config.yaml}
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file config/accelerate_config.yaml \
  src/training/train_sft.py --config "$CONFIG"
