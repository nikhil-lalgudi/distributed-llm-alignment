#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/eval_config.yaml}
export TOKENIZERS_PARALLELISM=false

python -m src.eval.eval_alignment --config "$CONFIG"
python -m src.eval.eval_latency --config "$CONFIG"
