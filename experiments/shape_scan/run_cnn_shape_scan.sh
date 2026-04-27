#!/usr/bin/env bash
# Run CNN shape scan (masked patch reconstruction + patch AGOP).
# Prerequisites: pip install -r ../../requirements.txt (from this directory)
#   or install PyTorch from https://pytorch.org/get-started/locally/
# Usage:
#   ./run_cnn_shape_scan.sh --dry-run-configs
#   ./run_cnn_shape_scan.sh --target-params 3000000 --layers 2 24
# Quick smoke test:
#   ./run_cnn_shape_scan.sh --target-params 100000 --layers 2 5 --max-steps 20 --aofe-images 128
set -euo pipefail
cd "$(dirname "$0")"
exec python3 cnn_shape_scan.py "$@"
