#!/usr/bin/env bash
# Run Tiny-GPT depth scan on OpenWebText (default in transformer_shape_scan.py).
# Results: ./results_tiny_gpt_depth_openwebtext/
#
# By default the Python script **streams** OpenWebText and caps materialized text
# (~1200 MiB train / 128 MiB valid) so small container roots are not filled.
# Full corpus: add --openwebtext-full-download (needs tens of GB free).
#
# If you see "No space left on device", free space first, e.g. remove an old hub
# cache: rm -rf ~/.cache/huggingface /workspace/.hf_home  (only if you do not need them)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep HuggingFace downloads under the repo data dir instead of a full root filesystem.
export HF_HOME="${SCRIPT_DIR}/data/openwebtext/hf_home"
mkdir -p "$HF_HOME"

if ! python3 -c "import datasets" 2>/dev/null; then
  echo "Installing HuggingFace datasets (required for OpenWebText)..."
  python3 -m pip install -U datasets
fi

# Extra CLI args are forwarded, e.g.:
#   ./run_openwebtext_shape_scan.sh --dry-run-configs
#   ./run_openwebtext_shape_scan.sh --max-steps 50
#   ./run_openwebtext_shape_scan.sh --openwebtext-max-train-mib 800
exec python3 transformer_shape_scan.py "$@"
