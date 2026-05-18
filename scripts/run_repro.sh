#!/usr/bin/env bash
set -euo pipefail

WORKING_DIR=""
TASK="Read task.md and reproduce exactly the task it describes. Configure the environment, download required datasets and checkpoints, run the specified commands, extract reproduced metrics, extract original metrics from README or the markdown, and write results_comparison.md with original value, reproduced value, absolute difference, relative difference, and fluctuation analysis. Datasets and checkpoints/pretrained models must be downloaded from documented sources or copied from user-provided local paths; do not train a model to replace a checkpoint. When any command fails, first classify whether it is an environment problem, dependency version problem, dataset problem, checkpoint problem, command/entrypoint problem, metric parsing problem, or unknown. For environment errors, inspect package versions and decide whether a dependency is too new, too old, missing, or incompatible before changing source code. Record category, evidence, and repair action in .trae_env/repair_history.md before retrying. Do not use Docker."
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen3-8B}"
PROVIDER="${PROVIDER:-openai_compatible}"
CONFIG_FILE="${CONFIG_FILE:-trae_config.repro.yaml.example}"
TRAJECTORY_FILE="${TRAJECTORY_FILE:-trajectories/task_repro.json}"
MAX_STEPS="${MAX_STEPS:-160}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_repro.sh --working-dir /path/to/repo [options]

Options:
  --working-dir PATH       Target repository directory. Required.
  --task TEXT              Task prompt. Default asks the agent to read task.md.
  --base-url URL           Model API base URL. Default: http://127.0.0.1:8000/v1
  --api-key KEY            Model API key. Default: EMPTY
  --model NAME             Served model name. Default: Qwen3-8B
  --provider NAME          Provider. Use openai_compatible for /v1/chat/completions APIs. Default: openai_compatible
  --config-file PATH       Config file. Default: trae_config.repro.yaml.example
  --trajectory-file PATH   Trajectory output path. Default: trajectories/task_repro.json
  --max-steps N            Max agent loop steps. Default: 160
  -h, --help               Show this help.

Environment overrides:
  BASE_URL API_KEY MODEL PROVIDER CONFIG_FILE TRAJECTORY_FILE MAX_STEPS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --working-dir)
      WORKING_DIR="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --trajectory-file)
      TRAJECTORY_FILE="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WORKING_DIR" ]]; then
  echo "Error: --working-dir is required." >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/trae-cli" ]]; then
  TRAE_CLI=".venv/bin/trae-cli"
else
  TRAE_CLI="uv run trae-cli"
fi

$TRAE_CLI run "$TASK" \
  --agent-type env_setup_agent \
  --config-file "$CONFIG_FILE" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --model-base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --working-dir "$WORKING_DIR" \
  --trajectory-file "$TRAJECTORY_FILE" \
  --max-steps "$MAX_STEPS"
