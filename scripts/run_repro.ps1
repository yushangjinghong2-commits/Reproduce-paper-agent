param(
    [Parameter(Mandatory = $true)]
    [string]$WorkingDir,

    [string]$Task = "Read task.md and reproduce exactly the task it describes. Configure the environment, download required datasets and checkpoints, run the specified commands, extract reproduced metrics, extract original metrics from README or the markdown, and write results_comparison.md with original value, reproduced value, absolute difference, relative difference, and fluctuation analysis. Datasets and checkpoints/pretrained models must be downloaded from documented sources or copied from user-provided local paths; do not train a model to replace a checkpoint. When any command fails, first classify whether it is an environment problem, dependency version problem, dataset problem, checkpoint problem, command/entrypoint problem, metric parsing problem, or unknown. For environment errors, inspect package versions and decide whether a dependency is too new, too old, missing, or incompatible before changing source code. Record category, evidence, and repair action in .trae_env/repair_history.md before retrying. Do not use Docker.",

    [string]$BaseUrl = "http://127.0.0.1:8000/v1",
    [string]$ApiKey = "EMPTY",
    [string]$Model = "Qwen3-8B",
    [string]$Provider = "openai_compatible",
    [string]$ConfigFile = "trae_config.repro.yaml.example",
    [string]$TrajectoryFile = "trajectories\task_repro.json",
    [int]$MaxSteps = 160
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& ".\.venv\Scripts\trae-cli.exe" run $Task `
    --agent-type env_setup_agent `
    --config-file $ConfigFile `
    --provider $Provider `
    --model $Model `
    --model-base-url $BaseUrl `
    --api-key $ApiKey `
    --working-dir $WorkingDir `
    --trajectory-file $TrajectoryFile `
    --max-steps $MaxSteps
