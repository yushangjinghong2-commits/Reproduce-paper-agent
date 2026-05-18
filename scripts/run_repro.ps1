param(
    [Parameter(Mandatory = $true)]
    [string]$WorkingDir,

    [Parameter(Mandatory = $true)]
    [string]$Target,

    [string]$Task = "Read only README.md and reproduce the user-specified target prompt. Use README text only. First plan the environment setup commands. Then identify the exact README-documented inference/evaluation command needed to complete the target. Then derive and download/copy only the datasets, checkpoints, pretrained models, or sample inputs required by that command. Do not inspect other repository files for planning. Write results_comparison.md for the target prompt with original README value/result, reproduced value/result, absolute difference when numeric, relative difference when numeric, and fluctuation analysis. Datasets and checkpoints/pretrained models must be downloaded from README-documented sources or copied from user-provided local paths; do not train a model to replace a checkpoint. When any command fails, first classify whether it is an environment problem, dependency version problem, dataset problem, checkpoint problem, command/entrypoint problem, metric parsing problem, or unknown. For environment errors, inspect logs and installed package versions, then decide whether a dependency is too new, too old, missing, or incompatible before changing source code. Record category, evidence, and repair action in .trae_env/repair_history.md before retrying. Do not use Docker.",

    [string]$BaseUrl = "http://127.0.0.1:8000/v1",
    [string]$ApiKey = "EMPTY",
    [string]$Model = "Qwen3-8B",
    [string]$Provider = "openai_compatible",
    [string]$ConfigFile = "trae_config.repro.yaml.example",
    [string]$TrajectoryFile = "",
    [int]$MaxSteps = 160
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TrajectoryFile)) {
    $repoName = Split-Path -Leaf (Resolve-Path -LiteralPath $WorkingDir)
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $TrajectoryFile = "trajectories\$repoName`_$timestamp.json"
}

$Task = "$Task User-specified reproduction target prompt: $Target. Reproduce only this target and do not expand to other README results."

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
