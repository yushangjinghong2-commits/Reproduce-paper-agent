# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""README-driven task reproduction agent for Python research repositories."""

import json
from pathlib import Path
from typing import override

from trae_agent.agent.agent_basics import AgentError, AgentStep
from trae_agent.agent.trae_agent import TraeAgent
from trae_agent.prompt.agent_prompt import ENV_SETUP_SYSTEM_PROMPT
from trae_agent.tools.base import ToolCall
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


class EnvSetupAgent(TraeAgent):
    """Trae agent variant for README-driven reproduction tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rejected_completion_count = 0

    @override
    def get_system_prompt(self) -> str:
        return ENV_SETUP_SYSTEM_PROMPT

    @override
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        super().new_task(task, extra_args, tool_names)

        if not extra_args:
            raise AgentError("Project path and reproduction task are required.")

        project_path = extra_args.get("project_path", "")
        issue = extra_args.get("issue", task)
        user_message = (
            f"[Project root path]:\n{project_path}\n\n"
            "[Reproduction task]:\n"
            f"{issue}\n\n"
            "[Required outputs]:\n"
            "- repro_plan.md\n"
            "- download_assets.sh\n"
            "- run_reproduction.sh\n"
            "- .trae_env/logs/\n"
            "- .trae_env/original_metrics.json\n"
            "- .trae_env/reproduced_metrics.json\n"
            "- .trae_env/reproduction_verification.json\n"
            "- .trae_env/asset_verification.json\n"
            "- .trae_env/repair_history.md when repairs are needed\n"
            "- results_comparison.md\n"
            "- failure_analysis.md if blocked\n"
            "- final_report.md\n\n"
            "[Completion rule]:\n"
            "Proceed strictly in order: README target -> repro_plan.md -> conda env -> one-by-one installs -> "
            "download_assets.sh -> asset_verification.json -> run_reproduction.sh -> metrics -> results_comparison.md. "
            "Start the next phase only after the previous phase succeeds. Use Linux/WSL bash, repository-relative scripts, "
            "default pip index, no Docker, and no setup.sh for environment installation. Use README Python or python=3.12. "
            "Torch/torchaudio/torchvision must be <2.6; for CUDA/torch errors, repair in the active target env with "
            "`pip install --force-reinstall \"torch<2.6\" \"torchaudio<2.6\" \"torchvision<2.6\"`. "
            "For Hugging Face assets use HF_ENDPOINT=https://hf-mirror.com. For flash-attn prefer the matching "
            "FlashAttention v2.8.3 wheel, falling back to `--no-build-isolation` only after recording the reason. "
            "Call task_done only after asset verification passes, run_reproduction.sh succeeds, real reproduced metrics exist, "
            "and results_comparison.md compares actual reproduced values with README values. Do not fabricate results.\n"
        )
        self._initial_messages = [
            LLMMessage(role="system", content=self.get_system_prompt()),
            LLMMessage(role="user", content=user_message),
        ]

    @override
    def _is_task_completed(self, llm_response: LLMResponse) -> bool:
        if not super()._is_task_completed(llm_response):
            return False

        verification_file = (
            Path(self.project_path) / ".trae_env" / "reproduction_verification.json"
        )
        reproduced_metrics_file = (
            Path(self.project_path) / ".trae_env" / "reproduced_metrics.json"
        )
        asset_verification_file = (
            Path(self.project_path) / ".trae_env" / "asset_verification.json"
        )
        comparison_file = Path(self.project_path) / "results_comparison.md"
        if not verification_file.exists():
            return False
        if not reproduced_metrics_file.exists():
            return False
        if not asset_verification_file.exists():
            return False
        if not comparison_file.exists():
            return False

        try:
            verification = json.loads(verification_file.read_text(encoding="utf-8"))
            reproduced_metrics = json.loads(
                reproduced_metrics_file.read_text(encoding="utf-8")
            )
            asset_verification = json.loads(
                asset_verification_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

        if int(verification.get("returncode", -1)) != 0:
            return False
        if not self._has_reproduced_result(reproduced_metrics):
            return False
        if not self._has_valid_asset_verification(asset_verification):
            return False
        if self._comparison_reports_failure(comparison_file):
            return False

        self._rejected_completion_count = 0
        return True

    @override
    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        if super().llm_indicates_task_completed(llm_response):
            return True
        if llm_response.tool_calls:
            return False
        response_lower = llm_response.content.lower()
        textual_completion_markers = [
            "task_done",
            "i have completed",
            "completed all required",
            "completed all deliverables",
            "reproduction completed",
        ]
        return any(marker in response_lower for marker in textual_completion_markers)

    @override
    async def _tool_call_handler(
        self, tool_calls: list[ToolCall] | None, step: AgentStep
    ) -> list[LLMMessage]:
        if self._tries_to_write_setup_script(tool_calls):
            return [LLMMessage(role="user", content=self.no_setup_script_message())]
        if self._should_redirect_cuda_bypass(tool_calls, step):
            return [LLMMessage(role="user", content=self.cuda_repair_message())]
        if self._should_redirect_environment_failure_summary(tool_calls, step):
            return [LLMMessage(role="user", content=self.environment_repair_message())]
        return await super()._tool_call_handler(tool_calls, step)

    def _tries_to_write_setup_script(self, tool_calls: list[ToolCall] | None) -> bool:
        if not tool_calls:
            return False
        for tool_call in tool_calls:
            if tool_call.name != "str_replace_based_edit_tool":
                continue
            command = str(tool_call.arguments.get("command", ""))
            path = str(tool_call.arguments.get("path", "")).lower()
            if command in {"create", "str_replace", "insert"} and path.endswith("setup.sh"):
                return True
        return False

    def no_setup_script_message(self) -> str:
        return (
            "Do not create or edit `setup.sh` for environment installation. "
            "Execute environment setup as direct bash commands one by one instead: "
            "`conda create -n <env> python=<version> -y`, then "
            "`conda run -n <env> pip install ...` for each README/requirements dependency. "
            "Record failed environment repairs in `.trae_env/repair_history.md`, then retry the failed command."
        )

    def _should_redirect_environment_failure_summary(
        self, tool_calls: list[ToolCall] | None, step: AgentStep
    ) -> bool:
        if not tool_calls or step.llm_response is None:
            return False
        response_lower = step.llm_response.content.lower()
        environment_failure_markers = [
            "unable to run",
            "unable to execute",
            "cannot run",
            "can't run",
            "environment issue",
            "environment issues",
            "environment problem",
            "dependency issue",
            "dependency issues",
            "import error",
            "import failure",
            "missing package",
        ]
        summary_markers = [
            "based on the readme",
            "expected values",
            "document that",
            "create a summary",
            "create the required files",
            "supposed to be reproduced",
        ]
        if not any(marker in response_lower for marker in environment_failure_markers):
            return False
        if not any(marker in response_lower for marker in summary_markers):
            return False
        return any(self._is_report_or_metric_write(call) for call in tool_calls)

    def _should_redirect_cuda_bypass(
        self, tool_calls: list[ToolCall] | None, step: AgentStep
    ) -> bool:
        if not tool_calls or step.llm_response is None:
            return False
        response_lower = step.llm_response.content.lower()
        cuda_markers = [
            "cuda library issue",
            "cuda libraries",
            "cuda unavailable",
            "cuda is unavailable",
            "cuda available: false",
            "pytorch_cuda",
            "torch/cuda",
        ]
        bypass_markers = [
            "simpler approach",
            "shouldn't prevent",
            "should not prevent",
            "just download",
            "continue with",
            "run the evaluation",
            "cpu fallback",
            "use cpu",
            "without cuda",
        ]
        if not any(marker in response_lower for marker in cuda_markers):
            return False
        if not any(marker in response_lower for marker in bypass_markers):
            return False
        return not any(self._is_cuda_repair_tool_call(call) for call in tool_calls)

    def _is_cuda_repair_tool_call(self, tool_call: ToolCall) -> bool:
        if tool_call.name != "bash":
            return False
        command = str(tool_call.arguments.get("command", "")).lower()
        repair_markers = [
            "repair_history.md",
            "pip install",
            "force-reinstall",
            "torch<2.6",
            "torch==",
            "pip show torch",
            "python -c",
            "torch.__version__",
            "torch.cuda",
        ]
        return any(marker in command for marker in repair_markers)

    def cuda_repair_message(self) -> str:
        return (
            "STOP. CUDA/PyTorch validation failure cannot be bypassed by downloading assets or running evaluation. "
            "Do not use CPU fallback. A CUDA error means the current torch, torchvision, and torchaudio packages are wrong. "
            "First repair the dedicated conda environment: "
            "1) inspect and record the CUDA/PyTorch error in `.trae_env/repair_history.md` with category `pytorch_cuda`; "
            "2) check versions with `conda run -n <env> python -c \"import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())\"`; "
            "3) activate or stay inside the target conda environment, then reinstall using the default configured pip index, e.g. "
            "`pip install --force-reinstall \"torch<2.6\" \"torchaudio<2.6\" \"torchvision<2.6\"`; "
            "4) verify torch imports and `torch.cuda.is_available()` is true; "
            "5) rerun the validation or failed reproduction command only after CUDA/PyTorch validation is repaired."
        )

    def _is_report_or_metric_write(self, tool_call: ToolCall) -> bool:
        if tool_call.name != "str_replace_based_edit_tool":
            return False
        command = str(tool_call.arguments.get("command", ""))
        if command not in {"create", "str_replace", "insert"}:
            return False
        path = str(tool_call.arguments.get("path", "")).lower()
        blocked_targets = [
            "results_comparison.md",
            "final_report.md",
            "failure_analysis.md",
            ".trae_env/reproduced_metrics.json",
        ]
        return any(path.endswith(target) for target in blocked_targets)

    def environment_repair_message(self) -> str:
        return (
            "STOP. Do not turn an environment failure into a README-based summary or expected-value report. "
            "The reproduced metrics must come from a successful run, not from README values. "
            "Continue repairing the environment instead: "
            "1) inspect the latest `.trae_env/logs/` error; "
            "2) if an import is missing, install that package inside the dedicated conda env with `conda run -n <env> pip install ...`; "
            "3) if an import/API error suggests the environment is too new, downgrade the relevant package version with a direct `conda run -n <env> pip install ...` command; "
            "4) if any CUDA error occurs, classify it as pytorch_cuda and reinstall in the active target environment with `pip install --force-reinstall \"torch<2.6\" \"torchaudio<2.6\" \"torchvision<2.6\"` using the default configured pip index; "
            "5) do not switch to CPU fallback; "
            "6) record the category, evidence, and repair action in `.trae_env/repair_history.md`; "
            "7) rerun setup/download/run as needed. "
            "Only write `results_comparison.md` or `.trae_env/reproduced_metrics.json` after an actual successful reproduction command produces real values."
        )

    def _has_reproduced_result(self, metrics: object) -> bool:
        """Accept only real, non-empty reproduced metrics/results."""
        failure_markers = {
            "blocked",
            "failed",
            "failure",
            "missing",
            "not_run",
            "not run",
            "not_executed",
            "not executed",
            "timeout",
            "unknown",
            "n/a",
            "none",
        }

        def valid_value(value: object) -> bool:
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str):
                normalized = value.strip().lower()
                return bool(normalized) and normalized not in failure_markers
            return False

        def walk(value: object, parent_key: str = "") -> bool:
            if isinstance(value, dict):
                status = value.get("status")
                if isinstance(status, str) and status.strip().lower() in failure_markers:
                    return False
                for key, child in value.items():
                    key_lower = str(key).lower()
                    if key_lower in {
                        "error",
                        "errors",
                        "failure",
                        "failure_reason",
                        "blocked_reason",
                    }:
                        continue
                    if walk(child, key_lower):
                        return True
                return False
            if isinstance(value, list):
                return any(walk(item, parent_key) for item in value)
            if parent_key in {"metric", "name", "target", "status", "note", "notes"}:
                return False
            return valid_value(value)

        return walk(metrics)

    def _has_valid_asset_verification(self, verification: object) -> bool:
        """Require explicit evidence that downloaded assets match the target."""
        if not isinstance(verification, dict):
            return False

        if verification.get("target_match") is not True:
            return False

        required_models = verification.get("required_models")
        downloaded_models = verification.get("downloaded_models")
        required_datasets = verification.get("required_datasets")
        downloaded_datasets = verification.get("downloaded_datasets")

        if not self._non_empty_asset_list(required_models):
            return False
        if not self._non_empty_asset_list(downloaded_models):
            return False
        if not self._non_empty_asset_list(required_datasets):
            return False
        if not self._non_empty_asset_list(downloaded_datasets):
            return False

        for key in ("models_match", "datasets_match", "all_required_assets_present"):
            if verification.get(key) is not True:
                return False

        return True

    def _non_empty_asset_list(self, value: object) -> bool:
        if not isinstance(value, list) or len(value) == 0:
            return False
        for item in value:
            if isinstance(item, str) and item.strip():
                continue
            if isinstance(item, dict) and any(str(v).strip() for v in item.values()):
                continue
            return False
        return True

    def _comparison_reports_failure(self, comparison_file: Path) -> bool:
        try:
            text = comparison_file.read_text(encoding="utf-8").lower()
        except OSError:
            return True
        failure_phrases = [
            "not reproduced",
            "not executed",
            "not run",
            "no reproduced result",
            "missing reproduced",
            "blocked",
            "failed to reproduce",
            "cannot execute",
            "无法执行",
            "没有实际",
            "未复现",
            "失败",
            "阻塞",
        ]
        return any(phrase in text for phrase in failure_phrases)

    @override
    def abort_on_rejected_completion(self) -> bool:
        self._rejected_completion_count += 1
        return self._rejected_completion_count >= 3

    @override
    def task_incomplete_message(self) -> str:
        return (
            "ERROR! The requested reproduction task is not complete. `task_done` is rejected. "
            "Do not repeat `task_done`. Continue the reproduction workflow now. "
            "Check which required artifact is missing or invalid: "
            "1) run `bash run_reproduction.sh` if it has not completed successfully; "
            "2) ensure `.trae_env/asset_verification.json` confirms the downloaded model/checkpoint and datasets match the user target; "
            "3) ensure `.trae_env/reproduction_verification.json` has returncode 0; "
            "4) extract real reproduced values from execution logs into `.trae_env/reproduced_metrics.json`; "
            "5) write `results_comparison.md` with the actual original/reproduced values and differences; "
            "6) if a command failed, inspect logs, classify the failure in `.trae_env/repair_history.md`, repair setup/download/run scripts, and retry; "
            "7) if torch was installed as `>=2.6` or any CUDA error occurred, reinstall `torch<2.6`, `torchaudio<2.6`, and `torchvision<2.6` inside the dedicated conda environment instead of using CPU fallback. "
            "Only call `task_done` after these checks pass. After three rejected completion attempts, the framework will stop the run as an error."
        )
