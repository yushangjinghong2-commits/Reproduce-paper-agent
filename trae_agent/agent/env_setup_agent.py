# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""README-driven task reproduction agent for Python research repositories."""

import json
from pathlib import Path
from typing import override

from trae_agent.agent.agent_basics import AgentError
from trae_agent.agent.trae_agent import TraeAgent
from trae_agent.prompt.agent_prompt import ENV_SETUP_SYSTEM_PROMPT
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


class EnvSetupAgent(TraeAgent):
    """Trae agent variant for README-driven reproduction tasks."""

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
            "- setup.sh\n"
            "- download_assets.sh\n"
            "- run_reproduction.sh\n"
            "- .trae_env/logs/\n"
            "- .trae_env/original_metrics.json\n"
            "- .trae_env/reproduced_metrics.json\n"
            "- .trae_env/reproduction_verification.json\n"
            "- .trae_env/repair_history.md when repairs are needed\n"
            "- results_comparison.md\n"
            "- failure_analysis.md if blocked\n"
            "- final_report.md\n\n"
            "[Completion rule]:\n"
            "Call task_done only after bash run_reproduction.sh succeeds and results_comparison.md compares "
            "the reproduced target result with the original README result/value when present. Use Linux/WSL bash commands "
            "and write generated scripts with repository-relative paths. Read only README.md for planning, "
            "first plan environment setup commands, then identify the README command that completes the target, "
            "then derive dataset/model/checkpoint downloads required by that command. When creating environments, specify "
            "the README Python version, or python=3.12 if README gives no version. Always create a dedicated conda "
            "environment for the target repository and run README setup/reproduction commands inside it; do not reuse the "
            "currently active environment. Use default pip index settings and do not add pip mirror options. If README "
            "omits versions, prefer PyTorch 2.6 with CUDA 12.4 and transformers 4.55.x, then adjust versions based on "
            "concrete errors. For Hugging Face datasets/models, use HF_ENDPOINT=https://hf-mirror.com. Install flash-attn "
            "with --no-build-isolation. "
            "Do not expand to unrelated README results. Do not use Docker.\n"
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
        comparison_file = Path(self.project_path) / "results_comparison.md"
        if not verification_file.exists():
            return False
        if not reproduced_metrics_file.exists():
            return False
        if not comparison_file.exists():
            return False

        try:
            verification = json.loads(verification_file.read_text(encoding="utf-8"))
            reproduced_metrics = json.loads(
                reproduced_metrics_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

        if int(verification.get("returncode", -1)) != 0:
            return False
        if not self._has_reproduced_result(reproduced_metrics):
            return False
        if self._comparison_reports_failure(comparison_file):
            return False

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
        return True

    @override
    def task_incomplete_message(self) -> str:
        return (
            "ERROR! The requested reproduction task is not complete. `task_done` is rejected. "
            "A valid completion requires successful `bash run_reproduction.sh`, "
            ".trae_env/reproduction_verification.json with returncode 0, "
            ".trae_env/reproduced_metrics.json containing real non-empty reproduced metrics/results, "
            "and results_comparison.md without blocked/failed/not-executed language. "
            "If the task is blocked, keep the run failed and write failure_analysis.md with exact evidence; "
            "do not call task_done without real reproduced scores."
        )
