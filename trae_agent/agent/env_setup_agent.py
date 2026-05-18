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
            "then derive dataset/model/checkpoint downloads required by that command. Do not expand to unrelated README results. Do not use Docker.\n"
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
        comparison_file = Path(self.project_path) / "results_comparison.md"
        if not verification_file.exists():
            return False
        if not comparison_file.exists():
            return False

        try:
            verification = json.loads(verification_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        return int(verification.get("returncode", -1)) == 0

    @override
    def task_incomplete_message(self) -> str:
        return (
            "ERROR! The requested reproduction task has not been recorded as successful. "
            "Run `bash run_reproduction.sh` from the project root, extract the reproduced target result, "
            "write results_comparison.md comparing it with the original README result/value when present, "
            "and only then call task_done. If the task is blocked, write failure_analysis.md with exact evidence instead of task_done."
        )
