# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 13 June 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

import asyncio
import json
import locale
import os
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from trae_agent.tools.run import decode_bytes


def _default_bash_timeout() -> float | None:
    raw_timeout = os.environ.get("TRAE_BASH_TIMEOUT", "").strip()
    if not raw_timeout:
        return None
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return None
    return timeout if timeout > 0 else None


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float | None = _default_bash_timeout()
    _sentinel: str = ",,,,bash-command-exit-__ERROR_CODE__-banner,,,,"  # `__ERROR_CODE__` will be replaced by `$?` or `!errorlevel!` later

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._started:
            return

        # Windows compatibility: os.setsid not available

        if os.name != "nt":  # Unix-like systems
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /v:on",  # enable delayed expansion to allow `echo !errorlevel!`
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

    async def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process is None:
            return
        if self._process.returncode is not None:
            return
        try:
            self._process.terminate()

            # Wait until the process has truly terminated.
            stdout, stderr = await asyncio.wait_for(self._process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
            try:
                # Set a shorter timeout for the cleanup process
                stdout, stderr = await asyncio.wait_for(self._process.communicate(), timeout=2.0)
            except asyncio.TimeoutError:
                # If it still timeout, return None.
                return None
        except Exception:
            return None

    async def run(self, command: str, timeout: float | None = None) -> ToolExecResult:
        """Execute a command in the bash shell."""
        timeout_value = timeout or self._timeout
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            if timeout_value is None:
                raise ToolError("bash was previously interrupted and must be restarted")
            raise ToolError(f"timed out: bash has not returned in {timeout_value} seconds")

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        error_code = 0

        sentinel_before, pivot, sentinel_after = self._sentinel.partition("__ERROR_CODE__")
        assert pivot == "__ERROR_CODE__"

        errcode_retriever = "!errorlevel!" if os.name == "nt" else "$?"
        command_sep = "&" if os.name == "nt" else ";"

        if os.name != "nt":
            command = "set -o pipefail\n" + command

        # send command to the process
        command_block = (
            "(\n"
            + command
            + f"\n){command_sep} echo {self._sentinel.replace('__ERROR_CODE__', errcode_retriever)}\n"
        )
        self._process.stdin.write(self._encode_command(command_block))
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(timeout_value):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output: str = decode_bytes(self._process.stdout._buffer)  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
                    if sentinel_before in output:
                        # strip the sentinel from output
                        output, pivot, exit_banner = output.rpartition(sentinel_before)
                        assert pivot

                        # get error code inside banner
                        error_code_str, pivot, _ = exit_banner.partition(sentinel_after)
                        if not pivot or not error_code_str.isdecimal():
                            continue

                        error_code = int(error_code_str)
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(f"timed out: bash has not returned in {timeout_value} seconds") from None

        if output.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            output = output[:-1]  # pyright: ignore[reportUnknownVariableType]

        error: str = decode_bytes(self._process.stderr._buffer)  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        if error.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            error = error[:-1]  # pyright: ignore[reportUnknownVariableType]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        return ToolExecResult(output=output, error=error, error_code=error_code)  # pyright: ignore[reportUnknownArgumentType]

    def _encode_command(self, command: str) -> bytes:
        if os.name == "nt":
            return command.encode(locale.getpreferredencoding(False), errors="replace")
        return command.encode()


class BashTool(Tool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)
        self._session: _BashSession | None = None
        self._command_counter = 0
        self._job_counter = 0
        self._background_jobs: dict[str, dict[str, object]] = {}

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "bash"

    @override
    def get_description(self) -> str:
        return """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Long setup/download/evaluation commands can take a long time. Use run_in_background=true for those commands, then poll with job_id to show progress from the log tail until the job finishes.
* To stop a background job after inspecting progress, call this tool with the same job_id and kill=true.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to run.",
                required=True,
            ),
            ToolParameter(
                name="restart",
                type="boolean",
                description="Set to true to restart the bash session.",
                required=restart_required,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Maximum seconds to wait for a foreground command. Omit for no fixed timeout; TRAE_BASH_TIMEOUT can set a default.",
                required=False,
            ),
            ToolParameter(
                name="run_in_background",
                type="boolean",
                description="Run a long command as a tracked background job and return immediately with a job_id.",
                required=False,
            ),
            ToolParameter(
                name="job_id",
                type="string",
                description="Poll a previously started background job. When this is set, command is ignored.",
                required=False,
            ),
            ToolParameter(
                name="kill",
                type="boolean",
                description="When job_id is set, terminate that background job instead of polling it.",
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        job_id = arguments.get("job_id")
        if job_id:
            if bool(arguments.get("kill")):
                return await self._kill_background_job(str(job_id))
            return await self._poll_background_job(str(job_id))

        if arguments.get("restart"):
            if self._session:
                await self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolExecResult(output="tool has been restarted.")

        if self._session is None:
            try:
                self._session = _BashSession()
                await self._session.start()
            except Exception as e:
                return ToolExecResult(error=f"Error starting bash session: {e}", error_code=-1)

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )

        safety_error = self._validate_command(command)
        if safety_error:
            return ToolExecResult(error=safety_error, error_code=-1)

        if bool(arguments.get("run_in_background")) or self._should_auto_background(command):
            try:
                return await self._start_background_job(command)
            except Exception as e:
                return ToolExecResult(
                    error=f"Error starting background bash command: {e}", error_code=-1
                )

        timeout = self._parse_timeout(arguments.get("timeout"))
        try:
            result = await self._session.run(command, timeout=timeout)
            return self._record_and_truncate_result(command, result)
        except Exception as e:
            return ToolExecResult(error=f"Error running bash command: {e}", error_code=-1)

    @override
    async def close(self):
        """Properly close self._process."""
        for job in self._background_jobs.values():
            process = job.get("process")
            if isinstance(process, asyncio.subprocess.Process) and process.returncode is None:
                if os.name != "nt" and process.pid is not None:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    if os.name != "nt" and process.pid is not None:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
        if self._session:
            ret = await self._session.stop()
            self._session = None
            return ret

    def _validate_command(self, command: str) -> str | None:
        normalized = command.lower()
        forbidden_patterns = [
            (r"(^|[\s;&|()])docker([\s;&|()]|$)", "Docker commands are forbidden for this environment setup task."),
            (r"(^|[\s;&|()])docker-compose([\s;&|()]|$)", "Docker Compose commands are forbidden for this environment setup task."),
            (r"docker\s+compose", "Docker Compose commands are forbidden for this environment setup task."),
            (r"rm\s+-[^\n;&|]*r[^\n;&|]*f\s+/", "Refusing dangerous recursive deletion from filesystem root."),
            (r"sudo\s+rm\s+-[^\n;&|]*r[^\n;&|]*f", "Refusing sudo recursive deletion."),
            (r"git\s+clean\s+-[^\n;&|]*f[^\n;&|]*d", "Refusing destructive git clean command."),
        ]
        for pattern, message in forbidden_patterns:
            if re.search(pattern, normalized):
                return message
        conda_error = self._validate_conda_create_python_version(command)
        if conda_error:
            return conda_error
        pip_index_error = self._validate_pip_default_index(command)
        if pip_index_error:
            return pip_index_error
        torch_version_error = self._validate_torch_version_constraint(command)
        if torch_version_error:
            return torch_version_error
        environment_error = self._validate_reproduction_environment_usage(command)
        if environment_error:
            return environment_error
        return None

    def _validate_conda_create_python_version(self, command: str) -> str | None:
        if self._has_conda_create_without_python(command):
            return (
                "Conda environment creation must specify a Python version. "
                "Use the README Python version, or `python=3.12` if README does not specify one."
            )
        if self._is_setup_command(command):
            setup_path = Path.cwd() / "setup.sh"
            try:
                setup_text = setup_path.read_text(encoding="utf-8")
            except OSError:
                return None
            if self._has_conda_create_without_python(setup_text):
                return (
                    "setup.sh contains `conda create` without an explicit Python version. "
                    "Patch setup.sh to use the README Python version, or `python=3.12` if README does not specify one."
                )
        return None

    def _has_conda_create_without_python(self, text: str) -> bool:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.search(r"\bconda\s+create\b", stripped) and not re.search(
                r"\bpython\s*=", stripped
            ):
                return True
        return False

    def _is_setup_command(self, command: str) -> bool:
        segment_prefix = r"(?:^|[;&]\s*|\|\|\s*|&&\s*)"
        optional_runner = r"(?:(?:bash|sh)\s+)?"
        script = r"(?:\./)?setup\.sh"
        terminator = r"(?:\s|$)"
        pattern = segment_prefix + optional_runner + script + terminator
        return re.search(pattern, command) is not None

    def _validate_pip_default_index(self, command: str) -> str | None:
        if re.search(r"\bpip(?:\d+(?:\.\d+)?)?\s+install\b", command) and re.search(
            r"(?<!\S)(?:-i|--index-url|--extra-index-url)\s+", command
        ):
            return (
                "pip install must use the default package index for this task. "
                "Remove -i/--index-url/--extra-index-url unless README explicitly requires it. "
                "For torch installs, use the default pip index, e.g. `pip install \"torch<2.6\" torchvision torchaudio`."
            )
        for script_name in ("setup.sh", "download_assets.sh", "run_reproduction.sh"):
            if not self._command_runs_script(command, script_name):
                continue
            script_path = Path.cwd() / script_name
            try:
                script_text = script_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if re.search(r"\bpip(?:\d+(?:\.\d+)?)?\s+install\b", script_text) and re.search(
                r"(?<!\S)(?:-i|--index-url|--extra-index-url)\s+", script_text
            ):
                return (
                    f"{script_name} contains pip mirror/index options. "
                    "Use the default pip package index; remove -i/--index-url/--extra-index-url unless README explicitly requires it. "
                    "For torch installs, use the default pip index, e.g. `pip install \"torch<2.6\" torchvision torchaudio`."
                )
        return None

    def _validate_torch_version_constraint(self, command: str) -> str | None:
        offending_line = self._find_torch_install_without_lt26(command)
        if offending_line:
            return (
                "torch installs must enforce torch<2.6 and use the default pip index. "
                "Use README/requirements torch constraints when present, but add/keep an upper bound below 2.6. "
                f"First offending command: {offending_line}"
            )
        for script_name in ("setup.sh", "download_assets.sh", "run_reproduction.sh"):
            if not self._command_runs_script(command, script_name):
                continue
            script_path = Path.cwd() / script_name
            try:
                script_text = script_path.read_text(encoding="utf-8")
            except OSError:
                continue
            offending_line = self._find_torch_install_without_lt26(script_text)
            if offending_line:
                return (
                    f"{script_name} contains a torch install without enforcing torch<2.6. "
                    "Use README/requirements torch constraints when present, but add/keep an upper bound below 2.6. "
                    f"First offending line: {offending_line}"
                )
        return None

    def _find_torch_install_without_lt26(self, text: str) -> str | None:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not re.search(r"\bpip(?:\d+(?:\.\d+)?)?\s+install\b", line):
                continue
            if not re.search(r"(?:^|\s)[\"']?torch(?:\[.*?\])?(?:[<=>!~]=?[^\"'\s]+)?[\"']?", line):
                continue
            if re.search(r"[\"']?torch(?:\[.*?\])?\s*<\s*2\.6[\"']?", line):
                continue
            if re.search(r"[\"']?torch(?:\[.*?\])?\s*==\s*(?:[01](?:\.\d+)*|2\.[0-5](?:\.\d+)*)", line):
                continue
            return raw_line.strip()
        return None

    def _validate_reproduction_environment_usage(self, command: str) -> str | None:
        direct_unscoped = self._find_unscoped_environment_commands(command)
        if direct_unscoped and not self._is_setup_command(command):
            return (
                "Commands that use python/pip/torchrun/accelerate must run inside the dedicated conda environment. "
                "Use `conda run -n <env> ...` instead of executing them directly. "
                f"First offending command: {direct_unscoped}"
            )
        if self._is_setup_command(command):
            setup_path = Path.cwd() / "setup.sh"
            try:
                setup_text = setup_path.read_text(encoding="utf-8")
            except OSError:
                return None
            if not re.search(r"\bconda\s+create\b", setup_text):
                return (
                    "setup.sh must create a dedicated conda environment for the target repository. "
                    "Patch setup.sh before running it."
                )
            if re.search(r"\bpip(?:\d+(?:\.\d+)?)?\s+install\b", setup_text) and not (
                re.search(r"\bconda\s+run\s+-n\b", setup_text)
                or re.search(r"\bconda\s+activate\b", setup_text)
            ):
                return (
                    "setup.sh installs packages without clearly running inside the dedicated conda environment. "
                    "Use `conda run -n <env> pip install ...` or activate the environment inside the script."
                )
            unscoped = self._find_unscoped_environment_commands(setup_text)
            if unscoped:
                return (
                    "setup.sh contains commands that are not scoped to the dedicated conda environment. "
                    "Prefer `conda run -n <env> ...` for every pip/python/torchrun/accelerate command. "
                    f"First offending line: {unscoped}"
                )
        if self._is_download_assets_command(command) or self._is_reproduction_command(command):
            script_name = "download_assets.sh" if self._is_download_assets_command(command) else "run_reproduction.sh"
            script_path = Path.cwd() / script_name
            try:
                script_text = script_path.read_text(encoding="utf-8")
            except OSError:
                return None
            if re.search(r"\bpython(?:\d+(?:\.\d+)?)?\b|\bpip(?:\d+(?:\.\d+)?)?\b", script_text) and not (
                re.search(r"\bconda\s+run\s+-n\b", script_text)
                or re.search(r"\bconda\s+activate\b", script_text)
            ):
                return (
                    f"{script_name} uses python/pip without clearly running inside the dedicated conda environment. "
                    "Use `conda run -n <env> ...` or activate the environment inside the script."
                )
            unscoped = self._find_unscoped_environment_commands(script_text)
            if unscoped:
                return (
                    f"{script_name} contains commands that are not scoped to the dedicated conda environment. "
                    "Prefer `conda run -n <env> ...` for every python/pip/torchrun/accelerate command. "
                    f"First offending line: {unscoped}"
                )
        return None

    def _find_unscoped_environment_commands(self, script_text: str) -> str | None:
        env_commands = (
            r"python(?:\d+(?:\.\d+)?)?",
            r"pip(?:\d+(?:\.\d+)?)?",
            r"torchrun",
            r"accelerate",
            r"pytest",
            r"jupyter",
        )
        command_pattern = r"\b(?:" + "|".join(env_commands) + r")\b"
        allowed_patterns = [
            r"\bconda\s+run\s+-n\b",
            r"\bconda\s+create\b",
            r"\bconda\s+install\b",
            r"\bconda\s+env\s+create\b",
            r"\bmamba\s+run\s+-n\b",
            r"\bmicromamba\s+run\s+-n\b",
        ]
        for raw_line in script_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not re.search(command_pattern, line):
                continue
            if any(re.search(pattern, line) for pattern in allowed_patterns):
                continue
            if re.search(r"\bpython\s*=", line):
                continue
            return raw_line.strip()
        return None

    def _is_download_assets_command(self, command: str) -> bool:
        return self._command_runs_script(command, "download_assets.sh")

    def _command_runs_script(self, command: str, script_name: str) -> bool:
        segment_prefix = r"(?:^|[;&]\s*|\|\|\s*|&&\s*)"
        optional_runner = r"(?:(?:bash|sh)\s+)?"
        script = rf"(?:\./)?{re.escape(script_name)}"
        terminator = r"(?:\s|$)"
        pattern = segment_prefix + optional_runner + script + terminator
        return re.search(pattern, command) is not None

    def _parse_timeout(self, value: object) -> float | None:
        if value is None:
            return None
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return None
        if timeout <= 0:
            return None
        return timeout

    def _should_auto_background(self, command: str) -> bool:
        long_command_patterns = [
            r"(?:^|[;&]\s*|\|\|\s*|&&\s*)(?:(?:bash|sh)\s+)?(?:\./)?setup\.sh(?:\s|$)",
            r"(?:^|[;&]\s*|\|\|\s*|&&\s*)(?:(?:bash|sh)\s+)?(?:\./)?download_assets\.sh(?:\s|$)",
            r"(?:^|[;&]\s*|\|\|\s*|&&\s*)(?:(?:bash|sh)\s+)?(?:\./)?run_reproduction\.sh(?:\s|$)",
            r"\bconda\s+(?:create|install|env\s+create)\b",
            r"\bpip\s+install\b",
            r"\bhuggingface-cli\s+download\b",
            r"\bwget\b",
            r"\bcurl\b.+(?:-O|-o)\b",
        ]
        return any(re.search(pattern, command) for pattern in long_command_patterns)

    async def _start_background_job(self, command: str) -> ToolExecResult:
        self._job_counter += 1
        job_id = f"job_{self._job_counter:04d}"
        env_dir = Path.cwd() / ".trae_env"
        logs_dir = env_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job_id}.log"
        header = (
            f"timestamp: {datetime.now().isoformat()}\n"
            f"job_id: {job_id}\n"
            f"command: {command}\n\n"
            "===== OUTPUT =====\n"
        )
        log_path.write_text(header, encoding="utf-8")

        command_to_run = command
        if os.name != "nt":
            command_to_run = "set -o pipefail\n" + command

        subprocess_kwargs: dict[str, object] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name != "nt":
            subprocess_kwargs["executable"] = "/bin/bash"
            subprocess_kwargs["start_new_session"] = True

        process = await asyncio.create_subprocess_shell(command_to_run, **subprocess_kwargs)
        job: dict[str, object] = {
            "command": command,
            "log_path": log_path,
            "process": process,
            "pid": process.pid,
            "started_at": datetime.now().isoformat(),
            "recorded": False,
        }
        self._background_jobs[job_id] = job
        job["capture_task"] = asyncio.create_task(
            self._capture_background_output(job_id, process, log_path)
        )
        return ToolExecResult(
            output=(
                f"Started background job {job_id}.\n"
                f"PID/process group: {process.pid}\n"
                f"Log: {log_path}\n"
                "Poll progress with the bash tool using this argument: "
                f'{{"job_id": "{job_id}"}}\n'
                "To stop it after inspecting progress, use: "
                f'{{"job_id": "{job_id}", "kill": true}}'
            )
        )

    async def _capture_background_output(
        self, job_id: str, process: asyncio.subprocess.Process, log_path: Path
    ) -> None:
        try:
            assert process.stdout is not None
            with log_path.open("ab") as log_file:
                while True:
                    chunk = await process.stdout.read(4096)
                    if not chunk:
                        break
                    log_file.write(chunk)
                    log_file.flush()
            returncode = await process.wait()
        except Exception as e:
            returncode = -1
            with log_path.open("ab") as log_file:
                log_file.write(f"\n[background capture error] {e}\n".encode())

        job = self._background_jobs.get(job_id)
        if job is not None:
            job["returncode"] = returncode
            job["finished_at"] = datetime.now().isoformat()

    async def _poll_background_job(self, job_id: str) -> ToolExecResult:
        job = self._background_jobs.get(job_id)
        if job is None:
            return ToolExecResult(error=f"Background job '{job_id}' not found.", error_code=-1)

        process = job["process"]
        if not isinstance(process, asyncio.subprocess.Process):
            return ToolExecResult(error=f"Background job '{job_id}' has invalid state.", error_code=-1)

        log_path = job["log_path"]
        if not isinstance(log_path, Path):
            return ToolExecResult(error=f"Background job '{job_id}' has invalid log path.", error_code=-1)

        returncode = process.returncode
        if returncode is None and "returncode" in job:
            recorded_returncode = job["returncode"]
            returncode = int(recorded_returncode) if isinstance(recorded_returncode, int) else None

        tail = self._read_log_tail(log_path)
        command = str(job.get("command", ""))
        if returncode is None:
            return ToolExecResult(
                output=(
                    f"Background job {job_id} is still running.\n"
                    f"PID/process group: {job.get('pid')}\n"
                    f"Log: {log_path}\n\n"
                    f"===== LOG TAIL =====\n{tail}"
                ),
                error_code=0,
            )

        if not bool(job.get("recorded")):
            result = ToolExecResult(output=tail, error_code=returncode)
            self._append_command_record(command, result, log_path)
            self._record_reproduction_verification(command, result, log_path)
            job["recorded"] = True

        return ToolExecResult(
            output=(
                f"Background job {job_id} finished with returncode {returncode}.\n"
                f"Log: {log_path}\n\n"
                f"===== LOG TAIL =====\n{tail}"
            ),
            error_code=returncode,
        )

    def _read_log_tail(self, log_path: Path, max_chars: int = 12000) -> str:
        try:
            data = log_path.read_bytes()
        except OSError as e:
            return f"[unable to read log: {e}]"
        text = decode_bytes(data)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    async def _kill_background_job(self, job_id: str) -> ToolExecResult:
        job = self._background_jobs.get(job_id)
        if job is None:
            return ToolExecResult(error=f"Background job '{job_id}' not found.", error_code=-1)

        process = job.get("process")
        if not isinstance(process, asyncio.subprocess.Process):
            return ToolExecResult(error=f"Background job '{job_id}' has invalid state.", error_code=-1)

        log_path = job.get("log_path")
        if not isinstance(log_path, Path):
            return ToolExecResult(error=f"Background job '{job_id}' has invalid log path.", error_code=-1)

        if process.returncode is None:
            try:
                if os.name != "nt" and process.pid is not None:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    if os.name != "nt" and process.pid is not None:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
            finally:
                job["returncode"] = process.returncode if process.returncode is not None else -9
                job["finished_at"] = datetime.now().isoformat()
                with log_path.open("ab") as log_file:
                    log_file.write(f"\n[killed by user request: {datetime.now().isoformat()}]\n".encode())

        tail = self._read_log_tail(log_path)
        return ToolExecResult(
            output=(
                f"Background job {job_id} was terminated by request.\n"
                f"Log: {log_path}\n\n"
                f"===== LOG TAIL =====\n{tail}"
            ),
            error_code=130,
        )

    def _record_and_truncate_result(
        self, command: str, result: ToolExecResult
    ) -> ToolExecResult:
        self._command_counter += 1
        env_dir = Path.cwd() / ".trae_env"
        logs_dir = env_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / f"bash_step_{self._command_counter:04d}.log"
        log_payload = (
            f"timestamp: {datetime.now().isoformat()}\n"
            f"command: {command}\n"
            f"returncode: {result.error_code}\n\n"
            "===== STDOUT =====\n"
            f"{result.output or ''}\n\n"
            "===== STDERR =====\n"
            f"{result.error or ''}\n"
        )
        log_path.write_text(log_payload, encoding="utf-8")
        self._append_command_record(command, result, log_path)
        self._record_reproduction_verification(command, result, log_path)

        result.output = self._truncate_text(result.output, log_path, "stdout")
        result.error = self._truncate_text(result.error, log_path, "stderr")
        return result

    def _append_command_record(
        self, command: str, result: ToolExecResult, log_path: Path
    ) -> None:
        commands_path = Path.cwd() / ".trae_env" / "commands.json"
        try:
            commands = json.loads(commands_path.read_text(encoding="utf-8"))
            if not isinstance(commands, list):
                commands = []
        except (FileNotFoundError, json.JSONDecodeError):
            commands = []

        commands.append(
            {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "returncode": result.error_code,
                "log": str(log_path),
            }
        )
        commands_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")

    def _record_reproduction_verification(
        self, command: str, result: ToolExecResult, log_path: Path
    ) -> None:
        if result.error_code != 0:
            return
        if not self._is_reproduction_command(command):
            return

        verification_path = Path.cwd() / ".trae_env" / "reproduction_verification.json"
        verification = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "returncode": result.error_code,
            "log": str(log_path),
        }
        verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    def _is_reproduction_command(self, command: str) -> bool:
        segment_prefix = r"(?:^|[;&]\s*|\|\|\s*|&&\s*)"
        optional_runner = r"(?:(?:bash|sh)\s+)?"
        script = r"(?:\./)?run_reproduction\.sh"
        terminator = r"(?:\s|$)"
        pattern = segment_prefix + optional_runner + script + terminator
        return re.search(pattern, command) is not None

    def _truncate_text(self, text: str | None, log_path: Path, stream_name: str) -> str | None:
        if text is None:
            return None

        max_chars = 12000
        if len(text) <= max_chars:
            return text

        head_chars = 6000
        tail_chars = 6000
        return (
            f"[{stream_name} output truncated; full log saved to {log_path}]\n"
            f"{text[:head_chars]}\n"
            "...[truncated]...\n"
            f"{text[-tail_chars:]}"
        )
