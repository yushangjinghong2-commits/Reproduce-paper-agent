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
from datetime import datetime
from pathlib import Path
from typing import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from trae_agent.tools.run import decode_bytes


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
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

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the bash shell."""
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

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
            async with asyncio.timeout(self._timeout):
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
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

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
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
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
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
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

        try:
            result = await self._session.run(command)
            return self._record_and_truncate_result(command, result)
        except Exception as e:
            return ToolExecResult(error=f"Error running bash command: {e}", error_code=-1)

    @override
    async def close(self):
        """Properly close self._process."""
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
        return None

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
