# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.bash_tool import BashTool


class TestBashTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = BashTool()

    async def asyncTearDown(self):
        # Cleanup any active session
        if self.tool._session:
            await self.tool._session.stop()

    async def test_tool_initialization(self):
        self.assertEqual(self.tool.get_name(), "bash")
        self.assertIn("Run commands in a bash shell", self.tool.get_description())

        params = self.tool.get_parameters()
        param_names = [p.name for p in params]
        self.assertIn("command", param_names)
        self.assertIn("restart", param_names)

    async def test_command_error_handling(self):
        result = await self.tool.execute(ToolCallArguments({"command": "invalid_command_123"}))

        # Fix assertion: Check if error message contains 'not found' or 'not recognized' (Windows system)
        self.assertTrue(any(s in result.error.lower() for s in ["not found", "not recognized"]))
        self.assertNotEqual(result.error_code, 0)

    async def test_session_restart(self):
        # Ensure session is initialized
        await self.tool.execute(ToolCallArguments({"command": "echo first session"}))

        # Fix: Check if session object exists
        self.assertIsNotNone(self.tool._session)

        # Restart and test new session
        restart_result = await self.tool.execute(ToolCallArguments({"restart": True}))
        self.assertIn("restarted", restart_result.output.lower())

        # Fix: Ensure new session is created
        self.assertIsNotNone(self.tool._session)

        # Verify new session works
        result = await self.tool.execute(ToolCallArguments({"command": "echo new session"}))
        self.assertIn("new session", result.output)

    async def test_successful_command_execution(self):
        result = await self.tool.execute(ToolCallArguments({"command": "echo hello world"}))

        # Fix: Check if return code is 0
        self.assertEqual(result.error_code, 0)
        self.assertIn("hello world", result.output)
        self.assertEqual(result.error, "")

    async def test_missing_command_handling(self):
        result = await self.tool.execute(ToolCallArguments({}))
        self.assertIn("no command provided", result.error.lower())
        self.assertEqual(result.error_code, -1)

    @unittest.skipIf(os.name == "nt", "pipefail behavior is specific to bash on Linux/WSL")
    async def test_pipeline_failure_is_not_hidden_by_tee(self):
        result = await self.tool.execute(
            ToolCallArguments({"command": "false | tee /tmp/trae-agent-pipefail-test.log"})
        )

        self.assertNotEqual(result.error_code, 0)

    @unittest.skipIf(os.name == "nt", "verification command matching is exercised through bash")
    async def test_reproduction_verification_requires_script_execution(self):
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            script = workdir / "run_reproduction.sh"
            script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)

            cleanup_result = await self.tool.execute(
                ToolCallArguments(
                    {"command": f"cd {workdir.as_posix()} && rm -f run_reproduction.sh"}
                )
            )
            self.assertEqual(cleanup_result.error_code, 0)
            self.assertFalse((workdir / ".trae_env" / "reproduction_verification.json").exists())

            script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            script.chmod(0o755)
            run_result = await self.tool.execute(
                ToolCallArguments(
                    {"command": f"cd {workdir.as_posix()} && bash run_reproduction.sh"}
                )
            )
            self.assertEqual(run_result.error_code, 0)
            self.assertTrue((workdir / ".trae_env" / "reproduction_verification.json").exists())


if __name__ == "__main__":
    unittest.main()
