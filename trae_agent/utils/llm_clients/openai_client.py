# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import json
import os
from typing import Any
from typing import override

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import retry_with


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.message_history: ResponseInputParam = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_openai_response(
        self,
        api_call_input: ResponseInputParam,
        model_config: ModelConfig,
        tool_schemas: list[ToolParam] | None,
    ) -> Response:
        """Create a response using OpenAI API. This method will be decorated with retry logic."""
        return self.client.responses.create(
            input=api_call_input,
            model=model_config.model,
            tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
            temperature=model_config.temperature
            if "o3" not in model_config.model
            and "o4-mini" not in model_config.model
            and "gpt-5" not in model_config.model
            else openai.NOT_GIVEN,
            top_p=model_config.top_p,
            max_output_tokens=model_config.max_tokens,
            timeout=_llm_request_timeout(),
        )

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support."""
        openai_messages: ResponseInputParam = self.parse_messages(messages)

        if reuse_history:
            self.message_history = self.message_history + openai_messages
        else:
            self.message_history = openai_messages
        self.message_history = _compact_response_history(self.message_history)

        tool_schemas = None
        if tools:
            tool_schemas = [
                FunctionToolParam(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.get_input_schema(),
                    strict=True,
                    type="function",
                )
                for tool in tools
            ]

        api_call_input: ResponseInputParam = self.message_history

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_openai_response,
            provider_name="OpenAI",
            max_retries=model_config.max_retries,
        )
        response = retry_decorator(api_call_input, model_config, tool_schemas)

        content = ""
        parse_error_content = ""
        tool_calls: list[ToolCall] = []
        for output_block in response.output:
            if output_block.type == "function_call":
                try:
                    arguments = (
                        json.loads(output_block.arguments) if output_block.arguments else {}
                    )
                except json.JSONDecodeError as e:
                    parse_error_content = (
                        "RECOVERABLE_TOOL_CALL_PARSE_ERROR: Tool call arguments were not valid "
                        f"JSON, likely because the response was truncated. finish_reason={response.status}. "
                        f"error={e}. Please retry the same action with shorter arguments."
                    )
                    tool_calls = []
                    break
                tool_calls.append(
                    ToolCall(
                        call_id=output_block.call_id,
                        name=output_block.name,
                        arguments=arguments,
                        id=output_block.id,
                    )
                )
                tool_call_param = ResponseFunctionToolCallParam(
                    arguments=_sanitize_tool_arguments_json(output_block.arguments),
                    call_id=output_block.call_id,
                    name=output_block.name,
                    type="function_call",
                )
                if output_block.status:
                    tool_call_param["status"] = output_block.status
                if output_block.id:
                    tool_call_param["id"] = output_block.id
                self.message_history.append(tool_call_param)
            elif output_block.type == "message":
                content = "".join(
                    content_block.text
                    for content_block in output_block.content
                    if content_block.type == "output_text"
                )

        if parse_error_content:
            content = parse_error_content

        if content != "":
            self.message_history.append(
                EasyInputMessageParam(content=content, role="assistant", type="message")
            )

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                cache_read_input_tokens=response.usage.input_tokens_details.cached_tokens or 0,
                reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens or 0,
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.status,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
        )
        if not llm_response.content and not llm_response.tool_calls:
            llm_response.content = (
                "RECOVERABLE_EMPTY_LLM_RESPONSE: The model provider returned an empty assistant "
                "response without tool calls. Retry the previous step with a concise tool call or "
                "short status response."
            )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_config.model,
                tools=tools,
            )

        return llm_response

    def parse_messages(self, messages: list[LLMMessage]) -> ResponseInputParam:
        """Parse the messages to OpenAI format."""
        openai_messages: ResponseInputParam = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                openai_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call(self, tool_call: ToolCall) -> ResponseFunctionToolCallParam:
        """Parse the tool call from the LLM response."""
        return ResponseFunctionToolCallParam(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=json.dumps(_sanitize_tool_arguments(tool_call.arguments)),
            type="function_call",
        )

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> FunctionCallOutput:
        """Parse the tool call result from the LLM response to FunctionCallOutput format."""
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()
        result_content = _truncate_history_text(result_content, _tool_result_history_max_chars())

        return FunctionCallOutput(
            type="function_call_output",  # Explicitly set the type field
            call_id=tool_call_result.call_id,
            output=result_content,
        )


def _history_max_chars() -> int:
    return _positive_int_from_env("TRAE_LLM_HISTORY_MAX_CHARS", 120000)


def _llm_request_timeout() -> float:
    raw_timeout = os.environ.get("TRAE_LLM_REQUEST_TIMEOUT", "120").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else 120.0


def _history_keep_messages() -> int:
    return _positive_int_from_env("TRAE_LLM_HISTORY_KEEP_MESSAGES", 30)


def _tool_result_history_max_chars() -> int:
    return _positive_int_from_env("TRAE_TOOL_RESULT_HISTORY_MAX_CHARS", 4000)


def _tool_argument_history_max_chars() -> int:
    return _positive_int_from_env("TRAE_TOOL_ARGUMENT_HISTORY_MAX_CHARS", 1200)


def _error_context_lines() -> int:
    return _positive_int_from_env("TRAE_ERROR_CONTEXT_LINES", 5)


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _history_char_count(messages: ResponseInputParam) -> int:
    return sum(len(str(message)) for message in messages)


def _compact_response_history(messages: ResponseInputParam) -> ResponseInputParam:
    if len(messages) <= 4:
        return messages

    prefix = messages[:2]
    summary: EasyInputMessageParam = {
        "role": "user",
        "content": (
            "Older interaction history was compacted to keep the LLM request within a usable size. "
            "Repository artifacts, scripts, logs, and trajectory files remain on disk. Continue from the "
            "latest visible tool result and inspect files/logs directly when needed."
        ),
    }
    if _history_char_count(messages) <= _history_max_chars():
        return messages

    keep_count = min(_history_keep_messages(), max(len(messages) - 2, 1))
    while keep_count > 0:
        suffix = [_sanitize_response_history_message(message) for message in messages[2:][-keep_count:]]
        while suffix and dict(suffix[0]).get("type") == "function_call_output":
            suffix = suffix[1:]
        compacted = prefix + [summary] + suffix
        if _history_char_count(compacted) <= _history_max_chars() or keep_count <= 4:
            return compacted
        keep_count = max(keep_count // 2, 4)

    return prefix + [summary]


def _truncate_history_text(text: str, max_chars: int) -> str:
    if _looks_like_error_output(text):
        return _condense_error_context(text)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        f"[history content truncated from {len(text)} chars]\n"
        f"{text[:head]}\n...[truncated for LLM history]...\n{text[-tail:]}"
    )


def _sanitize_tool_arguments_json(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return _truncate_history_text(arguments, _tool_argument_history_max_chars())
    if not isinstance(parsed, dict):
        return arguments
    sanitized: dict[str, Any] = {}
    max_chars = _tool_argument_history_max_chars()
    for key, value in parsed.items():
        if isinstance(value, str):
            sanitized[key] = _truncate_history_text(value, max_chars)
        else:
            sanitized[key] = value
    return json.dumps(sanitized)


def _sanitize_tool_arguments(arguments: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    max_chars = _tool_argument_history_max_chars()
    for key, value in arguments.items():
        if isinstance(value, str):
            sanitized[key] = _truncate_history_text(value, max_chars)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_response_history_message(message: object) -> object:
    raw = dict(message)  # type: ignore[arg-type]
    content = raw.get("content")
    message_type = raw.get("type")
    role = raw.get("role")
    if isinstance(content, str):
        max_chars = (
            _tool_result_history_max_chars()
            if message_type == "function_call_output"
            else 3000
        )
        raw["content"] = _truncate_history_text(content, max_chars)
    output = raw.get("output")
    if isinstance(output, str):
        raw["output"] = _truncate_history_text(output, _tool_result_history_max_chars())
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        raw["arguments"] = _sanitize_tool_arguments_json(arguments)
    if role == "assistant" and isinstance(content, str):
        raw["content"] = _truncate_history_text(content, 3000)
    return raw


def _looks_like_error_output(text: str) -> bool:
    lower = text.lower()
    markers = [
        "tool call failed with error",
        "traceback",
        "error:",
        "exception",
        "failed",
        "cuda",
        "importerror",
        "modulenotfounderror",
        "returncode",
        "dependency_too_new",
        "pytorch_cuda",
    ]
    return any(marker in lower for marker in markers)


def _condense_error_context(text: str) -> str:
    lines = text.splitlines()
    context = _error_context_lines()
    if len(lines) <= context * 2:
        return text

    important_lines = [
        line
        for line in lines
        if ".trae_env/logs" in line or line.startswith("Log:") or line.startswith("command:")
    ]
    head = lines[:context]
    tail = lines[-context:]
    omitted = len(lines) - len(head) - len(tail)
    parts = [
        f"[error output condensed to first {context} lines and last {context} lines; {omitted} middle lines omitted]",
        *head,
    ]
    if important_lines:
        parts.extend(["[important log references]", *important_lines[:5]])
    parts.extend(["...[middle error output omitted]...", *tail])
    return "\n".join(parts)
