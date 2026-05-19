# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Base class for OpenAI-compatible clients with shared logic."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, override

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from trae_agent.tools.base import Tool, ToolCall
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import retry_with


class ProviderConfig(ABC):
    """Abstract base class for provider-specific configurations."""

    @abstractmethod
    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create the OpenAI client instance."""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        pass

    @abstractmethod
    def get_extra_headers(self) -> dict[str, str]:
        """Get any extra headers needed for the API call."""
        pass

    @abstractmethod
    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """Base class for OpenAI-compatible clients with shared logic."""

    def __init__(self, model_config: ModelConfig, provider_config: ProviderConfig):
        super().__init__(model_config)
        self.provider_config = provider_config
        self.client = provider_config.create_client(self.api_key, self.base_url, self.api_version)
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_response(
        self,
        model_config: ModelConfig,
        tool_schemas: list[ChatCompletionToolParam] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Create a response using the provider's API. This method will be decorated with retry logic."""
        """Select the correct token parameter based on model configuration.
        If max_completion_tokens is set, use it. Otherwise, use max_tokens."""
        token_params = {}
        if model_config.should_use_max_completion_tokens():
            token_params["max_completion_tokens"] = model_config.get_max_tokens_param()
        else:
            token_params["max_tokens"] = model_config.get_max_tokens_param()

        request_params = {
            "model": model_config.model,
            "messages": self.message_history,
            "tools": tool_schemas if tool_schemas else openai.NOT_GIVEN,
            "temperature": model_config.temperature
            if "o3" not in model_config.model
            and "o4-mini" not in model_config.model
            and "gpt-5" not in model_config.model
            else openai.NOT_GIVEN,
            "top_p": model_config.top_p,
            "extra_headers": extra_headers if extra_headers else None,
            "n": 1,
            "timeout": _llm_request_timeout(),
            **token_params,
        }

        if self._should_disable_thinking(model_config):
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

        return self.client.chat.completions.create(**request_params)

    def _should_disable_thinking(self, model_config: ModelConfig) -> bool:
        provider_name = self.provider_config.get_provider_name().lower()
        base_url = (model_config.model_provider.base_url or "").lower()
        model_name = model_config.model.lower()
        return (
            "deepseek" in provider_name
            or "deepseek" in base_url
            or model_name.startswith("deepseek-v4")
        )

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages with optional tool support."""
        parsed_messages = self.parse_messages(messages)
        if reuse_history:
            self.message_history = self.message_history + parsed_messages
        else:
            self.message_history = parsed_messages
        self.message_history = _compact_chat_history(self.message_history)

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                )
                for tool in tools
            ]

        # Get provider-specific extra headers
        extra_headers = self.provider_config.get_extra_headers()

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_response,
            provider_name=self.provider_config.get_service_name(),
            max_retries=model_config.max_retries,
        )
        response = self._normalize_chat_completion_response(
            retry_decorator(model_config, tool_schemas, extra_headers)
        )

        choice = response.choices[0]

        parse_error_content = ""
        tool_calls: list[ToolCall] | None = None
        if choice.message.tool_calls:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                try:
                    arguments = (
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    )
                except json.JSONDecodeError as e:
                    parse_error_content = (
                        "RECOVERABLE_TOOL_CALL_PARSE_ERROR: Tool call arguments were not valid "
                        f"JSON, likely because the response was truncated. finish_reason={choice.finish_reason}. "
                        f"error={e}. Please retry the same action with shorter arguments."
                    )
                    tool_calls = None
                    break
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=arguments,
                    )
                )

        llm_response = LLMResponse(
            content=parse_error_content or choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=(
                LLMUsage(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )
                if response.usage
                else None
            ),
        )
        if not llm_response.content and not llm_response.tool_calls:
            llm_response.content = (
                "RECOVERABLE_EMPTY_LLM_RESPONSE: The model provider returned an empty assistant "
                "response without tool calls. Retry the previous step with a concise tool call or "
                "short status response."
            )

        # Update message history
        if parse_error_content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    content=parse_error_content, role="assistant"
                )
            )
        elif llm_response.tool_calls:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=llm_response.content,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.call_id,
                            function=Function(
                                name=tool_call.name,
                                arguments=json.dumps(
                                    _sanitize_tool_arguments(tool_call.arguments)
                                ),
                            ),
                            type="function",
                        )
                        for tool_call in llm_response.tool_calls
                    ],
                )
            )
        elif llm_response.content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(content=llm_response.content, role="assistant")
            )

        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider=self.provider_config.get_provider_name(),
                model=model_config.model,
                tools=tools,
            )

        return llm_response

    def _normalize_chat_completion_response(self, response: Any) -> ChatCompletion:
        if isinstance(response, ChatCompletion):
            return response

        raw_response = response
        if isinstance(response, str):
            try:
                raw_response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise TypeError(
                    "OpenAI-compatible API returned a plain string instead of a chat "
                    "completion object. Check that --model-base-url points to the "
                    "OpenAI-compatible API root, usually ending with /v1, and that the "
                    "server supports /v1/chat/completions."
                ) from exc

        if isinstance(raw_response, dict):
            try:
                return ChatCompletion.model_validate(raw_response)
            except Exception as exc:
                raise TypeError(
                    "OpenAI-compatible API returned JSON, but it is not a valid chat "
                    "completion response with choices[]. Check the base URL, model name, "
                    "and provider compatibility."
                ) from exc

        if not hasattr(response, "choices"):
            raise TypeError(
                "OpenAI-compatible API returned an unsupported response type "
                f"{type(response).__name__}. Expected a chat completion response with choices[]. "
                "Check that --model-base-url usually ends with /v1."
            )

        return response

    def parse_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        """Parse LLM messages to OpenAI format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            match msg:
                case msg if msg.tool_call is not None:
                    _msg_tool_call_handler(openai_messages, msg)
                case msg if msg.tool_result is not None:
                    _msg_tool_result_handler(openai_messages, msg)
                case _:
                    _msg_role_handler(openai_messages, msg)

        return openai_messages


def _msg_tool_call_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_call:
        messages.append(
            ChatCompletionFunctionMessageParam(
                content=json.dumps(
                    {
                        "name": msg.tool_call.name,
                        "arguments": msg.tool_call.arguments,
                    }
                ),
                role="function",
                name=msg.tool_call.name,
            )
        )


def _msg_tool_result_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.tool_result:
        result: str = ""
        if msg.tool_result.result:
            result = result + msg.tool_result.result + "\n"
        if msg.tool_result.error:
            result += "Tool call failed with error:\n"
            result += msg.tool_result.error
        result = result.strip()
        result = _truncate_history_text(result, _tool_result_history_max_chars())
        messages.append(
            ChatCompletionToolMessageParam(
                content=result,
                role="tool",
                tool_call_id=msg.tool_result.call_id,
            )
        )


def _msg_role_handler(messages: list[ChatCompletionMessageParam], msg: LLMMessage) -> None:
    if msg.role:
        match msg.role:
            case "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                messages.append(
                    ChatCompletionSystemMessageParam(content=msg.content, role="system")
                )
            case "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                messages.append(ChatCompletionUserMessageParam(content=msg.content, role="user"))
            case "assistant":
                if not msg.content:
                    raise ValueError("Assistant message content is required")
                messages.append(
                    ChatCompletionAssistantMessageParam(content=msg.content, role="assistant")
                )
            case _:
                raise ValueError(f"Invalid message role: {msg.role}")


def _history_max_chars() -> int:
    return _positive_int_from_env("TRAE_LLM_HISTORY_MAX_CHARS", 120000)


def _llm_request_timeout() -> float:
    raw_timeout = os.environ.get("TRAE_LLM_REQUEST_TIMEOUT", "180").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return 180.0
    return timeout if timeout > 0 else 180.0


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


def _history_char_count(messages: list[ChatCompletionMessageParam]) -> int:
    return sum(len(str(message)) for message in messages)


def _compact_chat_history(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    if len(messages) <= 4:
        return messages

    prefix = messages[:2]
    summary = ChatCompletionUserMessageParam(
        role="user",
        content=(
            "Older interaction history was compacted to keep the LLM request within a usable size. "
            "Repository artifacts, scripts, logs, and trajectory files remain on disk. Continue from the "
            "latest visible tool result and inspect files/logs directly when needed."
        ),
    )
    if _history_char_count(messages) <= _history_max_chars():
        return messages

    keep_count = min(_history_keep_messages(), max(len(messages) - 2, 1))
    while keep_count > 0:
        suffix = [_sanitize_history_message(message) for message in messages[2:][-keep_count:]]
        while suffix and dict(suffix[0]).get("role") == "tool":
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


def _sanitize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    max_chars = _tool_argument_history_max_chars()
    for key, value in arguments.items():
        if isinstance(value, str):
            sanitized[key] = _truncate_history_text(value, max_chars)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_history_message(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    raw = dict(message)
    role = raw.get("role")

    if isinstance(raw.get("content"), str):
        max_chars = _tool_result_history_max_chars() if role in {"tool", "function"} else 3000
        raw["content"] = _truncate_history_text(raw["content"], max_chars)

    tool_calls = raw.get("tool_calls")
    if isinstance(tool_calls, list):
        sanitized_tool_calls = []
        for tool_call in tool_calls:
            tool_call_raw = dict(tool_call)
            function = tool_call_raw.get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function = dict(function)
                    function["arguments"] = _sanitize_tool_arguments_json(arguments)
                    tool_call_raw["function"] = function
            sanitized_tool_calls.append(tool_call_raw)
        raw["tool_calls"] = sanitized_tool_calls

    return raw  # type: ignore[return-value]


def _sanitize_tool_arguments_json(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return _truncate_history_text(arguments, _tool_argument_history_max_chars())
    if not isinstance(parsed, dict):
        return arguments
    return json.dumps(_sanitize_tool_arguments(parsed))


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
