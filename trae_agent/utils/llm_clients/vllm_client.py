# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI-compatible chat completions client.

This client is used for local vLLM and third-party APIs that expose
`/v1/chat/completions`.
"""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class VLLMProvider(ProviderConfig):
    """Provider configuration for OpenAI-compatible chat completion servers."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        return openai.OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    def get_service_name(self) -> str:
        return "OpenAI-compatible"

    def get_provider_name(self) -> str:
        return "vllm"

    def get_extra_headers(self) -> dict[str, str]:
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        return True


class VLLMClient(OpenAICompatibleClient):
    """vLLM client using /v1/chat/completions instead of /v1/responses."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, VLLMProvider())
