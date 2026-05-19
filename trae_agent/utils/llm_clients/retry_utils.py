# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import random
import os
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """
    Decorator that adds retry logic with randomized backoff.

    Args:
        func: The function to decorate
        provider_name: The name of the model provider being called
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retry_forever = _retry_forever_on_timeout(e)

                if attempt >= max_retries and not retry_forever:
                    # Last attempt, re-raise the exception
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                retry_note = (
                    " Timeout/no-response retry is enabled, so this request will keep retrying."
                    if retry_forever and attempt >= max_retries
                    else ""
                )
                print(
                    f"{provider_name} API call failed: {this_error_message}. "
                    f"Will sleep for {sleep_time} seconds and will retry.{retry_note}\n"
                    f"{traceback.format_exc()}"
                )
                # Randomly sleep for 3-30 seconds
                time.sleep(sleep_time)
                attempt += 1

        # This should never be reached, but just in case
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper


def _retry_forever_on_timeout(error: Exception) -> bool:
    raw_enabled = os.environ.get("TRAE_LLM_RETRY_FOREVER_ON_TIMEOUT", "1").strip().lower()
    if raw_enabled in {"0", "false", "no", "off"}:
        return False

    error_name = type(error).__name__.lower()
    error_text = str(error).lower()
    timeout_markers = [
        "timeout",
        "timed out",
        "readtimeout",
        "connecttimeout",
        "apitimeout",
        "request timed out",
        "server disconnected without sending a response",
        "connection error",
        "remoteprotocolerror",
        "no response",
    ]
    return any(marker in error_name or marker in error_text for marker in timeout_markers)
