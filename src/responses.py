"""OpenAI-compatible chat completion response contract (see Phase 6)."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

MODEL_NAME = "agentic-mesh"


class Usage(BaseModel):
    """Token usage counters. Real counts are wired in Phase 6."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Message(BaseModel):
    """One assistant message inside a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """The response shape a client expects from POST /v1/chat/completions."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = MODEL_NAME
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def build_response(content: str, model: str = MODEL_NAME) -> ChatCompletionResponse:
    """Wrap the assembled final answer into the OpenAI response shape."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[Choice(message=Message(content=content))],
    )
