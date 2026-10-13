import json
import os
from typing import Any

import ollama


def _extract_content(response: Any) -> str:
    """Extract message string content from an Ollama ChatResponse or dict mock."""
    if hasattr(response, "message"):
        msg = response.message
        if hasattr(msg, "content"):
            return str(msg.content)
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
    if isinstance(response, dict):
        msg = response.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        if hasattr(msg, "content"):
            return str(msg.content)
        if "content" in response:
            return str(response["content"])
    if isinstance(response, str):
        return response
    raise ValueError(f"Could not extract message content from model response: {response!r}")


def _parse_json_dict(content: str) -> dict:
    """Parse JSON string and verify it produces a dictionary."""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object (dict), got {type(data).__name__}")
    return data


def generate_json(prompt: str, schema: dict, image_bytes: bytes | None = None) -> dict:
    """Generate structured JSON adhering to the given schema using the configured LLM provider.

    Provider is read from SHOTGUARD_PROVIDER (default: 'ollama').
    Model is read from SHOTGUARD_MODEL (default: 'gemma4:e4b').
    """
    provider = os.getenv("SHOTGUARD_PROVIDER", "ollama")
    model = os.getenv("SHOTGUARD_MODEL", "gemma4:e4b")

    if provider == "gemini":
        if os.getenv("SHOTGUARD_ALLOW_HOSTED") != "1":
            raise RuntimeError("Hosted provider 'gemini' requires SHOTGUARD_ALLOW_HOSTED=1")
        # TODO: Implement Gemini hosted provider call
        raise NotImplementedError("Gemini provider is not yet implemented")

    if provider != "ollama":
        raise ValueError(f"Unsupported provider: {provider}")

    def call_chat(prompt_text: str) -> dict:
        user_message: dict[str, Any] = {"role": "user", "content": prompt_text}
        if image_bytes is not None:
            user_message["images"] = [image_bytes]

        response = ollama.chat(
            model=model,
            messages=[user_message],
            format=schema,
            options={"temperature": 0},
        )
        content = _extract_content(response)
        return _parse_json_dict(content)

    try:
        return call_chat(prompt)
    except Exception:
        retry_prompt = f"{prompt}\nReply with valid JSON only."
        try:
            return call_chat(retry_prompt)
        except Exception as retry_err:
            raise ValueError(f"Failed to parse valid JSON from model response: {retry_err}") from retry_err
