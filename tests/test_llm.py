import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm import generate_json


def test_valid_json(monkeypatch):
    """Test generating JSON when ollama.chat returns valid JSON on the first attempt."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "ollama")
    monkeypatch.setenv("SHOTGUARD_MODEL", "gemma4:e4b")

    dummy_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    expected_result = {"name": "Alice"}
    mock_response = {"message": {"content": json.dumps(expected_result)}}

    with patch("core.llm.ollama.chat", return_value=mock_response) as mock_chat:
        result = generate_json("Extract name", dummy_schema)

    assert result == expected_result
    mock_chat.assert_called_once_with(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "Extract name"}],
        format=dummy_schema,
        options={"temperature": 0},
    )


def test_valid_json_with_images(monkeypatch):
    """Test passing image_bytes to ollama.chat."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "ollama")
    dummy_schema = {"type": "object"}
    mock_response = {"message": {"content": json.dumps({"status": "ok"})}}

    image_bytes = b"\x89PNG\r\n\x1a\nfakeimage"

    with patch("core.llm.ollama.chat", return_value=mock_response) as mock_chat:
        result = generate_json("Analyze image", dummy_schema, image_bytes=image_bytes)

    assert result == {"status": "ok"}
    mock_chat.assert_called_once_with(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "Analyze image", "images": [image_bytes]}],
        format=dummy_schema,
        options={"temperature": 0},
    )


def test_bad_then_good_json(monkeypatch):
    """Test retry when first response is invalid JSON, succeeding on the second attempt."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "ollama")
    monkeypatch.setenv("SHOTGUARD_MODEL", "gemma4:e4b")

    dummy_schema = {"type": "object"}
    first_response = {"message": {"content": "This is plain text, not JSON."}}
    second_response = {"message": {"content": json.dumps({"recovered": True})}}

    with patch("core.llm.ollama.chat", side_effect=[first_response, second_response]) as mock_chat:
        result = generate_json("Original prompt", dummy_schema)

    assert result == {"recovered": True}
    assert mock_chat.call_count == 2

    first_call_args = mock_chat.call_args_list[0].kwargs
    assert first_call_args["messages"] == [{"role": "user", "content": "Original prompt"}]

    second_call_args = mock_chat.call_args_list[1].kwargs
    assert "Reply with valid JSON only." in second_call_args["messages"][0]["content"]


def test_bad_then_bad_json_raises_value_error(monkeypatch):
    """Test that consecutive invalid JSON responses raise ValueError."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "ollama")
    dummy_schema = {"type": "object"}

    invalid_response = {"message": {"content": "Still not valid JSON"}}

    with patch("core.llm.ollama.chat", side_effect=[invalid_response, invalid_response]) as mock_chat:
        with pytest.raises(ValueError, match="Failed to parse valid JSON"):
            generate_json("Original prompt", dummy_schema)

    assert mock_chat.call_count == 2


def test_hosted_blocked_gemini(monkeypatch):
    """Test that provider='gemini' raises RuntimeError when SHOTGUARD_ALLOW_HOSTED is not set."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "gemini")
    monkeypatch.delenv("SHOTGUARD_ALLOW_HOSTED", raising=False)

    with pytest.raises(RuntimeError, match="requires SHOTGUARD_ALLOW_HOSTED=1"):
        generate_json("Test prompt", {})


def test_hosted_allowed_gemini_todo(monkeypatch):
    """Test that provider='gemini' with SHOTGUARD_ALLOW_HOSTED=1 hits TODO (NotImplementedError)."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "gemini")
    monkeypatch.setenv("SHOTGUARD_ALLOW_HOSTED", "1")

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        generate_json("Test prompt", {})


def test_custom_model_env(monkeypatch):
    """Test model name override via SHOTGUARD_MODEL environment variable."""
    monkeypatch.setenv("SHOTGUARD_PROVIDER", "ollama")
    monkeypatch.setenv("SHOTGUARD_MODEL", "custom-gemma:latest")

    mock_response = {"message": {"content": json.dumps({"ok": True})}}
    with patch("core.llm.ollama.chat", return_value=mock_response) as mock_chat:
        result = generate_json("Prompt", {})

    assert result == {"ok": True}
    assert mock_chat.call_args.kwargs["model"] == "custom-gemma:latest"
