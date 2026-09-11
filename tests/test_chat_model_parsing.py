"""Tests for chat model parsing helpers."""

from custom_components.extended_openai_conversation.helpers import (
    get_base_model_name,
    get_model_config,
    get_token_param_for_model,
    parse_chat_model,
)


def test_parse_chat_model_without_query() -> None:
    """Models without query parameters should be returned untouched."""
    model, overrides = parse_chat_model("gpt-5-mini")

    assert model == "gpt-5-mini"
    assert overrides == {}


def test_parse_chat_model_with_user_and_session_key() -> None:
    """Known query parameters should become OpenAI request overrides."""
    model, overrides = parse_chat_model(
        "openclaw:main?session_key=agent:main:openai:ha-spark&user=ha-voice"
    )

    assert model == "openclaw:main"
    assert overrides == {
        "user": "ha-voice",
        "extra_headers": {
            "x-openclaw-session-key": "agent:main:openai:ha-spark",
        },
    }


def test_parse_chat_model_supports_session_aliases() -> None:
    """OpenClaw session aliases should map to the same request header."""
    for param in ("session", "session_key", "openclaw_session_key"):
        model, overrides = parse_chat_model(f"openclaw:main?{param}=sticky-session")

        assert model == "openclaw:main"
        assert overrides == {
            "extra_headers": {
                "x-openclaw-session-key": "sticky-session",
            }
        }


def test_get_base_model_name_strips_query_string() -> None:
    """Model helpers should operate on the base model name."""
    assert (
        get_base_model_name(
            "gpt-5-mini?session_key=agent:main:openai:ha-spark&user=ha-voice"
        )
        == "gpt-5-mini"
    )


def test_model_capabilities_ignore_query_string() -> None:
    """Capability detection should ignore query-string overrides."""
    config = get_model_config("gpt-5-mini?user=ha-voice")

    assert config["supports_max_completion_tokens"] is True
    assert config["supports_temperature"] is False
    assert get_token_param_for_model("gpt-5-mini?session=sticky-session") == (
        "max_completion_tokens"
    )


def test_parse_chat_model_extra_body_json() -> None:
    model = 'deepseek-v4-flash?extra_body={"chat_template_kwargs":{"thinking":false}}'
    base_model, options = parse_chat_model(model)
    assert base_model == "deepseek-v4-flash"
    assert options == {"extra_body": {"chat_template_kwargs": {"thinking": False}}}


def test_parse_chat_model_extra_body_invalid_is_ignored() -> None:
    base_model, options = parse_chat_model("deepseek-v4-flash?extra_body=not-json")
    assert base_model == "deepseek-v4-flash"
    assert options == {}
