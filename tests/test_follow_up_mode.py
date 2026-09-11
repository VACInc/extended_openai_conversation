"""Tests for opt-in voice follow-up listening."""

from custom_components.extended_openai_conversation.const import (
    CONF_FOLLOW_UP_MODE,
    FOLLOW_UP_MODE_AUTO,
    FOLLOW_UP_MODE_OFF,
)
from custom_components.extended_openai_conversation.conversation import (
    should_continue_conversation,
)


def test_missing_follow_up_mode_does_not_continue() -> None:
    """Existing subentries without the key must never reopen the microphone."""
    assert should_continue_conversation({}, True) is False
    assert should_continue_conversation({}, False) is False


def test_follow_up_mode_off_never_continues() -> None:
    """Off is the default and ignores chat_log.continue_conversation."""
    assert (
        should_continue_conversation({CONF_FOLLOW_UP_MODE: FOLLOW_UP_MODE_OFF}, True)
        is False
    )


def test_follow_up_mode_auto_uses_chat_log() -> None:
    """Auto preserves today's question-mark follow-up behavior."""
    data = {CONF_FOLLOW_UP_MODE: FOLLOW_UP_MODE_AUTO}

    assert should_continue_conversation(data, True) is True
    assert should_continue_conversation(data, False) is False


def test_unknown_follow_up_mode_does_not_continue() -> None:
    """Unknown values fail closed to off."""
    assert should_continue_conversation({CONF_FOLLOW_UP_MODE: "always"}, True) is False
