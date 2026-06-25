"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result


class TestDeepSeekReasoningContentForwarding:
    """reasoning_content forwarding for multi-turn thinking-mode conversations.

    DeepSeek thinking mode requires `reasoning_content` on every assistant
    message sent back to the API; without it the API returns:
      "The content[].thinking in the thinking mode must be passed back to the API."

    These tests assert on the request body that the DeepSeek provider would
    actually send (the dict returned by transform_request), not on internal
    helper call shapes.
    """

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    @staticmethod
    def _messages_with_tool_calling_assistant() -> list:
        """A two-turn history whose assistant turn ended in a tool call.

        Tool-calling turns are the realistic failure case: front ends rarely
        round-trip `reasoning_content` on them, so DeepSeek 400s on turn two.
        A fresh list is returned each call so tests never share mutable state.
        """
        return [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
        ]

    def test_unregistered_v4_forwards_reasoning_content_when_thinking_enabled(self):
        """Unregistered V4 thinking models must forward reasoning_content (core regression).

        Fails on the current code: the fill routine is suppressed because the
        model is not in the capability registry (supports_reasoning == False).
        """
        messages = self._messages_with_tool_calling_assistant()
        body = self.config.transform_request(
            model="deepseek-v4-pro",
            messages=messages,
            optional_params={"thinking": {"type": "enabled"}},
            litellm_params={},
            headers={},
        )
        assistant_reasoning = [
            m.get("reasoning_content")
            for m in body["messages"]
            if m.get("role") == "assistant"
        ]
        assert assistant_reasoning, "expected at least one assistant message"
        assert all(
            rc for rc in assistant_reasoning
        ), f"every assistant message must carry non-empty reasoning_content, got {assistant_reasoning}"

    def test_unregistered_v4_unchanged_when_thinking_not_enabled(self):
        """Non-thinking requests must not be altered with spurious reasoning_content."""
        messages = self._messages_with_tool_calling_assistant()
        body = self.config.transform_request(
            model="deepseek-v4-pro",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert all(
            not m.get("reasoning_content") for m in body["messages"]
        ), "reasoning_content must not be injected when thinking is not enabled"

    def test_real_reasoning_content_preferred_over_placeholder(self):
        """Recovered prior-turn reasoning_content is used in preference to the placeholder."""
        messages = self._messages_with_tool_calling_assistant()
        messages[1]["provider_specific_fields"] = {
            "reasoning_content": "the real chain of thought from the prior turn"
        }
        body = self.config.transform_request(
            model="deepseek-v4-pro",
            messages=messages,
            optional_params={"thinking": {"type": "enabled"}},
            litellm_params={},
            headers={},
        )
        assistant = next(m for m in body["messages"] if m.get("role") == "assistant")
        assert (
            assistant.get("reasoning_content")
            == "the real chain of thought from the prior turn"
        )

    def test_deepseek_reasoner_still_forwards_reasoning_content(self):
        """Legacy deepseek-reasoner keeps forwarding reasoning_content (no regression)."""
        messages = self._messages_with_tool_calling_assistant()
        body = self.config.transform_request(
            model="deepseek-reasoner",
            messages=messages,
            optional_params={"thinking": {"type": "enabled"}},
            litellm_params={},
            headers={},
        )
        assistant_reasoning = [
            m.get("reasoning_content")
            for m in body["messages"]
            if m.get("role") == "assistant"
        ]
        assert all(
            rc for rc in assistant_reasoning
        ), f"reasoner assistant messages must still carry reasoning_content, got {assistant_reasoning}"

    async def test_async_transform_forwards_reasoning_content_for_v4(self):
        """Streaming (async) path behaves consistently with sync for V4 thinking."""
        messages = self._messages_with_tool_calling_assistant()
        body = await self.config.async_transform_request(
            model="deepseek-v4-pro",
            messages=messages,
            optional_params={"thinking": {"type": "enabled"}},
            litellm_params={},
            headers={},
        )
        assistant_reasoning = [
            m.get("reasoning_content")
            for m in body["messages"]
            if m.get("role") == "assistant"
        ]
        assert assistant_reasoning, "expected at least one assistant message"
        assert all(
            rc for rc in assistant_reasoning
        ), f"async path must forward reasoning_content, got {assistant_reasoning}"
