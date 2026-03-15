"""Tests for OpenAI-compatible client max_completion_tokens / max_tokens fallback."""
import unittest
from unittest.mock import MagicMock, patch


class TestOpenAITokenParam(unittest.TestCase):
    """Verify adaptive token parameter selection in OpenAICompatibleClient."""

    def _make_client(self):
        from llm_client import OpenAICompatibleClient
        client = OpenAICompatibleClient(
            base_url='http://localhost:8000/v1',
            api_key='test-key',
            default_model='test-model'
        )
        # Clear instance cache between tests
        client._token_param_cache.clear()
        client._client = MagicMock()
        return client

    def test_default_uses_max_completion_tokens(self):
        """First call for an uncached model should use max_completion_tokens."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "test-model"
        client._client.chat.completions.create.return_value = mock_response

        client.messages_create(
            model="test-model", max_tokens=100,
            system="test", messages=[{"role": "user", "content": "hi"}]
        )

        call_kwargs = client._client.chat.completions.create.call_args[1]
        self.assertIn('max_completion_tokens', call_kwargs)
        self.assertNotIn('max_tokens', call_kwargs)

    def test_fallback_to_max_tokens_on_rejection(self):
        """If max_completion_tokens is rejected, should retry with max_tokens."""
        from openai import BadRequestError
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "old-model"

        error_body = {'error': {'message': "Unsupported parameter: 'max_completion_tokens'"}}
        error = BadRequestError(
            message="Unsupported parameter: 'max_completion_tokens'",
            response=MagicMock(status_code=400, json=lambda: error_body,
                             headers={}, text=""),
            body=error_body
        )
        client._client.chat.completions.create.side_effect = [error, mock_response]

        result = client.messages_create(
            model="old-model", max_tokens=100,
            system="test", messages=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(result.content, "test")
        # Second call should use max_tokens
        second_call = client._client.chat.completions.create.call_args_list[1][1]
        self.assertIn('max_tokens', second_call)
        self.assertNotIn('max_completion_tokens', second_call)

    def test_cache_prevents_repeated_fallback(self):
        """After fallback, subsequent calls for the same model should use cached param."""
        client = self._make_client()
        # Pre-populate instance cache
        client._token_param_cache['cached-model'] = 'max_tokens'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "cached"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "cached-model"
        client._client.chat.completions.create.return_value = mock_response

        client.messages_create(
            model="cached-model", max_tokens=100,
            system="test", messages=[{"role": "user", "content": "hi"}]
        )

        # Should use cached max_tokens directly, only one call
        self.assertEqual(client._client.chat.completions.create.call_count, 1)
        call_kwargs = client._client.chat.completions.create.call_args[1]
        self.assertIn('max_tokens', call_kwargs)
        self.assertNotIn('max_completion_tokens', call_kwargs)


if __name__ == '__main__':
    unittest.main()
