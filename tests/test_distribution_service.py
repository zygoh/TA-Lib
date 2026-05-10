from contextlib import ExitStack
import unittest
from unittest.mock import patch

from app.services import distribution_service as ds


class FakePostResponse:
    def model_dump(self):
        return {"data": {"id": "456"}}


class FakePosts:
    def __init__(self, client):
        self._client = client

    def create(self, body):
        self._client.created_body = body
        return FakePostResponse()


class FakeXdkClient:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.oauth2_auth = False
        self.posts = FakePosts(self)
        self.created_body = None
        FakeXdkClient.last_instance = self


def _run_send_x_with_previous_id(previous_id: str):
    FakeXdkClient.last_instance = None
    patches = [
        patch.object(ds, "XdkClient", FakeXdkClient),
        patch.object(ds, "_read_oauth2_user_token", return_value={"access_token": "token"}),
        patch.object(ds, "_read_x_client_id", return_value="client-id"),
        patch.object(ds, "_read_x_client_secret", return_value="client-secret"),
        patch.object(ds, "_read_x_redirect_uri", return_value="https://example.com/callback"),
        patch.object(ds, "_read_env_with_source", return_value=("", "missing")),
        patch.object(ds, "_credential_snapshot", return_value={}),
        patch.object(ds, "_read_x_last_post_id", return_value=previous_id),
        patch.object(ds, "_persist_x_last_post_id"),
        patch.object(ds, "_x_status_permalink", side_effect=lambda tid: f"https://x.com/i/status/{tid}"),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        result = ds._send_x_sync(
            "market update",
            image_bytes=None,
            image_filename=None,
            image_content_type=None,
            reply_to_previous=True,
        )
    return result, FakeXdkClient.last_instance.created_body


class XQuotePreviousTests(unittest.TestCase):
    def test_x_quote_previous_uses_quote_tweet_id_without_appended_link(self):
        result, body = _run_send_x_with_previous_id("123")

        payload = body.model_dump(exclude_none=True)
        self.assertEqual(payload["text"], "market update")
        self.assertEqual(payload["quote_tweet_id"], "123")
        self.assertNotIn("https://x.com", payload["text"])
        self.assertEqual(result["quoted_previous_tweet_id"], "123")
        self.assertIs(result["quote_previous"], True)

    def test_x_quote_previous_missing_id_posts_original_text(self):
        result, body = _run_send_x_with_previous_id("")

        payload = body.model_dump(exclude_none=True)
        self.assertEqual(payload["text"], "market update")
        self.assertNotIn("quote_tweet_id", payload)
        self.assertIs(result["quote_previous"], False)
        self.assertIn("without quote tweet", result["quote_previous_note"])


if __name__ == "__main__":
    unittest.main()
