from contextlib import ExitStack
import json
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


class SquareSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_square_text_only_publish(self):
        captured: dict = {}

        class FakeResp:
            def __init__(self, status: int, payload: dict):
                self.status = status
                self._payload = payload

            async def text(self):
                return json.dumps(self._payload)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeSession:
            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                return FakeResp(200, {"code": "000000", "data": {"id": "sq1", "shareLink": "https://example/sq1"}})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch.object(ds, "_read_square_api_key", return_value=("key", "process_env")):
            with patch.object(ds.aiohttp, "ClientSession", return_value=FakeSession()):
                result = await ds._send_square("Hello #crypto $BTC")

        self.assertTrue(result["sent"])
        self.assertEqual(result["mode"], "text")
        self.assertEqual(captured["json"]["bodyTextOnly"], "Hello #crypto $BTC")
        self.assertEqual(captured["json"]["contentType"], 1)
        self.assertNotIn("imageList", captured["json"])

    async def test_square_image_post_includes_image_list(self):
        captured: dict = {"posts": []}

        class FakeResp:
            def __init__(self, status: int, payload: dict | str):
                self.status = status
                self._payload = payload

            async def text(self):
                return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeSession:
            async def post(self, url, json=None, headers=None):
                captured["posts"].append({"url": url, "json": json})
                if url.endswith("/image/presignedUrl"):
                    return FakeResp(
                        200,
                        {
                            "code": "000000",
                            "data": {"presignedUrl": "https://s3.example/upload", "fileTicket": "ticket-1"},
                        },
                    )
                if url.endswith("/image/imageStatus"):
                    return FakeResp(200, {"code": "000000", "data": {"status": 1, "imageUrl": "https://cdn.example/img.png"}})
                if url.endswith("/content/add"):
                    return FakeResp(200, {"code": "000000", "data": {"id": "sq2", "shareLink": "https://example/sq2"}})
                raise AssertionError(f"unexpected post url: {url}")

            async def put(self, url, data=None, headers=None):
                captured["put"] = {"url": url, "len": len(data or b""), "headers": headers}
                return FakeResp(200, "")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch.object(ds, "_read_square_api_key", return_value=("key", "process_env")):
            with patch.object(ds.aiohttp, "ClientSession", return_value=FakeSession()):
                result = await ds._send_square(
                    "Chart #SOL $SOL",
                    image_bytes=b"\x89PNG",
                    image_filename="chart.png",
                    image_content_type="image/png",
                )

        self.assertTrue(result["sent"])
        self.assertEqual(result["mode"], "image+text")
        publish = captured["posts"][-1]["json"]
        self.assertEqual(publish["imageList"], ["https://cdn.example/img.png"])
        self.assertEqual(publish["bodyTextOnly"], "Chart #SOL $SOL")


if __name__ == "__main__":
    unittest.main()
