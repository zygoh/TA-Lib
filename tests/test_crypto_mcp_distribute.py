import asyncio
import base64
import io
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile


def install_dependency_stubs() -> None:
    if "requests" not in sys.modules:
        requests_module = types.ModuleType("requests")

        class HTTPErrorStub(Exception):
            def __init__(self, *args, response=None, **kwargs):
                super().__init__(*args)
                self.response = response

        class RequestExceptionStub(Exception):
            pass

        def _unsupported_post(*args, **kwargs):
            raise NotImplementedError("requests.post should not be called in these tests")

        requests_module.HTTPError = HTTPErrorStub
        requests_module.RequestException = RequestExceptionStub
        requests_module.post = _unsupported_post
        sys.modules["requests"] = requests_module

    if "xdk" not in sys.modules:
        xdk_module = types.ModuleType("xdk")
        media_package = types.ModuleType("xdk.media")
        media_models = types.ModuleType("xdk.media.models")
        oauth1_module = types.ModuleType("xdk.oauth1_auth")
        posts_package = types.ModuleType("xdk.posts")
        posts_models = types.ModuleType("xdk.posts.models")

        class ClientStub:
            def __init__(self, *args, **kwargs):
                self.auth = kwargs.get("auth")
                self.oauth2_auth = None
                self.token = kwargs.get("token")
                self.media = types.SimpleNamespace(
                    initialize_upload=lambda *a, **k: None,
                    append_upload=lambda *a, **k: None,
                    finalize_upload=lambda *a, **k: None,
                )
                self.posts = types.SimpleNamespace(create=lambda *a, **k: None)

            def is_token_expired(self):
                return False

            def refresh_token(self):
                return self.token

        class AppendUploadRequestStub:
            @classmethod
            def model_validate(cls, data):
                return data

        class InitializeUploadRequestStub:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class OAuth1Stub:
            def __init__(self, *args, **kwargs):
                pass

            def build_request_header(self, *args, **kwargs):
                return "OAuth"

        class CreateRequestStub:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CreateRequestMediaStub:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        xdk_module.Client = ClientStub
        media_models.AppendUploadRequest = AppendUploadRequestStub
        media_models.InitializeUploadRequest = InitializeUploadRequestStub
        oauth1_module.OAuth1 = OAuth1Stub
        posts_models.CreateRequest = CreateRequestStub
        posts_models.CreateRequestMedia = CreateRequestMediaStub

        media_package.models = media_models
        posts_package.models = posts_models
        xdk_module.media = media_package
        xdk_module.posts = posts_package

        sys.modules["xdk"] = xdk_module
        sys.modules["xdk.media"] = media_package
        sys.modules["xdk.media.models"] = media_models
        sys.modules["xdk.oauth1_auth"] = oauth1_module
        sys.modules["xdk.posts"] = posts_package
        sys.modules["xdk.posts.models"] = posts_models


install_dependency_stubs()
os.environ.setdefault("XAI_API_KEY", "test-key")
os.environ.setdefault("XAI_MODEL", "test-model")

REAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+kX9sAAAAASUVORK5CYII="
)

from app.routers import crypto_mcp
from app.services import distribution_service


class CryptoMcpDistributeRouteTests(unittest.TestCase):
    def test_distribute_accepts_uploaded_image(self) -> None:
        response_payload = {
            "status": "success",
            "symbol": "BTCUSDT",
            "telegram_sent": True,
            "x_sent": True,
            "square_sent": True,
            "channels": {},
            "notes": [],
        }
        upload = UploadFile(
            file=io.BytesIO(REAL_PNG_BYTES),
            filename="cover.png",
            headers=Headers({"content-type": "image/png"}),
        )
        with (
            patch.object(crypto_mcp, "ensure_symbol_usdt", return_value="BTCUSDT"),
            patch.object(crypto_mcp, "distribute_post", AsyncMock(return_value=response_payload)) as distribute_mock,
        ):
            response = asyncio.run(
                crypto_mcp.distribute(
                    symbol="BTC",
                    text="压缩后正文",
                    image=upload,
                )
            )

        self.assertEqual(response["status"], "success")
        distribute_mock.assert_awaited_once_with(
            symbol_usdt="BTCUSDT",
            text="压缩后正文",
            image_bytes=REAL_PNG_BYTES,
            image_filename="cover.png",
            image_content_type="image/png",
        )

    def test_distribute_allows_text_only_mode(self) -> None:
        response_payload = {
            "status": "success",
            "symbol": "ETHUSDT",
            "telegram_sent": True,
            "x_sent": True,
            "square_sent": True,
            "channels": {},
            "notes": ["image not provided, telegram/x sent as text"],
        }
        with (
            patch.object(crypto_mcp, "ensure_symbol_usdt", return_value="ETHUSDT"),
            patch.object(crypto_mcp, "distribute_post", AsyncMock(return_value=response_payload)) as distribute_mock,
        ):
            response = asyncio.run(
                crypto_mcp.distribute(
                    symbol="ETH",
                    text="只有文字",
                    image=None,
                )
            )

        self.assertEqual(response["symbol"], "ETHUSDT")
        distribute_mock.assert_awaited_once_with(
            symbol_usdt="ETHUSDT",
            text="只有文字",
            image_bytes=None,
            image_filename=None,
            image_content_type=None,
        )

    def test_distribute_rejects_non_image_upload(self) -> None:
        upload = UploadFile(
            file=io.BytesIO(b"not-an-image"),
            filename="note.txt",
            headers=Headers({"content-type": "text/plain"}),
        )
        with (
            patch.object(crypto_mcp, "ensure_symbol_usdt", return_value="BTCUSDT"),
            patch.object(crypto_mcp, "distribute_post", AsyncMock()) as distribute_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    crypto_mcp.distribute(
                        symbol="BTC",
                        text="正文",
                        image=upload,
                    )
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "image 必须是图片文件")
        distribute_mock.assert_not_awaited()


class DistributionServiceTests(unittest.TestCase):
    def test_distribute_post_with_image_keeps_image_mode(self) -> None:
        with (
            patch.object(
                distribution_service,
                "_send_telegram",
                AsyncMock(return_value={"sent": True, "mode": "image+text"}),
            ) as telegram_mock,
            patch.object(
                distribution_service,
                "_send_x",
                AsyncMock(return_value={"sent": True, "mode": "image+text"}),
            ) as x_mock,
            patch.object(
                distribution_service,
                "_send_square",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ) as square_mock,
        ):
            result = asyncio.run(
                distribution_service.distribute_post(
                    symbol_usdt="BTCUSDT",
                    text="压缩后正文",
                    image_bytes=REAL_PNG_BYTES,
                    image_filename="cover.png",
                    image_content_type="image/png",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["notes"], [])
        telegram_mock.assert_awaited_once_with("压缩后正文", REAL_PNG_BYTES, "cover.png", "image/png")
        x_mock.assert_awaited_once_with("压缩后正文", REAL_PNG_BYTES, "cover.png", "image/png")
        square_mock.assert_awaited_once_with("压缩后正文")

    def test_distribute_post_without_image_records_text_fallback(self) -> None:
        with (
            patch.object(
                distribution_service,
                "_send_telegram",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ),
            patch.object(
                distribution_service,
                "_send_x",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ),
            patch.object(
                distribution_service,
                "_send_square",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ),
        ):
            result = asyncio.run(
                distribution_service.distribute_post(
                    symbol_usdt="ETHUSDT",
                    text="只有文字",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertIn("image not provided, telegram/x sent as text", result["notes"])

    def test_distribute_post_marks_partial_and_keeps_channel_reason(self) -> None:
        with (
            patch.object(
                distribution_service,
                "_send_telegram",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ),
            patch.object(
                distribution_service,
                "_send_x",
                AsyncMock(return_value={"sent": False, "mode": "none", "note": "x upload failed"}),
            ),
            patch.object(
                distribution_service,
                "_send_square",
                AsyncMock(return_value={"sent": True, "mode": "text"}),
            ),
        ):
            result = asyncio.run(
                distribution_service.distribute_post(
                    symbol_usdt="ETHUSDT",
                    text="只有文字",
                )
            )

        self.assertEqual(result["status"], "partial")
        self.assertIn("x: x upload failed", result["notes"])
        self.assertIn("partial success", result["notes"])
