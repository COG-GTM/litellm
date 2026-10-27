"""
Tests for RequestSizeLimitMiddleware.

The limit must apply before FastAPI parses Body/Form/File params, i.e. before
any route dependency (such as user_api_key_auth) runs.
"""

from unittest.mock import patch

import pytest
from fastapi import Body, Depends, FastAPI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_utils import (
    REQUEST_SIZE_CHECKED_SCOPE_KEY,
    check_if_request_size_is_safe,
)
from litellm.proxy.middleware.request_size_limit_middleware import (
    RequestSizeLimitMiddleware,
)

_ONE_MB = 1024 * 1024


def _make_client(handler_calls):
    async def handler(request: Request):
        handler_calls.append(request.scope.get(REQUEST_SIZE_CHECKED_SCOPE_KEY))
        # mirrors user_api_key_auth: must be a no-op once the middleware ran
        await check_if_request_size_is_safe(request=request)
        try:
            body = await request.body()
        except ProxyException as e:
            return JSONResponse(status_code=int(e.code), content={"error": e.message})
        return JSONResponse({"size": len(body)})

    app = Starlette(routes=[Route("/upload", handler, methods=["POST"])])
    app.add_middleware(RequestSizeLimitMiddleware)
    return TestClient(app)


@pytest.fixture
def limit_1mb():
    with patch(
        "litellm.proxy.proxy_server.general_settings", {"max_request_size_mb": 1}
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        yield


def test_rejects_large_content_length_before_route_runs(limit_1mb):
    handler_calls = []
    client = _make_client(handler_calls)

    response = client.post("/upload", content=b"x" * (2 * _ONE_MB))

    assert response.status_code == 400
    assert "Request size is too large" in response.json()["error"]["message"]
    assert handler_calls == []


def test_rejects_invalid_content_length(limit_1mb):
    handler_calls = []
    client = _make_client(handler_calls)

    response = client.post(
        "/upload", content=b"{}", headers={"content-length": "not-a-number"}
    )

    assert response.status_code == 400
    assert handler_calls == []


def test_aborts_chunked_body_once_limit_exceeded(limit_1mb):
    handler_calls = []
    client = _make_client(handler_calls)

    def body_chunks():
        for _ in range(64):
            yield b"x" * (_ONE_MB // 2)

    response = client.post("/upload", content=body_chunks())

    assert response.status_code == 400
    assert "Request size is too large" in response.json()["error"]["message"]
    assert handler_calls == [True]


def test_allows_body_under_limit(limit_1mb):
    handler_calls = []
    client = _make_client(handler_calls)

    response = client.post("/upload", content=b"x" * 1024)

    assert response.status_code == 200
    assert response.json() == {"size": 1024}
    assert handler_calls == [True]


def test_passthrough_when_limit_not_configured():
    handler_calls = []
    client = _make_client(handler_calls)

    with patch("litellm.proxy.proxy_server.general_settings", {}), patch(
        "litellm.proxy.proxy_server.premium_user", True
    ):
        response = client.post("/upload", content=b"x" * (2 * _ONE_MB))

    assert response.status_code == 200
    assert response.json() == {"size": 2 * _ONE_MB}
    assert handler_calls == [None]


def test_fastapi_body_params_are_not_parsed_when_too_large(limit_1mb):
    """FastAPI parses Body params before dependencies like user_api_key_auth run."""
    dependency_calls = []

    async def auth_dependency():
        dependency_calls.append(True)

    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware)

    @app.post("/key/generate", dependencies=[Depends(auth_dependency)])
    async def generate(data: dict = Body(...)):
        return {"keys": len(data)}

    client = TestClient(app)
    oversized = b'{"a": "' + b"x" * (2 * _ONE_MB) + b'"}'

    def chunked():
        yield oversized

    for content in (oversized, chunked()):
        response = client.post("/key/generate", content=content)
        assert response.status_code == 400
        assert "Request size is too large" in response.json()["error"]["message"]
    assert dependency_calls == []

    ok = client.post("/key/generate", json={"a": 1})
    assert ok.status_code == 200
    assert dependency_calls == [True]
