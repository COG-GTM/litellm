"""
Enforces `general_settings.max_request_size_mb` before any route runs.

FastAPI reads and parses `Body`/`Form`/`File` parameters before route
dependencies such as `user_api_key_auth` execute, so the limit has to be
applied at the ASGI layer to stop oversized payloads from being buffered.
"""

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_utils import (
    REQUEST_SIZE_CHECKED_SCOPE_KEY,
    check_content_length_is_safe,
    get_max_request_size_mb,
    wrap_receive_with_size_limit,
)


class RequestSizeLimitMiddleware:
    """
    - Rejects requests whose Content-Length exceeds the limit without reading the body.
    - Wraps `receive` so body reads abort once the limit is exceeded (chunked uploads,
      understated Content-Length).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_request_size_mb = get_max_request_size_mb()
        if max_request_size_mb is None:
            await self.app(scope, receive, send)
            return

        try:
            check_content_length_is_safe(
                content_length=Headers(scope=scope).get("content-length"),
                max_request_size_mb=max_request_size_mb,
            )
        except ProxyException as e:
            response = JSONResponse(
                status_code=int(e.code),
                content={"error": e.to_dict()},
                headers=e.headers,
            )
            await response(scope, receive, send)
            return

        scope[REQUEST_SIZE_CHECKED_SCOPE_KEY] = True
        await self.app(
            scope,
            wrap_receive_with_size_limit(receive, max_request_size_mb),
            send,
        )
