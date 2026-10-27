import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import hash_token
from litellm.proxy.utils import PrismaClient, ProxyLogging

SQLI_PAYLOAD = "' OR pg_sleep(5)--"


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def prisma_client():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=proxy_logging)
    client.db.query_first = AsyncMock(return_value=None)
    with patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()):
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [SQLI_PAYLOAD, hash_token("sk-1234")])
async def test_combined_view_lookup_binds_token_as_parameter(prisma_client, token):
    await prisma_client.get_data(
        token=token,
        table_name="combined_view",
        query_type="find_unique",
        check_deprecated=False,
    )

    prisma_client.db.query_first.assert_awaited_once()
    args = prisma_client.db.query_first.await_args.args
    sql_query, bound_args = args[0], args[1:]
    assert "WHERE v.token = $1" in sql_query
    assert token not in sql_query
    assert bound_args == (token,)


@pytest.mark.asyncio
async def test_combined_view_cached_plan_retry_keeps_bound_token(prisma_client):
    token = hash_token("sk-1234")
    prisma_client.db.query_first = AsyncMock(
        side_effect=[Exception("cached plan must not change result type"), None]
    )

    await prisma_client.get_data(
        token=token,
        table_name="combined_view",
        query_type="find_unique",
        check_deprecated=False,
    )

    assert prisma_client.db.query_first.await_count == 2
    retry_args = prisma_client.db.query_first.await_args_list[1].args
    assert "cache_invalidated_" in retry_args[0]
    assert retry_args[1:] == (token,)
