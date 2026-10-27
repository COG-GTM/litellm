from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.credential_endpoints.endpoints import (
    create_credential,
    delete_credential,
    update_credential,
)
from litellm.types.utils import CreateCredentialItem, CredentialItem

NON_ADMIN_ROLES = [
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.TEAM,
    None,
]


@pytest.fixture
def mock_prisma():
    prisma = MagicMock()
    table = prisma.db.litellm_credentialstable
    table.create = AsyncMock()
    table.update = AsyncMock()
    table.delete = AsyncMock()
    table.find_unique = AsyncMock(return_value=MagicMock())
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        yield table


@pytest.mark.asyncio
@pytest.mark.parametrize("user_role", NON_ADMIN_ROLES)
async def test_credential_mutations_reject_non_proxy_admin(user_role, mock_prisma):
    non_admin = UserAPIKeyAuth(user_id="mallory", user_role=user_role)

    with pytest.raises(HTTPException) as exc_info:
        await update_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CredentialItem(
                credential_name="azure-prod",
                credential_values={"api_base": "https://attacker.example"},
                credential_info={},
            ),
            credential_name="azure-prod",
            user_api_key_dict=non_admin,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await delete_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential_name="azure-prod",
            user_api_key_dict=non_admin,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await create_credential(
            request=MagicMock(),
            fastapi_response=MagicMock(),
            credential=CreateCredentialItem(
                credential_name="azure-new",
                credential_values={"api_key": "sk-x"},
                credential_info={},
            ),
            user_api_key_dict=non_admin,
        )
    assert exc_info.value.status_code == 403

    mock_prisma.update.assert_not_called()
    mock_prisma.delete.assert_not_called()
    mock_prisma.create.assert_not_called()


@pytest.mark.asyncio
async def test_credential_delete_allowed_for_proxy_admin(mock_prisma):
    admin = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.PROXY_ADMIN)

    response = await delete_credential(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        credential_name="azure-prod",
        user_api_key_dict=admin,
    )

    assert response == {"success": True, "message": "Credential deleted successfully"}
    mock_prisma.delete.assert_awaited_once_with(where={"credential_name": "azure-prod"})
