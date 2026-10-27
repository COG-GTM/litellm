"""
Regression tests: an org admin must not be able to manage teams/users that
belong to a different organization by supplying their own `organization_id`
in the request body.
"""

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.proxy._types import (
    BlockTeamRequest,
    DeleteTeamRequest,
    DeleteUserRequest,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.common_utils import (
    _assert_user_can_delete_users,
    _assert_user_can_manage_teams,
)

_NOW = datetime.now(timezone.utc)
ORG_A = "org-a"
ORG_B = "org-b"
CALLER_ID = "org-a-admin"


def _membership(user_id: str, org_id: str, role: str = "org_admin"):
    return LiteLLM_OrganizationMembershipTable(
        user_id=user_id,
        organization_id=org_id,
        user_role=role,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _caller_user(org_ids: List[str]) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=CALLER_ID,
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        organization_memberships=[_membership(CALLER_ID, o) for o in org_ids],
    )


def _caller_key(role: str = LitellmUserRoles.INTERNAL_USER.value) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=CALLER_ID, user_role=role, api_key="sk-caller")


def _team(team_id: str, org_id: Optional[str], admin_id: Optional[str] = None):
    members = [Member(user_id=admin_id, role="admin")] if admin_id else []
    return LiteLLM_TeamTable(
        team_id=team_id, organization_id=org_id, members_with_roles=members
    )


def _target_user(user_id: str, org_ids: List[str], role: str = "internal_user"):
    return LiteLLM_UserTable(
        user_id=user_id,
        user_role=role,
        organization_memberships=[
            _membership(user_id, o, role="internal_user") for o in org_ids
        ],
    )


def _status_code(exc: Exception) -> int:
    if isinstance(exc, HTTPException):
        return exc.status_code
    if isinstance(exc, ProxyException):
        return int(exc.code)
    raise AssertionError(f"unexpected exception type {type(exc)}")


@pytest.fixture
def caller_is_org_a_admin():
    lookup = AsyncMock(return_value=[ORG_A])
    with patch(
        "litellm.proxy.management_endpoints.common_utils._get_caller_org_admin_org_ids",
        new=lookup,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._get_caller_org_admin_org_ids",
        new=lookup,
    ), patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._get_caller_org_admin_org_ids",
        new=lookup,
    ):
        yield


@pytest.mark.asyncio
async def test_caller_org_admin_ids_read_from_db_not_cache(monkeypatch):
    """Revoked org_admin roles must take effect immediately, so the lookup
    queries the membership table directly instead of the cached user object."""
    from litellm.proxy.management_endpoints.common_utils import (
        _get_caller_org_admin_org_ids,
    )

    prisma = MagicMock()
    prisma.db.litellm_organizationmembership.find_many = AsyncMock(
        return_value=[_membership(CALLER_ID, ORG_A)]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    cached_lookup = AsyncMock(return_value=_caller_user([ORG_A, ORG_B]))
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_user_object", cached_lookup)

    assert await _get_caller_org_admin_org_ids(_caller_key()) == [ORG_A]
    prisma.db.litellm_organizationmembership.find_many.assert_awaited_once_with(
        where={"user_id": CALLER_ID, "user_role": LitellmUserRoles.ORG_ADMIN.value}
    )
    cached_lookup.assert_not_called()


@pytest.mark.asyncio
async def test_caller_org_admin_ids_empty_without_user_id():
    from litellm.proxy.management_endpoints.common_utils import (
        _get_caller_org_admin_org_ids,
    )

    key = UserAPIKeyAuth(user_id=None, api_key="sk-x")
    assert await _get_caller_org_admin_org_ids(key) == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manage_teams_allows_proxy_admin():
    await _assert_user_can_manage_teams(
        user_api_key_dict=_caller_key(LitellmUserRoles.PROXY_ADMIN.value),
        teams=[_team("t", ORG_B), _team("t2", None)],
        route="/team/delete",
    )


@pytest.mark.asyncio
async def test_manage_teams_allows_same_org(caller_is_org_a_admin):
    await _assert_user_can_manage_teams(
        user_api_key_dict=_caller_key(),
        teams=[_team("t", ORG_A)],
        route="/team/delete",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("team_org", [ORG_B, None])
async def test_manage_teams_rejects_other_org_or_no_org(
    caller_is_org_a_admin, team_org
):
    with pytest.raises(HTTPException) as exc:
        await _assert_user_can_manage_teams(
            user_api_key_dict=_caller_key(),
            teams=[_team("t-a", ORG_A), _team("t-x", team_org)],
            route="/team/delete",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_users_allows_same_org(caller_is_org_a_admin):
    await _assert_user_can_delete_users(
        user_api_key_dict=_caller_key(),
        users=[_target_user("u", [ORG_A])],
        route="/user/delete",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        _target_user("u-b", [ORG_B]),
        _target_user("u-ab", [ORG_A, ORG_B]),
        _target_user("u-none", []),
        _target_user("u-admin", [ORG_A], role=LitellmUserRoles.PROXY_ADMIN.value),
    ],
)
async def test_delete_users_rejects_unowned_targets(caller_is_org_a_admin, target):
    with pytest.raises(HTTPException) as exc:
        await _assert_user_can_delete_users(
            user_api_key_dict=_caller_key(), users=[target], route="/user/delete"
        )
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# endpoints: org A admin targeting org B resources (organization_id=org A)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_team_cross_org_returns_403(monkeypatch, caller_is_org_a_admin):
    from litellm.proxy.management_endpoints.team_endpoints import delete_team

    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=_team("team-b", ORG_B)
    )
    prisma.delete_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    data = DeleteTeamRequest(team_ids=["team-b"], organization_id=ORG_A)  # type: ignore
    with pytest.raises((HTTPException, ProxyException)) as exc:
        await delete_team(
            data=data,
            http_request=MagicMock(),
            user_api_key_dict=_caller_key(),
            litellm_changed_by=None,
        )
    assert _status_code(exc.value) == 403
    prisma.delete_data.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["block_team", "unblock_team"])
async def test_block_unblock_team_cross_org_returns_403(
    monkeypatch, caller_is_org_a_admin, endpoint_name
):
    from litellm.proxy.management_endpoints import team_endpoints

    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=_team("team-b", ORG_B)
    )
    prisma.db.litellm_teamtable.update = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await getattr(team_endpoints, endpoint_name)(
            data=BlockTeamRequest(team_id="team-b"),
            http_request=MagicMock(),
            user_api_key_dict=_caller_key(),
        )
    assert _status_code(exc.value) == 403
    prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["block_team", "unblock_team"])
async def test_block_unblock_team_same_org_allowed(
    monkeypatch, caller_is_org_a_admin, endpoint_name
):
    from litellm.proxy.management_endpoints import team_endpoints

    team_a = _team("team-a", ORG_A)
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_a)
    prisma.db.litellm_teamtable.update = AsyncMock(return_value=team_a)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    result = await getattr(team_endpoints, endpoint_name)(
        data=BlockTeamRequest(team_id="team-a"),
        http_request=MagicMock(),
        user_api_key_dict=_caller_key(),
    )
    assert result == team_a
    prisma.db.litellm_teamtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_block_team_not_found_returns_404(monkeypatch, caller_is_org_a_admin):
    from litellm.proxy.management_endpoints.team_endpoints import block_team

    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_teamtable.update = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await block_team(
            data=BlockTeamRequest(team_id="missing"),
            http_request=MagicMock(),
            user_api_key_dict=_caller_key(),
        )
    assert _status_code(exc.value) == 404
    prisma.db.litellm_teamtable.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        _target_user("user-b", [ORG_B]),
        _target_user("proxy-admin", [], role=LitellmUserRoles.PROXY_ADMIN.value),
    ],
)
async def test_delete_user_cross_org_returns_403(
    monkeypatch, caller_is_org_a_admin, target
):
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        delete_user,
    )

    prisma = MagicMock()
    prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[target])
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=target)
    prisma.delete_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    data = DeleteUserRequest(user_ids=[target.user_id], organization_id=ORG_A)  # type: ignore
    with pytest.raises((HTTPException, ProxyException)) as exc:
        await delete_user(
            data=data, user_api_key_dict=_caller_key(), litellm_changed_by=None
        )
    assert _status_code(exc.value) == 403
    prisma.delete_data.assert_not_called()


@pytest.mark.asyncio
async def test_delete_user_missing_user_returns_404(monkeypatch, caller_is_org_a_admin):
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        delete_user,
    )

    prisma = MagicMock()
    prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    prisma.delete_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await delete_user(
            data=DeleteUserRequest(user_ids=["missing"]),
            user_api_key_dict=_caller_key(),
            litellm_changed_by=None,
        )
    assert _status_code(exc.value) == 404
    prisma.delete_data.assert_not_called()


# ---------------------------------------------------------------------------
# /team/update, /user/new, /user/bulk_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_team_authz(caller_is_org_a_admin):
    from litellm.proxy.management_endpoints.team_endpoints import (
        _assert_user_can_update_team,
    )

    caller = _caller_key()
    # same org: allowed
    await _assert_user_can_update_team(caller, _team("t", ORG_A), None)
    # other org: denied even though caller is admin of org A
    with pytest.raises(HTTPException) as exc:
        await _assert_user_can_update_team(caller, _team("t", ORG_B), ORG_A)
    assert exc.value.status_code == 403
    # moving own team into an org the caller does not administer: denied
    with pytest.raises(HTTPException) as exc:
        await _assert_user_can_update_team(caller, _team("t", ORG_A), ORG_B)
    assert exc.value.status_code == 403
    # team admin (non org admin) can update but not change the org
    team_admin_team = _team("t", ORG_B, admin_id=CALLER_ID)
    await _assert_user_can_update_team(caller, team_admin_team, None)
    with pytest.raises(HTTPException):
        await _assert_user_can_update_team(caller, team_admin_team, "")


@pytest.mark.asyncio
async def test_new_user_organizations_must_be_administered(caller_is_org_a_admin):
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _assert_user_can_add_to_organizations,
    )

    await _assert_user_can_add_to_organizations(_caller_key(), [ORG_A])
    with pytest.raises(HTTPException) as exc:
        await _assert_user_can_add_to_organizations(_caller_key(), [ORG_A, ORG_B])
    assert exc.value.status_code == 403
    await _assert_user_can_add_to_organizations(
        _caller_key(LitellmUserRoles.PROXY_ADMIN.value), [ORG_B]
    )


@pytest.mark.asyncio
async def test_bulk_update_all_users_requires_proxy_admin(monkeypatch):
    from litellm.proxy._types import UpdateUserRequestNoUserIDorEmail
    from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
        BulkUpdateUserRequest,
    )
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        bulk_user_update,
    )

    prisma = MagicMock()
    prisma.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_usertable.update_many = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    data = BulkUpdateUserRequest(
        all_users=True,
        user_updates=UpdateUserRequestNoUserIDorEmail(max_budget=1.0),
    )
    with pytest.raises((HTTPException, ProxyException)) as exc:
        await bulk_user_update(
            data=data, user_api_key_dict=_caller_key(), litellm_changed_by=None
        )
    assert _status_code(exc.value) == 403
    prisma.db.litellm_usertable.update_many.assert_not_called()
