"""Slack Reconnect: ``GatewayOrchestrator.reconnect_slack`` + ``POST /api/slack/reconnect``.

Slack credentials are hoisted once, in ``GatewayOrchestrator.__init__``, and
nothing reassigned them afterwards, so a token or owner ID saved from the
dashboard could take effect only through ``POST /api/restart``. These tests
pin the in-place alternative:

* the orchestrator re-reads the store, recomputes ``_slack_enabled`` from the
  tokens now on disk (a stale ``False`` would make ``init_socket_mode`` a
  silent no-op), tears the old socket client down, re-awaits
  ``init_socket_mode`` on the running loop and records the outcome where the
  settings badge reads it;
* a store that cannot be read leaves the live connection untouched;
* the dashboard's Slack client mirror is cleared with the old socket and
  published again only behind a connected one, and the dashboard's
  ``owner_id`` follows the saved owner -- so a rejected workspace or a former
  owner never keeps dashboard access through a reconnect;
* the handler module's authorization subject (owner + allowlist) follows the
  saved owner on EVERY reconnect, including the ones that stop before a
  handshake, and an old socket client that will not close aborts the attempt
  with that subject cleared -- so a listener that outlives its credentials
  accepts no privileged command from the former owner;
* concurrent callers share one handshake;
* the route carries the PUT's direct-local gate and answers the
  ``connected`` / ``connect_error`` shape ``GET /api/slack/config`` documents.

The orchestrator is built through ``__new__`` (its ``__init__`` boots the
world); ``init_socket_mode`` and ``_connect_slack`` are replaced by recorders
because a real handshake needs Slack.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from kiro_crew.config.loader import CRED_OWNER_ID, CRED_SLACK_APP_TOKEN, CRED_SLACK_BOT_TOKEN

# Obvious placeholders: nothing here reaches Slack.
NEW_CREDS = {
    CRED_SLACK_APP_TOKEN: "xapp-new-not-a-real-value",
    CRED_SLACK_BOT_TOKEN: "xoxb-new-not-a-real-value",
    CRED_OWNER_ID: "U0NEWOWNER",
}
FORMER_OWNER = "U0FORMEROWNER"


def _orch(creds: dict[str, str] | Exception = NEW_CREDS, *, old_client: Any = None) -> Any:
    """A GatewayOrchestrator in the state a failed boot leaves it in.

    ``_slack_enabled`` is False and the tokens are stale, exactly the state the
    issue describes: ``init_socket_mode`` early-returns on that flag, so a
    reconnect that does not recompute it is a no-op.
    """
    from kiro_crew.slack.gateway import GatewayOrchestrator

    orch = GatewayOrchestrator.__new__(GatewayOrchestrator)
    orch._cfg = MagicMock()
    if isinstance(creds, Exception):
        orch._cfg.load_credentials.side_effect = creds
    else:
        orch._cfg.load_credentials.return_value = dict(creds)
    orch._app_token = "xapp-stale"
    orch._bot_token = "xoxb-stale"
    orch._owner_id = ""
    orch._allowed_users = set()
    orch._slack_enabled = False
    orch._slack_connect_error = "invalid_auth"
    orch.slack = None
    orch._socket_client = old_client
    orch._slack_seen = MagicMock(name="boot-seen-cache")
    orch._slack_reconnect_task = None
    orch.dashboard_state = MagicMock()
    orch.dashboard_state.slack_socket_connected = False
    orch.dashboard_state.slack_connect_error = "invalid_auth"
    orch.dashboard_state.slack_client = None
    orch.dashboard_state.owner_id = "U0FORMEROWNER"
    orch._tracking_channels = set()
    orch._background_tasks = set()
    orch._slack_links_team_id = ""
    orch.sessions = MagicMock(name="sessions")
    return orch


class _Recorder:
    """Stand-in for ``init_socket_mode`` that behaves like the real one's edges.

    On ``owner_missing`` it mirrors the real early return (flag off, no
    client); otherwise it installs a fresh client and records where it ran.
    """

    def __init__(self, *, owner_missing: bool = False, hold: asyncio.Event | None = None):
        self.calls: list[tuple[Any, Any]] = []
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.threads: list[threading.Thread] = []
        self.owner_missing = owner_missing
        self.hold = hold
        self.client = MagicMock(name="new-socket-client")

    async def __call__(self, orch: Any, seen: Any) -> None:
        self.calls.append((orch, seen))
        self.loops.append(asyncio.get_running_loop())
        self.threads.append(threading.current_thread())
        if self.hold is not None:
            await self.hold.wait()
        if not orch._slack_enabled:
            return
        if self.owner_missing or not orch._owner_id:
            orch._slack_enabled = False
            orch.slack = None
            return
        orch._socket_client = self.client


def _patches(recorder: _Recorder, connect: Any = None, client_cls: Any = None):
    """Patch the handshake seams; ``connect`` replaces ``_connect_slack``."""
    connect_mock = connect if connect is not None else AsyncMock(return_value=True)
    return (
        patch("kiro_crew.slack.events.init_socket_mode", recorder),
        patch("kiro_crew.slack.gateway.RealSlackClient", client_cls or MagicMock()),
        patch("kiro_crew.slack.gateway.GatewayOrchestrator._connect_slack", connect_mock),
    )


async def _run(
    orch: Any, recorder: _Recorder, connect: Any = None, client_cls: Any = None
) -> dict[str, object]:
    p_init, p_client, p_connect = _patches(recorder, connect, client_cls)
    with p_init, p_client, p_connect:
        return await orch.reconnect_slack()


# ── orchestrator ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconnect_recomputes_enabled_and_hoists_new_credentials() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch(old_client=old)
    rec = _Recorder()
    client_cls = MagicMock(name="RealSlackClient")

    result = await _run(orch, rec, client_cls=client_cls)

    # The stale False is recomputed from the tokens now on disk -- the whole
    # point: without it init_socket_mode returns before doing anything.
    assert orch._slack_enabled is True
    assert orch._app_token == NEW_CREDS[CRED_SLACK_APP_TOKEN]
    assert orch._bot_token == NEW_CREDS[CRED_SLACK_BOT_TOKEN]
    assert orch._owner_id == NEW_CREDS[CRED_OWNER_ID]
    assert orch._allowed_users == {NEW_CREDS[CRED_OWNER_ID]}
    # Old client torn down before the new handshake; new one installed.
    old.close.assert_awaited_once()
    assert orch._socket_client is rec.client
    # The boot-time dedup cache is reused, so redelivered envelopes stay deduped.
    assert rec.calls == [(orch, orch._slack_seen)]
    # Web API client rebuilt on the new bot token and, the socket being
    # connected, mirrored to the dashboard; the dashboard's owner follows.
    client_cls.assert_called_once_with(NEW_CREDS[CRED_SLACK_BOT_TOKEN])
    assert orch.slack is client_cls.return_value
    assert orch.dashboard_state.slack_client is orch.slack
    assert orch.dashboard_state.owner_id == NEW_CREDS[CRED_OWNER_ID]
    # Outcome recorded where GET /api/slack/config reads it, and returned.
    assert orch.dashboard_state.slack_socket_connected is True
    assert orch.dashboard_state.slack_connect_error == ""
    assert result == {"connected": True, "connect_error": ""}


@pytest.mark.asyncio
async def test_reconnect_awaits_init_socket_mode_on_the_running_loop() -> None:
    """WSSocketModeClient needs a current loop in the constructing thread."""
    orch = _orch()
    rec = _Recorder()

    await _run(orch, rec)

    assert rec.loops == [asyncio.get_running_loop()]
    assert rec.threads == [threading.current_thread()]


@pytest.mark.asyncio
async def test_load_failure_leaves_the_live_connection_untouched() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch(OSError("store unreadable"), old_client=old)
    rec = _Recorder()

    with pytest.raises(OSError):
        await _run(orch, rec)

    old.close.assert_not_awaited()
    assert orch._socket_client is old
    assert orch._app_token == "xapp-stale"
    assert orch._slack_enabled is False
    assert rec.calls == []
    # Nothing was recorded either: the badge still describes the live state.
    assert orch.dashboard_state.slack_connect_error == "invalid_auth"


@pytest.mark.asyncio
async def test_missing_tokens_disable_slack_without_a_handshake() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch({CRED_OWNER_ID: "U0NEWOWNER"}, old_client=old)
    rec = _Recorder()

    result = await _run(orch, rec)

    old.close.assert_awaited_once()  # a cleared token must not keep the old socket alive
    assert orch._socket_client is None
    assert orch._slack_enabled is False
    assert orch.slack is None
    assert orch.dashboard_state.slack_client is None
    assert rec.calls == []
    assert result == {"connected": False, "connect_error": "tokens_missing"}
    assert orch.dashboard_state.slack_socket_connected is False
    assert orch.dashboard_state.slack_connect_error == "tokens_missing"


@pytest.mark.asyncio
async def test_missing_owner_is_named_for_the_badge() -> None:
    creds = {k: v for k, v in NEW_CREDS.items() if k != CRED_OWNER_ID}
    orch = _orch(creds)
    rec = _Recorder()

    result = await _run(orch, rec)

    assert len(rec.calls) == 1  # init_socket_mode ran (the flag was recomputed True)
    assert orch._socket_client is None
    assert result == {"connected": False, "connect_error": "owner_id_missing"}


@pytest.mark.asyncio
async def test_connect_failure_reason_is_surfaced() -> None:
    orch = _orch()
    rec = _Recorder()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    result = await _run(orch, rec, connect=_connect)

    assert result == {"connected": False, "connect_error": "invalid_auth"}
    assert orch.dashboard_state.slack_socket_connected is False
    assert orch.dashboard_state.slack_connect_error == "invalid_auth"


def _bind_former_owner() -> None:
    """Put the handler module in the state a booted gateway leaves it in."""
    from kiro_crew.slack import handler

    handler.set_allowed_users({FORMER_OWNER})
    handler.set_owner_id(FORMER_OWNER)
    assert handler.is_allowed_user(FORMER_OWNER)


def _unbind_handler() -> None:
    from kiro_crew.slack import handler

    handler.set_allowed_users(set())
    handler.set_owner_id("")


@pytest.mark.asyncio
async def test_old_client_close_failure_aborts_and_revokes_the_former_owner() -> None:
    """GPT F1: a close that fails leaves a listener up; the reconnect must not
    then hoist new credentials around it (tokens now missing -> no handshake ->
    the old listener keeps the former owner). Abort, keep the client referenced,
    clear the authorization subject, name the outcome."""
    from kiro_crew.slack import handler

    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock(side_effect=RuntimeError("websocket would not close"))
    orch = _orch({CRED_OWNER_ID: "U0NEWOWNER"}, old_client=old)  # tokens cleared on disk
    rec = _Recorder()
    _bind_former_owner()
    try:
        result = await _run(orch, rec)

        assert result == {"connected": False, "connect_error": "previous_client_close_failed"}
        assert orch._socket_client is old  # still referenced: retry / shutdown close it again
        assert rec.calls == []  # no handshake attempted around a live listener
        # Nothing from the store was hoisted.
        assert orch._app_token == "xapp-stale"
        assert orch._owner_id == ""
        assert orch._slack_enabled is False
        # The surviving listener authorizes nobody.
        assert handler.is_allowed_user(FORMER_OWNER) is False
        assert handler.is_owner(FORMER_OWNER) is False
        # The badge reads the failure; the mirror stays empty.
        assert orch.dashboard_state.slack_socket_connected is False
        assert orch.dashboard_state.slack_connect_error == "previous_client_close_failed"
        assert orch.dashboard_state.slack_client is None
    finally:
        _unbind_handler()


@pytest.mark.asyncio
async def test_close_timeout_is_a_failed_close() -> None:
    """The bounded close's timeout is a failed close too (it is an Exception)."""
    old = MagicMock(name="old-socket-client")  # close() is a plain MagicMock: wait_for is patched
    orch = _orch(old_client=old)
    rec = _Recorder()

    with patch("kiro_crew.slack.gateway.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await _run(orch, rec)

    assert result["connect_error"] == "previous_client_close_failed"
    assert orch._socket_client is old
    assert rec.calls == []


@pytest.mark.asyncio
async def test_tokens_missing_rebinds_the_handler_subject_to_the_saved_owner() -> None:
    """The path that never reaches init_socket_mode must still move the
    handler module off the former owner (init_socket_mode is the only other
    writer of those globals)."""
    from kiro_crew.slack import handler

    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch({CRED_OWNER_ID: "U0NEWOWNER"}, old_client=old)
    rec = _Recorder()
    _bind_former_owner()
    try:
        result = await _run(orch, rec)

        assert result["connect_error"] == "tokens_missing"
        assert rec.calls == []
        assert handler.is_allowed_user(FORMER_OWNER) is False
        assert handler.is_owner("U0NEWOWNER") is True
    finally:
        _unbind_handler()


@pytest.mark.asyncio
async def test_cleared_owner_leaves_no_handler_subject() -> None:
    from kiro_crew.slack import handler

    creds = {k: v for k, v in NEW_CREDS.items() if k != CRED_OWNER_ID}
    orch = _orch(creds)
    rec = _Recorder()
    _bind_former_owner()
    try:
        result = await _run(orch, rec)

        assert result["connect_error"] == "owner_id_missing"
        assert handler.is_allowed_user(FORMER_OWNER) is False
        assert handler._owner_id == ""
        assert handler._allowed_users == set()
    finally:
        _unbind_handler()


@pytest.mark.asyncio
async def test_rejected_workspace_leaves_no_client_in_the_dashboard() -> None:
    """The enterprise gate's decline must not leave a sendable client behind.

    ``init_socket_mode`` clears ``orch.slack`` on that path but never touched
    the dashboard mirror, so a reconnect that published the mirror before the
    handshake would hand every dashboard Slack sender a client on a workspace
    the gate rejected.
    """
    old_web = MagicMock(name="old-web-client")
    orch = _orch()
    orch.slack = old_web
    orch.dashboard_state.slack_client = old_web
    seen: list[Any] = []

    class _Rejecting(_Recorder):
        async def __call__(self, orch: Any, seen_cache: Any) -> None:
            # What the dashboard held while the handshake ran.
            seen.append(orch.dashboard_state.slack_client)
            orch._slack_enabled = False
            orch.slack = None  # the real early return does this too

    result = await _run(orch, _Rejecting())

    assert seen == [None]  # the old client was gone BEFORE validation
    assert result == {"connected": False, "connect_error": "enterprise_validation_failed"}
    assert orch.dashboard_state.slack_client is None
    assert orch.dashboard_state.slack_socket_connected is False


@pytest.mark.asyncio
async def test_failed_handshake_publishes_no_client() -> None:
    """A client whose socket did not connect is not offered to the dashboard."""
    orch = _orch()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    await _run(orch, _Recorder(), connect=_connect)

    assert orch.slack is not None  # the orchestrator keeps its own handle, as at boot
    assert orch.dashboard_state.slack_client is None


@pytest.mark.asyncio
async def test_owner_change_moves_the_dashboard_owner() -> None:
    """``DashboardState.owner_id`` is the owner-only handlers' subject.

    It is set once from the boot-time owner; a reconnect that hoists a new
    owner without moving it would leave the former owner authorised.
    """
    orch = _orch()
    assert orch.dashboard_state.owner_id == "U0FORMEROWNER"

    await _run(orch, _Recorder())

    assert orch.dashboard_state.owner_id == NEW_CREDS[CRED_OWNER_ID]


@pytest.mark.asyncio
async def test_cleared_owner_clears_the_dashboard_owner() -> None:
    creds = {k: v for k, v in NEW_CREDS.items() if k != CRED_OWNER_ID}
    orch = _orch(creds)

    await _run(orch, _Recorder())

    assert orch.dashboard_state.owner_id == ""
    assert orch.dashboard_state.slack_client is None


@pytest.mark.asyncio
async def test_connected_reconnect_runs_the_tracked_channel_probe() -> None:
    """Boot warns about a tracked private channel the install cannot read; a
    reconnect that connects is the same moment and must warn the same way."""
    orch = _orch()
    orch._tracking_channels = {"C0TRACKED"}
    probe = AsyncMock()

    with patch("kiro_crew.slack.gateway.warn_unreadable_tracked_channels", probe):
        await _run(orch, _Recorder())
        await asyncio.sleep(0)  # let the fire-and-forget task start

    probe.assert_awaited_once()
    args, kwargs = probe.await_args
    assert args[0] is orch.slack
    assert args[1] == {"C0TRACKED"}
    assert kwargs["notify"] is orch.dashboard_state.notify


@pytest.mark.asyncio
async def test_failed_reconnect_skips_the_tracked_channel_probe() -> None:
    orch = _orch()
    orch._tracking_channels = {"C0TRACKED"}
    probe = AsyncMock()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    with patch("kiro_crew.slack.gateway.warn_unreadable_tracked_channels", probe):
        await _run(orch, _Recorder(), connect=_connect)
        await asyncio.sleep(0)

    probe.assert_not_awaited()


# ── workspace switch: persisted Slack destinations ───────────────────────────


def _team(team_id: str):
    """Pin the workspace the handshake 'validated' (what ``auth.test`` named)."""
    return patch(
        "kiro_crew.slack.gateway.GatewayOrchestrator._slack_validated_team_id",
        staticmethod(lambda: team_id),
    )


@pytest.mark.asyncio
async def test_workspace_switch_sweeps_persisted_links_before_publishing_the_client() -> None:
    """Credentials for ANOTHER workspace: every persisted Slack thread /
    channel link named a channel in the former workspace, so all of them go
    before the dashboard is handed the new client -- never a moment where a
    dashboard turn could combine the new client with an old destination."""
    orch = _orch()
    orch._slack_links_team_id = "T0FORMER"
    seen_client_at_sweep: list[Any] = []

    def _sweep() -> list[str]:
        seen_client_at_sweep.append(orch.dashboard_state.slack_client)
        return ["dashboard:one", "slack:171.2"]

    orch.sessions.clear_all_slack_links.side_effect = _sweep

    with _team("T0NEW"):
        result = await _run(orch, _Recorder())

    assert result["connected"] is True
    orch.sessions.clear_all_slack_links.assert_called_once_with()
    assert seen_client_at_sweep == [None]  # swept while the mirror was still empty
    assert orch.dashboard_state.slack_client is orch.slack  # published afterwards
    assert orch._slack_links_team_id == "T0NEW"


@pytest.mark.asyncio
async def test_same_workspace_keeps_persisted_links() -> None:
    """A token rotation inside one workspace is not a switch: the links still
    name reachable channels and stripping them would silently end every mirror."""
    orch = _orch()
    orch._slack_links_team_id = "T0SAME"

    with _team("T0SAME"):
        await _run(orch, _Recorder())

    orch.sessions.clear_all_slack_links.assert_not_called()
    assert orch.dashboard_state.slack_client is orch.slack
    assert orch._slack_links_team_id == "T0SAME"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous", "current", "after"),
    [
        ("", "T0NEW", "T0NEW"),  # no socket connected in this process yet
        ("T0FORMER", "", "T0FORMER"),  # the edition's gate records no identity
    ],
)
async def test_unknown_identity_on_either_side_is_not_a_switch(
    previous: str, current: str, after: str
) -> None:
    orch = _orch()
    orch._slack_links_team_id = previous

    with _team(current):
        await _run(orch, _Recorder())

    orch.sessions.clear_all_slack_links.assert_not_called()
    assert orch._slack_links_team_id == after


@pytest.mark.asyncio
async def test_failed_handshake_neither_sweeps_nor_moves_the_identity() -> None:
    """Nothing is published on a failed connect, so nothing can misroute; the
    links stay for the workspace that still owns them, and the recorded
    identity stays with them."""
    orch = _orch()
    orch._slack_links_team_id = "T0FORMER"

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    with _team("T0NEW"):
        await _run(orch, _Recorder(), connect=_connect)

    orch.sessions.clear_all_slack_links.assert_not_called()
    assert orch._slack_links_team_id == "T0FORMER"
    assert orch.dashboard_state.slack_client is None


def test_validated_team_id_reads_the_enterprise_cache() -> None:
    from kiro_crew.slack import enterprise
    from kiro_crew.slack.gateway import GatewayOrchestrator

    with patch.object(enterprise, "_validated_team_id", "T0CACHED"):
        assert enterprise.validated_team_id() == "T0CACHED"
        assert GatewayOrchestrator._slack_validated_team_id() == "T0CACHED"


def test_clear_all_slack_links_sweeps_only_slack_destinations(tmp_path: Any) -> None:
    """The map sweep: every row that names a Slack thread goes (fields,
    reverse index, mute marker), a non-Slack mirror and a legacy namespaced
    ``slack_channel_id`` with no thread -- ``set_channel`` bookkeeping, not a
    destination -- are left alone, and the result is on disk."""
    from kiro_crew.messaging.link import ChannelLink
    from kiro_crew.session_map import SessionMap

    with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
        smap = SessionMap()
        smap.set("dashboard:one", "sid-1")
        smap.set_slack_link("dashboard:one", "171.1", "C0FORMER")
        smap.set_slack_paused("dashboard:one", True)
        smap.set("slack:171.2", "sid-2")
        smap.set_slack_link("slack:171.2", "171.2", "C0FORMER")
        smap.set("dashboard:tg", "sid-3")
        smap.set_mirror_link("dashboard:tg", ChannelLink("telegram", channel_id="99"))
        smap.set("discord:55", "sid-4")
        smap._data["discord:55"]["slack_channel_id"] = "discord:55"  # legacy bucket, no thread

        cleared = smap.clear_all_slack_links()

        assert sorted(cleared) == ["dashboard:one", "slack:171.2"]
        assert smap.get_slack_link("dashboard:one") == (None, None)
        assert smap.get_slack_link("slack:171.2") == (None, None)
        assert smap.get_session_for_thread("171.1") is None
        assert smap.get_session_for_thread("171.2") is None
        assert smap.is_slack_paused("dashboard:one") is False
        assert smap.get_mirror_link("dashboard:tg") == ChannelLink("telegram", channel_id="99")
        assert smap._data["discord:55"]["slack_channel_id"] == "discord:55"
        assert smap._data["dashboard:one"]["sid"] == "sid-1"  # the sessions themselves survive

        reloaded = SessionMap()
        assert reloaded.get_slack_link("dashboard:one") == (None, None)
        assert reloaded.get_session_for_thread("171.2") is None
        assert reloaded.get_mirror_link("dashboard:tg") == ChannelLink("telegram", channel_id="99")


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_handshake() -> None:
    """A double-click must not race two Socket Mode handshakes."""
    orch = _orch()
    gate = asyncio.Event()
    rec = _Recorder(hold=gate)
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        first = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)  # first attempt is now parked inside init_socket_mode
        second = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(first, second)

    assert len(rec.calls) == 1
    assert results[0] == results[1] == {"connected": True, "connect_error": ""}
    assert orch._cfg.load_credentials.call_count == 1
    # The slot is released, so a later click starts a fresh attempt.
    assert orch._slack_reconnect_task is None


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_the_shared_attempt() -> None:
    orch = _orch()
    gate = asyncio.Event()
    rec = _Recorder(hold=gate)
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        first = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        second = await orch.reconnect_slack()

    assert len(rec.calls) == 1  # the in-flight attempt finished and was reused
    assert second["connected"] is True


def test_gateway_wires_the_callback_onto_dashboard_state() -> None:
    """Source pin: the dashboard init publishes reconnect_slack for the route."""
    import inspect

    from kiro_crew.slack.gateway import GatewayOrchestrator

    src = inspect.getsource(GatewayOrchestrator._init_dashboard)
    assert "self.dashboard_state._slack_reconnect = self.reconnect_slack" in src


# ── route ────────────────────────────────────────────────────────────────────


def _request(state: Any) -> web.Request:
    app = web.Application()
    app["state"] = state
    return make_mocked_request("POST", "/api/slack/reconnect", app=app)


@pytest.mark.asyncio
async def test_route_answers_the_config_get_shape() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock(
        return_value={"connected": False, "connect_error": "invalid_auth"}
    )
    sel = MagicMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: sel),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 200
    assert resp.text is not None
    import json

    assert json.loads(resp.text) == {"connected": False, "connect_error": "invalid_auth"}
    state._slack_reconnect.assert_awaited_once_with()
    kw = sel.log_api_access.call_args.kwargs
    assert kw["operation"] == "slack.reconnect"
    assert kw["outcome"] == "failed"
    assert kw["error"] == "invalid_auth"


@pytest.mark.asyncio
async def test_route_denies_remote_sessions_like_the_put() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: False),
        patch.object(mod, "_sel", lambda: MagicMock()),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 403
    assert json.loads(resp.body)["code"] == "remote_read_only"
    state._slack_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_is_503_when_no_gateway_owns_a_socket() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = None
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: MagicMock()),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 503
    assert json.loads(resp.body)["code"] == "slack_reconnect_unavailable"


@pytest.mark.asyncio
async def test_route_is_500_when_the_store_cannot_be_read() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock(side_effect=OSError("store unreadable"))
    sel = MagicMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: sel),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 500
    assert json.loads(resp.body)["code"] == "credential_store_unreadable"
    assert sel.log_api_access.call_args.kwargs["outcome"] == "denied"


@pytest.mark.asyncio
async def test_two_concurrent_requests_share_one_attempt_through_the_route() -> None:
    """The route must not serialize concurrent clicks into two full attempts.

    A lock held by the HTTP handler would make the second click wait for the
    first attempt to finish and then run a fresh handshake, tearing down the
    socket the first had just established. Both requests have to reach the
    orchestrator while the first attempt is still running, so its coalescing
    (``test_concurrent_callers_share_one_handshake``) can fold them.
    """
    import kiro_crew.dashboard.handlers.messaging as mod

    in_flight = 0
    peak = 0
    release = asyncio.Event()

    async def slow_reconnect() -> dict[str, object]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await release.wait()
            return {"connected": True, "connect_error": ""}
        finally:
            in_flight -= 1

    state = MagicMock()
    state._slack_reconnect = slow_reconnect
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: MagicMock()),
    ):
        first = asyncio.create_task(mod.api_slack_reconnect(_request(state)))
        second = asyncio.create_task(mod.api_slack_reconnect(_request(state)))
        for _ in range(5):
            await asyncio.sleep(0)
        assert peak == 2  # both reached the orchestrator before either finished
        release.set()
        r1, r2 = await asyncio.wait_for(asyncio.gather(first, second), 5)

    assert r1.status == r2.status == 200


@pytest.mark.asyncio
async def test_attempt_waits_for_the_config_lock_the_save_holds() -> None:
    """Reconnect must not read credentials while a save is writing them.

    Outside ``_get_config_lock()`` a Reconnect that lands mid-save snapshots
    the credentials the operator is replacing and hoists them AFTER the save
    commits: the former owner stays authorized on the live socket. The shared
    attempt takes the lock the PUT holds, so the read sees only a completed
    save.
    """
    from kiro_crew.dashboard.handlers.agents import _get_config_lock

    orch = _orch()
    rec = _Recorder()
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        async with _get_config_lock():  # a save in flight
            attempt = asyncio.create_task(orch.reconnect_slack())
            for _ in range(5):
                await asyncio.sleep(0)
            orch._cfg.load_credentials.assert_not_called()  # blocked behind the save
        result = await asyncio.wait_for(attempt, 5)

    assert result == {"connected": True, "connect_error": ""}
    orch._cfg.load_credentials.assert_called_once_with()


@pytest.mark.asyncio
async def test_shared_attempt_keeps_the_lock_until_it_ends() -> None:
    """A caller that gives up mid-handshake must not release the lock early.

    The attempt is shielded from the caller's cancel and keeps running; if the
    lock followed the caller, a save could commit under that still-running
    read -- the same window.
    """
    from kiro_crew.dashboard.handlers.agents import _get_config_lock

    orch = _orch()
    gate = asyncio.Event()
    rec = _Recorder(hold=gate)
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        caller = asyncio.create_task(orch.reconnect_slack())
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(rec.calls) == 1  # parked inside init_socket_mode
        caller.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert _get_config_lock().locked()  # still held while the attempt runs
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 5)
        shared = orch._slack_reconnect_task
        if shared is not None:
            await asyncio.wait_for(shared, 5)

    assert not _get_config_lock().locked()


def test_route_is_registered_as_post_beside_the_put() -> None:
    from kiro_crew.dashboard import handlers
    from kiro_crew.dashboard.routes import messaging as routes

    app = web.Application()
    routes.register(app)
    reconnect = [
        r
        for r in app.router.routes()
        if r.resource is not None and r.resource.canonical == "/api/slack/reconnect"
    ]
    assert [r.method for r in reconnect] == ["POST"]
    assert reconnect[0].handler is handlers.api_slack_reconnect


# ── Queued turns keep the client that received them across a reconnect ──


def _queue_orch(live_client: Any) -> MagicMock:
    """Orchestrator double for ``_dispatch_queued`` / ``_route_message``.

    ``orch.slack`` is whatever a reconnect made it; the test decides what the
    queue entry remembers. Mirrors test_message_queue's harness: transport off
    so the native ``handle_message`` patch is the one that runs.
    """
    from kiro_crew.config.loader import ACTIVATION_ALWAYS, KiroCrewConfig, MessagingConfig

    orch = MagicMock()
    orch._cfg = KiroCrewConfig(
        slack_channels={},
        slack_dm_activation=ACTIVATION_ALWAYS,
        messaging=MessagingConfig(use_transport=False),
    )
    orch.channel_history = MagicMock()
    orch.slack = live_client
    orch.sessions = MagicMock()
    orch.sessions.is_busy.return_value = False
    orch.sessions.enqueue = MagicMock(return_value=False)
    orch.sessions.dequeue = MagicMock(return_value=None)
    orch.sessions.cancel_queued = MagicMock(return_value=False)
    orch.sessions.is_cancelled = MagicMock(return_value=False)
    orch.sessions.clear_queue = MagicMock()
    orch.sessions.has_session = MagicMock(return_value=False)
    orch.ctx_builder = None
    orch.cron_svc = None
    orch.conv_log = None
    orch.consolidator = None
    orch.subagent_mgr = None
    orch.task_runner = None
    orch._handler_tasks = set()
    orch._session_tasks = {}
    orch._pending_queue = {}
    return orch


@pytest.mark.asyncio
async def test_queued_turn_answers_through_the_client_that_received_it() -> None:
    """A reconnect to workspace B between enqueue and drain must not carry a
    workspace-A turn onto B's client: the reaction removal and the turn itself
    both use the client the queue entry bound at enqueue."""
    from kiro_crew.slack import events

    workspace_a = AsyncMock(name="workspace_a")
    workspace_b = AsyncMock(name="workspace_b")
    orch = _queue_orch(workspace_b)  # the reconnect already happened
    kwargs = {"channel": "C_A", "thread_ts": "1.0", "slack_client": workspace_a}

    with patch.object(events, "handle_message", new_callable=AsyncMock) as hm:
        await events._dispatch_queued(orch, "1.0", "2.0", "follow up", kwargs)

    workspace_a.remove_reaction.assert_awaited_once_with("C_A", "2.0", "hourglass_flowing_sand")
    workspace_b.remove_reaction.assert_not_awaited()
    assert hm.await_args.args[0] is workspace_a


@pytest.mark.asyncio
async def test_queue_entry_without_a_bound_client_uses_the_live_one() -> None:
    """Entries queued before the key existed keep working."""
    from kiro_crew.slack import events

    live = AsyncMock(name="live")
    orch = _queue_orch(live)

    with patch.object(events, "handle_message", new_callable=AsyncMock) as hm:
        await events._dispatch_queued(orch, "1.0", "2.0", "follow up", {"channel": "C1"})

    live.remove_reaction.assert_awaited_once()
    assert hm.await_args.args[0] is live


@pytest.mark.asyncio
async def test_every_enqueue_site_binds_the_receiving_client() -> None:
    """All three queue writers in ``_route_message`` record ``orch.slack`` as
    it was when the message arrived: the session queue (busy task), the
    pre-session pending queue, and the semaphore-locked session queue."""
    from kiro_crew.slack.events import SeenCache, _route_message

    received_by = AsyncMock(name="received_by")
    patches = [
        patch("kiro_crew.slack.events.is_allowed_user", return_value=True),
        patch("kiro_crew.slack.enterprise.check_message_origin", return_value=True),
        patch("kiro_crew.slack.events.handle_message", new_callable=AsyncMock),
    ]
    for p in patches:
        p.start()
    try:
        # 1. Busy task, session object exists -> sessions.enqueue.
        orch = _queue_orch(received_by)
        orch._session_tasks["ts1"] = MagicMock()
        orch.sessions.enqueue.return_value = True
        event = {
            "user": "U1",
            "text": "q",
            "ts": "ts1",
            "channel": "D1",
            "channel_type": "im",
            "team": "T1",
        }
        await _route_message(orch, event, SeenCache(), is_mention=True)
        assert orch.sessions.enqueue.call_args.kwargs["slack_client"] is received_by

        # 2. Busy task, no session object yet -> orch._pending_queue.
        orch = _queue_orch(received_by)
        orch._session_tasks["thr"] = MagicMock()
        orch.sessions.enqueue.return_value = False
        event = {
            "user": "U1",
            "text": "q",
            "ts": "ts2",
            "thread_ts": "thr",
            "channel": "C1",
            "channel_type": "channel",
            "team": "T1",
        }
        await _route_message(orch, event, SeenCache(), is_mention=True)
        _ts, _text, kw = orch._pending_queue["thr"][0]
        assert kw["slack_client"] is received_by

        # 3. No task, but the session semaphore is locked -> sessions.enqueue.
        orch = _queue_orch(received_by)
        orch.sessions.enqueue.return_value = True
        event = {
            "user": "U1",
            "text": "q",
            "ts": "ts3",
            "channel": "D1",
            "channel_type": "im",
            "team": "T1",
        }
        await _route_message(orch, event, SeenCache(), is_mention=True)
        assert orch.sessions.enqueue.call_args.kwargs["slack_client"] is received_by
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_queue_binds_the_socket_client_even_when_routing_suspends() -> None:
    """``_route_message`` suspends before it enqueues (governance gate,
    ``users.info``, file downloads); a Reconnect in that gap swaps
    ``orch.slack`` to another workspace. The queue entry must carry the client
    of the socket that received the event, which the listener passes in --
    not whatever ``orch.slack`` is when the enqueue line runs."""
    from kiro_crew.slack.events import SeenCache, _route_message

    workspace_a = AsyncMock(name="workspace_a")
    workspace_b = AsyncMock(name="workspace_b")
    orch = _queue_orch(workspace_a)
    orch._session_tasks["ts1"] = MagicMock()
    orch.sessions.enqueue.return_value = True

    async def _gate_that_reconnects(_channel: str) -> bool:
        orch.slack = workspace_b  # POST /api/slack/reconnect landed mid-route
        return True

    with (
        patch("kiro_crew.slack.events.is_allowed_user", return_value=True),
        patch("kiro_crew.slack.enterprise.check_message_origin", return_value=True),
        patch("kiro_crew.slack.events.channel_inbound_permitted", _gate_that_reconnects),
        patch("kiro_crew.slack.events.handle_message", new_callable=AsyncMock),
    ):
        event = {
            "user": "U1",
            "text": "q",
            "ts": "ts1",
            "channel": "D1",
            "channel_type": "im",
            "team": "T1",
        }
        await _route_message(orch, event, SeenCache(), is_mention=True, slack_client=workspace_a)

    assert orch.slack is workspace_b  # the swap did happen mid-route
    assert orch.sessions.enqueue.call_args.kwargs["slack_client"] is workspace_a
    # The hourglass the drain removes through workspace_a was added through it.
    workspace_a.add_reaction.assert_awaited_once_with("D1", "ts1", "hourglass_flowing_sand")
    workspace_b.add_reaction.assert_not_awaited()


def test_listener_passes_its_own_client_to_route_message() -> None:
    """``init_socket_mode`` captures ``orch.slack`` once, beside the socket it
    builds, and hands it to every ``_route_message`` call."""
    import inspect

    from kiro_crew.slack import events

    src = inspect.getsource(events.init_socket_mode)
    assert "received_by = orch.slack" in src
    assert "slack_client=received_by," in src


def _swap_event() -> dict:
    return {
        "user": "U1",
        "text": "q",
        "ts": "ts1",
        "channel": "D1",
        "channel_type": "im",
        "team": "T1",
    }


@pytest.mark.asyncio
async def test_immediate_native_dispatch_uses_the_client_that_received_the_event() -> None:
    """The idle-session path dispatches ``handle_message`` straight away. A
    Reconnect that lands while routing is suspended must not make that turn
    answer through the new workspace: the handler gets the receiving socket's
    client, not ``orch.slack`` as it is at dispatch time."""
    from kiro_crew.slack.events import SeenCache, _route_message

    workspace_a = AsyncMock(name="workspace_a")
    workspace_b = AsyncMock(name="workspace_b")
    orch = _queue_orch(workspace_a)

    async def _gate_that_reconnects(_channel: str) -> bool:
        orch.slack = workspace_b  # POST /api/slack/reconnect landed mid-route
        return True

    handled = AsyncMock()
    with (
        patch("kiro_crew.slack.events.is_allowed_user", return_value=True),
        patch("kiro_crew.slack.enterprise.check_message_origin", return_value=True),
        patch("kiro_crew.slack.events.channel_inbound_permitted", _gate_that_reconnects),
        patch("kiro_crew.slack.events.handle_message", handled),
    ):
        await _route_message(
            orch, _swap_event(), SeenCache(), is_mention=True, slack_client=workspace_a
        )
        await asyncio.gather(*orch._handler_tasks)

    assert orch.slack is workspace_b
    handled.assert_awaited_once()
    assert handled.await_args.args[0] is workspace_a


@pytest.mark.asyncio
async def test_immediate_transport_dispatch_uses_the_client_that_received_the_event() -> None:
    """Same invariant on the transport path (``handle_message_transport``)."""
    from kiro_crew.config.loader import MessagingConfig
    from kiro_crew.slack.events import SeenCache, _route_message

    workspace_a = AsyncMock(name="workspace_a")
    workspace_b = AsyncMock(name="workspace_b")
    orch = _queue_orch(workspace_a)
    orch._cfg.messaging = MessagingConfig(use_transport=True)

    async def _gate_that_reconnects(_channel: str) -> bool:
        orch.slack = workspace_b
        return True

    handled = AsyncMock()
    with (
        patch("kiro_crew.slack.events.is_allowed_user", return_value=True),
        patch("kiro_crew.slack.enterprise.check_message_origin", return_value=True),
        patch("kiro_crew.slack.events.channel_inbound_permitted", _gate_that_reconnects),
        patch("kiro_crew.slack.events.handle_message_transport", handled),
    ):
        await _route_message(
            orch, _swap_event(), SeenCache(), is_mention=True, slack_client=workspace_a
        )
        await asyncio.gather(*orch._handler_tasks)

    assert orch.slack is workspace_b
    handled.assert_awaited_once()
    assert handled.await_args.args[0] is workspace_a


def test_route_message_never_reads_the_live_client_after_binding() -> None:
    """Enumeration guard: after ``received_by`` is bound on entry, no code line
    in ``_route_message`` reads ``orch.slack`` -- every event-scoped Web API
    call (user lookup, ephemeral denials, file download, stop/queue replies,
    both immediate dispatch branches, all enqueue sites) goes through the
    client of the socket that received the event."""
    import inspect

    from kiro_crew.slack import events

    lines = inspect.getsource(events._route_message).splitlines()
    bind = next(i for i, line in enumerate(lines) if "received_by = slack_client" in line)
    offenders = [
        line.strip()
        for line in lines[bind + 1 :]
        if "orch.slack" in line
        and not line.strip().startswith("#")
        and "orch.slack_command" not in line
    ]
    assert offenders == []
