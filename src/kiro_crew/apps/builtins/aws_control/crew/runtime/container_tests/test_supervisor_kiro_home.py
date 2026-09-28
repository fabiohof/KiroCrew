"""The task owns its agent-spec directory, so the backend is allowed to write it.

The bug this pins (kirodotdev/KiroCrew#14693). With no ``KIRO_HOME`` the agent specs
resolve to the process HOME's ``~/.kiro/agents``, which every instance under that
``$HOME`` shares. The backend runs on a non-default data home
(``KIROCREW_HOME=<data home>``), and Kiro Crew REFUSES to rewrite a shared agents dir
from one, because the specs it writes pin the writer's data home into every managed MCP
server entry and break strict session identity for a default-home gateway (#9690). So
the supervisor installed the crew's spec into the shared directory with no ownership
provenance, the backend read it as another home's, declined to write, and never created
the DEFAULT spec ``kirocrew.json``. Every turn then died:

    kiro_crew.agent.DerivedSpecStale: the default agent spec
    /var/lib/crew/.kiro/agents/kirocrew.json is missing, so the
    kirocrew-worker.json mirror cannot be verified or rebuilt

The fix does not touch that guard. It uses the guard's own documented private-target
case: ``<data home>/kiro/agents`` is exactly ``config.paths.isolated_agents_dir(data
home)``, a directory this task's teardown owns, so the guard stands aside.

Two invariants carry the fix, and each gets a test that fails if it breaks:

1. The installer and the backend resolve the SAME directory. They get there by
   different routes -- the installer mirrors kiro-cli's ``$KIRO_HOME`` rule from the
   environment, the backend goes through Kiro Crew's resolver -- so agreement is a
   property to prove, not to assume.
2. That directory is the one Kiro Crew's guard exempts. Asserted against the REAL
   ``isolated_agents_dir`` and the REAL guard, not against a copy of the path, so a
   change to either side of the contract reddens here rather than in production.
"""

from __future__ import annotations

import os

import pytest
from container.supervisor import __main__ as sup
from container.supervisor import backend as be
from container.supervisor import bundle as bundle_mod

from ._settings_helper import make_settings


def test_the_kiro_home_is_the_directory_the_guard_exempts(tmp_path):
    """``Settings.kiro_home`` must land on Kiro Crew's own private-target path.

    The exemption is an EXACT match on ``<data home>/kiro/agents`` -- deliberately not
    "anywhere beneath the data home", which would read the machine-wide dir as private.
    So the ``kiro`` segment is load-bearing and this compares against the real
    definition rather than re-spelling it.
    """
    from kiro_crew.config.paths import isolated_agents_dir

    settings = make_settings(tmp_path)

    assert settings.kiro_home / "agents" == isolated_agents_dir(settings.data_home)


def test_the_backend_environment_carries_the_task_owned_kiro_home(tmp_path):
    settings = make_settings(tmp_path)

    env = be.build_backend_env(settings, {"PATH": "/usr/bin"})

    assert env[be.ENV_KIRO_HOME] == str(settings.kiro_home)


def test_a_stale_inherited_kiro_home_does_not_reach_the_backend(tmp_path):
    """The value is a function of the settings, never of what the base mapping holds.

    A base carrying someone else's ``KIRO_HOME`` -- the image's, a previous task's, an
    operator's -- would otherwise point the backend at an agents dir the crew was not
    installed into, and nothing in the backend reports that: it just serves the agents
    it finds.
    """
    settings = make_settings(tmp_path)

    env = be.build_backend_env(settings, {"KIRO_HOME": "/somewhere/else", "PATH": "/usr/bin"})

    assert env[be.ENV_KIRO_HOME] == str(settings.kiro_home)


def test_export_points_the_installer_at_the_same_directory(tmp_path, monkeypatch):
    """The installer resolves its destination from the environment, so the export is
    what makes the two agree. Checked through the installer's OWN resolver."""
    settings = make_settings(tmp_path)
    monkeypatch.delenv("KIRO_HOME", raising=False)

    agents = sup.export_kiro_home(settings)

    assert os.environ["KIRO_HOME"] == str(settings.kiro_home)
    assert bundle_mod.default_kiro_agents_dir() == agents == settings.kiro_home / "agents"
    assert agents.is_dir(), "the export must leave the directory ready for the install"


def test_export_refuses_when_the_two_resolvers_disagree(tmp_path, monkeypatch):
    """The drift tripwire. A future change to either spelling must fail at boot.

    Simulated by making the installer's resolver answer a different directory while the
    settings keep theirs: that is precisely the shape a rename would produce, and
    without the check the deployment boots, installs the crew in one directory and
    serves agents out of another.
    """
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        bundle_mod, "default_kiro_agents_dir", lambda: tmp_path / "elsewhere" / "agents"
    )

    with pytest.raises(sup.common.ConfigError) as err:
        sup.export_kiro_home(settings)

    assert "agents directory" in str(err.value)


def test_export_refuses_when_the_directory_cannot_be_created(tmp_path, monkeypatch):
    """Fail closed rather than let the backend decline for a second reason later."""
    settings = make_settings(tmp_path)
    blocker = settings.data_home / "kiro"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.delenv("KIRO_HOME", raising=False)

    with pytest.raises(sup.common.ConfigError) as err:
        sup.export_kiro_home(settings)

    assert "agent-spec directory" in str(err.value)


def _let_the_real_resolver_answer(monkeypatch, agent_mod) -> None:
    """Release the suite's per-test agent-spec pin so the guard sees a real layout.

    Required, and the reason the control test below exists. The guard's FIRST act is to
    answer "no decline" when the write target is not what the ambient environment would
    produce -- a privately redirected target is private by definition -- and this
    repository's conftest installs exactly such a redirect (``KIRO_AGENTS_DIR``) for
    every test. Left in place, both tests below answer "no decline" for a reason that
    has nothing to do with the layout, which is the false green the control catches.

    Clearing the documented hook is enough: the conftest's own resolver already defers
    to the real resolution once a test has moved ``HOME`` or set ``KIRO_HOME``, so the
    target and the ambient directory then agree by construction rather than by a patch
    that asserts they do. Both tests move ``HOME`` to a tmp path, so nothing here can
    reach the operator's own agents directory.
    """
    monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", None)


def test_the_guard_stands_aside_for_the_container_layout(tmp_path, monkeypatch):
    """The decisive one: the REAL guard must not decline under the container's env.

    This is the assertion that would have caught the bug. It runs Kiro Crew's own
    ``_decline_shared_agent_home`` with the variables the supervisor sets, against a
    spec file carrying no ownership provenance -- which is what the crew installer
    writes -- and requires a verdict of "no decline". The control below runs the same
    setup on the OLD layout and requires a decline, so a green verdict here cannot come
    from a guard that has stopped declining anything.
    """
    from kiro_crew import agent as agent_mod

    settings = make_settings(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "crew"))
    monkeypatch.setenv("KIROCREW_HOME", str(settings.data_home))
    monkeypatch.delenv("KIROCREW_POD", raising=False)
    _let_the_real_resolver_answer(monkeypatch, agent_mod)

    agents = sup.export_kiro_home(settings)
    (agents / f"{settings.crew_name}.json").write_text('{"name": "crew1"}', encoding="utf-8")

    assert agent_mod._decline_shared_agent_home(audit=False) is None


def test_the_old_layout_still_declines(tmp_path, monkeypatch):
    """The control: the bug reproduces on the directory the container used to use.

    Same data home, same unvouched crew spec, only the directory differs -- the process
    HOME's shared ``~/.kiro/agents`` instead of the task-owned one. A decline here is
    what makes the test above a statement about the layout rather than about the guard.
    """
    from kiro_crew import agent as agent_mod

    settings = make_settings(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "crew"))
    monkeypatch.setenv("KIROCREW_HOME", str(settings.data_home))
    monkeypatch.delenv("KIRO_HOME", raising=False)
    monkeypatch.delenv("KIROCREW_POD", raising=False)
    _let_the_real_resolver_answer(monkeypatch, agent_mod)
    # The directory the container used to land in, taken from the installer's OWN
    # resolver with no KIRO_HOME set -- which is what the old code did. Not spelled out
    # as a literal: ``test_agent_home_isolation`` forbids a hard-coded copy of the
    # machine-wide agents dir anywhere in ``src``, and rightly so.
    shared = bundle_mod.default_kiro_agents_dir()
    shared.mkdir(parents=True)
    # The crew installer's spec: present, and vouched for by nobody.
    (shared / f"{settings.crew_name}.json").write_text('{"name": "crew1"}', encoding="utf-8")

    declined = agent_mod._decline_shared_agent_home(audit=False)

    assert declined is not None, (
        "the shared agents dir no longer declines a non-default data home, so the "
        "test above proves nothing about the fix"
    )
    assert declined.parent == shared
