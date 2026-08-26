"""Colleague accounts must reach the live site automatically (src/auth._sync_live).

Regression for 2026-08-26: Kevin Hobbs and Dave Pinder were both added on the local
Terminal and neither could log in, because pushing the account file to basisterminal.com
was a separate manual button that is easy to forget. The failure is silent here and lands
on the colleague, so the push now fires on every add and remove.

Two things must not regress:
  • a FAILED push has to be loud — the account exists locally while the person still
    cannot log in, which is precisely the state this change exists to prevent;
  • the push must NOT fire on the VPS itself, where the file already IS the live list.

No network: push_users_to_vps is stubbed throughout.
"""
from __future__ import annotations

import pytest

from src import auth


@pytest.fixture
def session(monkeypatch):
    """A stand-in for st.session_state that behaves like the dict auth uses."""
    store: dict = {}
    monkeypatch.setattr(auth.st, "session_state", store, raising=False)
    return store


def _stub_push(monkeypatch, ok: bool, msg: str = "boom"):
    calls = []

    def _fake():
        calls.append(True)
        return ok, msg

    monkeypatch.setattr(auth, "push_users_to_vps", _fake)
    return calls


def test_successful_push_is_reported(session, monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_LOGIN", False)
    calls = _stub_push(monkeypatch, True, "Pushed — live now.")
    auth._sync_live("Added kevin@example.com (colleague)")
    assert len(calls) == 1
    kind, text = session[auth._SYNC_MSG_KEY]
    assert kind == "ok"
    assert "kevin@example.com" in text and "basisterminal.com" in text


def test_failed_push_is_loud_and_says_it_is_not_live(session, monkeypatch):
    """The whole point: a local-only account must never look like a finished job."""
    monkeypatch.setattr(auth, "REQUIRE_LOGIN", False)
    _stub_push(monkeypatch, False, "ssh: connect to host ... timed out")
    auth._sync_live("Added kevin@example.com (colleague)")
    kind, text = session[auth._SYNC_MSG_KEY]
    assert kind == "err"
    assert "NOT live" in text
    assert "timed out" in text, "the underlying reason must reach the operator"
    assert "retry" in text.lower()


def test_no_push_from_the_vps_itself(session, monkeypatch):
    """On the live deployment this file already IS the account list — pushing would be
    a machine copying over itself."""
    monkeypatch.setattr(auth, "REQUIRE_LOGIN", True)
    calls = _stub_push(monkeypatch, True)
    auth._sync_live("Added someone")
    assert calls == [], "must not push when running as the live site"
    assert auth._SYNC_MSG_KEY not in session


def test_message_survives_one_rerun_then_clears(session, monkeypatch):
    """Callers rerun immediately after, so the outcome is stashed rather than rendered —
    it must be readable exactly once."""
    monkeypatch.setattr(auth, "REQUIRE_LOGIN", False)
    _stub_push(monkeypatch, True, "ok")
    auth._sync_live("Removed dave@example.com")
    assert session.pop(auth._SYNC_MSG_KEY, None) is not None
    assert session.pop(auth._SYNC_MSG_KEY, None) is None


def test_passwords_are_never_stored_in_the_clear(tmp_path, monkeypatch):
    """Adjacent guarantee, worth pinning while we are here: the account file holds a hash,
    which is why the admin table cannot show anyone their password."""
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(auth, "SETTINGS_FILE", tmp_path / "settings.json")
    auth.add_user("kevin@example.com", "Kevin Hobbs", "correct horse battery", auth.ROLE_COLLEAGUE)
    raw = (tmp_path / "users.json").read_text(encoding="utf-8")
    assert "correct horse battery" not in raw
    rec = auth.load_users()["kevin@example.com"]
    assert "password_hash" in rec and "password" not in rec
