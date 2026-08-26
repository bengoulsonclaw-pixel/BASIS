"""Per-user login for BASIS.

Two roles: ``admin`` (Ben — full read/write access, unchanged from before) and ``colleague``
(view every report page, generate PDFs, email a report to their own logged-in address only —
no config edits, no automation/recipient-list changes, no arbitrary-recipient email).

Accounts are provisioned manually by an admin from the in-app "Colleague accounts" panel
(``render_user_admin`` in app.py) — there is no self-serve signup. The username *is* the
colleague's work email, and that's the only address the "email me" buttons will ever send to
(see ``email_report_ui`` in app.py).

Storage is local to each install (data/users.json, gitignored — see .gitignore, never committed
through git, same as data/automation.json / data/email_recipients.json). Accounts are managed
from the local Terminal, then pushed to the live VPS over SSH with an explicit button
(``push_users_to_vps``) — the VPS's own copy is overwritten wholesale, so it's the local file
that's the source of truth. Passwords are bcrypt-hashed (via streamlit_authenticator's Hasher)
before they ever touch disk, in transit, or on the VPS.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

ROOT = Path(__file__).parent.parent
USERS_FILE = ROOT / "data" / "users.json"
COOKIE_KEY_FILE = ROOT / "data" / ".auth_cookie_key.txt"
SEND_LOG_FILE = ROOT / "data" / "email_send_log.jsonl"
SETTINGS_FILE = ROOT / "data" / "auth_settings.json"
USAGE_LOG_FILE = ROOT / "data" / "usage_log.jsonl"

# Login is opt-in per deployment, not baked into the app: your local Terminal (this same app.py,
# run straight off your machine) stays exactly as it always was -- zero login, full access, one
# implicit admin. Only a deployment that explicitly sets BASIS_REQUIRE_LOGIN=1 (the VPS site --
# see deploy/vps/docker-compose.yml) enforces the login/role system below.
REQUIRE_LOGIN = os.environ.get("BASIS_REQUIRE_LOGIN", "0") == "1"
LOCAL_ADMIN = {"email": "local", "name": "Local Terminal", "role": "admin"}

ROLE_ADMIN = "admin"
ROLE_COLLEAGUE = "colleague"

_SEND_COOLDOWN_S = 20   # per-user, per-session — guards against a double-click firing two sends


# ----- account store --------------------------------------------------------------------------
def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2, sort_keys=True), encoding="utf-8")


def load_allowed_domains() -> list[str]:
    """Colleague accounts must use a work email on one of these domains (checked on add, not
    just at login) — set from the Colleague Accounts panel. Empty = no restriction, which is
    only safe while you're the only account that exists."""
    if not SETTINGS_FILE.exists():
        return []
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("allowed_email_domains", [])
    except Exception:
        return []


def save_allowed_domains(domains: list[str]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"allowed_email_domains": domains}, indent=2), encoding="utf-8")


def domain_allowed(email: str) -> bool:
    domains = load_allowed_domains()
    if not domains:
        return True
    email = email.strip().lower()
    return any(email.endswith("@" + d.lower().lstrip("@")) for d in domains)


def add_user(email: str, name: str, password: str, role: str) -> None:
    """Admin action: create or overwrite an account. Raises ValueError on anything invalid —
    callers should catch it and show the message, not let it propagate as a crash."""
    email = (email or "").strip().lower()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    if not domain_allowed(email):
        raise ValueError(f"Email must be on an allowed domain ({', '.join(load_allowed_domains())}).")
    if role not in (ROLE_ADMIN, ROLE_COLLEAGUE):
        raise ValueError("Invalid role.")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    users = load_users()
    users[email] = {
        "name": (name or "").strip() or email,
        "password_hash": stauth.Hasher.hash(password),
        "role": role,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_users(users)


def remove_user(email: str) -> None:
    users = load_users()
    users.pop((email or "").strip().lower(), None)
    save_users(users)


def _cookie_key() -> str:
    """Stable per-install secret for signing the login cookie. Generated once and kept out of git
    (see .gitignore) — each environment (your machine, the VPS) has its own, so a cookie issued by
    one is meaningless to the other."""
    if COOKIE_KEY_FILE.exists():
        return COOKIE_KEY_FILE.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    COOKIE_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_KEY_FILE.write_text(key, encoding="utf-8")
    return key


def _credentials() -> dict:
    users = load_users()
    return {"usernames": {
        email: {"name": u["name"], "password": u["password_hash"], "email": email}
        for email, u in users.items()
    }}


# ----- session / gating -----------------------------------------------------------------------
def current_user() -> dict | None:
    """The logged-in user's {email, name, role}, or None if not authenticated. On a deployment
    that doesn't require login (REQUIRE_LOGIN False -- your local Terminal), always the implicit
    local admin -- there's no session to check."""
    if not REQUIRE_LOGIN:
        return LOCAL_ADMIN
    if not st.session_state.get("authentication_status"):
        return None
    email = st.session_state.get("username")
    rec = load_users().get(email)
    if not rec:
        return None
    return {"email": email, "name": rec["name"], "role": rec["role"]}


def is_admin() -> bool:
    if not REQUIRE_LOGIN:
        return True
    u = current_user()
    return bool(u and u["role"] == ROLE_ADMIN)


def _render_bootstrap_admin() -> None:
    st.title("BASIS — set up the admin account")
    st.caption("No accounts exist yet on this install. Create your own admin account below — "
               "you'll be able to add colleague accounts from inside the app afterwards.")
    with st.form("bootstrap_admin"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        pw = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        if st.form_submit_button("Create admin account", type="primary"):
            if pw != pw2:
                st.error("Passwords don't match.")
            else:
                try:
                    add_user(email, name, pw, ROLE_ADMIN)
                    st.success("Admin account created — log in below.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))


def require_login() -> dict:
    """Gate the whole app behind login -- but only on a deployment that opts in (REQUIRE_LOGIN;
    see the module docstring). Call once, at the very top of app.py, before anything else
    renders — stops the script until a valid session exists. Returns the logged-in user."""
    if not REQUIRE_LOGIN:
        return LOCAL_ADMIN
    if not load_users():
        _render_bootstrap_admin()
        st.stop()

    authenticator = stauth.Authenticate(
        _credentials(), cookie_name="basis_auth", cookie_key=_cookie_key(),
        cookie_expiry_days=14, auto_hash=False,
    )
    st.session_state["_authenticator"] = authenticator   # reused by the sidebar logout button

    if not st.session_state.get("authentication_status"):
        authenticator.login(location="main")

    user = current_user()
    if user is None:
        if st.session_state.get("authentication_status") is False:
            st.error("Email or password is incorrect.")
        st.stop()
    if not st.session_state.get("_login_logged"):
        # Once per browser session (not every rerun) -- fires whether they typed a password or
        # came in on the 14-day cookie, since either way it's a real visit worth counting.
        record_login(user["email"])
        st.session_state["_login_logged"] = True
    return user


def render_logout_button() -> None:
    authenticator = st.session_state.get("_authenticator")
    if authenticator is not None:
        authenticator.logout("Log out", "sidebar")


# ----- admin: manage colleague accounts -------------------------------------------------------
def render_user_admin() -> None:
    st.subheader("👥 Colleague accounts")
    st.caption("Accounts added here can log in and use every report page, but can't change any "
               "settings, run a data pull, or email a report to anyone but themselves.")
    # Result of the automatic push, stashed by _sync_live so it survives the st.rerun()
    # that follows an add/remove (Ben, 2026-08-26).
    _m = st.session_state.pop(_SYNC_MSG_KEY, None)
    if _m:
        (st.success if _m[0] == "ok" else st.error)(_m[1])
    users = load_users()
    if users:
        import pandas as pd
        rows = [{"email": e, "name": u["name"], "role": u["role"], "created": u.get("created", "")}
                for e, u in sorted(users.items())]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    colleague_emails = [e for e, u in users.items() if u["role"] == ROLE_COLLEAGUE]
    if colleague_emails:
        rm = st.selectbox("Remove a colleague", [""] + sorted(colleague_emails), key="user_admin_rm")
        if rm and st.button(f"Remove {rm}", key="user_admin_rm_btn"):
            remove_user(rm)
            # revoking access must reach the live site at once — and `drop` stops the
            # reconcile step adopting the account straight back off the VPS
            _sync_live(f"Removed {rm}", drop=rm)
            st.rerun()
    domains = load_allowed_domains()
    with st.expander("⚙️ Email domain restriction" + (f" — {', '.join('@' + d for d in domains)}"
                                                       if domains else " — none set"),
                     expanded=False):
        st.caption("Only email addresses on these domains can be added below. Leave blank to "
                   "allow any address (fine while you're the only account that exists).")
        _dom_text = st.text_input(
            "Allowed domains — comma-separated (e.g. yourbroker.com)",
            value=", ".join(domains), key="allowed_domains_input")
        if st.button("Save", key="save_allowed_domains"):
            new_domains = [d.strip().lstrip("@").lower() for d in _dom_text.split(",") if d.strip()]
            save_allowed_domains(new_domains)
            st.success("Saved.")
            st.rerun()

    st.markdown("**Add an account**" if not REQUIRE_LOGIN else "**Add a colleague**")
    if domains:
        st.caption(f"Email must end in: {', '.join('@' + d for d in domains)}")
    else:
        st.caption("⚠️ No email domain restriction set (see above) — any email address can be "
                   "added right now.")
    with st.form("add_colleague", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Work email — becomes their username and their only "
                              "'email me' destination")
        pw = st.text_input("Temporary password", type="password",
                           help="Share this with them directly; there's no self-service reset yet.")
        # Only offered locally -- e.g. to seed your OWN admin account before the first push to the
        # VPS, so you claim it instead of leaving a public bootstrap screen for whoever visits first.
        role = ROLE_COLLEAGUE
        if not REQUIRE_LOGIN:
            role = ROLE_ADMIN if st.radio(
                "Role", ["Colleague", "Admin"], horizontal=True, key="add_acct_role",
                help="Admin = full access, same as your local Terminal. Colleague = view/generate/"
                     "email-to-self only.") == "Admin" else ROLE_COLLEAGUE
        if st.form_submit_button("Add account" if not REQUIRE_LOGIN else "Add colleague",
                                 type="primary"):
            try:
                add_user(email, name, pw, role)
                _sync_live(f"Added {email.strip().lower()} ({role})")
                st.rerun()                   # surfaces the push result + refreshes the table
            except ValueError as e:
                st.error(str(e))

    if not REQUIRE_LOGIN:
        # Only shown on the local Terminal -- accounts are managed here, then pushed to the VPS.
        # (On the VPS itself this file already IS the live account list, so there's nothing to push.)
        st.divider()
        st.markdown("**Live site**")
        st.caption("Adding or removing an account now pushes to basisterminal.com automatically. "
                   "This button is the manual retry — use it if a push failed (no network, VPS "
                   "down), or to force the live list back in step with this one.")
        if st.button("🚀 Push accounts to basisterminal.com", type="primary", key="push_users_vps"):
            with st.spinner("Pushing to the VPS…"):
                ok, msg = push_users_to_vps()
            (st.success if ok else st.error)(msg)


# ----- push local account changes to the live VPS -----------------------------------------------
# Accounts are managed from the local Terminal (this file's admin panel), never by visiting the
# live site directly. This copies the local account file over SSH -- the VPS app picks it up on
# its very next page load (load_users() always reads fresh off disk, no restart needed).
#
# 2026-08-26: this now fires AUTOMATICALLY on add/remove (Ben's call, after Kevin Hobbs and Dave
# Pinder both sat local-only and couldn't log in). It was manual on the reasoning that it reaches
# a production machine -- but the failure mode of forgetting is silent and lands on the colleague,
# not on you, which is worse. The manual button remains as the retry when a push fails.
_SYNC_MSG_KEY = "user_sync_msg"


def merge_vps_users(drop: str | None = None) -> tuple[bool, str]:
    """Adopt any account that exists on the live site but not here, then save locally.

    The live site's admin panel writes only the VPS copy, and Ben does add colleagues from
    his phone — so a straight local-over-VPS push silently DELETES them. That happened once
    already (the "Pep" account, 2026-08-11). Now that the push fires automatically it would
    happen far more often, so every sync reconciles first. `drop` is the account being
    removed, which must NOT be adopted back off the VPS."""
    import json as _json
    import subprocess
    import tempfile
    if not VPS_SSH_KEY.exists():
        return False, f"SSH key not found at {VPS_SSH_KEY}."
    tmp = Path(tempfile.gettempdir()) / "basis_vps_users.json"
    try:
        r = subprocess.run(
            ["scp", "-i", str(VPS_SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{VPS_HOST}:{VPS_USERS_PATH}", str(tmp)],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "scp failed").strip()
        remote = _json.loads(tmp.read_text(encoding="utf-8"))
    except Exception as e:
        return False, str(e)
    local = load_users()
    adopted = [e for e in remote
               if e not in local and e != (drop or "").strip().lower()]
    if adopted:
        for e in adopted:
            local[e] = remote[e]
        save_users(local)
    return True, (f"adopted {len(adopted)} account(s) added on the live site "
                  f"({', '.join(sorted(adopted))})" if adopted else "")


def _sync_live(what: str, drop: str | None = None) -> None:
    """Mirror a local account change to basisterminal.com straight away. The outcome is stashed
    in session_state rather than rendered here, because every caller reruns immediately after and
    would otherwise wipe the message. A failed sync must be LOUD: the account exists locally but
    the colleague still can't log in, which is exactly the state that prompted this change.

    Reconciles before pushing — see merge_vps_users. If the pull fails we do NOT push: a blind
    overwrite is how a phone-added account gets deleted, and that is worse than a delayed one."""
    if REQUIRE_LOGIN:            # on the VPS this file already IS the live list
        return
    got, note = merge_vps_users(drop=drop)
    if not got:
        st.session_state[_SYNC_MSG_KEY] = (
            "err", f"⚠️ {what} locally, but the live account list could NOT be read "
                   f"({note}) — nothing was pushed, because overwriting it blind can delete "
                   "an account added from your phone. The change is NOT live yet; fix the "
                   "connection and press “Push accounts to basisterminal.com” below.")
        return
    ok, msg = push_users_to_vps()
    _note = f" (also {note})" if note else ""
    st.session_state[_SYNC_MSG_KEY] = (
        ("ok", f"✅ {what} — pushed to basisterminal.com, live on their next page load.{_note}")
        if ok else
        ("err", f"⚠️ {what} locally, but the push to basisterminal.com FAILED: {msg} — the change "
                "is NOT live yet. Press “Push accounts to basisterminal.com” below to retry."))
VPS_SSH_KEY = Path.home() / ".ssh" / "basis_vps"
VPS_HOST = "root@2.24.221.3"
VPS_USERS_PATH = "/docker/basis/app/data/users.json"


def push_users_to_vps() -> tuple[bool, str]:
    import subprocess
    if not VPS_SSH_KEY.exists():
        return False, f"SSH key not found at {VPS_SSH_KEY}."
    try:
        r = subprocess.run(
            ["scp", "-i", str(VPS_SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             str(USERS_FILE), f"{VPS_HOST}:{VPS_USERS_PATH}"],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return True, "Pushed — live on basisterminal.com now."
        return False, (r.stderr or r.stdout or "scp failed").strip()
    except Exception as e:
        return False, str(e)


# ----- pull activity from the live VPS -----------------------------------------------------------
VPS_USAGE_LOG_PATH = "/docker/basis/app/data/usage_log.jsonl"


def pull_usage_log_from_vps() -> tuple[bool, str]:
    """Copy the VPS's usage log down to the local Terminal for viewing -- the reverse of
    push_users_to_vps(). Colleagues only ever log in on the live site (local has no login), so
    this is the one thing that flows VPS-to-local rather than the other way around."""
    import subprocess
    if not VPS_SSH_KEY.exists():
        return False, f"SSH key not found at {VPS_SSH_KEY}."
    try:
        r = subprocess.run(
            ["scp", "-i", str(VPS_SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{VPS_HOST}:{VPS_USAGE_LOG_PATH}", str(USAGE_LOG_FILE)],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return True, "Pulled — showing the latest activity from basisterminal.com."
        err = (r.stderr or r.stdout or "scp failed").strip()
        if "No such file" in err:
            return False, "No activity recorded on the live site yet."
        return False, err
    except Exception as e:
        return False, str(e)


# ----- usage tracking: who logged in, when, and what they looked at -----------------------------
def record_login(email: str) -> None:
    _append_usage_event(email, "login")


def record_page_view(email: str, page: str) -> None:
    _append_usage_event(email, "page_view", page=page)


def _append_usage_event(email: str, event: str, **extra) -> None:
    try:
        USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "email": email, "event": event, **extra,
            }) + "\n")
    except Exception:
        pass   # usage tracking is best-effort — never block the app over it


def _load_usage_events() -> list[dict]:
    if not USAGE_LOG_FILE.exists():
        return []
    out = []
    for line in USAGE_LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def render_activity() -> None:
    st.subheader("📊 Colleague activity")
    st.caption("Who has logged in, how often, and which pages they've used.")
    if not REQUIRE_LOGIN:
        # Colleagues only ever log in on the live site -- local has no login, so this data only
        # ever accumulates on the VPS. Pull it down to view it, same pattern as pushing accounts.
        _mtime = (datetime.fromtimestamp(USAGE_LOG_FILE.stat().st_mtime, tz=timezone.utc)
                  .isoformat(timespec="minutes")) if USAGE_LOG_FILE.exists() else None
        st.caption(f"Last pulled: {_mtime or 'never'}")
        if st.button("🔄 Pull latest activity from basisterminal.com", type="primary",
                     key="pull_activity"):
            with st.spinner("Pulling…"):
                ok, msg = pull_usage_log_from_vps()
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
        st.divider()

    events = _load_usage_events()
    if not events:
        st.info("No activity recorded yet.")
        return

    import pandas as pd
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df["ts"])
    names = {e: u["name"] for e, u in load_users().items()}
    df["name"] = df["email"].map(lambda e: names.get(e, e))

    st.markdown("**Summary**")
    logins = df[df["event"] == "login"]
    if logins.empty:
        st.caption("No logins recorded yet.")
    else:
        summary = (logins.groupby(["name", "email"])["ts"]
                   .agg(logins="count", last_seen="max", first_seen="min")
                   .reset_index().sort_values("last_seen", ascending=False))
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("**Most-viewed pages**")
    views = df[df["event"] == "page_view"]
    if views.empty:
        st.caption("No page views recorded yet.")
    else:
        pivot = (views.groupby(["name", "page"]).size().reset_index(name="views")
                 .sort_values(["name", "views"], ascending=[True, False]))
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    with st.expander("Raw recent events", expanded=False):
        st.dataframe(df.sort_values("ts", ascending=False).head(200),
                    use_container_width=True, hide_index=True)


# ----- self-send rate limit + audit log -------------------------------------------------------
def can_send(email: str) -> bool:
    """Per-session cooldown so a double-click / accidental resubmit doesn't fire duplicate sends."""
    ts = st.session_state.get(f"_last_send_{email}", 0)
    return (time.time() - ts) >= _SEND_COOLDOWN_S


def record_send(email: str, report_key: str) -> None:
    st.session_state[f"_last_send_{email}"] = time.time()
    try:
        SEND_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SEND_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "email": email, "report": report_key,
            }) + "\n")
    except Exception:
        pass   # audit log is best-effort — never block a send over it
