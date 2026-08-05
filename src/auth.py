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
                st.success(f"Added {email.strip().lower()} ({role}).")
            except ValueError as e:
                st.error(str(e))

    if not REQUIRE_LOGIN:
        # Only shown on the local Terminal -- accounts are managed here, then pushed to the VPS.
        # (On the VPS itself this file already IS the live account list, so there's nothing to push.)
        st.divider()
        st.markdown("**Live site**")
        st.caption("Changes above are local until you push them — basisterminal.com won't see a "
                   "new or removed colleague until you do.")
        if st.button("🚀 Push accounts to basisterminal.com", type="primary", key="push_users_vps"):
            with st.spinner("Pushing to the VPS…"):
                ok, msg = push_users_to_vps()
            (st.success if ok else st.error)(msg)


# ----- push local account changes to the live VPS -----------------------------------------------
# Accounts are managed from the local Terminal (this file's admin panel), never by visiting the
# live site directly. This copies the local account file over SSH -- the VPS app picks it up on
# its very next page load (load_users() always reads fresh off disk, no restart needed). A manual,
# explicit action, never fired automatically, since it reaches a real production machine.
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
