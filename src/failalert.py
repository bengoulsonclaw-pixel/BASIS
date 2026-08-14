"""Failure alerts for the scheduled report emailers.

Every scheduled emailer's __main__ runs through `guard(label, main)`: if the run
crashes (an exception) or exits non-zero (e.g. "Report generation FAILED"), an
alert email with the captured output/traceback goes to the failure-alert list —
so a broken automation can never fail silently again (the Aug 2026 WASDE lesson:
the task failed ~40 times over two days and nothing surfaced it).

Design points:
  * The alert list is its OWN list (data/failure_alert_recipients.json, edited on
    the Alert Settings page) — failures go to Ben only, never the client/desk
    report lists.
  * Throttled to ONE alert per report per calendar day (the tasks repeat every
    10-30 min; without this a broken build = ~40 identical emails/day). The
    throttle marker lives in data/signals/failalerts/.
  * Normal no-op runs (outside window / already sent / automation off) exit 0 and
    never alert. A clean run never alerts.
  * The alert itself can never mask the original failure (fully try/excepted),
    and the original exit code is preserved for Task Scheduler.
  * FAILALERT_DRY=1 in the environment prints the would-send instead of emailing
    (used by tests).
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "failure_alert_recipients.json"
MARK_DIR = ROOT / "data" / "signals" / "failalerts"
DEFAULTS = ["benjamin.goulson@xpi.com.br", "bengoulson@gmail.com"]
TAIL_CHARS = 4000                # how much captured output/traceback goes in the email


def load_recipients() -> list:
    """The failure-alert address list (the saved file, else the defaults)."""
    try:
        lst = json.loads(FILE.read_text(encoding="utf-8"))
        clean = [str(e).strip() for e in lst if str(e).strip()]
        if clean:
            return clean
    except Exception:
        pass
    return list(DEFAULTS)


def save_recipients(lst) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    clean = [str(e).strip() for e in (lst or []) if str(e).strip()]
    FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "report"


def _already_alerted_today(label: str) -> bool:
    try:
        return (MARK_DIR / f"{_slug(label)}.txt").read_text().strip() == date.today().isoformat()
    except Exception:
        return False


def _mark_alerted(label: str) -> None:
    try:
        MARK_DIR.mkdir(parents=True, exist_ok=True)
        (MARK_DIR / f"{_slug(label)}.txt").write_text(date.today().isoformat())
    except Exception:
        pass


def send_failure_alert(label: str, detail: str) -> bool:
    """Email the failure to the alert list. Returns True if (dry-)sent."""
    recipients = load_recipients()
    if not recipients:
        print("[failalert] no failure-alert recipients configured; not alerting.")
        return False
    stamp = datetime.now().strftime("%a %d %b %Y %H:%M")
    subject = f"[FAILED] BASIS scheduled report - {label}"
    tail = (detail or "").strip()[-TAIL_CHARS:] or "(no output captured)"
    if os.environ.get("FAILALERT_DRY"):
        print(f"[failalert DRY] would email '{subject}' to {recipients}\n----- detail tail -----\n{tail}")
        return True
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from html import escape
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cot_scheduled_email import load_email_cfg     # Gmail creds (single source of truth)
    sender, app_pw, _desk = load_email_cfg()
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#222;font-size:14px;">'
        f'<p><b style="color:#B71C1C">The scheduled &ldquo;{escape(label)}&rdquo; report FAILED</b> '
        f'at {stamp} and did NOT email its recipients.</p>'
        '<p>Captured output / error (tail):</p>'
        f'<pre style="background:#F5F5F5;border:1px solid #DDD;padding:8px;font-size:11px;'
        f'white-space:pre-wrap;">{escape(tail)}</pre>'
        '<p style="font-size:12px;color:#666">Repeats of this failure today are suppressed (one alert '
        'per report per day) &mdash; the task keeps retrying on its schedule and sends the report '
        'normally once fixed. Manage who gets these alerts: BASIS &rarr; Alert Settings &rarr; '
        'Failure alerts.</p></div>')
    text = (f'The scheduled "{label}" report FAILED at {stamp} and did NOT email its recipients.\n\n'
            f"Captured output / error (tail):\n{tail}\n\n"
            "Repeats today are suppressed (one alert per report per day). "
            "Manage recipients: BASIS -> Alert Settings -> Failure alerts.")
    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, app_pw)
        s.sendmail(sender, recipients, msg.as_string())
    print(f"[failalert] emailed failure alert to {', '.join(recipients)}.")
    return True


class _Tee:
    """Mirror writes to the real stream AND a capture buffer (for the alert email)."""

    def __init__(self, real, buf):
        self._real, self._buf = real, buf

    def write(self, s):
        try:
            self._buf.write(s)
        except Exception:
            pass
        return self._real.write(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def reconfigure(self, *a, **k):        # scripts call sys.stdout.reconfigure(encoding=...)
        try:
            self._real.reconfigure(*a, **k)
        except Exception:
            pass

    def __getattr__(self, name):           # encoding, errors, isatty, ... -> the real stream
        return getattr(self._real, name)


def guard(label: str, fn) -> None:
    """Run a scheduled emailer's main() with failure alerting. On an exception or a
    non-zero sys.exit, email the alert list (once per report per day), then exit with
    the original code so Task Scheduler still records the failure."""
    buf = io.StringIO()
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, buf), _Tee(err, buf)
    code = 0
    try:
        fn()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:              # a manual Ctrl-C is not a task failure
        code = 130
        raise
    except BaseException:
        code = 1
        traceback.print_exc()              # lands in the tee -> captured for the email
    finally:
        sys.stdout, sys.stderr = out, err
    if code != 0:
        try:
            if _already_alerted_today(label):
                print(f"[failalert] failure already alerted today for '{label}'; not re-emailing.")
            elif send_failure_alert(label, buf.getvalue()):
                _mark_alerted(label)
        except Exception as e:
            print(f"[failalert] could not send the failure alert: {e}")
        sys.exit(code)
