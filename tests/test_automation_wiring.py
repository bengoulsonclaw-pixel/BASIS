"""Every automated report must be reachable from the Recipients page.

Regression for 2026-08-26: automation.REPORTS["macroradar"] pointed at a recipients key
that recipients.REPORTS didn't define. The editor therefore never drew a row for it, so it
could never be given a recipient, so its auto-email could never be switched on — a closed
loop that left a finished report unreachable from the UI entirely. Nothing in the app
noticed, because each half was internally consistent.
"""
from __future__ import annotations

from src import automation, recipients


def test_every_automated_report_has_a_recipients_row():
    missing = {k: m["recipients"] for k, m in automation.REPORTS.items()
               if m["recipients"] not in recipients.REPORTS}
    assert not missing, (
        f"these reports can never be enabled — their recipients key has no editor row: {missing}")


def test_every_automated_report_names_a_task():
    """A report with no task name can't be created, enabled or diagnosed."""
    for k, m in automation.REPORTS.items():
        assert m.get("tasks"), f"{k} names no Windows task"
        assert all(isinstance(t, str) and t.strip() for t in m["tasks"]), k
