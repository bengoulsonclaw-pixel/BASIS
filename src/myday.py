"""My Day — per-seat private task lists for the desk home (2026-08-20 redesign).

A "seat" is a person: locally the implicit admin seat; on the VPS each colleague
login (the same accounts Colleague Access manages). One JSON per seat under
data/myday/ — deliberately file-per-person so a colleague's list never travels
with anyone else's session. No Streamlit here; app.py owns the widgets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "myday"


def seats() -> list[dict]:
    """[{id, name, desk}] — the admin seat first, then every colleague account."""
    out = [{"id": "admin", "name": "Ben", "desk": "Admin"}]
    try:
        from . import auth
        for uid, u in sorted(auth.load_users().items()):
            if isinstance(u, dict):
                out.append({"id": uid, "name": u.get("name", uid),
                            "desk": (u.get("role") or "desk").title()})
    except Exception:
        pass
    return out


def _file(seat: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.@") else "_" for ch in str(seat))[:80]
    return DIR / f"{safe or 'admin'}.json"


def load(seat: str) -> list[dict]:
    try:
        items = json.loads(_file(seat).read_text(encoding="utf-8"))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def save(seat: str, items: list[dict]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    _file(seat).write_text(json.dumps(items, indent=1), encoding="utf-8")


def add(seat: str, title: str, date_s: str, time_s: str) -> None:
    title = (title or "").strip()
    if not title:
        return
    items = load(seat)
    items.append({"id": f"t{int(time.time() * 1000)}", "title": title,
                  "date": date_s or "", "time": time_s or "", "done": False})
    save(seat, items)


def toggle(seat: str, tid: str) -> None:
    from datetime import date as _date
    items = load(seat)
    for i in items:
        if i.get("id") == tid:
            i["done"] = not i.get("done")
            # undated tasks recur every day until done; stamping the completion day
            # lets the Today view keep the struck row visible (and un-toggleable)
            # until midnight, then drop it
            if i["done"]:
                i["done_date"] = _date.today().isoformat()
            else:
                i.pop("done_date", None)
    save(seat, items)


def remove(seat: str, tid: str) -> None:
    save(seat, [i for i in load(seat) if i.get("id") != tid])
