#!/usr/bin/env python
"""BASIS morning digest — emails the NEW strategy flags since the last digest.

Reads data/signals/opportunities.parquet, diffs the flagged signals against
data/signals/digest_seen.json (what the previous digest had), and emails a short
summary of what is *newly* flagged plus the day's cross-strategy confluence.

DRY-RUN BY DEFAULT — it never sends unless you pass --send.

  python digest_email.py                  # dry-run: print the email, don't send
  python digest_email.py --refresh        # run_daily first (recompute), then build
  python digest_email.py --send           # actually send (to Ben) + update marker
  python digest_email.py --seed           # record current flags as 'seen', no email
  python digest_email.py --to a@b.com     # override recipient(s)

Recipient defaults to Ben only (the first address in the Morning Coffee _EMAIL_TO).
Not scheduled — run it by hand, or wire a Windows task later (like the COT emailer).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import smtplib
import ssl
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SIGNALS = ROOT / "data" / "signals" / "opportunities.parquet"
MARKER = ROOT / "data" / "signals" / "digest_seen.json"
MC_MAIN = Path(os.getenv("BASIS_MC_DIR", r"C:\Users\Ben\OneDrive\Personal\AI\Futures_Movements")) / "main.py"

_SHORT = {
    "Mean Reversion": "MeanRev", "Trend": "Trend", "MA Crossover": "MA-cross",
    "MA Swing": "MA-swing", "Volatility": "Vol", "Skew Volatility": "Skew",
    "Vol Term Structure": "VolTerm", "COT Reports": "COT", "Put/Call Ratios": "Put/Call",
    "AG Fundamentals": "AG",
}


def load_email_cfg():
    """(_EMAIL_FROM, _EMAIL_APP_PW, _EMAIL_TO) from the Morning Coffee project, read
    via ast so we don't import it (it pulls heavy deps). Single source of creds."""
    tree = ast.parse(MC_MAIN.read_text(encoding="utf-8", errors="ignore"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("_EMAIL_FROM", "_EMAIL_APP_PW", "_EMAIL_TO"):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out["_EMAIL_FROM"], out["_EMAIL_APP_PW"], out["_EMAIL_TO"]


def _managed_recipients(report, fallback):
    """Recipients set on the dashboard's Recipients page, if any; else fallback."""
    try:
        p = ROOT / "data" / "email_recipients.json"
        if p.exists():
            lst = [e for e in (json.loads(p.read_text(encoding="utf-8")).get(report) or [])
                   if e and str(e).strip()]
            if lst:
                return lst
    except Exception:
        pass
    return fallback


def _norm(m) -> str:
    return str(m).split(" · ")[0].strip()


def load_flags() -> pd.DataFrame:
    df = pd.read_parquet(SIGNALS)
    fl = df[df["signal"].ne("—")].copy()
    try:                                       # honour the dashboard's sector/product filter
        from src import universe
        if universe.filter_active():
            en = universe.enabled_tickers()
            fl = fl[fl["instruments"].map(lambda i: all(p.strip() in en for p in str(i).split("/")))]
    except Exception:
        pass
    fl["key"] = fl["strategy"].astype(str) + "|" + fl["instruments"].astype(str)
    fl["name"] = fl["market"].map(_norm)
    return fl


def read_marker() -> dict:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.exists() else {}
    except Exception:
        return {}


def build(fl: pd.DataFrame, seen: dict):
    """Return (new_flags_df, confluence_list)."""
    new = fl[fl.apply(lambda r: seen.get(r["key"]) != r["signal"], axis=1)].copy()
    conf = []
    for inst, sub in fl.groupby("instruments"):
        strats = list(dict.fromkeys(sub["strategy"].tolist()))
        if len(strats) < 2:
            continue
        conf.append({"name": sub["name"].iloc[0], "n": len(strats),
                     "strats": ", ".join(_SHORT.get(s, s) for s in strats)})
    conf.sort(key=lambda d: (-d["n"], d["name"]))
    return new, conf


def render_text(date_s, new, conf) -> str:
    out = [f"BASIS morning digest  -  {date_s}", "=" * 48, ""]
    if new.empty:
        out += ["No new flags since the last digest.", ""]
    else:
        out += [f"NEW FLAGS  ({len(new)})", "-" * 24]
        for strat, sub in new.groupby("strategy"):
            out.append(f"{strat}")
            for _, r in sub.iterrows():
                out.append(f"   - {r['name']}: {r['signal']}")
            out.append("")
    if conf:
        out += [f"CONFLUENCE  (flagged by 2+ strategies, top {min(12, len(conf))})", "-" * 24]
        for c in conf[:12]:
            out.append(f"   - {c['name']} ({c['n']}): {c['strats']}")
        out.append("")
    out += ["-- BASIS Strategy Monitor"]
    return "\n".join(out)


def render_html(date_s, new, conf) -> str:
    p = [f'<div style="font-family:Segoe UI,Arial,sans-serif;color:#1f2329">',
         f'<h2 style="margin:0 0 2px">BASIS morning digest</h2>',
         f'<div style="color:#6a7078;font-size:13px">{date_s}</div><hr>']
    if new.empty:
        p.append('<p>No new flags since the last digest.</p>')
    else:
        p.append(f'<h3>New flags <span style="color:#6a7078">({len(new)})</span></h3>')
        for strat, sub in new.groupby("strategy"):
            p.append(f'<p style="margin:8px 0 2px"><b>{strat}</b></p><ul style="margin:0">')
            for _, r in sub.iterrows():
                p.append(f'<li>{r["name"]}: <b>{r["signal"]}</b></li>')
            p.append('</ul>')
    if conf:
        p.append('<h3 style="margin-top:16px">Confluence <span style="color:#6a7078">'
                 '(flagged by 2+ strategies)</span></h3><ul>')
        for c in conf[:12]:
            p.append(f'<li><b>{c["name"]}</b> ({c["n"]}): <span style="color:#6a7078">'
                     f'{c["strats"]}</span></li>')
        p.append('</ul>')
    p.append('<hr><div style="color:#9aa0a8;font-size:12px">Auto-generated by the '
             'BASIS Strategy Monitor.</div></div>')
    return "".join(p)


def main():
    ap = argparse.ArgumentParser(description="BASIS morning digest emailer (dry-run by default).")
    ap.add_argument("--send", action="store_true", help="actually send the email (default is dry-run)")
    ap.add_argument("--refresh", action="store_true", help="run_daily first to recompute signals")
    ap.add_argument("--seed", action="store_true", help="record current flags as seen, do not email")
    ap.add_argument("--to", nargs="+", default=None, help="override recipient(s)")
    args = ap.parse_args()

    if args.refresh:
        print("Recomputing signals (run_daily)...")
        import run_daily
        run_daily.run()

    if not SIGNALS.exists():
        print(f"No signals file at {SIGNALS} — run the dashboard / --refresh first.")
        sys.exit(1)

    fl = load_flags()
    seen = read_marker()
    new, conf = build(fl, seen)
    date_s = datetime.now().strftime("%d %b %Y, %H:%M")
    text = render_text(date_s, new, conf)

    if args.seed:
        MARKER.write_text(json.dumps({r["key"]: r["signal"] for _, r in fl.iterrows()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Seeded {len(fl)} current flags as 'seen' — next digest shows only changes.")
        return

    sender, app_pw, email_to = load_email_cfg()
    recipients = args.to or _managed_recipients("digest", [email_to[0]])   # managed list, else Ben
    subject = f"BASIS morning digest - {datetime.now():%d %b %Y}  ({len(new)} new)"

    if not args.send:
        print(f"[dry-run] would send to {recipients}\nSubject: {subject}\n")
        print(text)
        print("\n[dry-run] marker NOT updated. Use --send to email + record, or --seed to record only.")
        return

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(render_html(date_s, new, conf), "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, app_pw)
        s.sendmail(sender, recipients, msg.as_string())
    MARKER.write_text(json.dumps({r["key"]: r["signal"] for _, r in fl.iterrows()},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sent '{subject}' to {', '.join(recipients)}; marker updated ({len(fl)} flags).")


if __name__ == "__main__":
    from src.failalert import guard          # emails Ben on a crash/non-zero exit (one alert/day)
    guard("Daily signal digest", main)
