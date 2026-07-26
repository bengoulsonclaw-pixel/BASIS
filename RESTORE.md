# RESTORE.md — rebuilding BASIS on a fresh PC

This is the **disaster-recovery** runbook: laptop stolen, dead, or replaced, and you are
starting from a blank Windows machine. (For copying a working install onto the office PC
with offline wheels, see `SETUP_WORK_PC.md` instead — different scenario.)

Last verified by an actual clone-and-rebuild test: **2026-07-25**.

---

## Where everything lives

| What | Where | Notes |
|---|---|---|
| BASIS code + full history + data stores | GitHub `bengoulsonclaw-pixel/BASIS` (private) | includes non-regenerable history: `data/snapshot/own30_history.parquet`, `stir_curve_history.parquet`, `data/equities/fundamentals.parquet` |
| OPEC report, ag-fundamentals, Futures_Movements stub | GitHub `bengoulsonclaw-pixel/basis-aux` (private) | |
| Secrets, user settings, generated PDFs | OneDrive only (never in git) | see the secrets list below |
| Live Morning Coffee code + briefing archive | GitHub `bengoulsonclaw-pixel/morning-coffee` (private) | working copy: `OneDrive\Personal\AI\Futures_Movements` |
| Morning Coffee Gmail OAuth (`credentials.json`, `token.json`) + its `.env` | OneDrive only (`OneDrive\Personal\AI\Futures_Movements`) | never in git |

## Rebuild steps

1. **Install** Python **3.14.x** (`py --version` to check), Git for Windows, and the
   Bloomberg Terminal (if this machine will do live pulls).

2. **Clone:**
   ```
   git clone https://github.com/bengoulsonclaw-pixel/BASIS.git
   git clone https://github.com/bengoulsonclaw-pixel/basis-aux.git
   ```
   The first push/pull opens a browser window to sign in to GitHub — that's normal.

3. **Environment** (inside the BASIS folder):
   ```
   py -3.14 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python -m playwright install chromium
   ```
   `requirements.txt` already carries the `--extra-index-url` for Bloomberg's `blpapi`
   (it is not on PyPI). Internet required; behind the office proxy use `SETUP_WORK_PC.md`.

4. **Recreate secrets** (none of these are in git — by design):
   - `data/eia_key.txt` — free key from https://www.eia.gov/opendata/register.php
   - `..\Futures_Movements\.env` — `ANTHROPIC_API_KEY=` (new key from console.anthropic.com;
     the old one should be revoked there if the laptop was stolen)
   - Gmail OAuth (`credentials.json` + `token.json` in the **Personal** Futures_Movements
     folder): restore from OneDrive, or re-run `gmail_auth.py` there (Google Cloud project
     `morning-coffee-501213`, OAuth app must be Published)

5. **Restore user settings** from OneDrive (not in git, so in-app choices are never
   clobbered by git): `data/automation.json`, `data/email_recipients.json`,
   `data/sector_filter.json`, `data/ui_prefs.json`, `data/alerts.json`. Without them the
   app starts with defaults and you reconfigure in the Recipients / Scheduled reports panel.

6. **Re-register the scheduled tasks** from the XML exports in `scheduled_tasks\`:
   ```
   schtasks /create /tn "OPEC MOMR Synopsis" /xml "scheduled_tasks\OPEC_MOMR_Synopsis.xml"
   ```
   …and likewise for the others (COT, PM Release Synopses, Precious Metals Monitor,
   USDA Reaction, WASDE, and **BASIS Nightly Backup Push** — the 22:00 job that
   auto-commits and pushes all three repos via `nightly_backup_push.bat`, so the daily
   data pulls reach GitHub without anyone touching git). **The XMLs hard-code paths under
   `C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard`** — edit them first if the new
   machine uses a different user name or folder.

7. **Verify:** `.venv\Scripts\streamlit run app.py` → the app should open with data as of
   the last push. **The first launch takes several minutes** on "Running load_signals()" —
   `data/signals/` is not in git, so it rebuilds the whole opportunities cache from cold
   (~6 min in the 2026-07-25 test; instant afterwards). With the Bloomberg Terminal open
   and logged in, run the morning pull to go live.

## After a theft, also do

- Revoke the old Anthropic API key (console.anthropic.com) and mint a new one.
- Revoke the Gmail token: myaccount.google.com → Security → Third-party access →
  remove Morning Coffee, then re-run `gmail_auth.py`.
- Change the Microsoft (OneDrive) and GitHub passwords; sign the old device out of both.
