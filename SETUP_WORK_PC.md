# Running BASIS on the work PC

This guide gets the **BASIS Strategy Monitor** running on your work PC with **live Bloomberg**,
without needing the work PC to download anything (your dev PC does that part).

You keep developing on the home PC exactly as now. When you've made changes, you re-run one
command here, copy the folder across, and re-run the setup over there. Nothing on the home PC
changes.

---

## The short version

**On the home PC (this one), once per update:**
```
.venv\Scripts\python build_deploy.py
```
→ builds `C:\Users\<you>\BASIS_deploy` (code + offline packages + the browser for PDFs).

**Copy that whole `BASIS_deploy` folder to the work PC.** (Network share, approved USB, etc. —
it's a few hundred MB.)

**On the work PC, once:**
1. Double-click **`setup_work_pc.bat`** → installs everything offline.
2. Open + log into the **Bloomberg Terminal**.
3. Double-click **`run_dashboard_bloomberg.bat`** → BASIS opens in your browser.

That's it. The rest of this file is detail and troubleshooting.

---

## Before you start — two things to check

**1. Python version match.** The offline packages are built for the Python on the *home* PC:

> **Python 3.14.4**

On the work PC, open a terminal and run `python --version` (or `py --version`). It should be
**3.14.x**. If it's a different version (e.g. 3.12), the offline install won't work — see
*Troubleshooting → "Wrong Python version"* below (it's a one-line rebuild on the home PC).

**2. Bloomberg.** For live data the **Bloomberg Terminal must be open and logged in** on the
work PC. BASIS talks to it through the local Desktop API (the same machine). No Terminal =
use snapshot mode instead (`run_dashboard.bat`).

---

## Step-by-step (work PC)

### 1. Copy the folder
Copy the entire **`BASIS_deploy`** folder somewhere sensible on the work PC, e.g.
`C:\BASIS` or `Documents\BASIS`. Avoid a OneDrive-synced location if your firm restricts that.

### 2. Install (offline — your proxy is never touched)
Double-click **`setup_work_pc.bat`**. It will:
- find your Python,
- create a private environment (`.venv`) inside the folder,
- install every package from the bundled `wheels\` folder — **no internet, no proxy**.

If it finishes with "Setup complete", you're done installing.

### 3. First run
1. Make sure the **Bloomberg Terminal is running and logged in**.
2. (Optional but recommended) confirm Bloomberg works:
   `.venv\Scripts\python check_bloomberg.py`
   — it does a tiny test pull and prints OK.
3. Pull a full set of signals: `.venv\Scripts\python run_daily.py`
   — or just launch and click **"Pull Bloomberg Snapshot"** on the Home page.
4. Launch: double-click **`run_dashboard_bloomberg.bat`**.
   BASIS opens at `http://localhost:8501`. Leave the black window open while you use it;
   close it to stop.

### Day-to-day
- **Live Bloomberg:** `run_dashboard_bloomberg.bat` (Terminal must be running).
- **No Bloomberg / offline look:** `run_dashboard.bat` (reads the last cached snapshot).
- The PDF client reports work offline — rendered by the bundled browser if the bundle has a
  `playwright-browsers\` folder, otherwise automatically by this PC's own **Edge or Chrome**
  (the slim bundle uses the latter — nothing to set up, as long as Edge or Chrome is installed).

---

## Updating later (your deploy workflow)

1. Develop and test on the **home PC** as usual.
2. When ready, on the home PC run: `.venv\Scripts\python build_deploy.py`
3. Copy the refreshed `BASIS_deploy` folder to the work PC (overwrite the old one, **except**
   keep the work PC's `.venv` if you like — or just delete `.venv` over there and re-run
   `setup_work_pc.bat`; offline install is quick after the first time).
4. Your saved settings on the work PC (sector defaults, trigger defaults, recipient lists)
   live in its `data\` folder — copying code over them is fine; the build doesn't ship throwaway
   session files, so it won't clobber your work-PC preferences unless you overwrite `data\`.

> Tip: if you only changed code (not data), you can copy just the changed `.py` files /
> `src\` / `templates\` over and skip re-running setup.

---

## Bloomberg notes
- BASIS auto-selects the data source from the `DATAFEED_MODE` environment variable;
  `run_dashboard_bloomberg.bat` sets it to `bloomberg`, `run_dashboard.bat` to `snapshot`.
- `blpapi` + `xbbg` are included in the offline bundle. If they weren't (the build prints a
  warning if Bloomberg's index was unreachable at build time), install them **online** on the
  work PC — Bloomberg's package index is normally whitelisted on a trading-floor network:
  ```
  .venv\Scripts\python -m pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi==3.26.4.2
  .venv\Scripts\python -m pip install xbbg==1.2.6
  ```

## The "Morning Coffee" briefing (optional, not included)
BASIS can show a macro briefing + heatmap that comes from a **separate** project
(`Futures_Movements`) which is **not** part of this bundle. On the work PC those panels just say
"not found" and everything else works normally. If you copy that project over too, point BASIS at
it by setting an environment variable before launching:
```
set BASIS_MC_DIR=C:\path\to\Futures_Movements
```
(or add that `set` line into the launcher `.bat`).

---

## Troubleshooting

**"Python was not found"** — install Python 3.14 (match the version above), then re-run
`setup_work_pc.bat`. Tick "Add Python to PATH" in the installer.

**"Wrong Python version" / offline install fails with version errors** — the work PC's Python
differs from the home PC's. Easiest fix: on the **home PC**, rebuild the bundle for the work
PC's version, e.g. for 3.12:
```
.venv\Scripts\python build_deploy.py --pyver 3.12
```
then copy the new `BASIS_deploy` across. (Or install Python 3.14 on the work PC to match.)

**Offline install fails mentioning `blpapi`** — Bloomberg wasn't bundled. Install it online with
the two pip commands in *Bloomberg notes* above, then re-run `setup_work_pc.bat`.

**PDF reports don't generate** — BASIS renders them with the bundled Chromium if present, else
this PC's **Edge or Chrome**. On the **full** bundle, make sure the `playwright-browsers\` folder
came across (the launchers wire it up; manually, `set PLAYWRIGHT_BROWSERS_PATH=%cd%\playwright-browsers`).
On the **slim** bundle (no `playwright-browsers\`), it uses the machine's Edge/Chrome automatically —
so just make sure Edge or Chrome is installed. To force a specific one, set
`BASIS_PDF_CHANNEL=msedge` (or `chrome`) before launching.

**"localhost refused to connect"** — give it a few seconds after the window opens, then refresh
`http://localhost:8501`. The launcher already clears a stuck previous instance on that port.

**Bloomberg errors / no data** — confirm the Terminal is open and logged in, then run
`.venv\Scripts\python check_bloomberg.py` and read what it prints.

**Need pip online through the corporate proxy** (rare — the offline bundle avoids this):
```
set HTTPS_PROXY=http://USER:PASS@proxyhost:port
set HTTP_PROXY=http://USER:PASS@proxyhost:port
.venv\Scripts\python -m pip install -r requirements.txt
```
