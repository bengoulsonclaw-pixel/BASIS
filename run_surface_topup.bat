@echo off
rem BASIS Surface Topup — opportunistic vol-surface healer (see surface_topup.py).
rem Scheduled hourly 11:00-18:00 weekdays; silent no-op unless data is missing
rem AND the Bloomberg Terminal is reachable.
cd /d "C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard"
.venv\Scripts\python.exe surface_topup.py >> "%LOCALAPPDATA%\basis_surface_topup.log" 2>&1
