' No-flash launcher for the BASIS equities Auto-pull (2026-09-08).
' The always-on server keeper (run_basis_server.ps1) fires this when the day's
' equities pull is due and hasn't yet succeeded — replacing the old "BASIS Equities
' Auto Pull" scheduled task that flashed a console and never retried a missed run.
' wscript.exe has no console of its own and intWindowStyle 0 = hidden, so
' run_eq_autopull.bat (snapshot.py --equities, ~5-7 min) runs with NO window at all.
' (0 = hidden window, False = don't wait for it to finish.)
CreateObject("WScript.Shell").Run _
  """C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard\run_eq_autopull.bat""", _
  0, False
