' No-flash launcher for the BASIS server keeper (2026-09-07).
' Windows 11's default terminal (Windows Terminal) IGNORES PowerShell's own
' -WindowStyle Hidden, so the "BASIS Server" scheduled task flashed a console window
' every time it re-fired (every 5 min). WScript.Shell.Run with intWindowStyle = 0 is
' genuinely hidden, and wscript.exe itself has no console — so this wrapper launches the
' keeper with no window at all. (0 = hidden window, False = don't wait for it to exit.)
CreateObject("WScript.Shell").Run _
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\Users\Ben\OneDrive\Desktop\AI\strategy-dashboard\run_basis_server.ps1""", _
  0, False
