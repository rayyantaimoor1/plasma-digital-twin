@echo off
REM Double-click launcher for the Streamlit dashboard (FUTURE.md item 2), so the
REM live demo/viva never needs a typed command. Streamlit opens the default
REM browser to the dashboard automatically once the server is up.

REM Resolve to this script's own folder first, so double-clicking it from
REM anywhere (e.g. a desktop shortcut) still finds the project's venv and
REM dashboard/app.py by a path relative to the repo root, not the caller's cwd.
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo Could not find .venv\Scripts\streamlit.exe
    echo Set up the virtual environment first: see HOW_TO_RUN.md.
    pause
    exit /b 1
)

echo Starting the Plasma Digital Twin dashboard...
echo Close this window ^(or press Ctrl+C^) to stop the server.
echo.

".venv\Scripts\streamlit.exe" run dashboard\app.py

REM Keeps the window open after Streamlit exits (including an immediate crash)
REM so an error message is readable instead of the window flashing shut.
pause
