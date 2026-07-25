@echo off
REM Double-click launcher that starts BOTH halves of the demo together
REM (FUTURE.md item 2): the Reactor Control Room's FastAPI backend (in its own
REM window) and the Streamlit dashboard (in this window), opening both in the
REM browser, so the live demo/viva never needs a typed command.

REM Resolve to this script's own folder first, so double-clicking it from
REM anywhere still finds the project's venv, dashboard/app.py, and the
REM reactor_control_room package by a path relative to the repo root.
cd /d "%~dp0"

if not exist ".venv\Scripts\uvicorn.exe" (
    echo Could not find .venv\Scripts\uvicorn.exe
    echo Install the companion app's extra dependencies first:
    echo   pip install -r reactor_control_room\requirements.txt
    pause
    exit /b 1
)
if not exist ".venv\Scripts\streamlit.exe" (
    echo Could not find .venv\Scripts\streamlit.exe
    echo Set up the virtual environment first: see HOW_TO_RUN.md.
    pause
    exit /b 1
)

echo Starting the Reactor Control Room backend in its own window...
start "Reactor Control Room" cmd /k ".venv\Scripts\uvicorn.exe reactor_control_room.backend.app:app --port 8000"

echo Waiting for the backend to come up...
REM `ping` rather than `timeout` for the delay: `timeout` requires a real
REM console input handle and errors out ("Input redirection is not supported")
REM in some launch contexts; `ping` never touches stdin, so it works everywhere.
ping -n 3 127.0.0.1 >nul
start http://127.0.0.1:8000/

echo.
echo Starting the Streamlit dashboard in this window...
echo Streamlit opens its own browser tab automatically once it's up.
echo When the demo is done, close BOTH windows: this one, and "Reactor Control Room".
echo.

".venv\Scripts\streamlit.exe" run dashboard\app.py

REM Keeps this window open after Streamlit exits so any error is readable, and
REM as a reminder that the Reactor Control Room window still needs closing.
pause
