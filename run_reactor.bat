@echo off
REM Double-click launcher for the Reactor Control Room companion app on its own
REM (FUTURE.md item 2) - starts its FastAPI backend, which also serves the
REM frontend at /, and opens it in the browser, so the live demo/viva never
REM needs a typed command.

REM Resolve to this script's own folder first, so double-clicking it from
REM anywhere (e.g. a desktop shortcut) still finds the project's venv and the
REM reactor_control_room package by a path relative to the repo root, not the
REM caller's cwd.
cd /d "%~dp0"

if not exist ".venv\Scripts\uvicorn.exe" (
    echo Could not find .venv\Scripts\uvicorn.exe
    echo Install the companion app's extra dependencies first:
    echo   pip install -r reactor_control_room\requirements.txt
    pause
    exit /b 1
)

echo Starting the Reactor Control Room backend...
echo Close this window ^(or press Ctrl+C^) to stop the server.
echo.

REM uvicorn does not auto-open a browser the way Streamlit does, so open it
REM ourselves in the background, giving the server a couple seconds to bind
REM its port first. Runs as a separate spawned process so it doesn't block the
REM foreground uvicorn command below. Uses `ping` rather than `timeout` for the
REM delay: `timeout` requires a real console input handle and errors out
REM ("Input redirection is not supported") in some launch contexts; `ping`
REM never touches stdin, so it works everywhere.
start "" cmd /c "ping -n 3 127.0.0.1 >nul && start http://127.0.0.1:8000/"

".venv\Scripts\uvicorn.exe" reactor_control_room.backend.app:app --port 8000

REM Keeps the window open after uvicorn exits (including an immediate crash)
REM so an error message is readable instead of the window flashing shut.
pause
