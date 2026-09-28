@echo off
rem Runs the backend and starts it again whenever it exits with an error: the
rem watchdog in src/app.py exits with code 1 when the pipeline dies or stalls.
rem A clean stop (Ctrl+C) ends this script too. To start with Windows, point a
rem Task Scheduler task ("At log on") at this file -- see README.
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

:loop
%PY% src\app.py
if %errorlevel% equ 0 goto :eof
echo [%date% %time%] Backend exited with code %errorlevel%; restarting in 5 s...
rem Not `timeout`: it fails without a console, as under Task Scheduler.
ping -n 6 127.0.0.1 >nul
goto loop
