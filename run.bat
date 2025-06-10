@echo off
echo Starting User Authentication Application...

REM Run the application in a new window
start cmd /k python start_server.py

REM Wait a moment for the server to start
timeout /t 3 /nobreak

REM Open the web browser
start http://localhost:5000

echo Application started. The webpage should open automatically.
echo You can close this window now. 