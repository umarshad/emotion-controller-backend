@echo off
cd /d "%~dp0"
echo Starting Emotion Detection API Server...
echo.
echo Make sure you have installed all dependencies:
echo   pip install -r requirements.txt
echo.
echo Server will start on http://0.0.0.0:5000
echo.
echo For Android Emulator, use: http://10.0.2.2:5000
echo For iOS Simulator, use: http://localhost:5000
echo.
python api_server.py
pause

