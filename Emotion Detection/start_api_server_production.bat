@echo off
cd /d "%~dp0"
echo ============================================================
echo Emotion Detection API Server - Production Mode
echo ============================================================
echo.

REM Check if .env file exists
if exist .env (
    echo Loading environment variables from .env file...
    REM Note: Windows batch doesn't natively support .env files
    REM For production, set environment variables in system or use python-dotenv
    echo.
    echo To use .env file, install python-dotenv:
    echo   pip install python-dotenv
    echo.
    echo Or set environment variables manually:
    echo   set API_HOST=0.0.0.0
    echo   set API_PORT=5000
    echo   set EMOTION_API_KEYS=your-production-keys-here
    echo.
) else (
    echo .env file not found. Using defaults or system environment variables.
    echo.
    echo For production, create .env file from .env.example
    echo.
)

echo Production Configuration:
echo   - Host: 0.0.0.0 (listens on all interfaces)
echo   - Port: 5000 (or set API_PORT environment variable)
echo   - API Keys: Set EMOTION_API_KEYS environment variable
echo   - Log Level: INFO (or set LOG_LEVEL environment variable)
echo.
echo Make sure:
echo   1. Firewall allows connections on port 5000
echo   2. API keys are set in EMOTION_API_KEYS environment variable
echo   3. Your computer and mobile devices are on the same network
echo.
echo Finding your IP address for mobile device connection...
python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print('Your IP address:', s.getsockname()[0]); s.close()" 2>nul
if errorlevel 1 (
    echo Could not determine IP address automatically.
    echo Find it manually: ipconfig (Windows) or ifconfig (Mac/Linux)
)
echo.
echo ============================================================
echo Starting API Server...
echo ============================================================
echo.

python api_server.py

pause


