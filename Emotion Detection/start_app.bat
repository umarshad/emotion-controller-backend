@echo off
cd /d "%~dp0"
echo Starting Streamlit Emotion Detection App...
echo.
python -m streamlit run detection.py --server.port 8502
pause

