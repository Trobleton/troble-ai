@echo off
cd /d %~dp0
call ..\.venv\Scripts\activate.bat
python check_audio_devices.py
pause