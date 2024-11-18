@echo off
REM Drag and drop a file onto this .bat file
py "%~dp0test_labeling.py" "%~1"
pause
