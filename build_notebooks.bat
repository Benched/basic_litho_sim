@echo off
REM =====================================================
REM  Build Marimo notebooks and prepare GitHub Pages docs
REM =====================================================


cd /d "%~dp0"
call litho_sim_venv\Scripts\activate

echo Building GitHub Pages site...
python scripts\build_pages.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete! All files ready in the docs folder.
echo.

REM Keep console open
cmd /k
