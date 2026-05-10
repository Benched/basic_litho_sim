@echo off
REM =====================================================
REM  Build Marimo notebooks and prepare GitHub Pages docs
REM =====================================================


cd /d "%~dp0"
set "VENV_ACTIVATE=%~dp0litho_sim_venv\Scripts\activate.bat"

if not exist "%VENV_ACTIVATE%" (
    for /d %%D in ("%~dp0..\*") do (
        if exist "%%~fD\litho_sim_venv\Scripts\activate.bat" (
            set "VENV_ACTIVATE=%%~fD\litho_sim_venv\Scripts\activate.bat"
        )
    )
)

if not exist "%VENV_ACTIVATE%" (
    echo Virtual environment not found in this worktree or a sibling checkout.
    echo Run setup_env.bat first to create a local environment.
    exit /b 1
)
call "%VENV_ACTIVATE%"
if errorlevel 1 exit /b %errorlevel%


echo Cleaning docs folder...
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath 'docs') { Remove-Item -LiteralPath 'docs' -Recurse -Force -ErrorAction Stop }; New-Item -ItemType Directory -Path 'docs' -Force | Out-Null"
if errorlevel 1 exit /b %errorlevel%

echo Cleaning build artifacts...
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach ($path in 'build','dist') { if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop } }"
if errorlevel 1 exit /b %errorlevel%

echo Building wheel...
python -m build --wheel
if errorlevel 1 exit /b %errorlevel%

for %%f in (dist\*.whl) do copy /Y "%%f" "docs\" >nul
if errorlevel 1 exit /b %errorlevel%

echo Exporting Marimo notebooks...
python -m marimo export html-wasm content\notebook_index.py -o docs\index.html
if errorlevel 1 exit /b %errorlevel%
python -m marimo export html-wasm content\notebook_introduction.py -o docs\notebook_introduction.html
if errorlevel 1 exit /b %errorlevel%
python -m marimo export html-wasm content\notebook_optics_basics.py -o docs\notebook_optics_basics.html
if errorlevel 1 exit /b %errorlevel%
python -m marimo export html-wasm content\notebook_zernikes_and_gratings.py -o docs\notebook_zernikes_and_gratings.html
if errorlevel 1 exit /b %errorlevel%
python -m marimo export html-wasm content\notebook_zernikes_and_gratings.py -o docs\notebook_zernikes_and_gratings_editable.html --mode edit
if errorlevel 1 exit /b %errorlevel%

echo Copying notebook assets...
if exist docs\CLAUDE.md del /q docs\CLAUDE.md
if exist content\figures xcopy /E /I /Y content\figures docs\content\figures >nul
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete! All files ready in the docs folder.
echo.
