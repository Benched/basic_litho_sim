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
set "MARIMO_TMP=%TEMP%\basic_litho_sim_marimo_%RANDOM%%RANDOM%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -ItemType Directory -Path $env:MARIMO_TMP -Force | Out-Null"
if errorlevel 1 exit /b %errorlevel%
call :export_html content\notebook_introduction.py notebook_introduction.html
if errorlevel 1 goto build_failed
call :export_html content\notebook_optics_basics.py notebook_optics_basics.html
if errorlevel 1 goto build_failed
call :export_html content\notebook_zernikes_and_gratings.py notebook_zernikes_and_gratings.html
if errorlevel 1 goto build_failed
call :export_html content\notebook_zernikes_and_gratings.py notebook_zernikes_and_gratings_editable.html --mode edit
if errorlevel 1 goto build_failed
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath $env:MARIMO_TMP) { Remove-Item -LiteralPath $env:MARIMO_TMP -Recurse -Force -ErrorAction Stop }"
if errorlevel 1 goto build_failed

echo Copying notebook assets...
if exist docs\CLAUDE.md del /q docs\CLAUDE.md
if exist content\figures xcopy /E /I /Y content\figures docs\content\figures >nul
if errorlevel 1 exit /b %errorlevel%

echo Cleaning build artifacts...
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach ($path in 'build','dist') { if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop } }"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete! All files ready in the docs folder.
echo.

exit /b 0

:export_html
set "NOTEBOOK=%~1"
set "HTML_FILE=%~2"
set "EXTRA_ARGS=%~3 %~4 %~5 %~6 %~7 %~8 %~9"
set "EXPORT_DIR=%MARIMO_TMP%\%~n2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -ItemType Directory -Path $env:EXPORT_DIR -Force | Out-Null"
if errorlevel 1 exit /b %errorlevel%
python -m marimo export html-wasm "%NOTEBOOK%" -o "%EXPORT_DIR%\%HTML_FILE%" %EXTRA_ARGS%
if errorlevel 1 exit /b %errorlevel%
copy /Y "%EXPORT_DIR%\%HTML_FILE%" "docs\%HTML_FILE%" >nul
if errorlevel 1 exit /b %errorlevel%
exit /b 0

:build_failed
set "BUILD_EXIT=%errorlevel%"
if defined MARIMO_TMP powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath $env:MARIMO_TMP) { Remove-Item -LiteralPath $env:MARIMO_TMP -Recurse -Force }"
exit /b %BUILD_EXIT%
