@echo off
REM =====================================================
REM  Build Marimo notebooks and prepare GitHub Pages docs
REM =====================================================


cd /d "%~dp0"
call litho_sim_venv\Scripts\activate


if not exist docs mkdir docs

echo Cleaning docs folder...
del /q docs\*.* >nul 2>&1

rmdir /s /q docs
mkdir docs

echo Building wheel...
python -m build --wheel

for %%f in (dist\*.whl) do copy /Y "%%f" "docs\" >nul

echo Exporting Marimo notebooks...
marimo export html-wasm content\notebook_index.py -o docs\index.html
marimo export html-wasm content\notebook_introduction.py -o docs\notebook_introduction.html
marimo export html-wasm content\notebook_optics_basics.py -o docs\notebook_optics_basics.html
marimo export html-wasm content\notebook_zernikes_and_gratings.py -o docs\notebook_zernikes_and_gratings.html --editable

echo.
echo ✅ Build complete! All files ready in the docs folder.
echo.

REM Keep console open
cmd /k