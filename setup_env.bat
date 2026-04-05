@echo off
REM Create a Python virtual environment and install dependencies

cd /d "%~dp0"

REM Create the virtual environment
python -m venv litho_sim_venv

REM Activate the environment
call litho_sim_venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install project and development dependencies from pyproject.toml
python -m pip install -e ".[dev]"

echo.
echo Environment setup complete!
echo To activate it later, run:
echo     call litho_sim_venv\Scripts\activate
