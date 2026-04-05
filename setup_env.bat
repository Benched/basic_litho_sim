@echo off
REM Create a Python virtual environment and install dependencies

REM Create the virtual environment
python -m venv litho_sim_venv

REM Activate the environment
call litho_sim_venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo.
echo Environment setup complete!
echo To activate it later, run:
echo     call litho_sim_venv\Scripts\activate