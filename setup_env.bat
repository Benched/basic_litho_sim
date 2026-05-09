@echo off
REM Create a Python virtual environment and install dependencies

REM Create the virtual environment
python -m venv litho_sim_venv
if errorlevel 1 exit /b %errorlevel%

REM Activate the environment
if not exist "litho_sim_venv\Scripts\activate.bat" (
    echo Failed to create the virtual environment.
    exit /b 1
)
call "litho_sim_venv\Scripts\activate.bat"
if errorlevel 1 exit /b %errorlevel%

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo.
echo Environment setup complete!
echo To activate it later, run:
echo     .\activate_venv.bat
