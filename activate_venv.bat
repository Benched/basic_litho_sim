@echo off
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

echo Using virtual environment at "%VENV_ACTIVATE%"
"%ComSpec%" /k ""%VENV_ACTIVATE%""
