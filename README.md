# basic_litho_sim

Interactive Marimo notebooks and helper library code for exploring optics concepts used in lithography.

## Repository layout

- `content/`: Marimo notebooks and notebook assets.
- `src/goodoptics/`: reusable numerical helpers for Fourier optics, shapes, plotting, and Zernikes.
- `tests/`: automated test coverage for the library code.
- `docs/`: exported static notebook output for GitHub Pages.

## Development setup

Create the local virtual environment and install dependencies:

```powershell
.\setup_env.bat
```

Activate the environment in later sessions:

```powershell
.\activate_venv.bat
```

## Common tasks

Run the test suite:

```powershell
python -m pytest -q
```

Open a notebook for editing:

```powershell
marimo edit content\notebook_index.py
```

Build the library wheel:

```powershell
python -m build --wheel
```

Export the notebooks for GitHub Pages:

```powershell
.\build_notebooks.bat
```
