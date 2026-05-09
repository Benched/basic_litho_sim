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

## Testing strategy

The test suite is split into two layers:

- Library tests in `tests/` check the numerical behavior of the reusable `goodoptics` modules.
- Notebook integration tests in `tests/test_notebooks.py` treat the main Marimo notebooks as entry points and verify that they start in headless mode, export to HTML/WASM, and keep using the local `goodoptics` bootstrap instead of legacy bare helper imports.

Use these checks as a simple rule of thumb:

- Run `python -m pytest -q` before merging any change.
- Run `python -m pytest -q tests\test_notebooks.py` after changing notebook imports, notebook bootstrap code, the `src/` package layout, or notebook export behavior.
- Rebuild the static notebooks with `.\build_notebooks.bat` when a notebook change affects rendered content and you want a final manual check of the GitHub Pages output.
