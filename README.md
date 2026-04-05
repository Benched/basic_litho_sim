# basic_litho_sim

Interactive Marimo notebooks and a small `goodoptics` helper library for
exploring Fourier optics and lithography topics.

## Setup

Create the virtual environment and install the project with development tools:

```bat
setup_env.bat
```

To activate the environment later:

```bat
call litho_sim_venv\Scripts\activate
```

## Working With Notebooks

Open a notebook for editing with Marimo:

```bat
marimo edit <path_of_notebook>
```

## Testing

Run the test suite with:

```bat
litho_sim_venv\Scripts\python -m pytest
```

## Packaging

Build the wheel with:

```bat
python -m build --wheel
```

## GitHub Pages Build

Export the notebook site into `docs` with:

```bat
build_notebooks.bat
```

## Notes

TODO:
- automate build and serve
- expand usage instructions
- add a roadmap sketch
