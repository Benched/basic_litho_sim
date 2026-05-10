# Agent Instructions

## Project Shape

- Source notebooks live in `content/*.py`.
- GitHub Pages output lives in `docs/` and is intentionally committed.
- The helper Python package lives under `src/goodoptics`.
- Tests live in `tests/`.

## Notebook Docs Build

- Do not hand-edit generated `docs/*.html` files.
- To change notebook content, edit the corresponding `content/notebook_*.py` file and regenerate docs.
- Use `cmd /c build_notebooks.bat` to rebuild the committed deployment output.
- The build must copy `content/figures` to `docs/content/figures`; otherwise images in the introduction notebook break after deployment.
- The build should fail fast. If cleanup, wheel building, notebook export, or asset copying fails, stop and report the failing step.
- Do not create alternate deployment folders such as `docs_build` unless the user explicitly asks for that.

## Generated Output Checks

After rebuilding `docs/`, verify the generated introduction page:

- `docs/notebook_introduction.html` should not contain `app._unparsable_cell`.
- `docs/notebook_introduction.html` should not contain absolute local checkout paths such as `C:\Users\...`.
- `docs/content/figures` should contain the local introduction images.
- `docs/CLAUDE.md` should not be present.

Also verify the generated wheel:

- `docs/goodoptics-0.1.0-py3-none-any.whl` should not contain `goodoptics/test`.

## Validation

Before committing or opening a PR, run:

```bat
cmd /c build_notebooks.bat
litho_sim_venv\Scripts\python -m pytest -q
```

If the virtualenv command is blocked by local permissions, rerun it with the normal project approval flow rather than switching to a different Python environment. 
When adding new pages or behaviors that affect build, update the build_notebooks.bat file accordingly.

## Cleanup Problems

If `docs` cleanup fails because a generated asset cannot be removed:

- Stop and report the exact stuck file.
- Do not continue with a partial `docs` directory.
- Do not stage partially regenerated docs.

## Git Hygiene

- Stage only files related to the requested task.
- For docs rebuilds, this usually means `build_notebooks.bat` and `docs/`.
- Do not revert unrelated user changes.
