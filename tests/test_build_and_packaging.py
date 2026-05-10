from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
BUILD_SCRIPT = ROOT / "build_notebooks.bat"
BUILD_AVAILABLE = importlib.util.find_spec("build") is not None
MARIMO_AVAILABLE = importlib.util.find_spec("marimo") is not None
IGNORED_COPY_PATTERNS = shutil.ignore_patterns(
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "*.pyc",
    "*.pyo",
)

DOC_EXPORTS = [
    (CONTENT_DIR / "notebook_index.py", "index.html", ()),
    (CONTENT_DIR / "notebook_introduction.py", "notebook_introduction.html", ()),
    (CONTENT_DIR / "notebook_optics_basics.py", "notebook_optics_basics.html", ()),
    (
        CONTENT_DIR / "notebook_zernikes_and_gratings.py",
        "notebook_zernikes_and_gratings.html",
        (),
    ),
    (
        CONTENT_DIR / "notebook_zernikes_and_gratings.py",
        "notebook_zernikes_and_gratings_editable.html",
        ("--mode", "edit"),
    ),
]

pytestmark = pytest.mark.skipif(
    not BUILD_AVAILABLE,
    reason="build is not installed",
)


def _normalized_batch_text(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower())


def _build_wheel(project_root: Path, outdir: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", outdir],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        "Wheel build failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    wheel_paths = sorted(outdir.glob("*.whl"))
    assert len(wheel_paths) == 1, f"Expected exactly one wheel in {outdir}, found {wheel_paths!r}"
    return wheel_paths[0]


@pytest.fixture(scope="module")
def clean_project_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project_root = tmp_path_factory.mktemp("project") / "repo"
    shutil.copytree(ROOT, project_root, ignore=IGNORED_COPY_PATTERNS)
    return project_root


@pytest.fixture(scope="module")
def built_wheel(
    clean_project_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    dist_dir = tmp_path_factory.mktemp("dist")
    return _build_wheel(clean_project_root, dist_dir)


def test_build_notebooks_script_exports_expected_targets() -> None:
    normalized = _normalized_batch_text(BUILD_SCRIPT)

    assert (
        "marimo export html-wasm content\\notebook_index.py -o docs\\index.html"
        in normalized
    )
    assert (
        'marimo export html-wasm "%notebook%" -o "%export_dir%\\%html_file%" %extra_args%'
        in normalized
    )
    assert (
        "copy /y content\\introduction_logic_viz.py docs\\content\\"
        in normalized
    )

    expected_helper_calls = [
        "call :export_html content\\notebook_introduction.py notebook_introduction.html",
        "call :export_html content\\notebook_optics_basics.py notebook_optics_basics.html",
        "call :export_html content\\notebook_zernikes_and_gratings.py notebook_zernikes_and_gratings.html",
        "call :export_html content\\notebook_zernikes_and_gratings.py notebook_zernikes_and_gratings_editable.html --mode edit",
    ]

    for command in expected_helper_calls:
        assert command in normalized, f"Missing export helper call in build_notebooks.bat: {command}"


@pytest.mark.skipif(not MARIMO_AVAILABLE, reason="marimo is not installed")
def test_docs_build_pipeline_smoke(
    tmp_path: Path,
    built_wheel: Path,
    clean_project_root: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    copied_wheel = docs_dir / built_wheel.name
    shutil.copy2(built_wheel, copied_wheel)

    temp_export_root = tmp_path / "marimo_exports"

    for export_index, (notebook_path, output_name, extra_args) in enumerate(DOC_EXPORTS):
        if export_index == 0:
            output_path = docs_dir / output_name
        else:
            export_dir = temp_export_root / Path(output_name).stem
            export_dir.mkdir(parents=True)
            output_path = export_dir / output_name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html-wasm",
                notebook_path,
                "-o",
                output_path,
                *extra_args,
            ],
            cwd=clean_project_root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

        assert result.returncode == 0, (
            f"{notebook_path.name} failed to export in docs build smoke test.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert output_path.exists(), f"Expected docs build output {output_name} to exist."
        exported_html = output_path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "<html" in exported_html, f"Expected {output_name} to contain HTML markup."
        if output_path.parent != docs_dir:
            shutil.copy2(output_path, docs_dir / output_name)

    produced_names = {path.name for path in docs_dir.iterdir()}
    expected_names = {output_name for _, output_name, _ in DOC_EXPORTS} | {built_wheel.name}
    assert expected_names.issubset(produced_names)
    assert "assets" in produced_names


def test_built_wheel_contains_expected_package_files(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = set(archive.namelist())

    assert "goodoptics/fourier.py" in names
    assert "goodoptics/shapes.py" in names
    assert "goodoptics/utils.py" in names
    assert "goodoptics/zernikes.py" in names
    assert not any(name.startswith("tests/") for name in names)
    assert not any(name.startswith("goodoptics/test/") for name in names)


def test_built_wheel_imports_goodoptics_modules(built_wheel: Path, tmp_path: Path) -> None:
    import_code = """
import sys
from pathlib import Path

wheel_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
sys.path = [
    str(wheel_path),
    *[
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != repo_root
    ],
]

import goodoptics.fourier as fourier
import goodoptics.plot as plot
import goodoptics.shapes as shapes
import goodoptics.utils as utils
import goodoptics.zernikes as zernikes

sample = shapes.top_hat_da(8, -0.5, 0.5, -1.0, 1.0)
assert sample.shape == (8,)
assert wheel_path.name in fourier.__file__
assert wheel_path.name in plot.__file__
assert wheel_path.name in shapes.__file__
assert wheel_path.name in utils.__file__
assert wheel_path.name in zernikes.__file__
"""

    result = subprocess.run(
        [sys.executable, "-c", import_code, built_wheel, ROOT],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        "Wheel import smoke test failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
