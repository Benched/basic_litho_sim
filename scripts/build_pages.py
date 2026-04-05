from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO_ROOT / "content"
DIST_DIR = REPO_ROOT / "dist"
DOCS_DIR = REPO_ROOT / "docs"
DOCS_CONTENT_DIR = DOCS_DIR / "content"
FIGURES_DIR = CONTENT_DIR / "figures"
PUBLISHED_FIGURES_DIR = DOCS_CONTENT_DIR / "figures"
WHEEL_MANIFEST_PATH = DOCS_DIR / "goodoptics-wheel.json"
UNPUBLISHED_DOCS_FILES = (
    DOCS_DIR / "CLAUDE.md",
)
FORBIDDEN_DOC_SUBSTRINGS = (
    "https://benched.github.io/basic_litho_sim/",
    "C:\\Users\\",
)
NOTEBOOK_EXPORTS = (
    ("notebook_index.py", "index.html", None),
    ("notebook_introduction.py", "notebook_introduction.html", None),
    ("notebook_optics_basics.py", "notebook_optics_basics.html", None),
    ("notebook_zernikes_and_gratings.py", "notebook_zernikes_and_gratings.html", None),
    ("notebook_zernikes_and_gratings.py", "notebook_zernikes_and_gratings_editable.html", "edit"),
)


def run_command(*args: str) -> None:
    print(f"> {' '.join(args)}")
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def clean_docs_dir() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")


def build_wheel() -> Path:
    run_command(sys.executable, "-m", "build", "--wheel")
    wheels = sorted(DIST_DIR.glob("goodoptics-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not wheels:
        raise RuntimeError("No built wheel was found in dist after `python -m build --wheel`.")
    return wheels[0]


def write_wheel_manifest(wheel_path: Path) -> Path:
    target_wheel_path = DOCS_DIR / wheel_path.name
    shutil.copy2(wheel_path, target_wheel_path)
    WHEEL_MANIFEST_PATH.write_text(
        json.dumps({"wheel_filename": wheel_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_wheel_path


def publish_static_assets() -> None:
    if FIGURES_DIR.exists():
        shutil.copytree(FIGURES_DIR, PUBLISHED_FIGURES_DIR, dirs_exist_ok=True)


def export_notebooks() -> None:
    for notebook_name, output_name, mode in NOTEBOOK_EXPORTS:
        command = [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(CONTENT_DIR / notebook_name),
            "-o",
            str(DOCS_DIR / output_name),
        ]
        if mode is not None:
            command.extend(["--mode", mode])
        run_command(*command)
    for unpublished_path in UNPUBLISHED_DOCS_FILES:
        unpublished_path.unlink(missing_ok=True)


def validate_docs() -> None:
    html_files = [DOCS_DIR / output_name for _, output_name, _ in NOTEBOOK_EXPORTS]
    for html_path in html_files:
        if not html_path.exists():
            raise RuntimeError(f"Expected exported page {html_path} was not created.")
        html_text = html_path.read_text(encoding="utf-8")
        for forbidden_substring in FORBIDDEN_DOC_SUBSTRINGS:
            if forbidden_substring in html_text:
                raise RuntimeError(f"Forbidden path '{forbidden_substring}' found in {html_path.name}.")

    manifest = json.loads(WHEEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    wheel_filename = manifest["wheel_filename"]
    if not (DOCS_DIR / wheel_filename).exists():
        raise RuntimeError(f"Wheel manifest points to missing wheel {wheel_filename}.")

    for figure_path in FIGURES_DIR.glob("*"):
        published_path = PUBLISHED_FIGURES_DIR / figure_path.name
        if not published_path.exists():
            raise RuntimeError(f"Expected figure asset {published_path} was not published to docs.")


def main() -> None:
    clean_docs_dir()
    wheel_path = build_wheel()
    write_wheel_manifest(wheel_path)
    publish_static_assets()
    export_notebooks()
    validate_docs()
    print(f"Pages build ready in {DOCS_DIR}")


if __name__ == "__main__":
    main()
