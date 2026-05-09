from __future__ import annotations

import importlib.util
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
NOTEBOOKS_WITHOUT_GOODOPTICS = [
    CONTENT_DIR / "notebook_index.py",
    CONTENT_DIR / "notebook_introduction.py",
]
GOODOPTICS_NOTEBOOKS = [
    CONTENT_DIR / "notebook_fourier_design_and_test.py",
    CONTENT_DIR / "notebook_optics_basics.py",
    CONTENT_DIR / "notebook_zernikes.py",
    CONTENT_DIR / "notebook_zernikes_and_gratings.py",
]
NOTEBOOKS = NOTEBOOKS_WITHOUT_GOODOPTICS + GOODOPTICS_NOTEBOOKS
MARIMO_AVAILABLE = importlib.util.find_spec("marimo") is not None

pytestmark = pytest.mark.skipif(not MARIMO_AVAILABLE, reason="marimo is not installed")

GOODOPTICS_IMPORT_RE = re.compile(
    r"^\s*from goodoptics(?:\.[a-z_]+)? import\b",
    re.MULTILINE,
)
LEGACY_IMPORT_PATTERNS = [
    re.compile(r"^\s*import shapes\b", re.MULTILINE),
    re.compile(r"^\s*import fourier\b", re.MULTILINE),
    re.compile(r"^\s*import utils\b", re.MULTILINE),
    re.compile(r"^\s*import plot\b", re.MULTILINE),
    re.compile(r"^\s*import zernikes\b", re.MULTILINE),
    re.compile(r"^\s*from zernikes import\b", re.MULTILINE),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _marimo_command(*args: object) -> list[str]:
    return [sys.executable, "-m", "marimo", *(str(arg) for arg in args)]


def _fetch_status(url: str) -> int | None:
    try:
        with urlopen(url, timeout=1) as response:
            return int(response.status)
    except (HTTPError, URLError, TimeoutError, OSError):
        return None


def _wait_for_headless_notebook(notebook_path: Path, timeout_seconds: float = 20.0) -> None:
    port = _free_port()
    process = subprocess.Popen(
        _marimo_command(
            "run",
            notebook_path,
            "--headless",
            "--check",
            "--no-token",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        deadline = time.monotonic() + timeout_seconds
        url = f"http://127.0.0.1:{port}/"

        while time.monotonic() < deadline:
            if process.poll() is not None:
                output, _ = process.communicate(timeout=5)
                pytest.fail(
                    f"{notebook_path.name} exited before serving HTTP 200.\n\n{output}"
                )

            if _fetch_status(url) == 200:
                return

            time.sleep(0.5)

        pytest.fail(f"{notebook_path.name} did not become ready within {timeout_seconds} seconds.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_starts_in_headless_mode(notebook_path: Path) -> None:
    _wait_for_headless_notebook(notebook_path)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_exports_to_html_wasm(notebook_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / f"{notebook_path.stem}.html"
    result = subprocess.run(
        _marimo_command("export", "html-wasm", notebook_path, "-o", output_path),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"{notebook_path.name} failed to export.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert output_path.exists(), f"Expected export output for {notebook_path.name} to exist."
    exported_html = output_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "<html" in exported_html, f"Expected {output_path.name} to contain HTML markup."


@pytest.mark.parametrize("notebook_path", GOODOPTICS_NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_uses_local_goodoptics_bootstrap(notebook_path: Path) -> None:
    notebook_text = notebook_path.read_text(encoding="utf-8")

    assert GOODOPTICS_IMPORT_RE.search(notebook_text), (
        f"{notebook_path.name} should import from goodoptics rather than bare helper modules."
    )
    assert 'Path.cwd() / "src"' in notebook_text, (
        f"{notebook_path.name} should look for a local src directory during local execution."
    )
    assert 'Path.cwd().parent / "src"' in notebook_text, (
        f"{notebook_path.name} should support running from the content directory as well as the repo root."
    )
    assert "sys.path.insert(0, src_path_str)" in notebook_text, (
        f"{notebook_path.name} should prepend the discovered src directory to sys.path."
    )

    for pattern in LEGACY_IMPORT_PATTERNS:
        assert not pattern.search(notebook_text), (
            f"{notebook_path.name} still contains a legacy bare import matching {pattern.pattern!r}."
        )
