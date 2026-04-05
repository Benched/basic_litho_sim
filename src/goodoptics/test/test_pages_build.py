import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
CONTENT_DIR = REPO_ROOT / "content"
OPTICS_NOTEBOOK = CONTENT_DIR / "notebook_optics_basics.py"
GRATINGS_NOTEBOOK = CONTENT_DIR / "notebook_zernikes_and_gratings.py"
INTRO_NOTEBOOK = CONTENT_DIR / "notebook_introduction.py"
WHEEL_MANIFEST = DOCS_DIR / "goodoptics-wheel.json"
PUBLISHED_FIGURES_DIR = DOCS_DIR / "content" / "figures"
HTML_EXPORTS = (
    DOCS_DIR / "index.html",
    DOCS_DIR / "notebook_introduction.html",
    DOCS_DIR / "notebook_optics_basics.html",
    DOCS_DIR / "notebook_zernikes_and_gratings.html",
    DOCS_DIR / "notebook_zernikes_and_gratings_editable.html",
)
FORBIDDEN_DOC_SUBSTRINGS = (
    "https://benched.github.io/basic_litho_sim/",
    "C:\\Users\\",
)


def test_pages_notebooks_use_manifest_backed_wheel_loading():
    for notebook_path in (OPTICS_NOTEBOOK, GRATINGS_NOTEBOOK):
        notebook_source = notebook_path.read_text(encoding="utf-8")
        assert "goodoptics-wheel.json" in notebook_source
        assert "https://benched.github.io/basic_litho_sim/" not in notebook_source
        assert "goodoptics-0.1.0-py3-none-any.whl" not in notebook_source


def test_introduction_notebook_avoids_absolute_local_image_paths():
    intro_source = INTRO_NOTEBOOK.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in intro_source
    assert "content/figures/npn_transistor.png" in intro_source
    assert "content/figures/pnp_transistor.png" in intro_source
    assert "content/figures/CMOS_Inverter.svg" in intro_source


def test_docs_wheel_manifest_matches_exported_wheel():
    assert WHEEL_MANIFEST.exists(), "The Pages build should publish a wheel manifest in docs."
    manifest = json.loads(WHEEL_MANIFEST.read_text(encoding="utf-8"))
    wheel_filename = manifest["wheel_filename"]
    assert wheel_filename.startswith("goodoptics-")
    assert wheel_filename.endswith(".whl")
    assert (DOCS_DIR / wheel_filename).exists(), "The wheel referenced by the manifest is missing from docs."


def test_docs_publish_local_figure_assets_for_static_notebooks():
    expected_figures = (
        PUBLISHED_FIGURES_DIR / "npn_transistor.png",
        PUBLISHED_FIGURES_DIR / "pnp_transistor.png",
        PUBLISHED_FIGURES_DIR / "CMOS_Inverter.svg",
    )
    for figure_path in expected_figures:
        assert figure_path.exists(), f"Expected figure asset {figure_path.name} is missing from docs."


def test_exported_pages_do_not_contain_hardcoded_deployment_or_local_paths():
    for html_path in HTML_EXPORTS:
        assert html_path.exists(), f"Expected exported page {html_path.name} is missing."
        html_text = html_path.read_text(encoding="utf-8")
        for forbidden_substring in FORBIDDEN_DOC_SUBSTRINGS:
            assert forbidden_substring not in html_text, (
                f"Found forbidden path '{forbidden_substring}' in {html_path.name}."
            )
