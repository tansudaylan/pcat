import json
import re
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def test_readme_local_images_exist():
    readme_path = REPOSITORY_ROOT / "README.md"
    print(f"Reading from {readme_path}...")
    image_paths = re.findall(r"!\[[^]]*\]\((?!https?://)([^)]+)\)", readme_path.read_text())

    assert image_paths
    for image_path in image_paths:
        assert (REPOSITORY_ROOT / image_path).is_file(), image_path


def test_documentation_pages_are_in_the_toctree():
    index = (DOCS_ROOT / "index.rst").read_text()
    pages = {path.stem for path in DOCS_ROOT.glob("*.rst")} - {"index", "prospec_migration"}

    assert pages == {
        "api",
        "capabilities",
        "examples",
        "getting_started",
        "outputs",
        "troubleshooting",
    }
    for page in pages:
        assert f"   {page}\n" in index


def test_documented_example_commands_resolve():
    relative_paths = [
        "examples/chandra_point_source_catalog/chandra_point_source_catalog.py",
        "examples/simulated_hst_strong_lens/simulated_hst_strong_lens.py",
        "examples/run_all_examples.py",
        "examples/daylan_2017_fermi_point_sources/daylan_2017_fermi_point_sources.py",
        "examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py",
        "examples/roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog.py",
        "examples/catalog_association_completeness_purity/catalog_association_completeness_purity.py",
        "examples/subpixel_psf_reconstruction/subpixel_psf_reconstruction.py",
    ]

    for relative_path in relative_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file()


def test_every_example_script_has_a_notebook():
    examples = REPOSITORY_ROOT / "examples"
    for script in examples.rglob("*.py"):
        notebook = script.with_suffix(".ipynb")
        assert notebook.is_file(), f"Missing notebook for {script.relative_to(examples)}"
        print(f"Reading from {notebook}...")
        document = json.loads(notebook.read_text())
        assert document["nbformat"] == 4
        assert {cell["cell_type"] for cell in document["cells"]} == {"code", "markdown"}
        for cell in document["cells"]:
            assert cell["metadata"]["language"] == (
                "python" if cell["cell_type"] == "code" else "markdown"
            )
            assert cell["metadata"]["id"] == cell["id"]


def test_figure_examples_display_visuals_in_notebooks():
    examples = REPOSITORY_ROOT / "examples"
    figure_examples = (
        "gaussian_mixture_catalog/gaussian_mixture_catalog",
        "chandra_point_source_catalog/chandra_point_source_catalog",
        "simulated_hst_strong_lens/simulated_hst_strong_lens",
        "daylan_2017_fermi_point_sources/daylan_2017_fermi_point_sources",
        "daylan_2018_strong_lens_subhalos/daylan_2018_strong_lens_subhalos",
        "voigt_spectral_line_catalog/voigt_spectral_line_catalog",
        "roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog",
        "catalog_association_completeness_purity/catalog_association_completeness_purity",
        "subpixel_psf_reconstruction/subpixel_psf_reconstruction",
        "simulated_rubin_cluster_lens/simulated_rubin_cluster_lens",
        "run_all_examples",
    )
    for name in figure_examples:
        notebook = examples / f"{name}.ipynb"
        print(f"Reading from {notebook}...")
        cells = json.loads(notebook.read_text())["cells"]
        code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        assert "display_example_visuals(" in code or "display(Image(" in code, name


def test_rubin_dp1_notebook_preserves_real_data_scope():
    notebook = REPOSITORY_ROOT / "examples/rubin_dp1_confirmed_strong_lenses/rubin_dp1_confirmed_strong_lenses.ipynb"
    print(f"Reading from {notebook}...")
    source = notebook.read_text()
    assert "get_siav2_service('dp1')" in source
    assert "cutout-sync-maskedimage" in source
    assert "otype IN ('gLS', 'gLe')" in source
    assert "Local fallback data are intentionally not substituted" in source
    assert "SIMBAD is curated but incomplete" in source
    assert "gaussian_lens_log_likelihood" in source


def test_documentation_excludes_obsolete_interface_terms():
    obsolete_terms = {
        "setup.py install",
        "pathbase/imag",
        "PCAT_DATA_PATH/imag",
        "truenumbpnts",
        "Daylan+2016",
        "daylan2016",
    }
    source = "\n".join(path.read_text() for path in DOCS_ROOT.glob("*.rst"))

    assert all(term not in source for term in obsolete_terms)


def test_documentation_dependencies_are_declared():
    metadata = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    requirements = (DOCS_ROOT / "requirements.txt").read_text().splitlines()

    assert metadata["project"]["optional-dependencies"]["docs"] == [
        "sphinx>=8",
        "sphinx-rtd-theme>=3",
    ]
    assert requirements == metadata["project"]["optional-dependencies"]["docs"]


def test_documentation_covers_primary_capabilities():
    capabilities = (DOCS_ROOT / "capabilities.rst").read_text()

    required_terms = {
        'typeexpr="gener"',
        "retr_llik",
        "lghtpnts",
        "lensed emission",
        "lghtlinevoig",
        "spectral data",
    }
    missing_terms = required_terms - {
        term for term in required_terms if term in capabilities
    }
    assert not missing_terms