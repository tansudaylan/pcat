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
        "examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py",
        "examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py",
        "examples/roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog.py",
        "examples/catalog_association_completeness_purity/catalog_association_completeness_purity.py",
    ]

    for relative_path in relative_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file()


def test_every_example_script_has_a_notebook():
    examples = REPOSITORY_ROOT / "examples"
    script_only_automation = {examples / "run_all_examples.py"}
    for script in examples.rglob("*.py"):
        if script in script_only_automation:
            continue
        notebook = script.with_suffix(".ipynb")
        assert notebook.is_file(), f"Missing notebook for {script.relative_to(examples)}"
        print(f"Reading from {notebook}...")
        document = json.loads(notebook.read_text())
        assert document["nbformat"] == 4
        assert {cell["cell_type"] for cell in document["cells"]} == {"code", "markdown"}
        for cell in document["cells"]:
            assert cell["id"]


def test_figure_examples_display_visuals_in_notebooks():
    examples = REPOSITORY_ROOT / "examples"
    figure_examples = (
        "gaussian_mixture_catalog/gaussian_mixture_catalog",
        "chandra_point_source_catalog/chandra_point_source_catalog",
        "simulated_hst_strong_lens/simulated_hst_strong_lens",
        "daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources",
        "daylan+2018_strong_lens_subhalos/daylan+2018_strong_lens_subhalos",
        "voigt_spectral_line_catalog/voigt_spectral_line_catalog",
        "roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog",
        "catalog_association_completeness_purity/catalog_association_completeness_purity",
        "simulated_rubin_cluster_lens/simulated_rubin_cluster_lens",
        "portillo+2017_crowded_sdss_m2/portillo+2017_crowded_sdss_m2",
        "feder+2020_multiband_sdss_deblending/feder+2020_multiband_sdss_deblending",
        "butler+2022_spire_sz_component_separation/butler+2022_spire_sz_component_separation",
        "feder+2023_point_diffuse_spire/feder+2023_point_diffuse_spire",
        "hall+2026_herschel_dsfg_multiplicity/hall+2026_herschel_dsfg_multiplicity",
    )
    for name in figure_examples:
        notebook = examples / f"{name}.ipynb"
        print(f"Reading from {notebook}...")
        cells = json.loads(notebook.read_text())["cells"]
        code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        assert "display_example_visuals(" in code or "display(Image(" in code, name


def test_verified_pcat_publications_have_examples():
    examples = REPOSITORY_ROOT / "examples"
    publication_examples = {
        "10.3847/1538-4357/aa679e": "daylan+2017_fermi_point_sources",
        "10.3847/1538-3881/aa8565": "portillo+2017_crowded_sdss_m2",
        "10.3847/1538-4357/aaa1f2": "daylan+2018_strong_lens_subhalos",
        "10.3847/1538-3881/ab74cf": "feder+2020_multiband_sdss_deblending",
        "10.3847/1538-4357/ac6c04": "butler+2022_spire_sz_component_separation",
        "10.3847/1538-3881/ace69b": "feder+2023_point_diffuse_spire",
        "10.3847/1538-4357/ae1e7a": "hall+2026_herschel_dsfg_multiplicity",
    }
    index = (examples / "README.md").read_text()

    for doi, directory in publication_examples.items():
        root = examples / directory
        assert root.is_dir(), directory
        assert (root / f"{directory}.py").is_file(), directory
        notebook = root / f"{directory}.ipynb"
        assert notebook.is_file(), directory
        assert doi in index, doi
        document = json.loads(notebook.read_text())
        for cell in document["cells"]:
            assert cell["metadata"]["id"] == cell["id"]
            expected_language = "python" if cell["cell_type"] == "code" else "markdown"
            assert cell["metadata"]["language"] == expected_language

        code = "\n".join(
            "".join(cell["source"])
            for cell in document["cells"]
            if cell["cell_type"] == "code"
        )
        if directory == "daylan+2017_fermi_point_sources":
            for posterior_product in (
                "read_posterior",
                "posterior.listpostcntpmodl",
                "posterior.listpostdictelem",
                "posterior.listpostnumbelem",
                "posterior.listpostlliktotl",
            ):
                assert posterior_product in code
        else:
            assert "load_example_posterior" in code


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
    assert "make_image_sequence_animation(" in source
    assert "[cutout['data'] for cutout in cutouts]" in source
    assert "rubin_dp1_confirmed_lens_cutouts.gif" in source
    assert "display(Image(filename=str(cutout_animation_path)))" in source


def test_simulated_rubin_cluster_notebook_contains_rendered_visuals():
    notebook = REPOSITORY_ROOT / "examples/simulated_rubin_cluster_lens/simulated_rubin_cluster_lens.ipynb"
    print(f"Reading from {notebook}...")
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in cells
        for output in cell.get("outputs", [])
    )
    assert image_count >= 2
    assert "run_lens_image_pipeline(" in code
    assert "savefig" not in code
    assert "matplotlib" not in code
    assert "plt." not in code


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