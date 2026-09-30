import json
import inspect
import re
import tomllib
from pathlib import Path

from pcat import plotting, sampling
from pcat.plotting import POSTERIOR_ANIMATION_PANELS
from pcat.main import retr_pathrun


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
        "likelihoods",
        "outputs",
        "sampling",
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
    getting_started = (DOCS_ROOT / "getting_started.rst").read_text().lower()

    assert metadata["project"]["optional-dependencies"]["docs"] == [
        "sphinx>=8",
        "sphinx-rtd-theme>=3",
    ]
    assert requirements == metadata["project"]["optional-dependencies"]["docs"]
    for dependency in metadata["project"]["dependencies"]:
        assert dependency.lower() in getting_started
    for dependency in metadata["project"]["optional-dependencies"]["examples"]:
        assert dependency.lower() in getting_started


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


def test_documentation_separates_likelihoods_from_catalog_products():
    capabilities = (DOCS_ROOT / "capabilities.rst").read_text()
    likelihoods = (DOCS_ROOT / "likelihoods.rst").read_text()

    assert "Point sources, extended emission, foreground light, and gravitational lenses" in capabilities
    assert "Unbinned Gaussian-mixture likelihood" in capabilities
    assert "without first accumulating them into image pixels" in capabilities
    assert inspect.signature(sampling.init).parameters["boolcondcatl"].default is True
    assert "default ``boolcondcatl=True``" in capabilities
    assert re.search(r"Set\s+``boolcondcatl=False``", likelihoods)
    assert "does\nnot condense one-dimensional line, flare, or Keplerian catalogs" in likelihoods
    assert "Neither\ncondensation nor catalog association is a likelihood" in likelihoods
    assert "The public ``typeexpr=\"gmix\"`` example currently" in likelihoods


def test_documented_api_signatures_match_public_entry_points():
    api = (DOCS_ROOT / "api.rst").read_text()

    assert list(inspect.signature(sampling.init).parameters)[:1] == ["dictglob"]
    assert list(inspect.signature(sampling.sample_parallel).parameters)[:2] == [
        "dictpcatinptvari",
        "listlablcnfg",
    ]
    assert list(inspect.signature(plotting.plot_grid).parameters)[:4] == [
        "path",
        "name",
        "listpara",
        "listlablparatotl",
    ]
    assert "pcat.sampling.init(dictglob, **options)" in api
    assert "pcat.sampling.sample_parallel(dictpcatinptvari, listlablcnfg" in api
    assert "scalpara=None, truepara=None, join=False" in api


def test_documented_output_layout_matches_runtime(tmp_path):
    outputs = (DOCS_ROOT / "outputs.rst").read_text()
    expected = tmp_path / "pcat_runs" / "example"

    assert Path(retr_pathrun(tmp_path, "example")) == expected
    assert "project_root / \"pcat_runs\" / \"gaussian_mixture_catalog\"" in outputs
    assert "retained_sample_count = state.numbsamp" in outputs


def test_documentation_lists_every_maintained_example():
    examples_root = REPOSITORY_ROOT / "examples"
    examples_page = (DOCS_ROOT / "examples.rst").read_text()
    script_directories = {
        path.parent.name
        for path in examples_root.glob("*/*.py")
        if "archive" not in path.parts
    }
    notebook_only_directories = {
        "legacy_external_analysis_commands",
        "rubin_dp1_confirmed_strong_lenses",
        "simulated_rubin_cluster_lens",
    }

    for directory in script_directories | notebook_only_directories:
        assert directory in examples_page, directory


def test_documentation_lists_stable_public_helpers():
    api = (DOCS_ROOT / "api.rst").read_text()

    for name in (
        "sample_allesfitter_pcat",
        "plot_population_grid",
        "binomial_wilson_interval",
    ):
        assert name in api


def test_documented_posterior_collage_matches_generator():
    examples = (DOCS_ROOT / "examples.rst").read_text()

    assert len(POSTERIOR_ANIMATION_PANELS) == 12
    assert "Twelve genuinely changing inference views from seven maintained workflows" in examples
    for domain in ("strong lenses", "stellar flares", "simulated spectral lines"):
        assert domain in examples


def test_documentation_distinguishes_posterior_and_candidate_animations():
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    outputs = (DOCS_ROOT / "outputs.rst").read_text()
    readme = (REPOSITORY_ROOT / "README.md").read_text()

    for source in (sampling, outputs, readme):
        assert "boolmakeanimprop" in source
        assert "proposal_candidates.gif" in source
    assert "Candidate frames are proposal diagnostics, not posterior samples" in sampling
    assert "including rejected candidates" in outputs
    assert "examples/proposal_state_animation/visuals/post/anim/proposal_candidates.gif" in readme


def test_readme_lists_daylan_applications_first_without_reproduction_language():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    applications = readme.split("## Applications\n", 1)[1].split("\n## ", 1)[0]
    headings = [line for line in applications.splitlines() if line.startswith("### ")]

    assert headings[:2] == [
        "### [Daylan et al. 2017 Fermi-LAT point-source populations](examples/daylan+2017_fermi_point_sources/)",
        "### [Daylan et al. 2018 Strong-lens subhalo catalogs](examples/daylan+2018_strong_lens_subhalos/)",
    ]
    assert "reproduc" not in applications.lower()


def test_burn_in_options_and_comparison_are_documented():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    outputs = (DOCS_ROOT / "outputs.rst").read_text()
    examples = (DOCS_ROOT / "examples.rst").read_text()

    for source in (readme, sampling):
        for option in ("numbburn", "booladaptstdp", "boolburntmpr", "factburntmpr"):
            assert option in source
        assert "gradient descent" in source.lower()
    assert "listpostfacttmpr" in sampling
    assert "listpostfacttmpr" in outputs
    assert "burn_in_strategies" in examples
    for name in (
        "burn_in_posterior_comparison.png",
        "burn_in_performance_comparison.png",
        "burn_in_temperature_schedule.png",
    ):
        assert name in readme
