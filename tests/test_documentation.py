import ast
import json
import inspect
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from pcat import plotting, sampling
from pcat.plotting import POSTERIOR_ANIMATION_PANELS
from pcat.main import retr_pathrun


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def test_pyproject_build_metadata_preserves_declared_requirements(tmp_path, monkeypatch):
    from email.parser import Parser
    import shutil
    from zipfile import ZipFile
    from packaging.requirements import Requirement
    from setuptools import build_meta

    for name in ("pyproject.toml", "README.md", "LICENSE"):
        source = REPOSITORY_ROOT / name
        print(f"Reading from {source}...")
        print(f"Writing to {tmp_path / name}...")
        shutil.copy2(source, tmp_path / name)
    print(f"Reading from {REPOSITORY_ROOT / 'pcat'}...")
    print(f"Writing to {tmp_path / 'pcat'}...")
    shutil.copytree(REPOSITORY_ROOT / "pcat", tmp_path / "pcat", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.chdir(tmp_path)
    metadata_path = tmp_path / "metadata"
    metadata_path.mkdir()
    directory = build_meta.prepare_metadata_for_build_wheel(str(metadata_path))
    path = metadata_path / directory / "METADATA"
    print(f"Reading from {path}...")
    generated = Parser().parsestr(path.read_text())
    path = tmp_path / "pyproject.toml"
    print(f"Reading from {path}...")
    configuration = tomllib.loads(path.read_text())
    assert "setuptools>=77" in configuration["build-system"]["requires"]
    project = configuration["project"]
    assert generated["Name"] == project["name"]
    assert generated["Version"] == project["version"]
    assert generated["Summary"] == project["description"]
    assert generated["Requires-Python"] == project["requires-python"]
    assert generated["License-Expression"] == project["license"] == "MIT"
    assert generated.get_all("License-File") == project["license-files"] == ["LICENSE"]
    assert generated["License"] is None
    expected = {str(Requirement(value)) for value in project["dependencies"]}
    for extra, requirements in project["optional-dependencies"].items():
        expected.update(str(Requirement(f'{value}; extra == "{extra}"')) for value in requirements)
    actual = {str(Requirement(value)) for value in generated.get_all("Requires-Dist", [])}
    assert actual == expected
    assert set(generated.get_all("Provides-Extra", [])) == set(project["optional-dependencies"])
    wheel_path = tmp_path / 'wheel'
    wheel_path.mkdir()
    print(f'Writing to {wheel_path}...')
    wheel_name = build_meta.build_wheel(str(wheel_path))
    print(f'Reading from {wheel_path / wheel_name}...')
    with ZipFile(wheel_path / wheel_name) as archive:
        paths = archive.namelist()
        metadata_name = next(name for name in paths if name.endswith('.dist-info/METADATA'))
        license_name = next(name for name in paths if name.endswith('.dist-info/licenses/LICENSE'))
        wheel_metadata = Parser().parsestr(archive.read(metadata_name).decode('utf-8'))
        print(f'Reading from {REPOSITORY_ROOT / "LICENSE"}...')
        assert archive.read(license_name) == (REPOSITORY_ROOT / 'LICENSE').read_bytes()
        assert wheel_metadata['License-Expression'] == 'MIT'
        assert wheel_metadata.get_all('License-File') == ['LICENSE']
        assert {str(Requirement(value)) for value in wheel_metadata.get_all('Requires-Dist', [])} == expected
        for module in (REPOSITORY_ROOT / 'pcat').glob('*.py'):
            assert f'pcat/{module.name}' in paths


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
        "architecture",
        "capabilities",
        "examples",
        "gallery",
        "getting_started",
        "likelihoods",
        "outputs",
        "sampling",
        "troubleshooting",
    }
    for page in pages:
        assert f"   {page}\n" in index


def test_readthedocs_examples_show_real_spectrum_and_flare_results():
    page = (DOCS_ROOT / "examples.rst").read_text()
    relative_images = (
        "jwst_miri_ngc7027_line_catalog/visuals/jwst_miri_ngc7027_spectrum_fit_20261025_175318.png",
        "jwst_miri_ngc7027_line_catalog/visuals/jwst_miri_ngc7027_line_catalog_samples_20261025_175318.png",
        "jwst_miri_ngc7027_line_catalog/visuals/jwst_miri_ngc7027_line_count_posterior_20261025_175318.png",
        "variable_number_stellar_flares/visuals/variable_number_stellar_flares_light_curve_fit.png",
        "variable_number_stellar_flares/visuals/variable_number_stellar_flares_catalog_samples.png",
        "variable_number_stellar_flares/visuals/variable_number_stellar_flares_count_posterior.png",
    )

    for relative_image in relative_images:
        assert f"../examples/{relative_image}" in page
        assert (REPOSITORY_ROOT / "examples" / relative_image).is_file()


def test_documented_example_commands_resolve():
    relative_paths = [
        "examples/chandra_point_source_catalog/chandra_point_source_catalog.py",
        "examples/simulated_hst_strong_lens/simulated_hst_strong_lens.py",
        "examples/run_all_examples.py",
        "examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py",
        "examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py",
        "examples/roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog.py",
        "examples/catalog_association_completeness_purity/catalog_association_completeness_purity.py",
        "examples/kl_divergence_gaussian/kl_divergence_gaussian.py",
    ]

    for relative_path in relative_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file()


def test_every_example_script_has_a_notebook():
    examples = REPOSITORY_ROOT / "examples"
    script_only_automation = set(examples.glob("run_all_examples*.py"))
    for script in examples.rglob("*.py"):
        if script in script_only_automation:
            continue
        if script.stem.endswith(" 2") and script.with_name(f"{script.stem[:-2]}{script.suffix}").is_file():
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
        "kl_divergence_gaussian/kl_divergence_gaussian",
    )
    for name in figure_examples:
        notebook = examples / f"{name}.ipynb"
        print(f"Reading from {notebook}...")
        cells = json.loads(notebook.read_text())["cells"]
        code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        assert (
            "display_example_visuals(" in code
            or "display(Image(" in code
            or "display_posterior_maps(" in code
            or "save_and_show(" in code and "posterior.listpostcntpmodl" in code
        ), name


def test_verified_pcat_publications_have_examples():
    examples = REPOSITORY_ROOT / "examples"
    publication_examples = {
        "10.3847/1538-4357/aa679e": "daylan+2017_fermi_point_sources",
        "10.3847/1538-3881/aa8565": "portillo+2017_crowded_sdss_m2",
        "10.3847/1538-4357/aaaa1e": "daylan+2018_strong_lens_subhalos",
        "10.3847/1538-3881/ab74cf": "feder+2020_multiband_sdss_deblending",
        "10.3847/1538-4357/ac6c04": "butler+2022_spire_sz_component_separation",
        "10.3847/1538-3881/ace69b": "feder+2023_point_diffuse_spire",
        "10.3847/1538-4357/ae1e7a": "hall+2026_herschel_dsfg_multiplicity",
    }
    index = (examples / "README.md").read_text()
    assert "10.3847/1538-4357/aaa1f2" not in index

    for doi, directory in publication_examples.items():
        root = examples / directory
        assert root.is_dir(), directory
        assert (root / f"{directory}.py").is_file(), directory
        notebook = root / f"{directory}.ipynb"
        assert notebook.is_file(), directory
        assert doi in index, doi
        document = json.loads(notebook.read_text())
        for cell in document["cells"]:
            expected_language = "python" if cell["cell_type"] == "code" else "markdown"
            assert cell["metadata"]["language"] == expected_language
            if "id" in cell["metadata"]:
                assert cell["metadata"]["id"] == cell["id"]
            else:
                assert "display_posterior_diagnostics(" in "".join(cell["source"])

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
            assert "display_posterior_maps(" in code
            assert "display_posterior_diagnostics(" in code


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
    assert "for main_id in system_ids" in source
    assert "_dp1_all_bands.png" in source
    assert "rubin_dp1_einstein_radius_summary.png" in source
    assert "np.percentile(result['draws'][:, 0], [16, 50, 84])" in source


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
        "PCAT_PATH/imag",
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
        "transit-timing likelihood",
        "unbinned_gaussian_mixture",
    }
    missing_terms = required_terms - {
        term for term in required_terms if term in capabilities
    }
    assert not missing_terms


def test_documentation_separates_likelihoods_from_catalog_products():
    capabilities = (DOCS_ROOT / "capabilities.rst").read_text()
    likelihoods = (DOCS_ROOT / "likelihoods.rst").read_text()

    assert "Unified Poisson imaging likelihood" in capabilities
    assert "strong-lens reconstructions" in capabilities
    assert "not separate likelihood" in capabilities
    assert "Unbinned Gaussian-mixture likelihood" in capabilities
    assert "fixed Gaussian" in capabilities
    assert inspect.signature(sampling.init).parameters["boolcondcatl"].default is True
    assert "default ``boolcondcatl=True``" in capabilities
    assert re.search(r"Set\s+``boolcondcatl=False``", likelihoods)
    assert "does not spatially condense one-dimensional line,\nflare, or Keplerian catalogs" in likelihoods
    assert "Every transdimensional model produces a catalog" in likelihoods
    assert "Condensation and either kind of association are posterior" in likelihoods
    assert "The maintained" in likelihoods
    assert "unbinned_gaussian_mixture" in likelihoods
    assert "PCAT does not currently provide a built-in unbinned" in likelihoods
    assert "dispatcher must be completed" not in likelihoods


def test_each_likelihood_has_a_configuration_section():
    likelihoods = (DOCS_ROOT / "likelihoods.rst").read_text()
    section_headings = (
        "Image likelihoods",
        "Poisson count-image likelihood",
        "Gaussian image likelihood",
        "Unbinned event-level likelihood",
        "Spectral-line likelihood",
        "Binned photometric time-series likelihood",
        "Radial-velocity likelihood",
        "Transit-timing likelihood",
        "Generic user-supplied likelihood",
    )
    configuration_options = (
        "liketype=\"pois\"",
        "liketype=\"gaus\"",
        "varidata",
        "lghtpnts",
        "lghtgausbgrd",
        "typeemishost=\"sers\"",
        "boollenssubh",
        "typeevalpsfn",
        "typemodlpsfn",
        "psfpexpr",
        "spectype",
        "lsftype=\"gaus\"",
        "lsfresolvingpower",
        "lsfkernel",
        "--profile pvoi",
        "--lsftype none|gaus|tabu",
        "edisintp",
        "flarfred",
        "--profile flardav",
        "spot_period_days",
        "retr_dictpcatrvel",
        "boollinetren",
        "numbjitt",
        "log_likelihood_transit_times",
        "make_transit_time_likelihood",
        "typeexpr=\"gener\"",
    )

    for heading in section_headings:
        assert heading in likelihoods
    for option in configuration_options:
        assert option in likelihoods


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
    assert list(inspect.signature(plotting.plot_lens_parameter_recovery).parameters) == [
        "output_path", "draws", "true_parameters", "typefileplot", "labels",
    ]
    assert 'pcat.plotting.plot_lens_parameter_recovery(output_path, draws, true_parameters, typefileplot="png", labels=(' in api
    assert "sample-by-parameter array with one injected value and one label per column." in api
    parameters = inspect.signature(plotting.plot_detection_diagnostic).parameters
    assert list(parameters) == ["records", "output_path", "examples", "detection_threshold", "example_prefix"]
    assert parameters["detection_threshold"].default == 0.5
    assert parameters["example_prefix"].default is None
    assert "pcat.plotting.plot_detection_diagnostic(records, output_path, examples=None, detection_threshold=0.5, example_prefix=None)" in api


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


def test_legacy_command_notebook_is_self_contained_and_only_prints_commands():
    import ast
    from contextlib import redirect_stdout
    import io
    import json

    notebook = json.loads(
        (REPOSITORY_ROOT / "examples" / "legacy_external_analysis_commands.ipynb").read_text()
    )
    output = io.StringIO()
    namespace = {}
    with redirect_stdout(output):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            assert all(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls)
            exec(compile(tree, "<command-reference>", "exec"), namespace)
    commands = output.getvalue().splitlines()
    assert len(commands) == 21
    assert all(command.startswith("python $") for command in commands)
    assert commands[0].endswith("pcat_lens_mock")
    assert commands[-1].endswith("test_psfn")


def test_documentation_lists_stable_public_helpers():
    api = (DOCS_ROOT / "api.rst").read_text()

    for name in (
        "sample_allesfitter_pcat",
        "plot_population_grid",
        "binomial_wilson_interval",
        "estimate_kl_divergence",
    ):
        assert name in api


def test_documented_posterior_collage_matches_generator():
    examples = (DOCS_ROOT / "examples.rst").read_text()
    normalized_examples = " ".join(examples.split())

    assert len(POSTERIOR_ANIMATION_PANELS) == 9
    static_panels = [
        panel for panel in POSTERIOR_ANIMATION_PANELS if panel.static_image is not None
    ]
    assert len(static_panels) == 1
    assert static_panels[0].static_image == "docs/_static/pcat_logo.png"
    assert "Eight changing inference views are synchronized" in examples
    assert "static PCAT logo panel" in examples
    assert "observed JWST/MIRI NGC 7027 spectral fit" in normalized_examples
    assert "starts from a nonempty catalog" in normalized_examples
    assert "follows burn-in for the first third" in normalized_examples
    for domain in (
        "lensed arcs", "stellar flares", "rotating starspots",
        "radial velocities", "transit times",
    ):
        assert domain in normalized_examples


def test_readme_separates_sampler_utilities_from_applications():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampler_utilities = readme.split("## Sampler utilities", 1)[1].split(
        "\n## Applications", 1
    )[0]
    applications = readme.split("## Applications", 1)[1].split("\n## ", 1)[0]

    assert "Dynamic convergence-based stopping" in sampler_utilities
    assert "Keplerian signals in radial-velocity data" in applications
    assert "Dynamic convergence-based stopping" not in applications


def test_variable_planet_rv_example_is_linked_and_describes_variable_catalog():
    examples = REPOSITORY_ROOT / "examples" / "variable_number_exoplanets_radial_velocity"
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    example_readme = (REPOSITORY_ROOT / "examples" / "README.md").read_text()
    example_docs = (DOCS_ROOT / "examples.rst").read_text()
    script = (examples / "variable_number_exoplanets_radial_velocity.py").read_text()
    notebook = json.loads(
        (examples / "variable_number_exoplanets_radial_velocity.ipynb").read_text()
    )
    notebook_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "variable_number_exoplanets_radial_velocity.ipynb" in readme
    assert "zero to five" in readme
    assert "--smoke" in readme and "--numbswep 100000" in readme
    assert "maxmnumbplan=5" in script
    assert "retr_dictpcatrvel" in script
    assert "run_example_script" in notebook_code
    assert "display_example_visuals" in notebook_code
    for filename in (
        "variable_number_exoplanets_rv_count_posterior.png",
        "variable_number_exoplanets_rv_phase_folded.png",
        "variable_number_exoplanets_rv_catalog_samples.png",
    ):
        assert filename in readme
        assert filename in example_readme
        assert filename in example_docs
        assert (examples / "visuals" / filename).is_file()


def test_documentation_distinguishes_posterior_and_candidate_animations():
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    outputs = (DOCS_ROOT / "outputs.rst").read_text()
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    example_guide = (DOCS_ROOT / "examples.rst").read_text()
    example_readme = (REPOSITORY_ROOT / "examples" / "README.md").read_text()
    example_script = (
        REPOSITORY_ROOT / "examples" / "proposal_state_animation"
        / "proposal_state_animation.py"
    ).read_text()

    for source in (sampling, outputs, readme, example_readme):
        assert "boolmakeanimprop" in source
    for source in (sampling, outputs, readme, example_guide, example_readme):
        assert "proposal_candidates.gif" in source
        assert "proposal_sequence.gif" in source
    sampling_text = " ".join(sampling.split())
    assert "Each frame shows a proposed state and its acceptance outcome." in sampling_text
    assert "Only retained chain states enter posterior samples." in sampling_text
    assert "including rejected candidates" in outputs
    assert "makeanim=True" in outputs
    assert "numbburn" in outputs
    assert "proposal_sequence.gif" in example_script
    assert "proposal_candidates.gif" in example_script
    assert "examples/proposal_state_animation/visuals/proposal_sequence_20261025_181311.gif" in readme
    assert "examples/proposal_state_animation/visuals/proposal_candidates_20261025_181311.gif" in readme
    for animation_name in ("proposal_sequence.gif", "proposal_candidates.gif"):
        stem = animation_name.removesuffix(".gif")
        assert f"../examples/proposal_state_animation/visuals/{stem}_20261025_181311.gif" in example_guide


def test_point_source_example_accepts_isolated_output_and_seed(tmp_path, monkeypatch):
    import runpy
    from pcat import demo

    script = REPOSITORY_ROOT / 'examples/chandra_point_source_catalog/chandra_point_source_catalog.py'
    namespace = runpy.run_path(script)
    calls = []
    monkeypatch.setattr(demo, 'run_pipeline_demo', lambda output, **configuration: calls.append((output, configuration)))
    namespace['main'](['--output-root', str(tmp_path), '--seed', '17'])
    output, configuration = calls[0]
    assert output == tmp_path and configuration['typeseed'] == 17
    assert configuration['truenumbelempop0'] == 2
    assert configuration['fittminmnumbelempop0'] == 1
    assert configuration['fittmaxmnumbelempop0'] == 3


def test_student_t_proposal_and_target_specific_efficiency_are_documented():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    example = (REPOSITORY_ROOT / "examples/burn_in_strategies/burn_in_strategies.py").read_text()

    assert "probstud" in readme
    assert "probdemc" in sampling and "probstud" in sampling
    assert "their sum must not exceed one" in sampling
    assert "student_t" in example
    assert "ESS per second" in sampling


def test_burn_in_benchmark_preserves_prior_runs_and_figures(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace
    import numpy as np

    namespace = runpy.run_path(REPOSITORY_ROOT / 'examples/burn_in_strategies/burn_in_strategies.py')
    state = namespace['run_benchmark'].__globals__
    monkeypatch.setitem(state, 'EXAMPLE_PATH', tmp_path)
    monkeypatch.setitem(state, 'VISUAL_PATH', tmp_path / 'visuals')
    monkeypatch.setitem(state, 'retr_pathrun', lambda base, name: str(base / name))
    monkeypatch.setitem(state, '_effective_sample_size', lambda samples: float(len(samples)))
    old_run = tmp_path / 'correlated_gaussian_fixed'
    old_run.mkdir()
    sentinel = old_run / 'retained.txt'
    sentinel.write_text('earlier result', encoding='utf-8')
    calls = []

    def sample(**configuration):
        directory = configuration['pathbase'] / configuration['strgcnfg']
        directory.mkdir(parents=True)
        calls.append(directory)
        number = configuration['numbsamp']
        samples = np.zeros((number, len(configuration['parameter_names'])))
        return SimpleNamespace(listpostparagenrscalbase=samples,
                               listpostboolpropaccp=np.ones(configuration['numbswep'], dtype=bool),
                               listpostfacttmpr=np.ones(configuration['numbswep']))

    monkeypatch.setattr(state['sampling'], 'sample_fixed', sample)
    first = namespace['run_benchmark'](40)
    first_runs = set(calls)
    second = namespace['run_benchmark'](40)
    assert len(set(calls)) == 16
    assert first_runs.isdisjoint(calls[8:])
    assert sentinel.read_text(encoding='utf-8') == 'earlier result'
    for result in (first, second):
        run_tag = next(iter(result.values()))['run_tag']
        metrics_path = tmp_path / 'data' / f'benchmark_{run_tag}' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        assert metrics['run_tag'] == run_tag and len(metrics['results']) == 8
        assert all(not Path(row['run_directory']).is_absolute() for row in metrics['results'].values())
        restored = namespace['load_benchmark'](metrics_path.parent)
        assert restored.keys() == result.keys()
        for key in result:
            np.testing.assert_array_equal(restored[key]['samples'], result[key]['samples'])
            np.testing.assert_array_equal(restored[key]['inverse_temperature'], result[key]['inverse_temperature'])
            assert restored[key]['elapsed_seconds'] == result[key]['elapsed_seconds']
    first_path = namespace['plot_temperature_schedule'](first)
    previous_bytes = first_path.read_bytes()
    second_path = namespace['plot_temperature_schedule'](second)
    assert first_path != second_path
    assert first_path.read_bytes() == previous_bytes
    assert namespace['plot_posterior_comparison'](second).is_file()
    assert namespace['plot_performance_comparison'](second).is_file()
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        namespace['plot_temperature_schedule'](first)


def test_neural_proposal_extra_and_transdimensional_usage_are_documented():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]
    benchmark = REPOSITORY_ROOT / "examples/neural_proposal_efficiency/neural_proposal_efficiency.py"

    assert any(requirement.startswith("torch>=") for requirement in project["optional-dependencies"]["neural"])
    assert benchmark.is_file()
    example_readme = (REPOSITORY_ROOT / "examples/README.md").read_text()
    assert "data-informed" in example_readme
    assert "--run-id" in example_readme and "before training" in example_readme
    for source in (readme, sampling):
        assert "NeuralElementProposal.fit" in source
        assert "uniform_fraction" in source
        assert "proposal.callbacks()" in source
        assert "neural_proposal_features" in source
        assert "transdimensional" in source
    assert "ess_per_call" in benchmark.read_text()
    assert "likelihood_ess_per_call" in benchmark.read_text()
    assert "count_posterior_total_variation" in benchmark.read_text()
    assert "_count_rhat" in benchmark.read_text()
    assert "_efficiency_gain_supported" in benchmark.read_text()
    assert 'boolmakepropelem=method != "prior"' in benchmark.read_text()
    assert "probjump=0.0" in benchmark.read_text()


@pytest.fixture
def neural_benchmark_control(tmp_path):
    path = REPOSITORY_ROOT / 'examples/neural_proposal_efficiency/neural_proposal_efficiency.py'
    print(f'Reading from {path}...')
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'run_benchmark')
    training = Mock(side_effect=RuntimeError('synthetic stop at training'))
    namespace = {
        'Path': Path, 'np': np, 'time': SimpleNamespace(perf_counter=Mock(return_value=0.0)),
        'RUNS_PATH': tmp_path / 'data', 'VISUALS_PATH': tmp_path / 'visuals',
        'retr_pathrun': retr_pathrun, '_training_data': training,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    arguments = SimpleNamespace(run_id='synthetic', seed=53, validation_scenes=2, chains_per_scene=2, training_scenes=2)
    return namespace, arguments, training


@pytest.fixture
def neural_count_ess():
    path = REPOSITORY_ROOT / 'examples/neural_proposal_efficiency/neural_proposal_efficiency.py'
    print(f'Reading from {path}...')
    functions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {'_count_trace', '_count_ess'}]
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_count_ess']


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 64])
def test_neural_count_ess_keeps_zero_policy_for_every_constant_trace(neural_count_ess, count):
    with np.errstate(all='raise'):
        assert neural_count_ess(np.full(count, 2)) == 0.0


@pytest.mark.parametrize('values', [
    [np.nan], [0, 1, np.nan, 2], [0, np.inf], [0, -np.inf],
    np.ma.array([0, 1, 2, 1], mask=[False, True, False, False]),
    np.ma.masked_all(4), np.ma.masked,
])
def test_neural_count_ess_rejects_unavailable_samples(neural_count_ess, values):
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Count ESS requires'):
        neural_count_ess(values)


@pytest.mark.parametrize('count', [2, 3, 8, 65])
def test_neural_count_ess_preserves_existing_varying_trace_calculation(neural_count_ess, count):
    values = np.random.default_rng(53).integers(0, 4, count).astype(float)
    if count < 4:
        expected = float(count)
    else:
        centered = values - np.mean(values)
        fft_size = 1 << (2 * count - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:count]
        covariance /= np.arange(count, 0, -1)
        correlation = covariance / covariance[0]
        paired_sum = 0.0
        for lag in range(1, count - 1, 2):
            pair = correlation[lag] + correlation[lag + 1]
            if pair <= 0.0:
                break
            paired_sum += pair
        expected = float(min(count, count / max(1.0, 1.0 + 2.0 * paired_sum)))

    with np.errstate(all='raise'):
        actual = neural_count_ess(values)
        unmasked = neural_count_ess(np.ma.asarray(values))

    assert actual == unmasked == expected


@pytest.mark.parametrize('artifact_kind', ['checkpoint', 'results', 'figure', 'prior', 'data-informed', 'neural', 'later_chain'])
def test_neural_benchmark_preserves_existing_artifacts_before_training(neural_benchmark_control, tmp_path, artifact_kind):
    namespace, arguments, training = neural_benchmark_control
    artifacts = {
        'checkpoint': namespace['RUNS_PATH'] / 'synthetic_proposal.pt',
        'results': namespace['RUNS_PATH'] / 'synthetic_results.json',
        'figure': namespace['VISUALS_PATH'] / 'synthetic_efficiency.png',
    }
    if artifact_kind in artifacts:
        artifact = artifacts[artifact_kind]
        preserved = artifact
        artifact.parent.mkdir(parents=True)
    else:
        method = 'neural' if artifact_kind == 'later_chain' else artifact_kind
        validation_seed = 64 if artifact_kind == 'later_chain' else 63
        chain_index = 1 if artifact_kind == 'later_chain' else 0
        run_id = f'synthetic_{method}_{validation_seed}_chain{chain_index}'
        artifact = Path(retr_pathrun(str(namespace['RUNS_PATH']), run_id)) / 'data' / 'outp' / run_id
        artifact.mkdir(parents=True)
        preserved = artifact / 'original.h5'
    original = b'original synthetic artifact'
    print(f'Writing to {preserved}...')
    preserved.write_bytes(original)
    original_paths = set(tmp_path.rglob('*'))

    with pytest.raises(FileExistsError, match='Choose a new --run-id') as caught:
        namespace['run_benchmark'](arguments)

    assert str(artifact) in str(caught.value)
    training.assert_not_called()
    namespace['time'].perf_counter.assert_not_called()
    print(f'Reading from {preserved}...')
    assert preserved.read_bytes() == original
    assert set(tmp_path.rglob('*')) == original_paths


def test_neural_benchmark_fresh_run_reaches_training_without_creating_outputs(neural_benchmark_control, tmp_path):
    namespace, arguments, training = neural_benchmark_control

    with pytest.raises(RuntimeError, match='synthetic stop at training'):
        namespace['run_benchmark'](arguments)

    training.assert_called_once()
    assert training.call_args.args[3:] == (arguments.training_scenes, arguments.seed + 1000)
    assert list(tmp_path.iterdir()) == []


def test_burn_in_benchmark_is_linked_and_compares_multiple_targets():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    script = (REPOSITORY_ROOT / "examples/burn_in_strategies/burn_in_strategies.py").read_text()

    assert "### Burn-in strategies" in readme
    assert "examples/burn_in_strategies/burn_in_strategies.ipynb" in readme
    assert "likelihood-\ntemperature annealing" in readme
    assert 'PROBLEMS = ("correlated_gaussian", "bimodal")' in script
    for strategy in ('"fixed"', '"adaptive"', '"tempered_adaptive"', '"student_t"'):
        assert strategy in script


def test_jacobian_example_is_linked_and_documents_unclipped_acceptance():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    script = (REPOSITORY_ROOT / "examples/proposal_state_animation/proposal_state_animation.py").read_text()
    animation = REPOSITORY_ROOT / "examples/proposal_state_animation/visuals/parameterization_jacobian_jumps_20261025_181311.gif"

    assert "visuals/jacobian_acceptance_effect_20261025_181311.png" in readme
    assert "visuals/parameterization_jacobian_jumps_20261025_181311.gif" in readme
    assert "parameterization_jacobian_jumps.gif" in sampling
    assert "make_parameterization_jump_animation" in script
    assert animation.is_file()
    with Image.open(animation) as gif:
        assert gif.n_frames == 41
        assert gif.size == (1280, 720)
        assert gif.info["duration"] == 180
    assert "listpostaccplprb" in script
    assert "listpostaccplprb" in sampling


def test_readme_lists_daylan_applications_first_without_reproduction_language():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    applications = readme.split("## Applications\n", 1)[1].split("\n## ", 1)[0]
    headings = [line for line in applications.splitlines() if line.startswith("### ")]

    assert headings[:2] == [
        "### [Daylan et al. (2017) Unresolved Fermi-LAT sources](examples/daylan+2017_fermi_point_sources/)",
        "### [Daylan et al. (2018) Strong-lens subhalos](examples/daylan+2018_strong_lens_subhalos/)",
    ]
    assert "reproduc" not in applications.lower()


def test_readthedocs_index_links_repository_and_pcat_publications():
    index = (DOCS_ROOT / "index.rst").read_text()

    for link in (
        "https://github.com/tansudaylan/pcat",
        "https://doi.org/10.3847/1538-4357/aa679e",
        "https://doi.org/10.3847/1538-4357/aaaa1e",
        "https://doi.org/10.3847/1538-3881/aa8565",
        "https://doi.org/10.3847/1538-3881/ab74cf",
        "https://doi.org/10.3847/1538-4357/ac6c04",
        "https://doi.org/10.3847/1538-3881/ace69b",
        "https://doi.org/10.3847/1538-4357/ae1e7a",
    ):
        assert link in index
    assert "10.3847/1538-4357/aaa1f2" not in index


def test_burn_in_options_and_comparison_are_documented():
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    sampling = (DOCS_ROOT / "sampling.rst").read_text()
    outputs = (DOCS_ROOT / "outputs.rst").read_text()
    examples = (DOCS_ROOT / "examples.rst").read_text()

    for source in (readme, sampling):
        for option in ("numbburn", "booladaptstdp", "boolburntmpr", "factburntmpr"):
            assert option in source
        source_lower = " ".join(source.lower().split())
        assert (
            "gradient descent" in source_lower
            or "no gradient-based burn-in mode" in source_lower
        )
    assert "listpostfacttmpr" in sampling
    assert "listpostfacttmpr" in outputs
    assert "burn_in_strategies" in examples
    for name in (
        "burn_in_posterior_comparison.png",
        "burn_in_performance_comparison.png",
        "burn_in_temperature_schedule.png",
    ):
        assert name in readme


def test_spectrum_and_flare_examples_use_shared_posterior_loader():
    for relative_path, cache_fragment in (
        ("examples/jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py",
         "data/outp/{RUN_NAME}/gdatfinlpost"),
        ("examples/variable_number_stellar_flares/variable_number_stellar_flares.py",
         "data/outp/{run_name}/gdatfinlpost"),
    ):
        path = REPOSITORY_ROOT / relative_path
        print(f"Reading from {path}...")
        source = path.read_text()
        assert "load_example_posterior" in source
        assert "def read_posterior(" not in source
        assert cache_fragment in source


def test_plotting_examples_use_shared_theme_and_save_helpers():
    for relative_path in (
        "examples/jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py",
        "examples/proposal_profiling/proposal_profiling.py",
        "examples/sampler_comparison_emcee_dynesty/sampler_comparison_emcee_dynesty.py",
        "examples/variable_number_stellar_flares/variable_number_stellar_flares.py",
    ):
        path = REPOSITORY_ROOT / relative_path
        print(f"Reading from {path}...")
        source = path.read_text()
        assert "configure_example_style" in source
        assert "save_example_figure" in source
        assert "def configure_style(" not in source
        assert "def save(" not in source


def test_three_sampler_comparison_documents_fixed_and_variable_dimensions():
    examples = REPOSITORY_ROOT / "examples" / "sampler_comparison_emcee_dynesty"
    script = (examples / "sampler_comparison_emcee_dynesty.py").read_text()
    notebook = json.loads(
        (examples / "sampler_comparison_emcee_dynesty.ipynb").read_text()
    )
    notebook_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    normalized_readme = " ".join(readme.split())
    example_readme = (REPOSITORY_ROOT / "examples" / "README.md").read_text()
    example_docs = (DOCS_ROOT / "examples.rst").read_text()

    assert "compare_fixed_dimension" in script
    assert "compare_variable_dimension" in script
    assert "'--part',\n    'both'" in notebook_code
    assert "emcee cannot compare models" in script
    assert "emcee requires one parameter dimension per chain" in normalized_readme
    assert "Each model fixes its parameter dimension." in normalized_readme
    assert "A metamodel combines candidate models with different dimensions." in normalized_readme
    assert "## Custom likelihoods" in readme
    assert "separate evidence calculation for each" in example_readme
    assert "emcee cannot vary the number of planets" in example_readme
    assert "--part both" in example_docs
    figure_names = (
        "sampler_comparison_marginal_posteriors.png",
        "sampler_comparison_efficiency.png",
        "sampler_comparison_planet_count_posterior.png",
        "sampler_comparison_planet_count_cost.png",
        "sampler_comparison_two_planet_periods.png",
    )
    for name in figure_names:
        assert (examples / "visuals" / name).is_file()
    assert all(name in readme for name in figure_names)
