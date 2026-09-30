import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


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
        "examples/chan_demo/generate_demo.py",
        "examples/hst_lens/generate_demo.py",
        "examples/run_examples.py",
        "examples/Daylan+2017/generate_reproduction.py",
        "examples/voigt-profile/pcat_voigt_profile_detection.py",
        "examples/roman_lens_catalog/roman_lens_catalog_diagnostic.py",
    ]

    for relative_path in relative_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file()


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