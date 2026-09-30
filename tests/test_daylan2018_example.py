import importlib.util
from pathlib import Path


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples" / "daylan+2018_strong_lens_subhalos"
    / "daylan+2018_strong_lens_subhalos.py"
)
specification = importlib.util.spec_from_file_location("daylan2018_example", EXAMPLE_SCRIPT)
example = importlib.util.module_from_spec(specification)
specification.loader.exec_module(example)


def test_lens_configurations_share_the_mock_and_change_catalog_prior(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(example, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("pcat.demo.run_pipeline_demo", lambda root, **cfg: calls.append((root, cfg)))

    example.run_reproduction(smoke=True)
    example.run_reproduction(one_subhalo=True, smoke=True, typefileplot="pdf")

    catalog_root, catalog = calls[0]
    fixed_root, fixed = calls[1]
    assert catalog_root != fixed_root
    assert catalog["typeexpr"] == "HST_WFC3_UVIS"
    assert catalog["typeelem"] == ["lens"]
    assert catalog["truenumbelempop0"] == fixed["truenumbelempop0"] == 3
    assert catalog["typeseed"] == fixed["typeseed"]
    assert catalog["fittmaxmnumbelempop0"] > 1
    assert fixed["fittminmnumbelempop0"] == fixed["fittmaxmnumbelempop0"] == 1
    assert fixed["typefileplot"] == "pdf"
    assert (fixed_root / "visuals/post/finl/varbscal/cova").is_dir()