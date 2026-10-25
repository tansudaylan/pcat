from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def test_readthedocs_covers_public_workflows():
    documentation = "\n".join(
        (DOCS_ROOT / name).read_text()
        for name in ("api.rst", "capabilities.rst", "examples.rst")
    )
    required_terms = {
        'typeexpr="gener"',
        "sample_parallel",
        "associate_catalogs",
        "posterior_convergence",
        "estimate_evidence",
        "psf_poly_fit",
        "daylan+2018_strong_lens_subhalos",
        "lghtpnts",
        "lensed emission",
        "lghtlinevoig",
    }

    assert not required_terms - {
        term for term in required_terms if term in documentation
    }


def test_readthedocs_embeds_maintained_visuals():
    visual_paths = {
        "../examples/pcat_posterior_samples.gif",
        "../examples/catalog_association_completeness_purity/visuals/catalog_association_completeness_purity.png",
        "../examples/roman_strong_lens_perturber_catalog/visuals/roman_strong_lens_perturber_catalog.png",
    }
    source = "\n".join(path.read_text() for path in DOCS_ROOT.glob("*.rst"))

    for relative_path in visual_paths:
        assert f".. image:: {relative_path}" in source
        assert (DOCS_ROOT / relative_path).is_file()


def test_readthedocs_documents_cubic_cartesian_psf_evaluation():
    source = (DOCS_ROOT / "likelihoods.rst").read_text()
    required_terms = (
        'kernevaltype="ulip"',
        'kernevaltype="bspx"',
        'typepixl="cart"',
        "cubic_psf_cost.png",
        "cubic_psf_accuracy.png",
        "--bokeh",
        "cubic_psf_benchmark.html",
    )

    assert all(term in source for term in required_terms)
    assert "does not bound errors over the full profile" in " ".join(source.split())

    readme = (REPOSITORY_ROOT / "README.md").read_text()
    assert 'kernevaltype="bspx"' in readme
    assert "not establish posterior equivalence or adequacy for observed data" in " ".join(readme.split())
    assert "adequacy for observed data" in readme
    assert "docs/_static/cubic_psf_cost.png" in readme
    assert "docs/_static/cubic_psf_accuracy.png" in readme
    assert "docs/_static/cubic_psf_benchmark.html" in readme
    assert ":download:`self-contained Bokeh document <_static/cubic_psf_benchmark.html>`" in source
    assert (DOCS_ROOT / "_static" / "cubic_psf_benchmark.html").is_file()
