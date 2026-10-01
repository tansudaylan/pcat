import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tdpy
from PIL import Image

from pcat import plotting
from pcat.plotting import (
    has_one_dimensional_prediction, plot_acceptance_decomposition, plot_catalog_trace,
    plot_compute_budget, plot_posterior_predictive, plot_proposal_ledger, plot_sampler_overview,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DOCS = REPOSITORY / "docs"


def synthetic_state(chain_count=2, sweep_count=600, sample_count=50, seed=3):
    """Return a two-chain state with the arrays PCAT stores for a 1D catalog run."""
    rng = np.random.default_rng(seed)
    proposal_count = chain_count * sweep_count
    move = rng.choice(5, size=proposal_count, p=(0.5, 0.15, 0.15, 0.1, 0.1))
    accepted = rng.random(proposal_count) < np.where(move == 0, 0.4, 0.05)
    axis = np.linspace(1.0, 2.0, 40)  # [day]
    catalogs, counts, models = [], [], []
    for _ in range(chain_count * sample_count):
        number = rng.integers(1, 4)
        position = rng.uniform(1.1, 1.9, number)  # [day]
        flux = rng.uniform(5.0, 50.0, number)  # [counts per bin]
        catalogs.append([{"elin": position, "flux": flux}])
        counts.append([number])
        models.append(100.0 + np.sum(flux[:, None] * np.exp(-0.5 * ((axis - position[:, None]) / 0.05) ** 2), 0))
    state = SimpleNamespace(
        nameproptype=np.array(["with", "brth", "deth", "splt", "merg"]),
        listpostindxproptype=move[:, None], listpostboolpropaccp=accepted[:, None],
        listpostboolpropfilt=(rng.random(proposal_count) > 0.05)[:, None],
        listpostaccplprb=rng.normal(-5.0, 20.0, (proposal_count, 1)),
        listpostltrp=np.where(move >= 3, 1.0, 0.0)[:, None],
        listpostljcb=np.where(move == 3, 2.0, np.where(move == 4, -2.0, 0.0))[:, None],
        listpostfacttmpr=np.minimum(1.0, np.repeat(np.arange(sweep_count), chain_count) / 200.0)[:, None],
        listnamechro=["totl", "prop", "llik", "plot"],
        listpostchrototl=rng.uniform(1e-3, 2e-3, (proposal_count, 1)),  # [s]
        listpostchroprop=rng.uniform(1e-4, 2e-4, (proposal_count, 1)),  # [s]
        listpostchrollik=rng.uniform(1e-5, 2e-5, (proposal_count, 1)),  # [s]
        listpostchroplot=np.full((proposal_count, 1), 1.0),  # [s], once per sweep
        numbproc=chain_count, numbswep=sweep_count, numbburn=sweep_count // 3, numbsamp=sample_count,
        listpostdictelem=catalogs, listpostnumbelem=np.array(counts),
        cntpdata=rng.poisson(models[0])[:, None, None].astype(float),
        listpostcntpmodl=np.array(models)[:, :, None, None],
        bctrpara=SimpleNamespace(ener=axis),
    )
    return state


def test_sampler_overview_writes_every_applicable_view(tmp_path):
    paths = plot_sampler_overview(synthetic_state(), tmp_path)
    assert set(paths) == {"proposal_ledger", "acceptance_decomposition", "compute_budget",
                          "catalog_trace", "posterior_predictive"}
    for path in paths.values():
        print(f"Reading from {path}...")
        with Image.open(path) as image:
            assert image.width > 400 and image.height > 300


def test_sampler_overview_skips_views_a_state_cannot_support(tmp_path):
    state = SimpleNamespace(listpostparagenrscalbase=np.ones((4, 1)))
    assert plot_sampler_overview(state, tmp_path) == {}
    assert not has_one_dimensional_prediction(state)


def test_views_reject_unusable_inputs(tmp_path):
    state = synthetic_state()
    with pytest.raises(ValueError, match="png"):
        plot_proposal_ledger(state, tmp_path / "ledger", typefileplot="jpg")
    with pytest.raises(ValueError, match="chain"):
        plot_catalog_trace(state, tmp_path / "trace", chain=5)
    with pytest.raises(ValueError, match="element parameter"):
        plot_catalog_trace(state, tmp_path / "trace", position="xpos")
    state.listpostindxproptype = np.full_like(state.listpostindxproptype, 9)
    with pytest.raises(ValueError, match="move names"):
        plot_proposal_ledger(state, tmp_path / "ledger")


def test_compute_budget_omits_stages_timed_once_per_sweep(tmp_path, monkeypatch):
    shown = []
    original = plotting.plt.Axes.set_xticks

    def record(axis, ticks, labels=None, **options):
        if labels is not None:
            shown.extend(labels)
        return original(axis, ticks, labels, **options)

    monkeypatch.setattr(plotting.plt.Axes, "set_xticks", record)
    plot_compute_budget(synthetic_state(), tmp_path / "budget")
    assert "Proposal" in shown and "Likelihood" in shown and "Plotting" not in shown


def test_posterior_predictive_tail_probability_is_calibrated_for_the_true_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    mean = np.full(400, 200.0)  # [counts per bin]
    state = SimpleNamespace(cntpdata=rng.poisson(mean)[:, None, None].astype(float),
                            listpostcntpmodl=np.repeat(mean[None, :, None, None], 300, axis=0))
    captured = {}
    original = plotting.plt.Axes.plot

    def record(axis, *arguments, **options):
        if options.get("color") == "#A50026":
            captured["tail"] = np.asarray(arguments[1])
        return original(axis, *arguments, **options)

    monkeypatch.setattr(plotting.plt.Axes, "plot", record)
    plot_posterior_predictive(state, tmp_path / "predictive")
    assert 0.4 < np.mean(captured["tail"]) < 0.6
    assert np.mean((captured["tail"] < 0.025) | (captured["tail"] > 0.975)) < 0.1


def test_acceptance_decomposition_requires_evaluated_proposals(tmp_path):
    state = synthetic_state()
    state.listpostboolpropfilt = np.zeros_like(state.listpostboolpropfilt)
    with pytest.raises(ValueError, match="five evaluated"):
        plot_acceptance_decomposition(state, tmp_path / "terms")


def test_split_and_merge_keep_unit_and_physical_parameters_consistent(tmp_path):
    from pcat.main import readfile, retr_pathrun
    from pcat.sampling import sample

    sample(
        typeexpr="fire", spectype=["voig"], strgexpo=1.0e5, spatdisttype=["line"],
        typeelem=["lghtlinevoig"], maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi), numbsidecart=1,
        anlytype="spec", probtran=0.7, probspmr=0.6, typeseed=0, typeseedelem=17, inittype="refr",
        truenumbelempop0=2, fittminmnumbelempop0=1, fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        numbswep=1500, numbburn=500, numbsamp=200, boolmakeplot=False, boolmakeplotinit=False,
        makeanim=False, booldiag=False, typeverb=0, pathbase=str(tmp_path), strgcnfg="spmr_unit",
    )
    state = readfile(str(Path(retr_pathrun(str(tmp_path), "spmr_unit")) / "data" / "outp"
                         / "spmr_unit" / "gdatfinlpost"))
    moves = np.asarray(state.listpostindxproptype).ravel()
    accepted = np.asarray(state.listpostboolpropaccp).ravel().astype(bool)
    evaluated = np.asarray(state.listpostboolpropfilt).ravel().astype(bool)
    assert np.any(accepted & np.isin(moves, (3, 4)))
    assert np.all(np.isfinite(np.asarray(state.listpostltrp).ravel()[evaluated & np.isin(moves, (3, 4))]))
    assert accepted[moves == 4].mean() < 0.9
    names = state.fitt.namepara.genrelem[0]
    base = int(state.fitt.numbparagenrbase)
    units = np.asarray(state.listpostparagenrunitfull)
    for sample_index, catalog in enumerate(state.listpostdictelem):
        for offset, name in enumerate(names):
            slots = base + offset + len(names) * np.arange((units.shape[1] - base) // len(names))
            rebuilt = tdpy.icdf_logt(units[sample_index, slots], getattr(state.fitt.minmpara, name),
                                     getattr(state.fitt.maxmpara, name))
            for value in np.asarray(catalog[0][name]):
                assert np.min(np.abs(rebuilt - value) / abs(value)) < 1e-6


def public_plotting_routines():
    names = [name for name, value in inspect.getmembers(plotting, inspect.isfunction)
             if value.__module__ == plotting.__name__ and re.match(r"(plot|make)_", name)]
    return names + ["plot_population_grid"]


def test_every_public_plotting_routine_is_documented_with_an_example_output():
    api = (DOCS / "api.rst").read_text()
    gallery = (DOCS / "gallery.rst").read_text()
    for name in public_plotting_routines():
        assert re.search(rf"py:function:: pcat\.(plotting\.)?{name}\(", api), name
        assert re.search(rf":func:`pcat\.(plotting\.)?{name}`", gallery), name
    for image in re.findall(r"^\.\. image:: (\S+)", gallery, flags=re.MULTILINE):
        assert (DOCS / image).resolve().is_file(), image


def test_gallery_is_part_of_the_documentation_tree():
    assert re.search(r"^\s+gallery$", (DOCS / "index.rst").read_text(), flags=re.MULTILINE)
