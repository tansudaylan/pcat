import numpy as np
from types import SimpleNamespace

import pytest
from scipy.integrate import trapezoid

from pcat.main import PointSpreadFunction, retr_indxswepanim
from pcat.plotting import animation_phase_label, animation_phase_labels, animation_states
from pcat.roman_lens import RomanLensConfig, render_lens


def test_animation_schedule_spends_one_third_in_burn_in_from_the_initial_state():
    sweeps = retr_indxswepanim(3000, 1000, 24)
    assert sweeps[0] == 0 and sweeps[-1] == 2999
    assert np.sum(sweeps < 1000) == 8 and np.sum(sweeps >= 1000) == 16
    assert retr_indxswepanim(3000, 1000, None).size == 0
    with pytest.raises(ValueError):
        retr_indxswepanim(3000, 1000, 1)


def test_generic_run_records_snapshots_from_a_prior_draw(tmp_path):
    from pcat.fixed import sample_fixed_chains

    chain, _, state = sample_fixed_chains(
        None, lambda values, data: -0.5 * np.sum((np.asarray(values) / 0.1) ** 2), None,
        ("x", "y"), ("self", "self"), (-5.0, -5.0), (5.0, 5.0), None, None, None,
        2, 300, 150, pathbase=str(tmp_path), typeverb=-1, numbframanim=9, return_state=True,
    )
    snapshots = animation_states(state)
    assert [snapshot["cntrswep"] for snapshot in snapshots][0] == 0
    assert sum(snapshot["boolburn"] for snapshot in snapshots) == 3
    # the run starts from a prior draw, far from the narrow posterior in most draws, and ends near it
    assert np.max(np.abs(snapshots[-1]["paragenrscalfull"][:2])) < 0.5
    assert animation_phase_label(snapshots[0]) == "Initial random draw from the prior"
    assert animation_phase_label(snapshots[-1]).startswith("Posterior sample")


def test_animation_phase_labels_reserve_width_for_every_frame():
    snapshots = [
        {"cntrswep": 0, "boolburn": True},
        {"cntrswep": 950, "boolburn": True},
        {"cntrswep": 1000, "boolburn": False},
    ]

    labels = animation_phase_labels(snapshots)

    assert len({len(label) for label in labels}) == 1
    assert labels[0].rstrip() == "Initial random draw from the prior"
    assert labels[1].rstrip() == "Burn-in, sweep 950"
    assert labels[2].rstrip() == "Posterior sample, sweep 1,000"


def test_animation_phase_titles_keep_the_same_rendered_geometry():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()
    bounds = []
    for phase_label in animation_phase_labels((
        {"cntrswep": 0, "boolburn": True},
        {"cntrswep": 9, "boolburn": True},
        {"cntrswep": 10000, "boolburn": False},
    )):
        title = axis.set_title(phase_label, loc="left", fontfamily="monospace")
        canvas.draw()
        assert title.get_ha() == "left"
        assert title.get_fontfamily() == ["monospace"]
        bounds.append(title.get_window_extent(canvas.get_renderer()).bounds)
    np.testing.assert_allclose(bounds, [bounds[0]] * 3)


@pytest.mark.parametrize("example_name", [
    "unbinned_gaussian_mixture", "transit_timing_variations",
    "roman_strong_lens_perturber_catalog", "variable_number_exoplanets_radial_velocity",
    "variable_number_stellar_spots",
])
def test_example_frame_titles_have_fixed_geometry(example_name, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    source = Path(__file__).resolve().parents[1] / "examples" / example_name / f"{example_name}.py"
    specification = importlib.util.spec_from_file_location("animation_title_" + example_name, source)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    snapshots = [
        {"cntrswep": sweep, "boolburn": burn, "numbelem": [count],
         "paragenrscalfull": np.array([0.1, 0.2, 0.3, 0.4]),
         "cntpmodl": np.full((4, 1, 1), 1000.0),
         "dictelem": [{name: np.ones(count) for name in ("elin", "flux", "ecce", "argp", "phas")}]}
        for sweep, burn, count in ((0, True, 1), (9, True, 9), (10000, False, 100))
    ]
    posterior = SimpleNamespace(listanimstate=snapshots)
    bounds = []
    original_savefig = Figure.savefig

    def capture_frame(figure, output, **kwargs):
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        title = figure.axes[0]._left_title
        assert title.get_text()
        assert title.get_ha() == "left"
        assert title.get_fontfamily() == ["monospace"]
        bounds.append(title.get_window_extent(canvas.get_renderer()).bounds)
        original_savefig(figure, output, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_frame)
    if example_name == "unbinned_gaussian_mixture":
        paths = example.render_posterior_frames(np.array([[0.0, 0.0], [1.0, 1.0]]), snapshots, tmp_path)
    elif example_name == "transit_timing_variations":
        monkeypatch.setattr(example, "ROOT", tmp_path)
        paths = example.render_ttv_posterior_frames(example.simulate_transits(), snapshots, "title_test")
    elif example_name == "roman_strong_lens_perturber_catalog":
        monkeypatch.setattr(example, "__file__", str(tmp_path / "example.py"))
        (tmp_path / "visuals").mkdir()
        monkeypatch.setattr(example, "run_lens_image_pipeline", lambda *args, **kwargs: SimpleNamespace(posterior=posterior))
        monkeypatch.setattr(example, "render_lens_counts", lambda config, *args: np.full((config.number_side, config.number_side), 1000.0))
        paths = example.render_posterior_lens_frames(smoke=True)
    elif example_name == "variable_number_exoplanets_radial_velocity":
        monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
        monkeypatch.setattr(example, "retr_rvelplan", lambda time, *args: np.ones_like(time))
        paths = example.render_rv_posterior_frames(
            example.TIME_START + np.arange(4.0), np.arange(4.0), np.ones(4),
            np.zeros(4, dtype=int), example.TIME_START, posterior,
        )
    else:
        monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
        paths = example.render_spot_posterior_frames(
            example.TIME_OFFSET_DAYS + np.arange(4.0), np.arange(1000.0, 1004.0), posterior,
        )
    assert len(paths) == len(bounds) == 3
    np.testing.assert_allclose(bounds, [bounds[0]] * 3)
    dimensions = []
    for output in paths:
        with Image.open(output) as image:
            dimensions.append(image.size)
    assert len(set(dimensions)) == 1


def test_animation_states_require_snapshots():
    with pytest.raises(ValueError, match="numbframanim"):
        animation_states(SimpleNamespace())


@pytest.mark.parametrize("profile, parameters", [
    ("singgaus", [0.01]),
    ("singking", [0.01, 2.5]),
    ("doubking", [0.01, 2.5, 0.03, 2.2, 0.8]),
])
def test_point_spread_functions_integrate_to_one(profile, parameters, monkeypatch):
    monkeypatch.delattr(np, 'trapz', raising=False)
    state = SimpleNamespace(numbener=1, numbdqlt=1, typeexpr="chan", indxenerincl=None, indxdqltincl=None)
    angle = np.linspace(0.0, 2.0, 200001)  # [rad]
    density = PointSpreadFunction(state, parameters, profile)(angle)[0, :, 0]  # [sr^-1]
    assert trapezoid(2 * np.pi * angle * density, angle) == pytest.approx(1.0, rel=2e-3)


def test_lens_image_has_no_central_image_and_conserves_counts():
    config = RomanLensConfig()
    image = render_lens(config, 0.9, 0.0, 0.0, 0.09, 0.8, 0.35)
    center = config.number_side // 2
    # a singular isothermal sphere has no central image of a source behind it
    assert image[center, center] < 1e-3 * image.max()
    assert image.sum() == pytest.approx(config.source_counts, rel=1e-6)
