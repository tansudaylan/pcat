import numpy as np
from types import SimpleNamespace

import pytest

from pcat.main import PointSpreadFunction, retr_indxswepanim
from pcat.plotting import animation_phase_label, animation_states
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


def test_animation_states_require_snapshots():
    with pytest.raises(ValueError, match="numbframanim"):
        animation_states(SimpleNamespace())


@pytest.mark.parametrize("profile, parameters", [
    ("singgaus", [0.01]),
    ("singking", [0.01, 2.5]),
    ("doubking", [0.01, 2.5, 0.03, 2.2, 0.8]),
])
def test_point_spread_functions_integrate_to_one(profile, parameters):
    state = SimpleNamespace(numbener=1, numbdqlt=1, typeexpr="chan", indxenerincl=None, indxdqltincl=None)
    angle = np.linspace(0.0, 2.0, 200001)  # [rad]
    density = PointSpreadFunction(state, parameters, profile)(angle)[0, :, 0]  # [sr^-1]
    assert np.trapz(2 * np.pi * angle * density, angle) == pytest.approx(1.0, rel=2e-3)


def test_lens_image_has_no_central_image_and_conserves_counts():
    config = RomanLensConfig()
    image = render_lens(config, 0.9, 0.0, 0.0, 0.09, 0.8, 0.35)
    center = config.number_side // 2
    # a singular isothermal sphere has no central image of a source behind it
    assert image[center, center] < 1e-3 * image.max()
    assert image.sum() == pytest.approx(config.source_counts, rel=1e-6)
