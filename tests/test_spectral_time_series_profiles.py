import numpy as np
import pytest

from pcat.spectral import (
    apply_gaussian_resolving_power,
    apply_line_spread_function,
    evaluate_line_profile,
    gaussian_lsf_kernel,
    spectral_profile_parameters,
)
from pcat.time_series import (
    evaluate_flare_profile,
    evaluate_rotating_spot_profile,
    flare_profile_parameters,
    rotating_spot_parameters,
)


@pytest.mark.parametrize(
    ("profile", "parameters", "tolerance"),
    [
        ("gaus", {"gaussian_width": [1.0]}, 1e-6),
        ("lore", {"lorentz_width": [1.0]}, 2e-2),
        ("voig", {"gaussian_width": [1.0], "lorentz_width": [0.5]}, 2e-2),
        ("pvoi", {"gaussian_width": [1.0], "lorentz_width": [1.0], "mixing_fraction": [0.3]}, 2e-2),
        ("sinc", {"gaussian_width": [1.0]}, 3e-3),
        ("skew", {"gaussian_width": [1.0], "skewness": [4.0]}, 1e-6),
        ("toph", {"gaussian_width": [1.0]}, 2e-3),
    ],
)
def test_line_profiles_preserve_integrated_flux(profile, parameters, tolerance):
    axis = np.linspace(-50.0, 50.0, 100_001)
    values = evaluate_line_profile(axis, profile, [2.0], [0.0], **parameters)[:, 0]
    assert np.trapezoid(values, axis) == pytest.approx(2.0, rel=tolerance)


def test_skew_gaussian_is_asymmetric():
    axis = np.linspace(-8.0, 8.0, 10_001)
    values = evaluate_line_profile(axis, "skew", [1.0], [0.0], gaussian_width=[1.0], skewness=[5.0])[:, 0]
    assert np.trapezoid(values[axis > 0.0], axis[axis > 0.0]) > 10.0 * np.trapezoid(values[axis < 0.0], axis[axis < 0.0])


def test_fixed_and_resolving_power_lsf_preserve_sampled_flux():
    axis = np.linspace(1.0, 2.0, 2001)
    spectrum = evaluate_line_profile(axis, "gaus", [2.0], [1.5], gaussian_width=[0.01])
    fixed = apply_line_spread_function(spectrum, gaussian_lsf_kernel(3.0))
    resolving = apply_gaussian_resolving_power(axis, spectrum, [1.5], 500.0)
    assert np.sum(fixed) == pytest.approx(np.sum(spectrum))
    assert np.sum(resolving) == pytest.approx(np.sum(spectrum))
    assert np.max(resolving) < np.max(spectrum)


def test_tabulated_lsf_longer_than_spectrum_keeps_shape_and_flux():
    spectrum = np.array([0.0, 1.0, 0.0])
    convolved = apply_line_spread_function(spectrum, np.ones(9))
    assert convolved.shape == spectrum.shape
    assert np.sum(convolved) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("profile", "parameters"),
    [
        ("flargauss", {"fwhm": [0.4]}),
        ("flarexpd", {"decay_time": [0.8]}),
        ("flarfred", {"rise_time": [0.2], "decay_time": [0.8]}),
        ("flardav", {"fwhm": [0.4]}),
    ],
)
def test_flare_profiles_are_nonnegative_and_peak_normalized(profile, parameters):
    time = np.linspace(-2.0, 4.0, 6001)
    values = evaluate_flare_profile(time, profile, [3.0], [0.0], **parameters)[:, 0]
    assert np.all(values >= 0.0)
    assert values[np.argmin(np.abs(time))] == pytest.approx(3.0, rel=0.02)


def test_fred_has_independent_rise_and_decay_times():
    values = evaluate_flare_profile(
        [-1.0, 0.0, 1.0], "flarfred", [1.0], [0.0], rise_time=[0.2], decay_time=[2.0]
    )[:, 0]
    assert values[0] < values[2] < values[1]


def test_rotating_spot_profile_repeats_and_only_dims_the_star():
    time_days = np.array([1.25, 4.45, 7.65, 2.85])  # [day]
    values = evaluate_rotating_spot_profile(
        time_days,
        [120.0],
        [0.25],
        [0.16],
        period_days=3.2,  # [day]
        reference_time_days=1.0,  # [day]
    )[:, 0]

    np.testing.assert_allclose(values[:3], -120.0)
    assert values[3] > -1e-8
    assert np.all(values <= 0.0)
    assert rotating_spot_parameters("spotrot") == ("flux", "elin", "fwhm")


def test_rotating_spot_profile_rejects_width_longer_than_period():
    with pytest.raises(ValueError, match="shorter than the rotation period"):
        evaluate_rotating_spot_profile(
            [0.0], [10.0], [0.0], [2.0], period_days=1.0  # [day]
        )


def test_native_parameter_sets_are_profile_specific():
    assert spectral_profile_parameters("pvoi") == ("flux", "elin", "sigm", "gamm", "frac")
    assert spectral_profile_parameters("skew") == ("flux", "elin", "sigm", "skew")
    assert flare_profile_parameters("flarfred") == ("flux", "elin", "scalrise", "scalfall")