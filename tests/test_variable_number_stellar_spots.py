import astropy.io.fits
import h5py
import numpy as np

from pcat import sampling
from pcat.time_series import evaluate_rotating_spot_profile


def test_pcat_samples_a_variable_number_of_stellar_spots(tmp_path):
    period_days = 2.0  # [day]
    reference_time_days = 1.0  # [day]
    cadence_days = 0.1  # [day]
    edges = reference_time_days + np.arange(81) * cadence_days
    time_days = 0.5 * (edges[1:] + edges[:-1])
    spot_model = evaluate_rotating_spot_profile(
        time_days,
        [240.0, 190.0],
        [0.35, 1.25],
        [0.15, 0.2],
        period_days,
        reference_time_days,
    ).sum(axis=1)
    expected_counts = 1400.0 + spot_model
    observed_counts = np.random.default_rng(8).poisson(expected_counts).astype(float)
    exposure = np.full(time_days.size, 1.0 / cadence_days)
    input_path = tmp_path / "variable_number_stellar_spots" / "data" / "inpt"
    input_path.mkdir(parents=True)
    astropy.io.fits.writeto(
        input_path / "sbrt.fits", observed_counts[:, None, None, None], overwrite=True
    )
    astropy.io.fits.writeto(
        input_path / "expo.fits", exposure[:, None, None], overwrite=True
    )

    sampling.sample(
        typeexpr="fire",
        typedata="inpt",
        strgexprsbrt="sbrt.fits",
        strgexpo="expo.fits",
        typeexpo="file",
        binsenerfull=edges,
        spectype=["spotrot"],
        spatdisttype=["line"],
        typeelem=["lghtlinevoig"],
        dictfitt={
            "typeelem": ["lghtlinevoig"],
            "spectype": ["spotrot"],
            "sbrtbacknorm": [np.full((time_days.size, 1, 1), 1400.0)],
            "listnamediff": ["back0000"],
        },
        spot_period_days=period_days,
        spot_reference_time_days=reference_time_days,
        limtparaelem={
            "flux": (80.0, 400.0),
            "elin": (0.05, period_days - 0.05),
            "fwhm": (0.08, 0.4),
        },
        stdvpropelemfire=[0.04, 0.04, 0.04],
        maxmgangdata=1e-4,
        anlytype="spec",
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=3,
        inittype="rand",
        typeseed=5,
        probtran=0.8,
        probspmr=0.4,
        numbswep=600,
        numbburn=150,
        numbsamp=450,
        boolmakeplot=False,
        boolmakeplotinit=False,
        boolmakeplotfram=False,
        booldiag=False,
        typeverb=0,
        pathbase=str(tmp_path / "variable_number_stellar_spots"),
        strgcnfg="variable_number_stellar_spots",
    )

    output_path = (
        tmp_path / "variable_number_stellar_spots" / "data" / "outp"
        / "variable_number_stellar_spots" / "gdatfinlpost.h5"
    )
    with h5py.File(output_path, "r") as saved:
        counts = np.asarray(saved["listpostnumbelem"][()]).reshape(-1)
        likelihood = np.asarray(saved["listpostlliktotl"][()])
    assert np.all((counts >= 0) & (counts <= 3))
    assert np.unique(counts).size > 1
    assert np.isfinite(likelihood).all()