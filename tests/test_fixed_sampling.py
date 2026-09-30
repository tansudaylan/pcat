import numpy as np

from pcat.main import sample


def gaussian_log_likelihood(gdat, strgmodl, values):
    gdat.generic_callback_count = getattr(gdat, 'generic_callback_count', 0) + 1
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    return -0.5 * values @ np.linalg.inv(covariance) @ values


def test_generic_model_uses_main_sampling_pipeline(tmp_path):
    result = sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        proposal_scales=(0.08, 0.08),
        pathbase=str(tmp_path),
        strgcnfg='fixed_gaussian',
        numbswep=80,
        numbburn=20,
        numbsamp=60,
        booladaptstdp=True,
        typeverb=-1,
    )

    assert result.typeexpr == 'gener'
    assert result.fitt.numbpopl == 0
    assert result.probtran == 0.0
    assert result.probspmr == 0.0
    assert np.all(result.listpostindxproptype == 0)
    assert result.listpostparagenrscalbase.shape == (60, 2)
    assert np.all(np.isfinite(result.listpostparagenrscalbase))