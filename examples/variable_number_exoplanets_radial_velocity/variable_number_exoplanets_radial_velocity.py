#!/usr/bin/env python3
"""Catalog a variable number of exoplanets in simulated radial-velocity (RV) time series with PCAT.

Data: simulated (not real observations). A star is observed at 120 irregular epochs over 6 years
with seasonal gaps, first by one spectrograph and then by a second one with a different velocity
zero point. Three planets on Keplerian orbits are injected, with semi-amplitudes of 12, 4, and
2.5 m/s, together with 2 m/s of white stellar jitter beyond the 1.5 m/s reported uncertainties.

PCAT samples catalogs of Keplerian signals (element type ``lghtlinekepl``) with birth, death,
jump, and within-model moves. Births and jumps draw periods from a periodogram-weighted density
with the matching Hastings correction. Instrument offsets are marginalized analytically and the
jitter numerically, so each catalog is scored by its marginal likelihood.
"""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tdpy.exoplanet import keplerian_radial_velocity

EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name

NUMB_EPOCH = 120
BASELINE = 2200.  # [day]
TIME_START = 7000.  # [BJD - 2450000 day]
STDV_REPORTED = 1.5  # [m/s]
JITTER = 2.0  # [m/s]
OFFSET_INSTRUMENTS = (-12.0, 25.0)  # [m/s]
# injected planets: period [day], semi-amplitude [m/s], eccentricity, argument of periastron [rad], mean anomaly [rad]
PLANETS = np.array([
    [5.77, 12.0, 0.05, 1.0, 0.3],
    [42.1, 4.0, 0.15, 2.5, 4.0],
    [410., 2.5, 0.0, 0.0, 1.5],
])


def simulate_data(rng):
    """Return epochs [day], velocities [m/s], uncertainties [m/s], instrument labels, and the time reference."""
    # observing seasons of 8 months per year
    time = []
    while len(time) < NUMB_EPOCH:
        epoch = rng.uniform(0., BASELINE)
        if np.mod(epoch, 365.25) < 240.:
            time.append(epoch)
    time = TIME_START + np.sort(np.array(time))
    indxinst = (time > TIME_START + 0.55 * BASELINE).astype(int)
    timerefr = 0.5 * (time[0] + time[-1])
    rvel = retr_rvelplan(time, PLANETS, timerefr) + np.array(OFFSET_INSTRUMENTS)[indxinst]
    rvel += rng.normal(0., np.hypot(STDV_REPORTED, JITTER), time.size)
    return time, rvel, np.full(time.size, STDV_REPORTED), indxinst, timerefr


def retr_rvelplan(time, planets, timerefr):
    """Total Keplerian RV [m/s] of a planet table at the given times [day]."""
    return np.sum(keplerian_radial_velocity(time[:, None], planets[:, 0], planets[:, 1], planets[:, 2],
                                            planets[:, 3], planets[:, 4], timerefr), axis=1)


def run_pcat(time, rvel, stdv, indxinst, numbswep):
    from pcat import sampling
    from pcat.radial_velocity import retr_dictpcatrvel

    dictpcat = retr_dictpcatrvel(
        time, rvel, stdv, indxinst, str(EXAMPLE_PATH), RUN_NAME, maxmnumbplan=5,
        numbswep=numbswep, numbburn=numbswep // 3, numbsamp=max(numbswep // 50, 20), numbproc=4,
        numbswepplot=max(numbswep // 20, 1), inittype='rand', typeseed=0, numbframanim=24,
        probtran=0.7, probspmr=0.4, probjump=0.2, makeanim=False,
        # a prior draw starts far from the data, so the likelihood is annealed during burn-in
        boolburntmpr=True, factburntmpr=0.8,
        # unit-cube proposal scales for K, P, phase, eccentricity, and argument of periastron
        stdvpropelemfire=[1e-2, 1e-4, 3e-2, 3e-2, 3e-2],
        boolmakeplot=False, boolmakeplotinit=False, typeverb=0,
    )
    sampling.sample(**dictpcat)
    from pcat.main import proc_anim
    proc_anim(RUN_NAME, pathbase=str(EXAMPLE_PATH))


def read_posterior():
    from pcat.main import readfile, retr_pathrun

    pathrun = retr_pathrun(str(EXAMPLE_PATH), RUN_NAME)
    return readfile(str(Path(pathrun) / 'data' / 'outp' / RUN_NAME / 'gdatfinlpost'))


def retr_nuisances(time, resi, stdv, indxinst):
    """Weighted least-squares instrument offsets [m/s] of residuals, used only to display the data."""
    weig = 1. / (stdv**2 + JITTER**2)
    return np.array([np.sum(weig[indxinst == k] * resi[indxinst == k]) / np.sum(weig[indxinst == k])
                     for k in range(2)])


def configure_style(typeplotback):
    colrfore = 'white' if typeplotback == 'dark' else 'black'
    colrback = 'black' if typeplotback == 'dark' else 'white'
    mpl.rcParams.update({
        'font.size': 10, 'text.usetex': False, 'axes.grid': False, 'figure.facecolor': colrback, 'axes.facecolor': colrback,
        'savefig.facecolor': colrback, 'axes.edgecolor': colrfore, 'axes.labelcolor': colrfore,
        'xtick.color': colrfore, 'ytick.color': colrfore, 'text.color': colrfore,
        'legend.fancybox': True, 'legend.framealpha': 1.0, 'legend.fontsize': 10, 'axes.titlesize': 10,
    })
    return colrfore


def save(figure, name, typefileplot):
    path = EXAMPLE_PATH / 'visuals' / f'{name}.{typefileplot}'
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing to {path}...')
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def retr_listelem(posterior):
    """Concatenated posterior element periods [day] and semi-amplitudes [m/s] over all saved catalogs."""
    listperi = np.concatenate([np.asarray(sample[0]['elin']) for sample in posterior.listpostdictelem])
    listrvsa = np.concatenate([np.asarray(sample[0]['flux']) for sample in posterior.listpostdictelem])
    return listperi, listrvsa


def retr_listplanpost(posterior, numbdraw=300):
    """Planet tables (period, K, e, omega, mean anomaly) of evenly spaced posterior catalogs."""
    listdictelem = posterior.listpostdictelem
    indxdraw = np.linspace(0, len(listdictelem) - 1, min(numbdraw, len(listdictelem))).astype(int)
    return [np.column_stack([np.asarray(listdictelem[k][0][name]) for name in ['elin', 'flux', 'ecce', 'argp', 'phas']])
            for k in indxdraw]


def render_rv_posterior_frames(time, rvel, stdv, indxinst, timerefr, posterior):
    """Show fixed-scale simulated RV data against PCAT catalogs from the prior draw to posterior samples."""
    from pcat.plotting import animation_phase_label, animation_states

    snapshots = animation_states(posterior)
    model_time = np.linspace(time.min(), time.max(), 450)  # [day]
    catalogs = [np.column_stack([np.asarray(snapshot['dictelem'][0][name])
                                 for name in ('elin', 'flux', 'ecce', 'argp', 'phas')]) for snapshot in snapshots]
    curves = [retr_rvelplan(model_time, catalog, timerefr) for catalog in catalogs]
    measured = rvel - np.asarray(OFFSET_INSTRUMENTS)[indxinst]  # [m/s]
    # data-based limits; early burn-in catalogs may leave the axes
    scale = 1.25 * np.max(np.abs(measured) + stdv)  # [m/s]
    output_directory = EXAMPLE_PATH / 'visuals'
    output_directory.mkdir(parents=True, exist_ok=True)
    for old_frame in output_directory.glob('rv_posterior_swep*.png'):
        print(f'Removing previous frame {old_frame}...')
        old_frame.unlink()
    paths = []
    for snapshot, catalog, curve in zip(snapshots, catalogs, curves):
        index = snapshot['cntrswep']
        figure, axis = plt.subplots(figsize=(6.0, 4.5), facecolor='white')
        axis.errorbar(time - TIME_START, measured, yerr=stdv, fmt='.', markersize=3,
                      color='#47565E', alpha=0.65, label='Simulated RV')
        axis.plot(model_time - TIME_START, curve, color='#A51C30', lw=1.8,
                  label=f'PCAT model ({len(catalog)} planets)')
        axis.set(xlim=(0, BASELINE), ylim=(-scale, scale), xlabel='Time since start [day]',
                 ylabel='Radial velocity [m/s]', title=animation_phase_label(snapshot))
        axis.grid(False)
        axis.legend(loc='upper right', facecolor='white', framealpha=1)
        path = output_directory / f'rv_posterior_swep{index:09d}.png'
        print(f'Writing to {path}...')
        figure.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(figure)
        paths.append(path)
    return paths


def plot_phase_folded(time, rvel, stdv, indxinst, timerefr, posterior, typefileplot, colrfore):
    """Simulated RVs folded on each planet of the maximum-posterior catalog, with the other planets subtracted."""
    listplan = retr_listplanpost(posterior)
    indxmaxm = int(np.argmax(np.asarray(posterior.listpostlpostotl).ravel()))
    planmaxm = np.column_stack([np.asarray(posterior.listpostdictelem[indxmaxm][0][name])
                                for name in ['elin', 'flux', 'ecce', 'argp', 'phas']])
    planmaxm = planmaxm[np.argsort(planmaxm[:, 0])]
    # instrument offsets, which PCAT marginalizes, are fit to the maximum-posterior model only for display
    offs = retr_nuisances(time, rvel - retr_rvelplan(time, planmaxm, timerefr), stdv, indxinst)
    rveldisp = rvel - offs[indxinst]
    numbplan = planmaxm.shape[0]
    figure, axes = plt.subplots(1, numbplan, figsize=(7.1, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    phasfine = np.linspace(0., 1., 400)
    for p in range(numbplan):
        peri = planmaxm[p, 0]  # [day]
        othr = np.delete(planmaxm, p, 0)
        rvelplan = rveldisp - retr_rvelplan(time, othr, timerefr)
        phas = np.mod((time - timerefr) / peri, 1.)
        # curves of the matching signal in each posterior catalog, evaluated over one orbit
        listcurv = []
        for plan in listplan:
            if plan.shape[0] == 0:
                continue
            indx = np.argmin(np.abs(np.log(plan[:, 0] / peri)))
            if abs(np.log(plan[indx, 0] / peri)) < 0.05:
                listcurv.append(retr_rvelplan(timerefr + phasfine * peri, np.array([[peri, *plan[indx, 1:]]]), timerefr))
        quan = np.percentile(np.array(listcurv), [16., 50., 84.], axis=0)
        axes[p].fill_between(phasfine, quan[0], quan[2], color='C0', alpha=0.4, lw=0, label='PCAT 68% interval')
        axes[p].plot(phasfine, quan[1], color='C0', lw=1., label='PCAT median')
        for k, labl in enumerate(['Spectrograph A', 'Spectrograph B']):
            indx = indxinst == k
            axes[p].errorbar(phas[indx], rvelplan[indx], np.hypot(stdv[indx], JITTER), fmt='o', ms=2.5,
                             color=f'C{2 * k + 1}', lw=0.5, label=f'{labl} (simulated)')
        axes[p].set_xlabel('Phase')
        axes[p].set_title('P = %.3g day\nin %.0f%% of catalogs' % (peri, 100. * len(listcurv) / len(listplan)))
        axes[p].set_xticks([0., 0.5] if p < numbplan - 1 else [0., 0.5, 1.])
    axes[0].set_ylabel('RV [m/s]')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
    figure.subplots_adjust(wspace=0.05)
    save(figure, 'variable_number_exoplanets_rv_phase_folded', typefileplot)


def plot_catalog_samples(posterior, typefileplot, colrfore):
    """Posterior Keplerian signals in the period and semi-amplitude plane, with the injected planets."""
    listperi, listrvsa = retr_listelem(posterior)
    figure, axis = plt.subplots(figsize=(7.1, 3.2))
    axis.scatter(listperi, listrvsa, s=4, alpha=0.3, color='C0', lw=0,
                 label=f'PCAT signals ({len(posterior.listpostdictelem)} catalogs)')
    axis.scatter(PLANETS[:, 0], PLANETS[:, 1], marker='*', s=200, facecolor='none', edgecolor='C3', lw=1.2, zorder=5,
                 label='Injected planets')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel('Period [day]')
    axis.set_ylabel('Semi-amplitude [m/s]')
    axis.legend(loc='upper right')
    save(figure, 'variable_number_exoplanets_rv_catalog_samples', typefileplot)


def plot_count_posterior(posterior, typefileplot, colrfore):
    """Posterior probability of the number of planets."""
    numbelem = np.asarray(posterior.listpostnumbelem).ravel().astype(int)
    values, counts = np.unique(numbelem, return_counts=True)
    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    axis.bar(values, counts / counts.sum(), color='C0', label='PCAT posterior')
    axis.axvline(PLANETS.shape[0], color='C3', ls='--', lw=1.2, label='Injected count')
    axis.set_xlabel('Number of planets')
    axis.set_ylabel('Posterior probability')
    axis.legend(loc='upper left')
    save(figure, 'variable_number_exoplanets_rv_count_posterior', typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numbswep', type=int, default=100_000)
    parser.add_argument('--smoke', action='store_true', help='Run a short pipeline check.')
    add_plot_arguments(parser)
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    parser.add_argument('--skip-sampling', action='store_true', help='Replot an existing chain.')
    arguments = parser.parse_args()
    numbswep = 1000 if arguments.smoke else arguments.numbswep

    rng = np.random.default_rng(7)
    time, rvel, stdv, indxinst, timerefr = simulate_data(rng)
    if not arguments.skip_sampling:
        run_pcat(time, rvel, stdv, indxinst, numbswep)
    posterior = read_posterior()
    colrfore = configure_style(arguments.typeplotback)
    render_rv_posterior_frames(time, rvel, stdv, indxinst, timerefr, posterior)
    plot_phase_folded(time, rvel, stdv, indxinst, timerefr, posterior, arguments.typefileplot, colrfore)
    plot_catalog_samples(posterior, arguments.typefileplot, colrfore)
    plot_count_posterior(posterior, arguments.typefileplot, colrfore)


if __name__ == '__main__':
    main()
