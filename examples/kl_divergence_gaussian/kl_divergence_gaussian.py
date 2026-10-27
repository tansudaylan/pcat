#!/usr/bin/env python3
"""Estimate KL divergence and mutual information for a Gaussian PCAT posterior."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm
from tdpy.cli import add_plot_arguments, add_smoke_argument

from pcat.demo import configure_example_style, save_example_figure
from pcat.diagnostics import estimate_kl_divergence, estimate_mutual_information
from pcat.fixed import sample_fixed_chains

EXAMPLE_PATH = Path(__file__).resolve().parent
PARAMETER_NAMES = ("x", "y")
PRIOR_MEAN = np.array([-0.3, 0.4])
PRIOR_STDV = np.array([1.1, 0.8])
PRIOR_COVARIANCE = np.diag(PRIOR_STDV**2)
OBSERVATION = np.array([0.9, -0.8])
LIKELIHOOD_COVARIANCE = np.array([[0.35, 0.12], [0.12, 0.5]])
LIKELIHOOD_PRECISION = np.linalg.inv(LIKELIHOOD_COVARIANCE)


def gaussian_log_likelihood(values, state):
    """Return the Gaussian observation log likelihood up to a constant."""
    residual = values - state["observation"]
    return float(-0.5 * residual @ state["precision"] @ residual)


def analytic_posterior():
    """Return exact mean and covariance for the Gaussian posterior."""
    precision_prior = np.linalg.inv(PRIOR_COVARIANCE)
    precision_likelihood = LIKELIHOOD_PRECISION
    covariance = np.linalg.inv(precision_prior + precision_likelihood)
    mean = covariance @ (
        precision_prior @ PRIOR_MEAN + precision_likelihood @ OBSERVATION
    )
    return mean, covariance


def analytic_gaussian_kl(mean_p, covariance_p, mean_q, covariance_q):
    """Return the exact forward KL divergence between two multivariate Gaussians."""
    dimension = mean_p.size
    difference = mean_q - mean_p
    sign_p, logdet_p = np.linalg.slogdet(covariance_p)
    sign_q, logdet_q = np.linalg.slogdet(covariance_q)
    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Gaussian covariance matrices must be positive definite")
    return float(0.5 * (
        np.trace(np.linalg.solve(covariance_q, covariance_p))
        + difference @ np.linalg.solve(covariance_q, difference)
        - dimension + logdet_q - logdet_p
    ))


def run_example(sample_count=3000, burn_count=1000, chain_count=4, typefileplot="png"):
    """Sample the exact Gaussian posterior with PCAT and estimate its KL to the prior."""
    configure_example_style("white")
    chains, _ = sample_fixed_chains(
        {"observation": OBSERVATION, "precision": LIKELIHOOD_PRECISION},
        gaussian_log_likelihood,
        None,
        PARAMETER_NAMES,
        ("gaus", "gaus"),
        PRIOR_MEAN - 8.0 * PRIOR_STDV,
        PRIOR_MEAN + 8.0 * PRIOR_STDV,
        PRIOR_MEAN,
        PRIOR_STDV,
        None,
        chain_count,
        sample_count,
        burn_count,
        pathbase=None,
        typeverb=0,
        typeseed=23,
        booladaptstdp=True,
        boolmakeplot=False,
        boolmakeplotinit=False,
        boolmakeplotfram=False,
        boolmakeplotfinlpost=False,
        makeanim=False,
    )
    mean_post, covariance_post = analytic_posterior()
    distribution_post = multivariate_normal(mean=mean_post, cov=covariance_post)
    distribution_prior = multivariate_normal(mean=PRIOR_MEAN, cov=PRIOR_COVARIANCE)
    estimate = estimate_kl_divergence(
        chains, distribution_post.logpdf, distribution_prior.logpdf
    )
    analytic = analytic_gaussian_kl(
        mean_post, covariance_post, PRIOR_MEAN, PRIOR_COVARIANCE
    )
    figure_path = plot_kl_comparison(estimate, analytic, typefileplot)
    mutual_information = estimate_mutual_information(
        chains, distribution_post.logpdf,
        lambda values: norm.logpdf(values[:, 0], loc=mean_post[0], scale=np.sqrt(covariance_post[0, 0])),
        lambda values: norm.logpdf(values[:, 0], loc=mean_post[1], scale=np.sqrt(covariance_post[1, 1])),
    )
    correlation = covariance_post[0, 1] / np.sqrt(covariance_post[0, 0] * covariance_post[1, 1])
    analytic_mutual_information = float(-0.5 * np.log1p(-correlation**2))
    mutual_information_path = plot_information_comparison(
        mutual_information["mutual_information"], mutual_information["standard_error"],
        analytic_mutual_information, typefileplot, "mutual_information_gaussian",
        r"Mutual information $I(X;Y)$ [nats]", "Posterior parameter dependence",
    )
    return {
        **estimate, "analytic_kl": analytic, "figure_path": figure_path,
        "chains": chains,
        "mutual_information": mutual_information,
        "analytic_mutual_information": analytic_mutual_information,
        "mutual_information_figure_path": mutual_information_path,
    }


def plot_kl_comparison(estimate, analytic, typefileplot):
    """Plot PCAT's Monte Carlo estimate against the analytic Gaussian KL."""
    return plot_information_comparison(
        estimate["kl_divergence"], estimate["standard_error"], analytic,
        typefileplot, "kl_divergence_gaussian",
        r"Forward KL divergence $D_{KL}(P\Vert Q)$ [nats]", "Posterior vs prior",
    )


def plot_information_comparison(value, standard_error, analytic, typefileplot, name, ylabel, category):
    """Compare a sampled information measure with its analytic Gaussian value."""
    figure, axis = plt.subplots(figsize=(5.6, 3.5))
    axis.axhline(analytic, color="#A51C30", ls="--", lw=1.5,
                 label=f"Analytic Gaussian value = {analytic:.3f} nats")
    axis.errorbar(
        [0],
        [value],
        yerr=[2.0 * standard_error],
        fmt="o",
        capsize=4,
        color="#007360",
        label="PCAT estimate ± 2 MCSE",
    )
    axis.set_xticks([0], [category])
    axis.set_ylabel(ylabel)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                frameon=True, fancybox=True, framealpha=1.0)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return save_example_figure(
        figure, EXAMPLE_PATH, name, typefileplot
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbsamp", type=int, default=3000)
    parser.add_argument("--numbburn", type=int, default=1000)
    parser.add_argument("--numbproc", type=int, default=4)
    add_smoke_argument(parser)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    numbsamp = 250 if arguments.smoke else arguments.numbsamp
    numbburn = 200 if arguments.smoke else arguments.numbburn
    result = run_example(numbsamp, numbburn, arguments.numbproc, arguments.typefileplot)
    print(
        f"PCAT D_KL(posterior || prior) = {result['kl_divergence']:.4f} "
        f"± {result['standard_error']:.4f} MCSE; "
        f"analytic = {result['analytic_kl']:.4f} nats; "
        f"ESS = {result['effective_sample_size']:.0f}"
    )
    mutual_information = result["mutual_information"]
    print(
        f"PCAT I(x;y) = {mutual_information['mutual_information']:.4f} "
        f"+/- {mutual_information['standard_error']:.4f} MCSE, "
        f"analytic = {result['analytic_mutual_information']:.4f} nats"
    )


if __name__ == "__main__":
    main()