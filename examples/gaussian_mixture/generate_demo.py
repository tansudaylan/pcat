#!/usr/bin/env python
"""Generate a compact Gaussian-mixture illustration using PCAT path setup."""

from __future__ import annotations

import os
from pathlib import Path
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pcat

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "pcat-output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["PCAT_DATA_PATH"] = str(OUTPUT_ROOT)


def make_figure(path: Path) -> None:
    x1 = np.random.default_rng(11).normal(loc=0.0, scale=0.7, size=600)
    x2 = np.random.default_rng(22).normal(loc=3.0, scale=0.9, size=600)
    values = np.concatenate([x1, x2])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=26, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axvline(0.0, color="black", ls="--", lw=1.0)
    ax.axvline(3.0, color="black", ls="--", lw=1.0)
    ax.set_xlabel("latent coordinate")
    ax.set_ylabel("count")
    ax.set_title("Gaussian-mixture toy catalog")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    gdat = types.SimpleNamespace(
        pathbase=str(OUTPUT_ROOT),
        liststrgfeatparalist=["minm", "maxm", "scal"],
        liststrgfeatpara=["minm", "maxm", "scal"],
        listscaltype=["self", "logt"],
        numbstdvgaus=4.0,
    )
    pcat.setup_pcat(gdat)
    visuals_dir = Path(gdat.pathvisu)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    fig_path = visuals_dir / "gaussian_mixture_demo.png"
    make_figure(fig_path)
    print(f"Wrote {fig_path}")

    try:
        from pcat import main as pcat_main
        cfg = {
            "typeexpr": "gmix",
            "booldiag": False,
            "typeverb": 0,
            "numbswep": 2,
            "numbsamp": 1,
            "boolmakeplot": True,
            "boolmakeplotinit": True,
            "boolmakeplotfram": True,
            "pathbase": str(OUTPUT_ROOT),
            "strgcnfg": "gmix_demo",
        }
        pcat_main.sample(**cfg)
        print("PCAT sampling hook executed successfully.")
    except Exception as exc:  # pragma: no cover - illustrative fallback
        print(f"PCAT sampling hook skipped: {exc}")


if __name__ == "__main__":
    main()
