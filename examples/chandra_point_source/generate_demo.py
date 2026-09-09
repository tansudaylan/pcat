#!/usr/bin/env python
"""Generate a point-source illustration using the PCAT output layout."""

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
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 10.0, 30)
    y = rng.uniform(0.0, 10.0, 30)
    flux = 0.8 + rng.random(30) * 2.3

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y, s=np.square(flux) * 18.0, c=flux, cmap="viridis", edgecolors="black", linewidth=0.5)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("point-source mock catalog")
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

    fig_path = visuals_dir / "chandra_point_source_demo.png"
    make_figure(fig_path)
    print(f"Wrote {fig_path}")

    try:
        from pcat import main as pcat_main
        cfg = {
            "typeexpr": "chan",
            "elemtype": ["lghtpnts"],
            "typedata": "simu",
            "booldiag": False,
            "typeverb": 0,
            "numbswep": 2,
            "numbsamp": 1,
            "boolmakeplot": True,
            "boolmakeplotinit": True,
            "pathbase": str(OUTPUT_ROOT),
            "strgcnfg": "chan_demo",
        }
        pcat_main.sample(**cfg)
        print("PCAT sampling hook executed successfully.")
    except Exception as exc:  # pragma: no cover - illustrative fallback
        print(f"PCAT sampling hook skipped: {exc}")


if __name__ == "__main__":
    main()
