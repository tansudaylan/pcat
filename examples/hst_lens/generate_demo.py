#!/usr/bin/env python
"""Generate a lensing-style illustration using the PCAT output layout."""

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
    x = np.linspace(-2.0, 2.0, 500)
    y = np.linspace(-2.0, 2.0, 500)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius = np.sqrt(xx**2 + yy**2)
    field = np.exp(-0.5 * radius**2 / 0.35**2)
    field += 0.35 * np.exp(-0.5 * ((xx + 0.7) ** 2 + (yy - 0.2) ** 2) / 0.18**2)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(field, cmap="magma", origin="lower", extent=(-2.0, 2.0, -2.0, 2.0))
    ax.set_title("strong-lens style mock field")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    fig.colorbar(im, ax=ax, label="relative surface brightness")
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

    fig_path = visuals_dir / "hst_lens_demo.png"
    make_figure(fig_path)
    print(f"Wrote {fig_path}")

    try:
        from pcat import main as pcat_main
        cfg = {
            "typeexpr": "HST_WFC3_IR",
            "typedata": "simu",
            "booldiag": False,
            "typeverb": 0,
            "numbswep": 2,
            "numbsamp": 1,
            "boolmakeplot": True,
            "boolmakeplotinit": True,
            "pathbase": str(OUTPUT_ROOT),
            "strgcnfg": "hst_lens_demo",
        }
        pcat_main.sample(**cfg)
        print("PCAT sampling hook executed successfully.")
    except Exception as exc:  # pragma: no cover - illustrative fallback
        print(f"PCAT sampling hook skipped: {exc}")


if __name__ == "__main__":
    main()
