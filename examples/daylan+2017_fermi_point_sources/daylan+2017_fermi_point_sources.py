#!/usr/bin/env python3
"""Run a synthetic PCAT analysis inspired by Daylan et al. (2017)."""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
import hashlib
import json
from tdpy.cli import add_plot_arguments
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "daylan2017_mock"
OUTPUT_ROOT = Path(__file__).resolve().parent
FIELD_HALF_WIDTH_DEG = 20.0  # [deg]
ENERGY_BIN_EDGES_GEV = np.array([0.3, 1.0, 3.0, 10.0])  # [GeV]
NUMBER_EVENT_CLASSES = 2


def build_ngpc_diffuse_template(number_side: int) -> np.ndarray:
    """Return a normalized, deterministic proxy for high-latitude dust emission."""
    pixel_size_deg = 2.0 * FIELD_HALF_WIDTH_DEG / number_side  # [deg]
    coordinate_deg = np.linspace(  # [deg]
        -FIELD_HALF_WIDTH_DEG + 0.5 * pixel_size_deg,
        FIELD_HALF_WIDTH_DEG - 0.5 * pixel_size_deg,
        number_side,
    )
    xpos_deg, ypos_deg = np.meshgrid(coordinate_deg, coordinate_deg, indexing="ij")

    # Broad, curved filaments approximate the dust-traced morphology of the
    # Fermi diffuse model at high Galactic latitude without claiming archival data.
    filament_one = np.exp(
        -0.5 * ((ypos_deg - 0.18 * xpos_deg - 3.0 * np.sin(xpos_deg / 7.0)) / 2.8) ** 2
    )
    filament_two = np.exp(
        -0.5 * ((xpos_deg + 0.28 * ypos_deg - 7.0 * np.cos(ypos_deg / 8.0)) / 3.6) ** 2
    )
    structure = (
        0.55
        + 0.30 * filament_one
        + 0.18 * filament_two
        + 0.07 * np.sin(xpos_deg / 3.4) * np.cos(ypos_deg / 4.7)
    )
    structure /= np.mean(structure)

    template = np.broadcast_to(
        structure.reshape(1, number_side**2, 1),
        (ENERGY_BIN_EDGES_GEV.size - 1, number_side**2, NUMBER_EVENT_CLASSES),
    )
    return np.array(template, copy=True)


def build_fermi_psf() -> tuple[np.ndarray, np.ndarray]:
    """Return approximate double-King parameters and angular scales for the mock."""
    angular_scale_deg = np.array(  # [deg]
        [
            [0.80, 1.20],
            [0.32, 0.48],
            [0.16, 0.24],
        ]
    )
    shape = np.empty((ENERGY_BIN_EDGES_GEV.size - 1, NUMBER_EVENT_CLASSES, 5))
    shape[...] = np.array([0.78, 2.4, 2.2, 2.0, 0.82])
    parameters = np.empty(shape.size)
    for event_class in range(NUMBER_EVENT_CLASSES):
        for energy_bin in range(ENERGY_BIN_EDGES_GEV.size - 1):
            start = event_class * 5 * (ENERGY_BIN_EDGES_GEV.size - 1) + energy_bin * 5
            parameters[start : start + 5] = shape[energy_bin, event_class]
    return parameters, np.deg2rad(angular_scale_deg)


def build_configuration(smoke: bool = False, typefileplot: str = "png",
                        quick: bool = False) -> dict[str, object]:
    """Return the published mock-population assumptions in a runnable PCAT setup."""
    number_sources = 40 if smoke else 300
    reduced_run = smoke or quick
    number_sweeps = 10_000 if reduced_run else 1_000_000
    number_samples = 1_000 if reduced_run else 10_000
    number_side = 48 if reduced_run else 100
    diffuse_template = build_ngpc_diffuse_template(number_side)
    psf_parameters, psf_scale = build_fermi_psf()
    model_overrides = {
        "typeelemspateval": ["full"],
        "listnamediff": ["back0000", "back0001"],
        "sbrtbacknorm": [np.ones_like(diffuse_template), diffuse_template],
        "psfpexpr": psf_parameters,
    }
    return {
        "typeexpr": "ferm",
        "typedata": "simu",
        "typepixl": "cart",
        "boolforccart": True,
        "boolbindspat": True,
        "booldiag": False,
        "typeelem": ["lghtpnts"],
        "numbelempop0reg0": number_sources,
        "truenumbelempop0": number_sources,
        "truemaxmnumbelempop0": number_sources,
        "truefluxdistslop": -1.8,
        "fittminmnumbelempop0": max(1, number_sources // 2),
        "fittmaxmnumbelempop0": 2 * number_sources,
        "dicttrue": model_overrides,
        "dictfitt": model_overrides,
        "typeseed": 160704637,
        "typeseedelem": 1607,
        "numbsidecart": number_side,
        "maxmgangdata": np.deg2rad(FIELD_HALF_WIDTH_DEG),
        "strgexpo": 1.0e10,  # [cm^2 s]
        "indxdqltfull": np.arange(NUMBER_EVENT_CLASSES),
        "indxdqltincl": np.arange(NUMBER_EVENT_CLASSES),
        "indxenerincl": np.arange(ENERGY_BIN_EDGES_GEV.size - 1),
        "fermscalfact": psf_scale,
        "lablxpos": r"\theta_1",
        "lablypos": r"\theta_2",
        "numbswep": number_sweeps,
        "numbsamp": number_samples,
        "numbburn": number_sweeps // 3,
        "numbproc": 4,
        "inittype": "rand",
        "numbframanim": 24,
        "numbswepplot": 1_000 if smoke else 100_000,
        "probtran": 0.7,
        "probspmr": 0.4,
        "boolcondcatl": True,
        "boolmakeplot": True,
        "boolmakeplotinit": True,
        "boolmakeplotfram": True,
        "boolmakeplotfinlpost": True,
        "makeanim": True,
        "typefileplot": typefileplot,
        "typeverb": 0,
        "strgcnfg": RUN_NAME,
        "pathbase": str(OUTPUT_ROOT),
    }


def run_reproduction(smoke: bool = False, typefileplot: str = "png",
                     quick: bool = False) -> object:
    """Run PCAT and write its chain and figures under the example directory."""
    from pcat.sampling import sample

    configuration = build_configuration(smoke, typefileplot, quick)
    configuration["strgcnfg"] = RUN_NAME + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return sample(**configuration)


def read_posterior(run: object) -> object:
    """Load the finalized posterior produced by a PCAT run."""
    from pcat.main import readfile

    return readfile(run.pathoutpcnfg + "gdatfinlpost")


def render_source_frames(posterior: object, *, provenance: dict | None = None) -> list[Path]:
    """Overlay PCAT source catalogs, from the prior draw to posterior samples, on the simulated count map."""
    import matplotlib.pyplot as plt
    from pcat.plotting import animation_phase_label, animation_states

    if np.ma.is_masked(posterior.cntpdata):
        raise ValueError("Fermi source frames require complete count data")
    data = np.asarray(posterior.cntpdata, dtype=float)
    if data.ndim != 3 or not data.size or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Fermi source frames require finite nonnegative energy/pixel/class counts")
    counts = data.sum(axis=(0, 2))
    side = int(np.sqrt(counts.size))
    if side * side != counts.size:
        raise ValueError("Fermi source frames require a square count map")
    counts = counts.reshape(side, side)
    minimum = float(np.arcsinh(counts.min()))
    maximum = max(float(np.arcsinh(counts.max())), minimum + np.finfo(float).eps)
    output_directory = OUTPUT_ROOT / "visuals"
    output_directory.mkdir(parents=True, exist_ok=True)
    time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    snapshots = animation_states(posterior, require_nonzero_catalog=True)
    maximum_sweep = max(snapshot["cntrswep"] for snapshot in snapshots)
    count_width = max(3, len(str(max(len(snapshot["dictelem"][0]["xpos"]) for snapshot in snapshots))))
    fluxes = np.concatenate([np.asarray(snapshot["dictelem"][0]["flux"]) for snapshot in snapshots])
    low, high = np.log10(fluxes.min()), np.log10(fluxes.max())
    for snapshot in snapshots:
        sources = snapshot["dictelem"][0]
        sizes = 20 + 120 * (np.log10(sources["flux"]) - low) / max(high - low, 1e-9)  # [point^2]
        figure, axis = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        artist = axis.imshow(np.arcsinh(counts).T, origin="lower", cmap="Greys", vmin=minimum,
                     vmax=maximum, extent=[-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG] * 2)
        axis.scatter(np.rad2deg(sources["xpos"]), np.rad2deg(sources["ypos"]), s=sizes,
                     facecolors="none", edgecolors="#c13d31", linewidths=1.5,
                     label=f"PCAT sources ({len(sources['xpos']):{count_width}d})")
        axis.set(xlim=(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG),
                 ylim=(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG),
                  xlabel="Field offset 1 [deg]", ylabel="Field offset 2 [deg]")
        axis.set_title(animation_phase_label(snapshot, maximum_sweeps=maximum_sweep),
                       loc="left", fontfamily="monospace", fontsize=10)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), facecolor="white", framealpha=1,
                    prop={"family": "monospace"})
        axis.grid(False)
        figure.colorbar(artist, ax=axis, label="asinh(total mock counts)", shrink=0.8)
        path = output_directory / f"fermi_ngpc_sources_{time_tag}_swep{snapshot['cntrswep']:09d}.png"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    manifest = {
        "input_kind": "simulated Daylan+2017 northern Galactic cap mock",
        "projection": "sum over every energy bin and event class",
        "count_cube_shape": list(data.shape),
        "count_cube_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "projected_count_min": float(counts.min()), "projected_count_max": float(counts.max()),
        "display_limits_asinh": [minimum, maximum],
        "native_frame_paths": [path.name for path in paths],
        "native_frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_run_provenance": provenance,
    }
    manifest_path = output_directory / f"fermi_ngpc_frames_{time_tag}.json"
    print(f"Writing to {manifest_path}...")
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small pipeline check.")
    parser.add_argument("--quick", action="store_true", help="Run the 300-source mock at reduced resolution and depth.")
    parser.add_argument("--fresh", action="store_true", help="Use a fresh run; earlier output is preserved.")
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run = run_reproduction(arguments.smoke, arguments.typefileplot, arguments.quick)
    render_source_frames(read_posterior(run))
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())