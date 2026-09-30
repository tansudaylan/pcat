"""Shared configurations for scaled PCAT publication examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .demo import run_pipeline_demo


def run_optical_catalog_example(
    output_root: Path,
    run_name: str,
    number_sources: int,
) -> None:
    """Run a scaled five-band SDSS crowded-field catalog calculation."""
    run_pipeline_demo(
        output_root,
        typeexpr="sdss",
        typeelem=["lghtpnts"],
        truenumbelempop0=number_sources,
        fittminmnumbelempop0=max(1, number_sources // 2),
        fittmaxmnumbelempop0=2 * number_sources,
        dicttrue={"typeelemspateval": ["full"]},
        dictfitt={"typeelemspateval": ["full"]},
        numbsidecart=24,
        strgexpo=50.0,  # [arbitrary exposure units]
        probspmr=0.0,
        strgcnfg=run_name,
    )


def _diffuse_template(number_side: int, band_count: int = 3) -> np.ndarray:
    """Return a normalized multiband, cirrus-like spatial template."""
    coordinate = np.linspace(-1.0, 1.0, number_side)
    xpos, ypos = np.meshgrid(coordinate, coordinate, indexing="ij")
    structure = 1.0 + 0.35 * np.sin(2.0 * np.pi * xpos) * np.cos(np.pi * ypos)
    structure += 0.2 * np.exp(-0.5 * ((ypos - 0.3 * xpos) / 0.18) ** 2)
    structure /= structure.mean()
    colors = np.array([1.0, 0.8, 0.6])[:band_count, None, None]
    template = (colors * structure[None, :, :]).reshape(band_count, number_side**2, 1)
    return np.broadcast_to(template, (band_count, number_side**2, 2)).copy()


def run_spire_component_example(
    output_root: Path,
    run_name: str,
    number_sources: int,
    diffuse_scale: float,
) -> None:
    """Run a scaled three-band point-source and diffuse-template calculation."""
    from . import main

    number_side = 24
    band_count = 3
    diffuse = _diffuse_template(number_side, band_count)
    uniform = np.ones_like(diffuse)
    psf_shape = np.empty((band_count, 2, 5))
    psf_shape[...] = np.array([0.78, 2.4, 2.2, 2.0, 0.82])
    psf_parameters = np.empty(psf_shape.size)
    for event_class in range(2):
        for band in range(band_count):
            start = event_class * 5 * band_count + band * 5
            psf_parameters[start : start + 5] = psf_shape[band, event_class]
    model = {
        "typeelemspateval": ["full"],
        "listnamediff": ["back0000", "back0001"],
        "sbrtbacknorm": [uniform, diffuse_scale * diffuse],
        "psfpexpr": psf_parameters,
    }
    main.sample(
        typeexpr="ferm",
        typedata="simu",
        typepixl="cart",
        boolforccart=True,
        boolbindspat=True,
        booldiag=False,
        typeelem=["lghtpnts"],
        truenumbelempop0=number_sources,
        numbelempop0reg0=number_sources,
        fittminmnumbelempop0=max(1, number_sources // 2),
        fittmaxmnumbelempop0=2 * number_sources,
        dicttrue=model,
        dictfitt=model,
        numbsidecart=number_side,
        maxmgangdata=np.deg2rad(0.05),
        strgexpo=2.0e8,  # [cm^2 s]
        indxenerincl=np.arange(band_count),
        indxdqltfull=np.arange(2),
        indxdqltincl=np.arange(2),
        fermscalfact=np.full((band_count, 2), np.deg2rad(18.0 / 3600.0)),
        probspmr=0.0,
        numbswep=4,
        numbsamp=2,
        numbswepplot=1,
        boolcondcatl=False,
        boolmakeplot=True,
        boolmakeplotinit=True,
        boolmakeplotfram=True,
        boolmakeplotfinlpost=True,
        makeanim=False,
        typefileplot="png",
        typeverb=0,
        strgcnfg=run_name,
        pathbase=str(output_root),
    )