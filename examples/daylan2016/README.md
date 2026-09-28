# Daylan et al. 2016 mock analysis

This example reproduces the central synthetic analysis in Section IV of
[Daylan, Portillo, and Finkbeiner (2016)](https://arxiv.org/abs/1607.04637),
later published as ApJ 839, 4 (2017). It generates a Poisson image from a
uniform population of 300 point sources whose flux distribution has the
published power-law slope of -1.8, runs transdimensional PCAT inference, and
writes PCAT's data, model, residual, flux-distribution, association, population,
and hyperparameter diagnostics.

Run the publication-scale configuration with:

```bash
python examples/daylan2016/generate_reproduction.py --fresh
```

Use `--typefileplot pdf` for PDF figures. A fast end-to-end check is available:

```bash
python examples/daylan2016/generate_reproduction.py --smoke --fresh
```

The original analysis used three Fermi-LAT energy bins, two Pass 7 point-spread
function classes, a 40 degree by 40 degree North Galactic Pole field, weeks
9--217 exposure, and mission diffuse templates. Those archival inputs are not
distributed with this repository. This example therefore reproduces the
paper's simulated population and PCAT analysis pattern on the maintained
Cartesian point-source pipeline. It does not reproduce the paper's numerical
Fermi-LAT measurements or claim agreement with its posterior values.