# Daylan et al. 2017 mock analysis

This example reproduces the central synthetic analysis in Section IV of
[Daylan, Portillo, and Finkbeiner (2017)](https://doi.org/10.3847/1538-4357/aa679e),
published as ApJ 839, 4. It generates a Fermi-LAT-like Poisson image on a
40 degree by 40 degree Cartesian projection about the North Galactic Pole from
a uniform population of 300 point sources whose flux distribution has the
published power-law slope of -1.8. The image uses the published three energy
bins and two event classes. It then runs transdimensional PCAT inference and
writes PCAT's data, model, residual, flux-distribution, association, population,
and hyperparameter diagnostics.

Run the publication-scale configuration with:

```bash
python 'examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py' --fresh
```

Use `--typefileplot pdf` for PDF figures. A fast end-to-end check is available:

```bash
python 'examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py' --smoke --fresh
```

The original analysis used Pass 7 source-class exposure from weeks 9--217 and
the mission diffuse template. Those archival inputs are not distributed with
this repository. This self-contained example uses constant exposure and a
deterministic, dust-like high-latitude morphology as an explicit synthetic proxy
for the diffuse template. It does not reproduce the paper's numerical Fermi-LAT
measurements or claim agreement with its posterior values.

The notebook runs the smoke configuration, loads the finalized posterior, and
saves posterior-derived count maps, source catalogs, source-count probabilities,
and likelihood traces directly under `visuals/`.

![Simulated counts, posterior model, and residual](visuals/fermi_posterior_count_maps.png)

![Posterior source positions and fluxes](visuals/fermi_posterior_source_catalog.png)

![Posterior probability of the source count](visuals/fermi_posterior_source_count.png)

![Posterior log-likelihood trace](visuals/fermi_posterior_log_likelihood.png)