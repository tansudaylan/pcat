Examples
========

Every Python example under ``examples/`` has a corresponding Jupyter notebook
in the same directory. Open the notebook in VS Code or Jupyter to run it
interactively. The Python scripts remain available for automated and command-line
runs. The batch-command notebook lists its external commands without executing
them.

Pipeline demonstrations
-----------------------

``chandra_point_source_catalog``
   Compact Chandra-style point-source detection with simulated data.

``gaussian_mixture_catalog``
   Two-dimensional variable-width Gaussian-mixture inference with genuine
   birth and death transitions.

``simulated_hst_strong_lens``
   Hubble Space Telescope Wide Field Camera 3 image inference with lens mass,
   foreground lens-galaxy emission, source emission, and lensed emission.

   .. code-block:: bash

      python examples/simulated_hst_strong_lens/simulated_hst_strong_lens.py

``simulated_rubin_cluster_lens``
   Simulated Rubin-like cluster-scale lens with a Gaussian point-spread function.
   The notebook samples the Einstein radius and source position with PCAT's
   Poisson image likelihood, then shows the fitted image, residuals, and
   posterior parameter distributions. It does not analyze Rubin observations.

``rubin_dp1_confirmed_strong_lenses``
   Rubin Science Platform notebook for real Data Preview 1 imaging. It queries
   confirmed SIMBAD lens-system classifications, checks exact DP1 image
   footprints, retrieves every resulting deep-coadd cutout, displays the full
   matched sample, and passes each usable cutout to PCAT. DP1 requires Rubin
   data rights, and SIMBAD does not constitute a complete census of lenses.

Run the maintained local smoke suite with:

.. code-block:: bash

   python examples/run_all_examples.py

The suite runs the compact image and publication-inspired simulations, the
stellar-flare and Voigt smoke configurations, the Roman and catalog-association
utilities, the proposal profiler, and the sampler comparison. It regenerates
their outputs. It intentionally excludes workflows requiring authenticated
services or mission downloads, the long radial-velocity analysis, and notebooks
whose role is interactive inspection.

Posterior samples from twelve maintained examples are synchronized below. The
panels span Gaussian mixtures, X-ray and gamma-ray point sources, strong lenses,
Sloan Digital Sky Survey and Herschel deblending, stellar flares, and spectral
lines. Every panel uses simulated data and a fixed intensity scale across its
frames.

.. image:: ../examples/pcat_posterior_samples.gif
   :alt: Animated posterior samples from twelve maintained PCAT examples
   :width: 760px
   :align: center

Publication and instrument examples
-----------------------------------

``daylan+2017_fermi_point_sources``
   Scaled mock-catalog reproduction of Daylan, Portillo, and Finkbeiner (2017).
   The smoke mode preserves the inference pattern while reducing the source
   count and chain length.

   .. code-block:: bash

      python 'examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py' --smoke --fresh

``daylan+2018_strong_lens_subhalos``
   Simulated Hubble Space Telescope Wide Field Camera 3 strong-lens analysis
   comparing a variable subhalo catalog with a fixed one-subhalo fit.

   .. code-block:: bash

      python 'examples/daylan+2018_strong_lens_subhalos/daylan+2018_strong_lens_subhalos.py' --smoke
      python 'examples/daylan+2018_strong_lens_subhalos/daylan+2018_strong_lens_subhalos.py' --smoke --one-subhalo

``voigt_spectral_line_catalog``
   Transdimensional detection of an unknown number of Voigt-profile emission
   lines in simulated spectral data.

   .. code-block:: bash

      python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration nomi

``jwst_miri_ngc7027_line_catalog``
   Transdimensional Voigt-line catalog for the public James Webb Space Telescope
   (JWST) MIRI Medium Resolution Spectrometer spectrum of NGC 7027 from program
   1523. The script downloads the level-3 spectrum from the Mikulski Archive for
   Space Telescopes when it is absent, maps the measured flux and uncertainty to
   effective Poisson counts, and fits a line catalog over 5.30--5.535 microns.

   .. code-block:: bash

      python examples/jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py --numbswep 200000

``variable_number_exoplanets_radial_velocity``
   Simulated six-year, two-instrument radial-velocity series containing three
   injected planets. PCAT samples a variable catalog of Keplerian signals while
   marginalizing instrument offsets and stellar jitter. The default 100,000
   sweeps are intended for the full demonstration; ``--smoke`` provides a short
   pipeline check.

   .. code-block:: bash

      python examples/variable_number_exoplanets_radial_velocity/variable_number_exoplanets_radial_velocity.py --smoke

``variable_number_stellar_flares``
   Simulated one-day, TESS-like light curve with fast-rise, exponential-decay
   flares. PCAT fits a variable catalog of symmetric Voigt bursts, making the
   profile mismatch and its effect on inferred component count explicit.

   .. code-block:: bash

      python examples/variable_number_stellar_flares/variable_number_stellar_flares.py --smoke

``roman_strong_lens_perturber_catalog``
   Seeded Roman strong-lens population benchmark and catalog-detection
   diagnostic.

   .. code-block:: bash

      python examples/roman_strong_lens_perturber_catalog/roman_strong_lens_perturber_catalog.py

   .. image:: ../examples/roman_strong_lens_perturber_catalog/visuals/roman_strong_lens_perturber_catalog.png
      :alt: Simulated Roman strong-lens images, residual, and catalog posterior
      :width: 720px
      :align: center

   The four panels trace one simulated detector image through the macro-lens
   model and residual, then summarize catalog posterior probabilities across
   the seeded lens population.

``fermi_lat_pg1553_event_filter``
   Fermi Large Area Telescope event-filter configuration for PG 1553+113. This
   utility requires the Fermi Science Tools and local mission data.

``portillo+2017_crowded_sdss_m2``
   Simulated Sloan Digital Sky Survey crowded-field deblending patterned after
   Portillo et al. (2017).

``feder+2020_multiband_sdss_deblending``
   Simulated multiband Sloan Digital Sky Survey deblending patterned after
   Feder et al. (2020).

``butler+2022_spire_sz_component_separation``
   Simulated Herschel SPIRE point-source and Sunyaev-Zeldovich component
   separation patterned after Butler et al. (2022).

``feder+2023_point_diffuse_spire``
   Simulated joint point-source and diffuse-emission inference patterned after
   Feder et al. (2023).

``hall+2026_herschel_dsfg_multiplicity``
   Simulated Herschel dusty star-forming galaxy multiplicity inference patterned
   after Hall et al. (2026).

Analysis utilities
------------------

``sampler_comparison_emcee_dynesty``
   Controlled simulated radial-velocity comparison of PCAT, emcee, and dynesty.
   The fixed-dimensional comparison uses the same likelihood and priors for all
   samplers. The variable-dimensional comparison contrasts one PCAT catalog run
   with separate dynesty evidence calculations for each planet count; emcee is
   excluded from that portion because it does not compare dimensions.

   .. code-block:: bash

      python examples/sampler_comparison_emcee_dynesty/sampler_comparison_emcee_dynesty.py --part fixed

``proposal_profiling``
   Simulated Voigt-line run reporting proposal attempts, acceptance fractions,
   time per sweep, and the recorded timing breakdown for major pipeline phases.

   .. code-block:: bash

      python examples/proposal_profiling/proposal_profiling.py --numbswep 20000

``proposal_state_animation``
   Short simulated Voigt-line run producing two complementary animations. The
   retained-state sequence follows the configured plotting cadence, while the
   candidate sequence shows every attempted move before acceptance or rejection.

   .. code-block:: bash

      python examples/proposal_state_animation/proposal_state_animation.py --numbswep 24

``population_grid``
   Seeded simulated populations used to demonstrate PCAT's marginal and pairwise
   posterior plotting. These arrays are illustrative rather than observations.

``legacy_external_analysis_commands``
   Command-reference notebook for external analyses. It records commands but
   does not execute them.

``catalog_association_completeness_purity``
   Seeded completeness and purity calculation for coordinate-and-value catalog
   matching across a range of association radii.

   .. code-block:: bash

      python examples/catalog_association_completeness_purity/catalog_association_completeness_purity.py

   .. image:: ../examples/catalog_association_completeness_purity/visuals/catalog_association_completeness_purity.png
      :alt: Catalog completeness and purity across association radii
      :width: 640px
      :align: center

Scientific interpretation
-------------------------

Generated examples are deterministic pipeline demonstrations unless their
README explicitly states otherwise. Synthetic outputs validate computation and
visualization but do not constitute measurements of real astrophysical systems.
The daylan+2017_fermi_point_sources README distinguishes its scaled simulation from the
archival Fermi-LAT analysis in the publication.