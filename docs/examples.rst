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

Run the image analyses, Roman benchmark, and reduced Voigt calculation with:

.. code-block:: bash

   python examples/run_all_examples.py

Posterior samples from four maintained examples are synchronized below. The
panels show model intensity for a Gaussian mixture, model counts for point
sources and a strong lens, and the model with data for spectral lines.

.. image:: ../examples/pcat_posterior_samples.gif
   :alt: Animated posterior samples from four maintained PCAT examples
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

Analysis utilities
------------------

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