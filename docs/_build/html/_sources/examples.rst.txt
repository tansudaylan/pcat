Maintained examples
===================

Each example owns one subdirectory under ``examples/``. Source, generated data,
and visuals stay together so outputs can be traced to the configuration that
created them.

Pipeline demonstrations
-----------------------

``chan_demo``
   Compact Chandra-style point-source detection with simulated data.

``gmix_demo``
   Two-dimensional variable-width Gaussian-mixture inference with genuine
   birth and death transitions.

``hst_lens``
   Hubble Space Telescope Wide Field Camera 3 lensing-style image inference.

Run all maintained pipeline checks and verify their figures with:

.. code-block:: bash

   python examples/run_examples.py

Publication and instrument examples
-----------------------------------

``Daylan+2017``
   Scaled mock-catalog reproduction of Daylan, Portillo, and Finkbeiner (2017).
   The smoke mode preserves the inference pattern while reducing the source
   count and chain length.

   .. code-block:: bash

      python 'examples/Daylan+2017/generate_reproduction.py' --smoke --fresh

``voigt-profile``
   Simulated nominal and high-signal Voigt-profile line detection.

   .. code-block:: bash

      python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration nomi

``roman_lens_catalog``
   Seeded Roman strong-lens population benchmark and catalog-detection
   diagnostic.

   .. code-block:: bash

      python examples/roman_lens_catalog/roman_lens_catalog_diagnostic.py

``fermi_lat_pg1553``
   Fermi Large Area Telescope event-filter configuration for PG 1553+113. This
   utility requires the Fermi Science Tools and local mission data.

Scientific interpretation
-------------------------

Generated examples are deterministic pipeline demonstrations unless their
README explicitly states otherwise. Synthetic outputs validate computation and
visualization but do not constitute measurements of real astrophysical systems.
The Daylan+2017 README distinguishes its maintained scaled simulation from the
archival Fermi-LAT analysis in the publication.