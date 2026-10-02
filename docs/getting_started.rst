Getting started
===============

Requirements
------------

PCAT requires Python 3.10 or newer. Its runtime dependencies are declared in
``pyproject.toml`` and include NumPy, SciPy, Astropy, Matplotlib, h5py, Numba,
Pillow, Seaborn, cloudpickle, Chalcedon, and TDpy. The optional ``examples``
dependency group adds dynesty, emcee, and Nicomedia for the sampler-comparison
and stellar-flare workflows.

For development with sibling repositories, install TDpy and PCAT in editable
mode:

.. code-block:: bash

   cd /path/to/tdpy
   python -m pip install -e .
   cd /path/to/pcat
   python -m pip install -e .

Install the documentation dependencies with:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Install dependencies used by every maintained local example with:

.. code-block:: bash

   python -m pip install -e ".[examples]"

First run
---------

Run the compact Chandra-style demonstration from the repository root:

.. code-block:: bash

   python examples/chandra_point_source_catalog/chandra_point_source_catalog.py

The script uses :func:`pcat.demo.run_pipeline_demo` to supply plotting defaults
and calls :func:`pcat.sampling.sample`. Its source is intentionally short and is a
runnable configuration template.

A programmatic demonstration has the same form:

.. code-block:: python

   from pathlib import Path

   from pcat.demo import run_pipeline_demo

   run_pipeline_demo(
       Path("analysis/point_sources"),
       typeexpr="chan",
       typeelem=["lghtpnts"],
       truenumbelempop0=2,
       fittminmnumbelempop0=1,
       fittmaxmnumbelempop0=3,
       dicttrue={"typeelemspateval": ["full"]},
       dictfitt={"typeelemspateval": ["full"]},
       numbsidecart=8,
       numbswep=100,
       numbsamp=10,
       numbswepplot=10,
       strgcnfg="point_sources",
   )

This configuration is a pipeline check, not a scientifically converged chain.
Production analyses require enough samples, convergence checks, and
problem-specific prior validation.

Runtime paths
-------------

``pathbase`` is the PCAT repository root. Each ``strgcnfg`` run becomes a
self-contained child of ``pathbase/pcat_runs`` containing ``data/`` and
``visuals/``. A ``pathbase`` that already identifies ``pcat_runs`` or ends in
``strgcnfg`` is used directly. If ``pathbase`` is omitted, PCAT uses
``PCAT_PATH``.

Configuration principles
------------------------

* Set ``typedata="simu"`` for generated data and ``typedata="inpt"`` for
  supplied data.
* Use a filesystem-safe ``strgcnfg`` run tag.
* Set deterministic seeds for reproducible simulations and chains.
* Keep true-model settings in ``dicttrue`` and fitted-model settings in
  ``dictfitt`` when defaults would otherwise overwrite the intended model.
* Use ``boolmakeplotinit``, ``boolmakeplotfram``, ``boolmakeplotfinlpost``, and
  ``makeanim`` to control visual products.
* Treat short smoke configurations as integration checks rather than evidence
  for scientific claims.