Likelihood configurations
=========================

PCAT separates data likelihoods from transdimensional catalog postprocessing.
Every transdimensional model produces a catalog at each retained draw. With
the default ``boolcondcatl=True``, final processing builds a condensed posterior
catalog by associating recurrent supported spatial elements across those
sampled catalogs. Associations stay within each population, even when elements
from different populations overlap spatially. Optional matching to an external reference catalog follows
this summary. Condensation and either kind of association are posterior
products, not likelihoods. Set ``boolcondcatl=False`` to skip condensation.
The current implementation does not spatially condense one-dimensional line,
flare, or Keplerian catalogs.

Image likelihoods
-----------------

The image workflow predicts counts in pixels from a source model, exposure,
background emission, and instrument response. Point sources, extended emission,
foreground light, and gravitational lenses are components of this one image
likelihood, not separate likelihood families.

Data and pixelization
~~~~~~~~~~~~~~~~~~~~~

Set ``typedata="simu"`` for generated counts or ``typedata="inpt"`` for
supplied data. For input data, ``strgexprsbrt`` identifies the surface-brightness
or count-rate FITS input. Exposure can be constant with ``typeexpo="cons"``
and ``strgexpo`` or read from a FITS map with ``typeexpo="file"`` and
``strgexpo``. PCAT converts the input to expected counts using the exposure and
its configured units before evaluating the likelihood.

``typepixl="cart"`` selects a Cartesian image. Set ``numbsidecart`` to choose
its side length. ``boolforccart=True`` forces a square grid when pixel
coordinates are supplied directly. ``typepixl="heal"`` selects HEALPix data
and uses ``numbsideheal`` for its resolution. Fermi-LAT defaults to HEALPix.
Other image experiments default to Cartesian pixels.

Poisson count-image likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``liketype="pois"`` is the default for count data. Omitting the data-only
factorial, its log likelihood is

.. math::

   \log L_{\mathrm{Pois}} = \sum_i D_i \log M_i - M_i,

where ``D`` is the observed count map and ``M`` is the predicted count map.
Gaussian image likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``liketype="gaus"`` for the built-in Gaussian residual likelihood. Supply
``varidata`` as a per-pixel variance map. If omitted, PCAT uses the observed
counts with a floor of one:

.. math::

   \log L_{\mathrm{Gaus}} = -\frac{1}{2}\sum_i \frac{(D_i-M_i)^2}{V_i}.

Source and lens components
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure populations with ``typeelem`` and their true and fitted settings in
``dicttrue`` and ``dictfitt``. Common image components include:

* ``lghtpnts`` for point sources with sampled positions and fluxes.
* ``lghtgausbgrd`` for extended Gaussian emission components.
* ``lens`` for lens deflectors and their associated mass parameters.
* ``clusvari`` for variable-width Gaussian components represented in an image count map.
* ``typeemishost="sers"`` selects a Sérsic foreground lens-galaxy light model.
  Set ``"none"`` to omit host emission.
* ``boollenssubh=True`` to include lens-substructure deflectors where the
  selected lens configuration supports them.

The ``typespatdist`` or ``spatdisttype`` setting selects the spatial prior for
each population. Source flux priors are selected with ``typeprioflux`` and
bounded or customized with the corresponding per-element limits and
``limtparaelem`` where available. HST strong-lens examples combine a fixed
macro lens and source/host light with a variable ``lens`` population for
additional deflectors.

Background emission and exposure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiment defaults define the available background components. Override their
templates with ``sbrtbacknorm`` in ``dicttrue`` or ``dictfitt``. The
``listnamediff`` entry selects the diffuse components added to the model. Names
such as ``back0000`` refer to entries in the template list. ``boolspecback``
controls whether a background amplitude is shared spectrally or varies by
energy bin. Fermi-LAT configurations can use a uniform component, a diffuse
template, or the ``anlytype="bfun"`` background-function basis. Exposure
controls how model brightness becomes expected counts and can be masked with
``listmask`` and ``typemaskexpo``.

Point-spread functions
~~~~~~~~~~~~~~~~~~~~~~

``typeevalpsfn`` controls where the PSF is applied:

* ``"none"`` leaves the source model unconvolved.
* ``"kern"`` applies a local PSF kernel to supported point-source elements.
* ``"conv"`` convolves eligible image components.
* ``"full"`` enables both kernel evaluation and image convolution where
  supported.

The experiment selects a PSF profile family through ``typemodlpsfn``. Current
experiment defaults include single Gaussian, single King, and double King
profiles. Supply profile parameters through ``psfpexpr`` in ``dicttrue`` and
``dictfitt``. Set ``boolmodipsfn=True`` when PSF parameters should vary in the
fit. Fermi-LAT can also use its energy- and event-class-dependent PSF scale
``fermscalfact``. PSF and source-kernel options depend on pixelization and
element type. Unsupported combinations should use ``typeevalpsfn="none"``.

Subpixel evaluation
~~~~~~~~~~~~~~~~~~~

``kernevaltype="ulip"`` evaluates the selected PSF profile at each source-to-
pixel angular separation. For Cartesian images, ``kernevaltype="bspx"`` instead
fits a two-dimensional cubic polynomial in the source's subpixel shifts and
evaluates its coefficients at the requested pixels. We compare the cubic
approximation with direct PSF evaluation at interleaved phase samples. Cells
with sampled errors above 0.1 percent of the profile peak use direct evaluation.
Nonfinite or negative polynomial values and pixels outside the fitted table
also use direct evaluation. The sampled error applies to tested phases and does
not bound errors over the full profile. The implementation does not renormalize
or clip the fitted PSF. ``bspx`` requires ``typepixl="cart"``.

The setup cost is paid once for each fitted PSF, so the speed depends on the
number of source evaluations and the image configuration. The
``psf_cubic_interpolation`` example measures setup and repeated evaluation
times for synthetic double-King profiles, and compares the direct and cubic
outputs over random subpixel shifts:

.. code-block:: console

  python examples/psf_cubic_interpolation/psf_cubic_interpolation.py --evaluations 600 --repeats 3 --bokeh

.. image:: _static/cubic_psf_cost.png
  :alt: One-time cubic setup and repeated direct and cubic PSF evaluation times for two synthetic energy-bin configurations

.. image:: _static/cubic_psf_accuracy.png
  :alt: Maximum cubic PSF density error relative to the profile peak for held-out random subpixel shifts

The same run writes an interactive HTML companion to
``examples/psf_cubic_interpolation/visuals/cubic_psf_benchmark.html``. The
published interactive view is available as a
:download:`self-contained Bokeh document <_static/cubic_psf_benchmark.html>`.

Unbinned event-level likelihood
-------------------------------

PCAT can sample an unbinned likelihood through a user callback. The maintained
``unbinned_gaussian_mixture`` example passes the 140 simulated event coordinates
directly to ``pcat.fixed.sample_fixed_chains`` and sums their Gaussian-mixture
log densities with no spatial bins or count map. It samples the four center
coordinates of two equal-weight Gaussian components. The component count and
width are fixed, so this callback demonstrates inference within one model.

The native ``typeexpr="gmix"`` experiment remains the binned Poisson workflow
shown in ``gaussian_mixture_catalog``. Its variable-width ``clusvari`` elements
are fit to an 8-by-8 count grid. ``boolbinsener=False`` only indicates that this
example has no energy axis. PCAT does not currently provide a built-in unbinned
``typeexpr="gmix"`` dispatcher for variable-count mixture components.

Spectral-line likelihood
------------------------

The one-dimensional ``typeexpr="fire"`` workflow uses ``anlytype="spec"`` for
spectral bins. Set ``binsenerfull`` to the monotonically increasing spectral
axis edges and use ``typedata="inpt"``, ``strgexprsbrt``, ``typeexpo``, and
``strgexpo`` for supplied data and exposure. The built-in count likelihood is
Poisson by default. ``liketype="gaus"`` selects the Gaussian residual
likelihood when ``varidata`` is supplied. A continuum or background template
can be provided through ``dictfitt["sbrtbacknorm"]`` and activated with
``dictfitt["listnamediff"]``.

The ``spectype`` setting selects an analytic spectral shape:

* ``"powr"`` for a power law.
* ``"gaus"`` for a Gaussian profile.
* ``"lore"`` for a Lorentzian profile.
* ``"voig"`` for a Voigt profile.
* ``"pvoi"`` for a pseudo-Voigt profile with mixing fraction ``frac``.
* ``"sinc"`` for a sinc-squared profile.
* ``"skew"`` for a skew-Gaussian profile with shape parameter ``skew``.
* ``"toph"`` for a top-hat profile.
* ``"edis"`` for an energy-dispersion profile using the configured
  ``edisintp`` response.
* ``"colr"``, ``"curv"``, or ``"expc"`` for color-index, curved, or
  exponential-cutoff continua.

Line elements use integrated flux ``flux`` and line center ``elin``. Gaussian,
sinc-squared, skew-Gaussian, and top-hat profiles add ``sigm``. Lorentzian
profiles add ``gamm``. Voigt profiles use both widths. Pseudo-Voigt profiles
also use ``frac``. Set parameter bounds with ``limtparaelem`` and proposal
scales with ``stdvpropelemfire``.

Line-spread function
~~~~~~~~~~~~~~~~~~~~

Set ``lsftype="gaus"`` and ``lsfresolvingpower`` to convolve each line at a
constant instrument resolving power. Set ``lsftype="tabu"`` and pass an
odd-length, nonnegative ``lsfkernel`` for a measured line-spread function.
Both paths preserve sampled line flux. The default ``lsftype="none"`` leaves
the intrinsic profiles unchanged. Parameter bounds are supplied through
``limtparaelem`` and within-model step scales through ``stdvpropelemfire``.

The ``voigt_spectral_line_catalog`` example demonstrates the built-in Voigt
profile on simulated spectra. ``jwst_miri_ngc7027_line_catalog`` applies the
same binned count likelihood to measured MIRI flux converted to effective
Poisson counts using its uncertainties and calibration floor.
The simulated example also accepts ``--profile`` for intrinsic line shapes and
``--lsftype none|gaus|tabu`` for instrument broadening. For example,
``--profile pvoi --lsftype gaus --lsf-resolving-power 3000`` fits a
pseudo-Voigt line catalog with constant-resolving-power broadening. The
response-dependent ``edis`` profile is selected through the Python
configuration API with an instrument ``edisintp`` callable.

Binned photometric time-series likelihood
------------------------------------------

Use ``typeexpr="fire"`` and ``anlytype="spec"`` for a time-binned photometric
series, with time-bin edges in ``binsenerfull`` and observed counts/exposure
configured as for other supplied one-dimensional data. This is the same
one-dimensional binned likelihood used for spectra. Flare and spot profiles
are alternative element models within this likelihood. Fixed baselines can
be supplied with ``dictfitt["sbrtbacknorm"]`` and enabled with
``dictfitt["listnamediff"]``.

Flare ``spectype`` options are ``"flargauss"``, ``"flarexpd"``,
``"flarfred"``, and ``"flardav"``. They sample peak amplitude ``flux``, peak
time ``elin``, and profile-specific widths: ``fwhm`` for Gaussian and
Davenport profiles, ``scalfall`` for one-sided exponential decay, and
independent ``scalrise``/``scalfall`` for FRED. Set physical bounds with
``limtparaelem`` and proposal scales with ``stdvpropelemfire``. The
``variable_number_stellar_flares`` example accepts ``--profile`` with any of
these four models. For example, ``--profile flardav`` selects the Davenport
template. The generated time series remains FRED-injected, so alternate fits
compare model misspecification against the same data.

Periodic Gaussian spot dips use ``spectype=["spotrot"]`` with the
``lghtlinevoig`` element container, plus ``spot_period_days`` and
``spot_reference_time_days``. Spot depth, phase epoch, and width use the
element parameters ``flux``, ``elin``, and ``fwhm``. The rotation period is
fixed. See ``variable_number_stellar_spots`` for a complete simulated example.

Radial-velocity likelihood
--------------------------

``pcat.radial_velocity.retr_dictpcatrvel`` configures the dedicated Gaussian
likelihood for irregular stellar RV observations. Pass observation times
``time`` in BJD minus the stated offset [day], velocities ``rvel`` [m/s],
reported uncertainties ``stdvrvel`` [m/s], and integer instrument labels
``indxinst``. The helper sorts observations, removes duplicate epochs, builds
time-bin edges, and returns configuration for a variable catalog of
``lghtlinekepl`` Keplerian elements. Each planet uses ``flux`` for semi-amplitude
``K``, ``elin`` for period, ``phas`` for mean anomaly, ``ecce`` for eccentricity,
and ``argp`` for argument of periastron.

Set ``maxmnumbplan`` for the catalog-size limit, ``minmrvsa``/``maxmrvsa`` and
``minmperi``/``maxmperi`` for amplitude and period bounds, and ``maxmecce`` for
the eccentricity ceiling. Per-instrument velocity offsets and an optional
linear trend are marginalized analytically. Enable the trend with
``boollinetren=True``. Stellar jitter is marginalized on the log-uniform grid
``listjittrvel``, configured through ``minmjitt``, ``maxmjitt``, and
``numbjitt``. The helper also supplies a data-informed period/amplitude/phase
birth density and its matching Hastings correction. Its proposal can be tuned
with ``numbperi``, ``expodchi``, ``fracunif``, ``fracgaus``, ``stdvlogtrvsa``,
and ``stdvphas``. See
``variable_number_exoplanets_radial_velocity`` for a multi-instrument example.

Transit-timing likelihood
-------------------------

For transit-time data, use ``pcat.time_series.log_likelihood_transit_times``
inside a fixed-dimensional PCAT likelihood callback. The
``pcat.time_series.make_transit_time_likelihood`` factory connects a forward
predictor to PCAT's ``(parameters, data)`` callback. The data contain transit
epochs, observed times, and positive timing errors. The Gaussian likelihood is
uncertainty-weighted, up to a parameter-independent normalization constant, and
returns negative infinity for nonfinite model times. Ephesos supplies linear,
combined sinusoidal-TTV, and optional N-body time predictions. PCAT handles
parameter priors and posterior sampling through ``sample_fixed`` or
``sample_fixed_chains``. The predictor must align observed epochs with model
epochs. See ``transit_timing_variations`` for a posterior-sampling example.

Generic user-supplied likelihood
--------------------------------

Use ``typeexpr="gener"`` for inference within one model with a custom
likelihood. The ``retr_llik(gdat, strgmodl, values)`` callback returns one
log-likelihood value for the supplied parameter vector. PCAT handles configured priors, proposals,
accept/reject steps, persistence, and saved sample summaries. Use this path
when none of the built-in binned likelihoods matches the data model. See
:doc:`capabilities` for a complete callback example and :doc:`sampling` for
proposal and convergence controls.