Likelihood configurations
=========================

PCAT separates the likelihood for the data from the transdimensional catalog
summary. Choose the likelihood family to match the data representation, then
configure its geometry, response, background, and source model. In particular,
every transdimensional likelihood produces samples of catalogs. With the
default ``boolcondcatl=True``, final processing condenses those samples into a
posterior catalog and can then associate its elements with a reference catalog
when the element model has supported ``xpos`` and ``ypos`` coordinates. Set
``boolcondcatl=False`` to skip condensation. The current implementation does
not condense one-dimensional line, flare, or Keplerian catalogs. Neither
condensation nor catalog association is a likelihood.

Poisson image likelihood
------------------------

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
its side length; ``boolforccart=True`` forces a square grid when pixel
coordinates are supplied directly. ``typepixl="heal"`` selects HEALPix data
and uses ``numbsideheal`` for its resolution. Fermi-LAT defaults to HEALPix;
other image experiments default to Cartesian pixels.

``liketype="pois"`` is the default for count data. Omitting the data-only
factorial, its log likelihood is

.. math::

   \log L_{\mathrm{Pois}} = \sum_i D_i \log M_i - M_i,

where ``D`` is the observed count map and ``M`` is the predicted count map.
Set ``liketype="gaus"`` for the built-in Gaussian residual likelihood, which
uses ``varidata`` (by default, the observed counts with a floor of one) as its
per-pixel variance:

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
* ``typeemishost="sers"`` for a Sérsic foreground lens-galaxy light model;
  ``"none"`` disables this host-emission component.
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
``listnamediff`` entry selects the diffuse components added to the model; names
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
``dictfitt``; set ``boolmodipsfn=True`` when PSF parameters should vary in the
fit. Fermi-LAT can also use its energy- and event-class-dependent PSF scale
``fermscalfact``. PSF and source-kernel options depend on pixelization and
element type; unsupported combinations should use ``typeevalpsfn="none"``.

Unbinned Gaussian-mixture likelihood
------------------------------------

An unbinned Gaussian-mixture likelihood evaluates the mixture density at every
observed coordinate. Its data are a list of events rather than an image, and
its transdimensional elements are Gaussian components with inferred positions,
widths, amplitudes, and population size. This likelihood is distinct from the
Poisson image likelihood because it does not accumulate events into pixels.

The public ``typeexpr="gmix"`` example currently uses variable-width Gaussian
elements (``typeelem=["clusvari"]``) on an 8-by-8 spatial grid and evaluates
binned Poisson counts. ``boolbinsener=False`` only indicates that the example
has no energy axis. It does not make the spatial likelihood unbinned. The
unbinned Gaussian-mixture dispatcher must be completed and tested before this
example can represent the unbinned likelihood described above.

Spectral and time-series likelihood
-----------------------------------

The one-dimensional ``typeexpr="fire"`` workflow uses ``anlytype="spec"`` for
spectral bins. Set ``binsenerfull`` to the axis edges and use
``typedata="inpt"``, ``strgexprsbrt``, ``typeexpo``, and ``strgexpo`` for
supplied spectra and exposure. A continuum or background template can be
provided through ``dictfitt["sbrtbacknorm"]`` and activated with
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

Set ``lsftype="gaus"`` and ``lsfresolvingpower`` to convolve each line at a
constant instrument resolving power. Set ``lsftype="tabu"`` and pass an
odd-length, nonnegative ``lsfkernel`` for a measured line-spread function.
Both paths preserve sampled line flux. The default ``lsftype="none"`` leaves
the intrinsic profiles unchanged.

The same one-dimensional machinery can model time-binned data by treating time
as the axis. ``flargauss`` uses peak amplitude ``flux``, peak time ``elin``,
and full width at half maximum ``fwhm``. ``flarexpd`` adds a one-sided decay
time ``scalfall``. ``flarfred`` uses independent ``scalrise`` and ``scalfall``
for a fast-rise, exponential-decay profile. ``flardav`` uses ``fwhm`` for the
empirical Davenport flare template. The maintained stellar-flare example fits
native ``flarfred`` elements. Radial-velocity models use Keplerian
``lghtlinekepl`` elements and their dedicated likelihood in
:mod:`pcat.radial_velocity`.

User-supplied likelihood
------------------------

Use ``typeexpr="gener"`` for a fixed-dimensional custom likelihood. The
``retr_llik(gdat, strgmodl, values)`` callback returns one log-likelihood value
for the supplied parameter vector; PCAT handles configured priors, proposals,
accept/reject steps, persistence, and diagnostics. This is the appropriate path
when none of the built-in binned likelihoods matches the data model. See
:doc:`capabilities` for a complete callback example and :doc:`sampling` for
proposal and convergence controls.