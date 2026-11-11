# Synthetic Mejiro Time-Delay Cosmography

This example uses Lenstronomy to solve a four-image SIE, shear, and convergence
lens at fixed redshifts and reference cosmology. The computed delays are stored
in a Mejiro `LensedSupernova` object. A seeded synthetic observation adds
independent timing errors, and PCAT samples $H_0$ with a fixed lens potential
and fixed $\Omega_m$.

```bash
PYTHONPATH=astromusers/mejiro:${PYTHONPATH:-} \
	python tansudaylan/pcat/examples/mejiro_time_delay_cosmography/mejiro_time_delay_cosmography.py --smoke
```

The reference delays scale as $H_{0,\rm ref}/H_0$ because the time-delay
distance is inversely proportional to $H_0$ when the dimensionless cosmology
and lens model are fixed. The example omits lens-model uncertainty, source
redshift uncertainty, and the mass-sheet degeneracy. Its posterior is a code
demonstration, not an observational constraint on $H_0$.