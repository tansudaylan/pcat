# Variable-number stellar spots

This example simulates 25.6 days of Poisson photometry with three rotating
dark spots. PCAT fits a catalog containing zero to five spots. The stellar
rotation period is held at 3.2 days; each spot has a sampled depth, phase, and
width. The frame sequence plots observed counts and a PCAT model sample above
their residuals.

Run a short pipeline check with:

```bash
python examples/variable_number_stellar_spots/variable_number_stellar_spots.py --smoke
```

The simulated data are illustrative and are not observations of a specific
star.