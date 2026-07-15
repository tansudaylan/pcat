# PCAT Coding Standards

This document outlines the comprehensive coding standards to be applied across the PCAT codebase for consistency, readability, and maintainability.

## Variable Naming

### Core Principle
Variable names must be concatenated 4-letter chunks, starting with the root description and appending chunks to distinguish further.

### Examples
- `numbelem` - number of elements
- `xpossour` - x position of source  
- `deflsing` - deflection angle at single point
- `convhost` - convolution of host
- `lliktotl` - total log-likelihood
- `cntpmodl` - counts in model
- `strgmodl` - string model identifier
- `minmpara` - minimum parameter bounds
- `maxmpara` - maximum parameter bounds

### Exception
File and folder names should remain readable and follow standard conventions:
- `main.py`, `test.py`, `collect_garbage.py` (not abbreviated)

---

## Function Naming

### Format
Action verb + underscore + 4-letter chunks

### Common Action Verbs
- `retr_` - retrieve/return a value
- `calc_` - calculate a value
- `srch_` - search for something
- `init_` - initialize
- `exec_` - execute (for test runners)
- `eval_` - evaluate (for test evaluators)
- `proc_` - process
- `make_` - create/make
- `plot_` - plotting routines

### Examples
```python
def retr_deflprof(gmod, indxelem):
    # Returns deflection profile
    
def calc_lliktotl(gdat, cntpdata, cntpmodl):
    # Calculates total log-likelihood
    
def srch_maxmflux(listflux):
    # Searches for maximum flux
```

---

## Function Structure

### Spacing
- **2 blank lines before** function definition
- **1 blank line after** `def func_name():` line
- Docstring on next line
- **1 blank line** before code begins
- **1 blank line** before return statement

### Template
```python


def retr_valu_name(parm1, parm2):

    '''
    
    One-line description of what function does.
    
    Longer description explaining logic, parameters, and return values if needed.
    Additional context about edge cases or important considerations.
    
    '''
    
    # Initialize return variable
    valu = None
    
    # Perform calculation or retrieval
    # ... code here with comments explaining logic ...
    
    # Sanity check if value is reasonable
    if valu < 0:
        raise RuntimeError('Value cannot be negative')

    return valu
```

---

## Code Style

### Line Length
Maximum 100 characters per line. Split longer statements:

```python
# BAD - 112 characters
result = pcat.main.initarry(dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec)

# GOOD - split across lines
result = pcat.main.initarry(
    dictargsvari,
    dictargs,
    listnamecnfgextn,
    strgcnfgextnexec=strgcnfgextnexec,
)
```

### Return Variable Convention
Always define the return variable first, then assign to it:

```python
# BAD
def calc_sum(a, b):
    return a + b

# GOOD
def calc_sum(a, b):
    summ = None
    summ = a + b
    return summ
```

---

## Comments and Documentation

### Physical Units
When a variable has physical units, add as end-of-line comment in brackets:

```python
xposelm = 0.5  # xposelm: element position [arcsec]
fluxelm = 1e-10  # fluxelm: element flux [erg/s/cm^2]
masshost = 1e12  # masshost: host galaxy mass [solar masses]
```

### Explaining Logic
Add comments explaining why code does what it does:

```python
# Exclude negative pixel values (instrument artifacts)
indxvalid = cntpdata > 0

# Subtract background following standard spectral analysis
cntpsubtracted = cntpdata - cntpback
```

### Task Completion Messages
Use print statements to narrate major task progress:

```python
print('Initializing strong-lens mock count test...')
# ... configuration code ...

print('Running transdimensional and fixed-dimension fits...')
# ... execution code ...

print('Completed strong-lens mock count test successfully.')
```

---

## Plotting Standards

### Grid Lines
Always disable gridlines:

```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.grid(False)  # Disable gridlines
plt.savefig(path, bbox_inches="tight")
```

### File Structure
Use standard directory structure:

```python
pathbase = os.environ['TDGU_DATA_PATH'] + '/' + 'ProjectName/'
pathdata = pathbase + 'data/'
pathvisu = pathbase + 'visuals/'
os.system('mkdir -p %s' % pathdata)
os.system('mkdir -p %s' % pathvisu)

path = pathvisu + 'PlotName' + '.png'
if not os.path.exists(path):
    # Create plot here
    plt.tight_layout()
    print('Writing to %s...' % path)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
else:
    print('%s already exists...' % path)
```

### Julian Date Axes
When time axis is in Julian dates (BJD, HJD, etc), subtract offset and include in unit label:

```python
bjdoff = 2470000  # Offset from reference epoch
timexaxi = bjddata - bjdoff

ax.set_xlabel('Time [BJD - %d]' % bjdoff)
```

### Literature References
Add references in plot legends and as code comments:

```python
# Reference: Smith et al. 2020, ApJ 900, 123
# https://arxiv.org/abs/2001.12345
reference_value = 42.0

ax.plot(x, y, label='Data (Smith et al. 2020)')
```

---

## Sanity Checks

Add physicality checks for key values, but keep them fast:

```python
# Check value is within reasonable range
if numbelem < 0 or numbelem > 1e6:
    raise RuntimeError('Number of elements outside valid range')

# Check flux positive (sanity check)
if fluxelm < 0:
    raise RuntimeError('Element flux cannot be negative')
```

---

## Import Conventions

Use standard 4-letter abbreviations for imports:

```python
import numpy as nump
import scipy as scip
import matplotlib.pyplot as pypl
import astropy as astr
from astropy.io import fits as fits
import pandas as pand
import os as oper
import sys as syst
import traceback as trbk
```

---

## Code Shortening

Look for opportunities to reduce code without changing functionality:

```python
# VERBOSE (3 lines)
a = [1, 2, 3]
b = [x * 2 for x in a]
c = sum(b)

# CONCISE (1 line, same result)
c = sum([x * 2 for x in [1, 2, 3]])
```

However, prioritize readability:

```python
# Don't sacrifice clarity for brevity
# UNCLEAR
result = [y for x in data if (lambda z: z > 0)(x) for y in [x**2]]

# CLEAR
result = []
for x in data:
    if x > 0:
        result.append(x**2)
```

---

## Refactoring Priority

Apply standards in this order:

1. **Critical path code** (test.py, main.py init functions)
2. **Frequently-used functions** (plotting, likelihood, parameter handling)
3. **Utility functions** (helpers, data processing)
4. **Legacy/deprecated code** (lower priority)

---

## Checklist for Code Review

Before committing changes:

- [ ] Function has 2 blank lines before definition
- [ ] Docstring properly formatted with blank lines
- [ ] Variables use 4-letter chunks
- [ ] No lines exceed 100 characters
- [ ] Return variable defined before use
- [ ] Physical units in comments for numeric values
- [ ] Print statements for major tasks
- [ ] Comments explain *why* not just *what*
- [ ] Plot code disables gridlines
- [ ] Sanity checks for unreasonable values
- [ ] Code follows function naming (retr_, calc_, etc.)

---

## Examples by Category

### Data Processing Function
```python


def calc_cntpback_subt(cntpdata, cntpback):

    '''
    
    Calculate background-subtracted counts with error handling.
    
    Subtracts background from observed counts and flags negative pixels
    that may indicate instrument artifacts or calibration issues.
    
    '''
    
    # Initialize result
    cntpsubt = None
    
    # Verify input shapes match
    if cntpdata.shape != cntpback.shape:
        raise RuntimeError('Data and background shapes do not match')
    
    # Subtract background
    print('Subtracting background from %d pixels...' % len(cntpdata))
    cntpsubt = cntpdata - cntpback
    
    # Check for unphysical values
    indxnegat = cntpsubt < 0
    if nump.sum(indxnegat) > 0:
        print('Warning: %d pixels have negative counts after background'
              ' subtraction' % nump.sum(indxnegat))
    
    print('Background subtraction complete.')

    return cntpsubt
```

### Retrieval Function
```python


def retr_indxelem_by_flux(fluxelm, fluxmin):

    '''
    
    Retrieve indices of elements with flux above minimum threshold.
    
    '''
    
    # Initialize index array
    indx = None
    
    # Filter by minimum flux
    print('Selecting elements with flux >= %.2e...' % fluxmin)
    indx = fluxelm >= fluxmin
    
    # Report results
    numbsele = nump.sum(indx)
    print('Selected %d/%d elements.' % (numbsele, len(fluxelm)))

    return indx
```

### Test Function
```python


def exec_model_comparis():

    '''
    
    Execute comparison between different model configurations.
    
    '''
    
    # Initialize output
    dictoutp = None
    
    # Configure test parameters
    print('Setting up model comparison test...')
    dictargs = {}
    dictargs["modeltype"] = "lens"
    dictargs["truenumbelempop0"] = 10
    
    # Run configurations
    print('Executing fits with different parameterizations...')
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
    )
    
    # Store results
    dictoutp = dictglob

    return dictoutp
```

---

## Notes

- These standards apply to **all new code** and **refactored sections**
- Legacy code can be gradually updated as it's touched during maintenance
- For large refactoring tasks, create separate commits per logical section
- Test frequently (after each major refactoring step)

