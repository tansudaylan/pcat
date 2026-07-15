import os as oper
import sys as syst
import astropy as astr
import matplotlib.pyplot as pypl
import numpy as nump
import pcat
import scipy as scip
import shutil as shut
import subprocess as subp
import tdpy
import traceback as trbk
from astropy.io import fits as fits


if not hasattr(tdpy, "retr_labltotlsing") and hasattr(tdpy, "retr_labltotl"):
    tdpy.retr_labltotlsing = tdpy.retr_labltotl


if not hasattr(tdpy.util, "mexp"):
    def _mexp_compat(valu):
        return f"{valu:g}"

    tdpy.util.mexp = _mexp_compat


def _normalize_legacy_pcat_args(dictargs):

    dictnorm = dict(dictargs)
    dictname = {
        "exprtype": "typeexpr",
        "datatype": "typedata",
        "pixltype": "typepixl",
        "priofactdoff": "factpriodoff",
        "seedtypeelem": "typeseedelem",
        "psfnevaltype": "typeevalpsfn",
    }
    for nameold, namenew in dictname.items():
        if nameold in dictnorm and namenew not in dictnorm:
            dictnorm[namenew] = dictnorm.pop(nameold)

    if dictnorm.get("forccart") and "typepixl" not in dictnorm:
        dictnorm["typepixl"] = "cart"

    if "mockonly" in dictnorm and "boolsimuonly" not in dictnorm:
        dictnorm["boolsimuonly"] = dictnorm.pop("mockonly")

    if "lionmode" in dictnorm:
        dictnorm.pop("lionmode")

    return dictnorm


def _apply_test_defaults(dictargs):

    dictout = dict(dictargs)
    maxm_numbswep = int(oper.environ.get("PCAT_FAST_NUMBSWEP", "200"))
    maxm_numbsamp = int(oper.environ.get("PCAT_FAST_NUMBSAMP", "25"))
    maxm_numbswepplot = int(oper.environ.get("PCAT_FAST_NUMBSWEPLOT", "50"))

    dictout.setdefault("typeverb", 0)
    dictout.setdefault("numbproc", 1)
    dictout.setdefault("makeanim", False)
    dictout.setdefault("boolmakeplot", False)
    dictout.setdefault("boolmakeplotinit", False)
    dictout.setdefault("boolmakeplotintr", False)
    dictout.setdefault("numbswepplot", maxm_numbswepplot)

    if dictout.get("lionmode"):
        dictout.setdefault("boolsimuonly", True)

    if "numbswep" not in dictout or dictout["numbswep"] is None or dictout["numbswep"] > maxm_numbswep:
        dictout["numbswep"] = maxm_numbswep
    if "numbburn" not in dictout or dictout["numbburn"] is None or dictout["numbburn"] >= dictout["numbswep"]:
        dictout["numbburn"] = max(10, int(0.1 * dictout["numbswep"]))
    if "numbsamp" not in dictout or dictout["numbsamp"] is None or dictout["numbsamp"] > maxm_numbsamp:
        dictout["numbsamp"] = maxm_numbsamp
    if "numbswepplot" not in dictout or dictout["numbswepplot"] is None or dictout["numbswepplot"] > dictout["numbswep"]:
        dictout["numbswepplot"] = max(1, min(maxm_numbswepplot, dictout["numbswep"]))

    return dictout


_pcat_init_native = pcat.main.init
_pcat_sample_native = pcat.main.sample


def _clear_forced_run_state(dictargs):

    if not dictargs.get("forcneww"):
        return

    strgcnfg = dictargs.get("strgcnfg")
    if strgcnfg is None:
        return

    pathbase = dictargs.get("pathbase") or oper.environ.get("PCAT_DATA_PATH") or "/Users/tdaylan/Documents/work/data/pcat"
    pathoutpcnfg = oper.path.join(pathbase, "data", "outp", strgcnfg)
    pathvisucnfg = oper.path.join(pathbase, "visuals", strgcnfg)

    for path in [pathoutpcnfg, pathvisucnfg]:
        if oper.path.exists(path):
            shut.rmtree(path)


def _init_compat(*args, **kwargs):

    if len(args) == 1 and isinstance(args[0], dict) and len(kwargs) == 0:
        return _pcat_init_native(_apply_test_defaults(_normalize_legacy_pcat_args(args[0])))

    if len(kwargs) > 0:
        dictargs = {}
        if len(args) == 1 and isinstance(args[0], dict):
            dictargs.update(args[0])
        elif len(args) > 0:
            raise TypeError("init compatibility wrapper received an unsupported call signature.")
        dictargs.update(kwargs)
        dictargs = _apply_test_defaults(_normalize_legacy_pcat_args(dictargs))
        return pcat.main.sample(**dictargs)

    return _pcat_init_native(*args)


pcat.main.init = _init_compat


def _sample_compat(*args, **kwargs):

    if len(args) == 0 and len(kwargs) > 0:
        dictargs = _apply_test_defaults(_normalize_legacy_pcat_args(kwargs))
        _clear_forced_run_state(dictargs)
        return _pcat_sample_native(**dictargs)

    return _pcat_sample_native(*args, **kwargs)


pcat.main.sample = _sample_compat


if not hasattr(pcat.main, "initarry") and hasattr(pcat.main, "sample_parallel"):
    def _initarry_compat(*args, **kwargs):
        if len(args) == 2 and "dictpcatinpt" in kwargs:
            dictargsvari, listnamecnfgextn = args
            dictargs = dict(kwargs.pop("dictpcatinpt"))
        elif len(args) >= 3:
            dictargsvari, dictargs, listnamecnfgextn = args[:3]
            dictargs = dict(dictargs)
        else:
            raise TypeError("initarry compatibility wrapper received an unsupported call signature.")
        dictparallel = {}
        for namekw in [
            "strgcnfgextnexec",
            "boolexecpara",
            "execpara",
            "namexaxi",
            "lablxaxi",
            "listtickxaxi",
            "scalxaxi",
            "namexaxivari",
            "lablxaxivari",
            "tickxaxivari",
            "scalxaxivari",
            "listnamevarbcomp",
            "listscalvarbcomp",
            "listlablvarbcomp",
            "listtypevarbcomp",
            "listpdfnvarbcomp",
            "listgdatvarbcomp",
            "forcneww",
            "forcprev",
            "strgpara",
        ]:
            if namekw in kwargs:
                dictparallel[namekw] = kwargs.pop(namekw)
        liststrgvarboutp = kwargs.pop("liststrgvarboutp", None)
        dictargs.update(kwargs)
        if "execpara" in dictparallel and "boolexecpara" not in dictparallel:
            dictparallel["boolexecpara"] = dictparallel.pop("execpara")
        if "namexaxi" in dictparallel:
            dictparallel["namexaxivari"] = dictparallel.pop("namexaxi")
        if "lablxaxi" in dictparallel:
            dictparallel["lablxaxivari"] = dictparallel.pop("lablxaxi")
        if "listtickxaxi" in dictparallel:
            dictparallel["tickxaxivari"] = dictparallel.pop("listtickxaxi")
        if "scalxaxi" in dictparallel:
            dictparallel["scalxaxivari"] = dictparallel.pop("scalxaxi")
        dictargs = _apply_test_defaults(_normalize_legacy_pcat_args(dictargs))
        dictargsvari = {
            namecnfg: _apply_test_defaults(_normalize_legacy_pcat_args(dictcnfg))
            for namecnfg, dictcnfg in dict(dictargsvari).items()
        }
        dictparallel.setdefault("boolexecpara", False)
        liststrgcnfg = pcat.main.sample_parallel(
            dictargsvari,
            listnamecnfgextn,
            dictpcatinpt=dictargs,
            **dictparallel,
        )

        if liststrgvarboutp is None:
            return liststrgcnfg

        typegdat = "init" if dictargs.get("boolsimuonly", False) else "finlpost"
        listgdat = pcat.main.retr_listgdat(liststrgcnfg, typegdat=typegdat)
        dictoutp = {}
        for namevarb in liststrgvarboutp:
            if len(listgdat) == 1:
                dictoutp[namevarb] = getattr(listgdat[0], namevarb)
            else:
                dictoutp[namevarb] = [getattr(gdat, namevarb) for gdat in listgdat]
        return listgdat, dictoutp

    pcat.main.initarry = _initarry_compat


def narr_open(path, mode="r"):

    pathnorm = oper.path.normpath(path)
    if mode.startswith("r") and "+" not in mode:
        action = "Reading"
    else:
        action = "Writing"
    print(f"{action} {pathnorm}...")
    return open(pathnorm, mode)


def narr_fits_open(path):

    pathnorm = oper.path.normpath(path)
    print(f"Reading {pathnorm}...")
    return fits.open(pathnorm)


def narr_astr_fits_open(path):

    pathnorm = oper.path.normpath(path)
    print(f"Reading {pathnorm}...")
    return astr.io.fits.open(pathnorm)



def exec_lionwrap():

    '''
    
    Initialize Cartesian TESS input-data run in LION compatibility mode.
    
    Provides minimal wrapper test verifying PCAT can construct and return global
    configuration expected by LION interface.
    
    '''

    # Initialize output dictionary
    dictoutp = None
    
    # Configure parameters for LION test
    print('Initializing LION compatibility test...')
    dictglob = pcat.main.init(
        pixltype="cart",
        listnameback="isot",
        backtype=[[1.0]],
        datatype="inpt",
        exprtype="tess",
        strgexpo=1.0,
        lionmode=True,
    )
    
    # Store result
    dictoutp = dictglob

    return dictoutp


def exec_fittminmflux_fittparsnone(strgcnfgextnexec=None):

    '''
    
    Measure fitted minimum point-source flux effect on recovery.
    
    Measures impact when source-count prior is disabled. Chandra mock contains 100
    point sources; fit threshold scanned across five decades while likelihood at
    true model recorded for comparison.
    
    '''

    # Initialize output dictionary
    dictoutp = None
    
    # Configure base parameters for Chandra test
    print('Setting up minimum flux test (no source-count prior)...')
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["priofactdoff"] = 0.0  # Disable source-count prior
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["indxenerincl"] = nump.array([2])
    
    # Define configurations to test with different flux thresholds
    listnamecnfgextn = [
        "fittvlow", "fittloww", "fittnomi", "fitthigh", "fittvhig"
    ]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    
    # Set flux thresholds (units: cm^-2 s^-1 keV^-1)
    dictargsvari["fittvlow"]["fittminmflux"] = 3e-10
    dictargsvari["fittloww"]["fittminmflux"] = 1e-09
    dictargsvari["fittnomi"]["fittminmflux"] = 3e-09
    dictargsvari["fitthigh"]["fittminmflux"] = 1e-08
    dictargsvari["fittvhig"]["fittminmflux"] = 3e-08
    
    # Configure plot labels and scales
    lablxaxi = ("$f_{min}$ "
                "[cm$^{-2}$ s$^{-1}$ keV$^{-1}$]")
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["fittminmflux"])
        for namecnfgextn in listnamecnfgextn
    ]
    
    # Configure comparison variables
    listnamevarbcomp = ["truelliktotl"]
    listscalvarbcomp = ["self"]
    listlablvarbcomp = ["$\\ln P(D|M_{true})$"]
    listtypevarbcomp = [""]
    listpdfnvarbcomp = [""]
    
    # Execute test configurations
    print('Running test configurations...')
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        strgpara="$PCAT_PATH/cnfg.py",
        namexaxi="fittminmflux",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
        listnamevarbcomp=listnamevarbcomp,
        listscalvarbcomp=listscalvarbcomp,
        listlablvarbcomp=listlablvarbcomp,
        listtypevarbcomp=listtypevarbcomp,
        listpdfnvarbcomp=listpdfnvarbcomp,
    )
    
    # Store result
    dictoutp = dictglob

    return dictoutp


def exec_fittminmflux_fittparsnomi(strgcnfgextnexec=None):

    '''
    Measure how the fitted minimum point-source flux affects a nominal Chandra point-source fit.

    The function scans five fitting thresholds for a 100-source mock to test the sensitivity of
    inference outputs to the lower flux boundary under the default prior configuration.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["fittvlow", "fittloww", "fittnomi", "fitthigh", "fittvhig"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["fittvlow"]["fittminmflux"] = 3e-10
    dictargsvari["fittloww"]["fittminmflux"] = 1e-09
    dictargsvari["fittnomi"]["fittminmflux"] = 3e-09
    dictargsvari["fitthigh"]["fittminmflux"] = 1e-08
    dictargsvari["fittvhig"]["fittminmflux"] = 3e-08
    lablxaxi = "$f_{min}$ [cm$^{-2}$ s$^{-1}$ keV$^{-1}$]"
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["fittminmflux"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="fittminmflux",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_fittparstrueback(strgcnfgextnexec=None):

    '''
    Test the effect of the parsimony-prior strength in a background-only Chandra field.

    With the true point-source population fixed to zero, the prior factor is varied from negative to
    strongly positive values to quantify false-source suppression or promotion.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truemaxmnumbelempop0reg0"] = 0
    dictargs["truenumbelempop0reg0"] = 0
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["parsnega", "parsnone", "parsloww", "parsnomi", "parshigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["parsnega"]["priofactdoff"] = -0.5
    dictargsvari["parsnone"]["priofactdoff"] = 0.0
    dictargsvari["parsloww"]["priofactdoff"] = 0.5
    dictargsvari["parsnomi"]["priofactdoff"] = 1.0
    dictargsvari["parshigh"]["priofactdoff"] = 1.5
    scalxaxi = "self"
    lablxaxi = "$\\alpha_p$"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["priofactdoff"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="priofactdoff",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_fittparstruepnts(strgcnfgextnexec=None):

    '''
    Test the effect of the parsimony-prior strength when fitting a populated Chandra field.

    The mock contains 100 point sources above a fixed flux floor, allowing source recovery to be
    compared as the prior penalty on additional model elements is varied.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["minmflux"] = 1e-08
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["parsnega", "parsnone", "parsloww", "parsnomi", "parshigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["parsnega"]["priofactdoff"] = -0.5
    dictargsvari["parsnone"]["priofactdoff"] = 0.0
    dictargsvari["parsloww"]["priofactdoff"] = 0.5
    dictargsvari["parsnomi"]["priofactdoff"] = 1.0
    dictargsvari["parshigh"]["priofactdoff"] = 1.5
    scalxaxi = "self"
    lablxaxi = "$\\alpha_p$"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["priofactdoff"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="priofactdoff",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_fittparstruepntsfittminmfluxloww(strgcnfgextnexec=None):

    '''
    Test parsimony-prior strength for a populated Chandra field with a low fitted flux threshold.

    The fit extends to 1e-9 in flux, so the scan probes how strongly the prior must regulate the
    many faint sources that the transdimensional model can introduce.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["fittminmflux"] = 1e-09
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["parsnega", "parsnone", "parsloww", "parsnomi", "parshigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["parsnega"]["priofactdoff"] = -0.5
    dictargsvari["parsnone"]["priofactdoff"] = 0.0
    dictargsvari["parsloww"]["priofactdoff"] = 0.5
    dictargsvari["parsnomi"]["priofactdoff"] = 1.0
    dictargsvari["parshigh"]["priofactdoff"] = 1.5
    scalxaxi = "self"
    lablxaxi = "$\\alpha_p$"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["priofactdoff"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="priofactdoff",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_fittparstruepntsfittminmfluxhigh(strgcnfgextnexec=None):

    '''
    Test parsimony-prior strength for a populated Chandra field with a high fitted flux threshold.

    The fit is restricted to sources above 1e-8 in flux, isolating prior effects when fewer faint
    candidate sources are available to the transdimensional model.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["fittminmflux"] = 1e-08
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["parsnega", "parsnone", "parsloww", "parsnomi", "parshigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["parsnega"]["priofactdoff"] = -0.5
    dictargsvari["parsnone"]["priofactdoff"] = 0.0
    dictargsvari["parsloww"]["priofactdoff"] = 0.5
    dictargsvari["parsnomi"]["priofactdoff"] = 1.0
    dictargsvari["parshigh"]["priofactdoff"] = 1.5
    scalxaxi = "self"
    lablxaxi = "$\\alpha_p$"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["priofactdoff"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="priofactdoff",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_truenumbelem(strgcnfgextnexec=None):

    '''
    Measure inference behavior as the true number of Chandra point sources increases.

    The mock population is scanned from 10 to 1000 sources while the fitted and true population caps
    remain high enough to test source-crowding and scaling effects.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["priofactdoff"] = 0.0
    dictargs["fittmaxmnumbelempop0reg0"] = 1000
    dictargs["truemaxmnumbelempop0reg0"] = 1000
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["numbvlow", "numbloww", "numbnomi", "numbhigh", "numbvhig"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["numbvlow"]["truenumbelempop0reg0"] = 10
    dictargsvari["numbloww"]["truenumbelempop0reg0"] = 30
    dictargsvari["numbnomi"]["truenumbelempop0reg0"] = 100
    dictargsvari["numbhigh"]["truenumbelempop0reg0"] = 300
    dictargsvari["numbvhig"]["truenumbelempop0reg0"] = 1000
    lablxaxi = "$N_{pts}$"
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["truenumbelempop0reg0"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="truenumbelempop0reg0",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_trueminmflux(strgcnfgextnexec=None):

    '''
    Measure inference behavior as the true minimum Chandra point-source flux changes.

    The function holds the population size fixed and varies the faint-end cutoff to test sensitivity
    to increasingly weak sources and the resulting source-confusion regime.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["indxenerincl"] = nump.array([2])
    listnamecnfgextn = ["truevlow", "trueloww", "truenomi", "truehigh", "truevhig"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["truevlow"]["trueminmflux"] = 3e-10
    dictargsvari["trueloww"]["trueminmflux"] = 1e-09
    dictargsvari["truenomi"]["trueminmflux"] = 3e-09
    dictargsvari["truehigh"]["trueminmflux"] = 1e-08
    dictargsvari["truevhig"]["trueminmflux"] = 3e-08
    lablxaxi = "$f_{min}$ [cm$^{-2}$ s$^{-1}$ keV$^{-1}$]"
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["trueminmflux"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        execpara=True,
        namexaxi="trueminmflux",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_perf(strgcnfgextnexec=None):

    '''
    Benchmark PCAT runtime, memory use, autocorrelation, and local-pixel bookkeeping versus source
    clustering scale.

    Three true angular correlation widths are simulated with a long chain so computational scaling
    can be separated from changes in the spatial concentration of the catalog.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["priofactdoff"] = 0.2
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["numbswep"] = 1000000
    dictargs["numbsamp"] = 1000
    listnamecnfgextn = ["sigcloww", "sigcnomi", "sigchigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["sigcloww"]["truesigc"] = 0.5 / anglfact
    dictargsvari["sigcnomi"]["truesigc"] = 2.0 / anglfact
    dictargsvari["sigchigh"]["truesigc"] = 4.0 / anglfact
    lablxaxi = "$\\sigma_c$ [$^\\circ$]"
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(anglfact * dictargsvari[namecnfgextn]["truesigc"])
        for namecnfgextn in listnamecnfgextn
    ]
    listnamevarbcomp = [
        "timereal",
        "timeproctotl",
        "timeproctotlswep",
        "timeatcrcntpmaxm",
        "timeprocnorm",
        "meanmemoresi",
        "timerealtotl",
    ]
    listnamevarbcomp += ["numbpixlprox%04d" % indxiter for indxiter in range(3)]
    listscalvarbcomp = ["self" for namevarbcomp in listnamevarbcomp]
    listlablvarbcomp = [
        "$t$ [s]",
        "$t_{CPU}$ [s]",
        "$t_{CPU}^\\prime$ [s]",
        "$t_{MC}$",
        "$t_{CPU}^{\\prime\\prime}$ [s]",
        "$\\bar{M}$",
        "$\\partial_t\\bar{M}$",
        "$t$ [s]",
    ]
    listlablvarbcomp += ["$N_{pxp,%d}$" % indxiter for indxiter in range(3)]
    listtypevarbcomp = ["" for namevarbcomp in listnamevarbcomp]
    listpdfnvarbcomp = ["" for namevarbcomp in listnamevarbcomp]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="truesigc",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
        listnamevarbcomp=listnamevarbcomp,
        listscalvarbcomp=listscalvarbcomp,
        listlablvarbcomp=listlablvarbcomp,
        listtypevarbcomp=listtypevarbcomp,
        listpdfnvarbcomp=listpdfnvarbcomp,
    )
    dictoutp = dictglob

    return dictoutp


def exec_psfn(strgcnfgextnexec=None):

    '''
    Quantify the impact of point-spread-function treatment on Chandra point-source inference.

    The comparison allows the PSF width to float, fixes it to the true value, or fixes it to an
    incorrect value, thereby isolating bias caused by PSF uncertainty and misspecification.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = 200000000.0
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["numbsidecart"] = 10
    dictargs["maxmgangdata"] = 0.492 / anglfact * 10 / 2.0
    dictargs["minmflux"] = 1e-07
    dictargs["priofactdoff"] = 0.2
    dictargs["truenumbelempop0reg0"] = 10
    listnamecnfgextn = ["nomi", "psfntfix", "psfnwfix"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["psfntfix"]["proppsfp"] = False
    dictargsvari["psfnwfix"]["proppsfp"] = False
    dictargsvari["psfnwfix"]["sigcen00evt0"] = 0.5 / anglfact
    lablxaxi = "PSF"
    scalxaxi = "self"
    listtickxaxi = ["Float", "Fixed/Wrong", "Fixed/True"]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="truesigc",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_anglassc(strgcnfgextnexec=None):

    '''
    Test source-catalog inference across a range of angular clustering scales.

    The function simulates 400 point sources with five spatial correlation lengths and compares how
    the sampler behaves as the field changes from strongly clustered to comparatively diffuse.
    '''

    dictoutp = None
    anglfact = 180.0 / nump.pi
    dictargs = {}
    dictargs["truemaxmnumbelempop0reg0"] = 400
    dictargs["truenumbelempop0reg0"] = 400
    dictargs["listnameback"] = ["isot"]
    dictargs["backtype"] = [[10.0]]
    dictargs["truenumbpopl"] = 1
    dictargs["refrlegdpopl"] = ["PS"]
    dictargs["trueelemtype"] = ["lghtpnts"]
    dictargs["maxmgangdata"] = 10.0 / anglfact
    dictargs["truespatdisttype"] = ["self"]
    dictargs["spectype"] = ["powr"]
    dictargs["psfnevaltype"] = "kern"
    dictargs["trueelemregitype"] = [True]
    dictargs["proppsfp"] = False
    dictargs["fittnumbpopl"] = 1
    dictargs["fittelemtype"] = ["lghtpnts"]
    dictargs["fittspatdisttype"] = ["self"]
    dictargs["fittmaxmnumbelempop0reg0"] = 1000
    dictargs["forccart"] = True
    dictargs["pixltype"] = "cart"
    dictargs["numbsidecart"] = 100
    dictargs["numbswep"] = 100000
    dictargs["inittype"] = "refr"
    dictargs["numbsamp"] = 1000
    listnamecnfgextn = ["vlow", "loww", "nomi", "high", "vhig"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["vlow"]["anglassc"] = 4.0 / anglfact
    dictargsvari["loww"]["anglassc"] = 2.0 / anglfact
    dictargsvari["nomi"]["anglassc"] = 1.0 / anglfact
    dictargsvari["high"]["anglassc"] = 0.5 / anglfact
    dictargsvari["vhig"]["anglassc"] = 0.2 / anglfact
    lablxaxi = "$\\theta_{asc}$ [deg]"
    scalxaxi = "self"
    listtickxaxi = [
        tdpy.util.mexp(anglfact * dictargsvari[namecnfgextn]["anglassc"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="anglassc",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
    )
    dictoutp = dictglob

    return dictoutp


def exec_errr(strgcnfgextnexec=None):

    '''
    Evaluate count-prediction errors as the spectral interpolation fraction is varied.

    Using a synthetic-dynamics cluster model, the function compares maximum and mean count errors
    for coarse, nominal, and fine spectral evaluation settings.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "sdyn"
    dictargs["backtype"] = [["tgasback.fits"]]
    dictargs["numbsidecart"] = 200
    dictargs["strgexpo"] = 1.0
    dictargs["elemtype"] = ["clus"]
    dictargs["psfnevaltype"] = "kern"
    dictargs["numbswep"] = 1000000
    dictargs["numbsamp"] = 1000
    listnamecnfgextn = ["nomi", "loww", "high"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["nomi"]["specfraceval"] = 0.1
    dictargsvari["loww"]["specfraceval"] = 0.01
    dictargsvari["high"]["specfraceval"] = 1.0
    lablxaxi = "$\\Delta_f$ "
    scalxaxi = "logt"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["specfraceval"])
        for namecnfgextn in listnamecnfgextn
    ]
    listnamevarbcomp = ["cntperrrreg0pop0maxm", "cntperrrreg0pop0mean"]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="specfraceval",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
        listnamevarbcomp=listnamevarbcomp,
    )
    dictoutp = dictglob

    return dictoutp


def exec_plot(strgcnfgextnexec=None):

    '''
    Generate the initialization and intermediate diagnostic plots for a nominal Chandra mock.

    Inference is disabled so this function specifically validates plotting paths and visual products
    without the cost or variability of a full sampling run.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["mockonly"] = True
    dictargs["boolmakeplotinit"] = True
    dictargs["boolmakeplotintr"] = True
    listnamecnfgextn = ["nomi"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def exec_elemspatevaltype(strgcnfgextnexec=None):

    '''
    Compare full and local-hash spatial likelihood evaluation for a Chandra point-source mock.

    The configurations test two local interaction radii against the exact full calculation, using
    the likelihood at the true model to identify approximation error.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["numbswep"] = 1000000
    dictargs["numbsamp"] = 1000
    dictargs["mockonly"] = True
    dictargs["boolmakeplot"] = False
    listnamecnfgextn = ["full", "locllarg", "loclsmal"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["full"]["elemspatevaltype"] = ["full"]
    dictargsvari["locllarg"]["elemspatevaltype"] = ["loclhash"]
    dictargsvari["locllarg"]["maxmangleval"] = nump.array([5.0, 7.0, 10.0]) / anglfact
    dictargsvari["loclsmal"]["elemspatevaltype"] = ["loclhash"]
    dictargsvari["loclsmal"]["maxmangleval"] = nump.array([2.0, 3.0, 4.0]) / anglfact
    scalxaxi = "self"
    lablxaxi = ""
    listtickxaxi = ["Full", "Local, LK", "Local, SK"]
    listnamevarbcomp = ["truelliktotl"]
    listscalvarbcomp = ["self"]
    listlablvarbcomp = ["$\\ln P(D|M_{true})$"]
    listtypevarbcomp = [""]
    listpdfnvarbcomp = [""]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="elemspatevaltype",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
        listnamevarbcomp=listnamevarbcomp,
        listscalvarbcomp=listscalvarbcomp,
        listlablvarbcomp=listlablvarbcomp,
        listtypevarbcomp=listtypevarbcomp,
        listpdfnvarbcomp=listpdfnvarbcomp,
    )
    dictoutp = dictglob

    return dictoutp


def exec_spmr(strgcnfgextnexec=None):

    '''
    Measure split-merge proposal performance as the split-merge search radius changes.

    The function compares acceptance fractions and autocorrelation time for compact and broad
    proposal neighborhoods in a 100-source Chandra mock.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["exprtype"] = "chan"
    dictargs["strgexpo"] = "expochanhome7msc06000000.fits"
    dictargs["elemtype"] = ["lghtpnts"]
    dictargs["truenumbelempop0reg0"] = 100
    dictargs["probtran"] = 1.0
    dictargs["probspmr"] = 1.0
    dictargs["numbswep"] = 1000000
    dictargs["numbsamp"] = 1000
    dictargs["boolmakeplot"] = False
    listnamecnfgextn = ["radiloww", "radihigh"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["radiloww"]["radispmr"] = 0.5 / anglfact
    dictargsvari["radihigh"]["radispmr"] = 2.0 / anglfact
    scalxaxi = "self"
    lablxaxi = "$\\alpha_p$"
    listnamevarbcomp = ["accpsplt", "accpmerg", "timeatcrcntpmaxm"]
    listscalvarbcomp = ["self" for namevarbcomp in listnamevarbcomp]
    listlablvarbcomp = ["$\\alpha_{splt}$", "$\\alpha_{merg}$", "$\\tau_{ac}$"]
    listtypevarbcomp = ["", "", ""]
    listpdfnvarbcomp = ["", "", ""]
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["radispmr"]) for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        strgcnfgextnexec=strgcnfgextnexec,
        namexaxi="radispmr",
        lablxaxi=lablxaxi,
        scalxaxi=scalxaxi,
        listtickxaxi=listtickxaxi,
        listnamevarbcomp=listnamevarbcomp,
        listscalvarbcomp=listscalvarbcomp,
        listlablvarbcomp=listlablvarbcomp,
        listtypevarbcomp=listtypevarbcomp,
        listpdfnvarbcomp=listpdfnvarbcomp,
    )
    dictoutp = dictglob

    return dictoutp


def exec_ferm():

    '''
    Create a simple isotropic Fermi-like input cube and run a Cartesian PCAT tutorial analysis.

    The function validates the expected HEALPix dimensions, writes the synthetic background map, and
    initializes a long point-source sampling configuration.
    '''

    dictoutp = None
    numbener = 5
    numbside = 256
    numbpixl = 12 * numbside**2
    numbevtt = 4
    if numbener < 1 or numbside < 1 or numbpixl != 12 * numbside**2 or numbevtt < 1:
        raise RuntimeError("Something has gone wrong..")
    fluxisot = 1e-06 * nump.ones((numbener, numbpixl, numbevtt))
    path = oper.environ["PCAT_DATA_PATH"] + "/data/inpt/isottuto.fits"
    fits.writeto(path, fluxisot, overwrite=True)
    cmnd = (
        "wget https://faun.rc.fas.harvard.edu/tansu/pcat/tuto/"
        "psf_P7REP_SOURCE_V15_back.fits "
        "$PCAT_DATA_PATH/data/inpt/psf_P7REP_SOURCE_V15_back.fits"
    )
    dictoutp = pcat.init(
        forccart=True,
        pixltype="cart",
        diagmode=False,
        backtype=[1.0],
        numbswep=2000000,
        strgexpo=100000000000.0,
        probbrde=0.5,
    )

    return dictoutp


def exec_gausmixt():

    '''
    Run the default Gaussian-mixture demonstration with binned data handling.

    This is a minimal initialization test for the generic transdimensional mixture-model interface.
    '''

    dictoutp = None
    dictglob = dict()
    dictoutp = pcat.init(dictglob)
    dictoutp = dictglob

    return dictoutp


def exec_gausmixtunbi():

    '''
    Run the Gaussian-mixture demonstration with unbinned data handling.

    This provides a direct counterpart to the binned mixture test and validates the unbinned
    likelihood path.
    '''

    dictoutp = None
    dictoutp = pcat.init(boolbins=False)

    return dictoutp


def exec_lenssimu():

    '''
    Initialize an HST WFC3/IR strong-lens image simulation without substructure inference.

    The function sets the image scale, exposure, reconstruction state, and a tightly constrained
    background level for a selected simulated lens data set.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    sizepixl = 0.05 / anglfact
    namedatasets = "lens29075550"
    strgexpo = 7.37487548893e21
    numbside = 400
    maxmgangdata = numbside * 0.5 * sizepixl
    strgexprsbrt = namedatasets + "_%04d.fits" % numbside
    if namedatasets == "lens29075550":
        initbacpbac0en00 = 1.1e-07
        fittmeanbacpbac0en00 = 1.1e-07
        fittstdvbacpbac0en00 = fittmeanbacpbac0en00 * 0.001
        fittscalbacpbac0en00 = "gaus"
    else:
        initbacpbac0en00 = None
        fittmeanbacpbac0en00 = None
        fittstdvbacpbac0en00 = None
        fittscalbacpbac0en00 = None
    boolinfe = False
    maxmnumbelem = nump.array([0])
    dictfitt = dict()
    dictfitt["init"] = dict()
    dictfitt["fitt"] = dict()
    dictfitt["fitt"]["maxm"] = dict()
    dictfitt["fitt"]["maxm"]["maxmnumbelem"] = maxmnumbelem
    dictfitt["init"]["initbacpbac0en00"] = initbacpbac0en00
    dictfitt["init"]["fittmeanbacpbac0en00"] = fittmeanbacpbac0en00
    dictfitt["init"]["fittstdvbacpbac0en00"] = fittstdvbacpbac0en00
    dictfitt["init"]["fittscalbacpbac0en00"] = fittscalbacpbac0en00
    dictoutp = pcat.main.init_image(
        typeexpr="HST_WFC3_IR",
        typedata="simu",
        dictfitt=dictfitt,
        strgexpo=strgexpo,
        inittype="reco",
        namerecostat="pcat_lens_inpt",
        maxmgangdata=maxmgangdata,
        strgexprsbrt=strgexprsbrt,
        boolinfe=boolinfe,
    )

    return dictoutp


def eval_lenscntpresi():

    '''
    
    Evaluate the numerical precision of the strong-lens count calculation.

    One element is placed in each of three populations so the internal count-evaluation
    routine can be tested in a small, controlled HST WFC3/IR configuration.
    
    '''

    # Initialize output dictionary
    dictoutp = None
    
    # Configure controlled three-population case for precision evaluation
    print('Setting up strong-lens count precision evaluation...')
    numbsidecart = 80
    numbpixl = numbsidecart**2
    expo = nump.ones((2, numbpixl, 1))
    cntpdata = nump.ones_like(expo)
    fitt = tdpy.gdatstrt()
    # Keep fitting model formally non-empty to satisfy strict init guard.
    fitt.numbparagenrbase = 1
    fitt.numbpopl = 0
    fitt.indxpopl = nump.array([], dtype=int)
    # Seed a minimal isotropic background component so model emission is non-zero.
    fitt.numbparagenr = 1
    fitt.maxmnumbpara = 1
    fitt.boollens = True
    fitt.boollenshost = True
    fitt.boollenssubh = False
    fitt.boolemishost = True
    fitt.typeemishost = 'sers'
    fitt.indxsersfgrd = nump.array([0], dtype=int)
    fitt.listnamediff = ["back0000"]
    fitt.listnamegcom = ["modl"]
    fitt.indxback = nump.array([0], dtype=int)
    fitt.sbrtbacknorm = [nump.ones((1, numbpixl, 1))]
    fitt.boolspecback = [False]
    fitt.boolunifback = [True]
    fitt.indxbacpback = [nump.array([0], dtype=int)]
    fitt.typeevalpsfn = "none"
    fitt.convdiffanyy = False
    fitt.indxpara = tdpy.gdatstrt()
    fitt.indxpara.bacp = nump.array([0], dtype=int)
    fitt.indxpara.genrbase = nump.array([0], dtype=int)
    fitt.indxpara.genrbasescal = nump.array([0], dtype=int)
    fitt.indxpara.genrbasepert = nump.array([0], dtype=int)
    fitt.namepara = tdpy.gdatstrt()
    fitt.namepara.genrbase = ['bacpback0000en00']
    fitt.namepara.scal = []
    fitt.namepara.genrelem = [[]]
    fitt.scalpara = tdpy.gdatstrt()
    fitt.scalpara.genrbase = ['self']
    fitt.minmpara = tdpy.gdatstrt()
    fitt.maxmpara = tdpy.gdatstrt()
    fitt.minmpara.bacpback0000en00 = 1e-6
    fitt.maxmpara.bacpback0000en00 = 1e3
    fitt.numblpri = 1

    dictglob = {
        "typeexpr": "HST_WFC3_IR",
        "typedata": "simu",
        "intrevalcntpresi": True,
        # Force a meaningful image cube shape.
        "typepixl": "cart",
        "boolbindspat": True,
        "numbsidecart": numbsidecart,
        "numbpixl": numbpixl,
        "numbpixlcart": numbpixl,
        "indxpixl": nump.arange(numbpixl, dtype=int),
        "indxpixlrofi": nump.arange(numbpixl, dtype=int),
        "numbener": 1,
        "indxener": nump.array([0], dtype=int),
        "numbdqlt": 1,
        "indxdqlt": nump.array([0], dtype=int),
        "expo": expo,
        "cntpdata": cntpdata,
        "varidata": nump.ones_like(expo),
        "numbdata": expo.size,
        "apix": 1.0,
        "fitt": fitt,
        # Keep this as a setup/evaluation diagnostic rather than long sampling.
        "numbswep": 50,
        "numbburn": 10,
    }
    dictoutp = pcat.main.init(dictglob)
    
    return dictoutp


def eval_lenscntpmodl():

    '''
    Run a native PCAT HST strong-lens mock through the existing mock-run helper.

    This keeps the image generation inside PCAT and uses the repository's own simulation
    machinery instead of constructing a synthetic map in the wrapper.
    '''

    return exec_lensmockfittnumb()


def exec_lensmocktrueminmdefs(strgcnfgextnexec=None):

    '''
    Study sensitivity to the true minimum subhalo deflection scale in HST lens mocks.

    Five lower cutoffs are compared to determine how the detectable and inferred substructure
    population changes as progressively weaker perturbers are included.
    '''

    dictoutp = None
    numbelem = int(25.0 * 10.0**0.9)
    anglfact = 3600.0 * 180.0 / nump.pi
    dictpcatinpt = pcat.retr_dictpcatinpt()
    listlablcnfg = ["", "", "", "", ""]
    listnamecnfgextn = ["truevlow", "trueloww", "nomi", "truehigh", "truevhig"]
    dictpcatinptvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictpcatinptvari[namecnfgextn] = pcat.retr_dictpcatinpt()
    dictpcatinpt["typeexpr"] = "HST_WFC3_IR"
    dictpcatinpt["boolbinsener"] = False
    dictpcatinpt["boolmakeplot"] = True
    dictpcatinpt["boolmakeplotinit"] = True
    dictpcatinpt["fittminmnumbelempop0"] = 1
    dictpcatinpt["fittmaxmnumbelempop0"] = 25
    dictpcatinpt["typepixl"] = "cart"
    dictpcatinpt["boolbindspat"] = True
    dictpcatinpt["numbsidecart"] = 80
    dictpcatinpt["numbpixl"] = dictpcatinpt["numbsidecart"]**2
    dictpcatinpt["numbpixlcart"] = dictpcatinpt["numbpixl"]
    dictpcatinpt["indxpixl"] = nump.arange(dictpcatinpt["numbpixl"], dtype=int)
    dictpcatinpt["indxpixlrofi"] = nump.arange(dictpcatinpt["numbpixl"], dtype=int)
    dictpcatinpt["numbswep"] = 20000
    dictpcatinpt["numbswepplot"] = 5000
    dictpcatinpt["numbburn"] = 1000
    dictpcatinpt["numbsamp"] = 1000
    dictpcatinpt["plot_func"] = pcat.plot_lens
    lablxaxi = "$\\alpha_{min}$ [^{\\prime\\prime}]"
    scalxaxi = "logt"
    dictpcatinpt["typeexpr"] = "HST_WFC3_IR"
    dictpcatinpt["dictboth"] = dict()
    dictpcatinpt["dictboth"]["typeelem"] = ["lens"]
    dictglob = pcat.main.initarry(
        dictpcatinptvari,
        listlablcnfg,
        dictpcatinpt=dictpcatinpt,
        boolexecpara=False,
        namexaxivari="trueminmdefs",
        lablxaxivari=lablxaxi,
        scalxaxivari=scalxaxi,
        strgcnfgextnexec=strgcnfgextnexec,
    )
    dictoutp = dictglob

    return dictoutp


def exec_lensmockpars(strgcnfgextnexec=None):

    '''
    Measure how the parsimony prior controls inferred substructure in HST lens mocks.

    The prior factor is scanned from zero to one, with one configuration sampling the prior, to test
    the balance between fitting genuine perturbers and suppressing unnecessary components.
    '''

    dictoutp = None
    numbelem = int(25.0 * 10.0**0.9)
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["truenumbelempop0"] = 25
    dictargs["typeelem"] = ["lens"]
    listnamecnfgextn = ["none", "loww", "nomi", "high", "vhig"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["none"]["priofactdoff"] = 0.0
    dictargsvari["loww"]["priofactdoff"] = 0.25
    dictargsvari["nomi"]["boolsampprio"] = True
    dictargsvari["nomi"]["priofactdoff"] = 0.5
    dictargsvari["high"]["priofactdoff"] = 0.75
    dictargsvari["vhig"]["priofactdoff"] = 1.0
    lablxaxi = "$\\alpha_{p}$"
    scalxaxi = "self"
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["priofactdoff"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        namexaxivari="priofactdoff",
        lablxaxivari=lablxaxi,
        scalxaxivari=scalxaxi,
        tickxaxivari=listtickxaxi,
        strgcnfgextnexec=strgcnfgextnexec,
    )
    dictoutp = dictglob

    return dictoutp



def exec_lensmockfittnumb(strgcnfgextnexec=None):

    '''
    
    Test strong-lens recovery under fixed and transdimensional fitted counts.
    
    Fits constrained to 0, 1, 2, or 10 subhalos are compared with nominal
    transdimensional model for mock containing 25 true perturbers.
    
    '''

    print('Setting up strong-lens mock count test...')
    print('Running PCAT sampling runtime for HST lens image generation and posterior inference...')
    strgcnfg = 'eval_lenscntpmodl'
    if strgcnfgextnexec is not None:
        strgcnfg += '_%s' % strgcnfgextnexec
    dictoutp = pcat.main.sample(
        typeexpr="HST_WFC3_IR",
        typedata="simu",
        strgcnfg=strgcnfg,
        truenumbelempop0=25,
        typeelem=["lens"],
        boolmakeplot=True,
        boolmakeplotinit=True,
        forcneww=True,
        typepixl="cart",
        boolbindspat=True,
        numbsidecart=80,
        numbpixl=80**2,
        numbpixlcart=80**2,
        indxpixl=nump.arange(80**2, dtype=int),
        indxpixlrofi=nump.arange(80**2, dtype=int),
        # Allow transdimensional exploration so all proposal types
        # (within-model, birth/death, split/merge) can be exercised.
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=10,
        probtran=0.6,
        probspmr=0.5,
        numbswep=20000,
        numbburn=1000,
        numbsamp=1000,
        plot_func=pcat.plot_lens,
        boolsimuonly=False,
    )

    return dictoutp


def exec_lensmocktrueback(strgcnfgextnexec=None):

    '''
    
    Measure true image-background effect on HST lens-substructure inference.
    
    Low, nominal, and high backgrounds simulated while subhalo population held
    fixed, isolating sensitivity changes caused by background photon counts.
    
    '''

    # Initialize output dictionary
    dictoutp = None
    
    # Configure base parameters for background sensitivity test
    print('Setting up background-level sensitivity test...')
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["truenumbelempop0"] = 15
    dictargs["typeelem"] = ["lens"]
    
    # Define configurations with different background levels
    listnamecnfgextn = ["loww", "nomi", "high"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    
    # Set true background counts [photons/pixel]
    dictargsvari["loww"]["truecntsback"] = 0.1
    dictargsvari["nomi"]["truecntsback"] = 1.0
    dictargsvari["high"]["truecntsback"] = 10.0
    
    # Configure plot axis labels
    lablxaxi = "True background [photons/pixel]"
    listtickxaxi = ["0.1", "1.0", "10"]
    
    # Execute test configurations
    print('Running fits with varying background levels...')
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        namexaxivari="truecntsback",
        lablxaxivari=lablxaxi,
        listtickxaxi=listtickxaxi,
        strgcnfgextnexec=strgcnfgextnexec,
    )
    
    # Store result
    dictoutp = dictglob

    return dictoutp

    dictoutp = None
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["truenumbelempop0"] = 25
    dictargs["typeelem"] = ["lens"]
    dictargs["numbswep"] = 10000
    dictargs["numbsamp"] = 100
    anglfact = 3600.0 * 180.0 / nump.pi
    listnamecnfgextn = ["loww", "nomi", "high"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["loww"]["truebacpbac0en00"] = 1e-07
    dictargsvari["nomi"]["truebacpbac0en00"] = 2e-07
    dictargsvari["high"]["truebacpbac0en00"] = 4e-07
    listtickxaxi = [
        tdpy.util.mexp(dictargsvari[namecnfgextn]["truebacpbac0en00"])
        for namecnfgextn in listnamecnfgextn
    ]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        namexaxivari="truebacp",
        tickxaxivari=listtickxaxi,
        strgcnfgextnexec=strgcnfgextnexec,
    )
    dictoutp = dictglob

    return dictoutp


def exec_lensmocksour(strgcnfgextnexec=None):

    '''
    Stress-test HST lens inference against changes in source structure, subhalo population, prior,
    exposure, and signal amplitude.

    The configuration suite includes null data, fixed-count, low-deflection, high-signal-to-noise,
    and alternative source-profile cases to identify major modeling sensitivities.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["typeelem"] = ["lens", "lghtgausbgrd"]
    dictargs["numbelempop0"] = 20
    dictargs["maxmnumbelempop0"] = 100
    dictargs["numbelempop1"] = 10
    dictargs["maxmnumbelempop1"] = 100
    dictargs["spatdisttype"] = ["unif", "dsrcexpo"]
    numbelem = int(25.0 * 10.0**0.9)
    listnamecnfgextn = [
        "nomi",
        "datanone",
        "subhsing",
        "truelowr",
        "parsnone",
        "truevlow",
        "s2nrhigh",
        "s2nrvhig",
        "amplhigh",
        "bgrdunif",
    ]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["datanone"]["killexpo"] = True
    dictargsvari["bgrdunif"]["spatdisttype"] = ["unif", "unif"]
    dictargsvari["subhsing"]["fittminmnumbelempop0"] = 1
    dictargsvari["subhsing"]["fittmaxmnumbelempop0"] = 1
    dictargsvari["truevlow"]["truenumbelempop0"] = 30
    dictargsvari["truelowr"]["fittminmdefs"] = 0.01 / anglfact
    dictargsvari["truevlow"]["truenumbelempop0"] = int(25.0 * 10.0**0.9)
    dictargsvari["truevlow"]["trueminmdefs"] = 0.0003 / anglfact
    dictargsvari["truevlow"]["fittminmdefs"] = 0.01 / anglfact
    dictargsvari["parsnone"]["priofactdoff"] = 0.0
    dictargsvari["s2nrhigh"]["strgexpo"] = 10000.0 / 1.6305e-19
    dictargsvari["s2nrvhig"]["strgexpo"] = 100000.0 / 1.6305e-19
    dictargsvari["amplhigh"]["minmdefs"] = 0.1 / anglfact
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def test_lensmocksele(strgcnfgextnexec=None):

    '''
    Estimate subhalo selection functions from repeated HST strong-lens mock realizations.

    For each mock, the function records true deflection, cutoff, and relevance distributions before
    and after parameter-based or relevance-based selection, enabling completeness comparisons.
    '''

    dictoutp = None
    pathbase = oper.environ["TDGU_DATA_PATH"] + "/" + "pcat_lens_mock_sele/"
    pathdata = pathbase + "data/"
    pathvisu = pathbase + "visuals/"
    oper.system("mkdir -p %s" % pathdata)
    oper.system("mkdir -p %s" % pathvisu)
    numbitermacr = 30
    numbiterelem = 10
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["mockonly"] = True
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["minmdefs"] = 0.0005 / anglfact
    dictargs["numbelempop0"] = 1000
    dictargs["variasca"] = False
    dictargs["variacut"] = False
    dictargs["allwfixdtrue"] = False
    dictargs["maxmnumbelempop0"] = 1000
    listnamesele = ["pars", "nrel"]
    numbsele = len(listnamesele)
    listnamefeatsele = ["defs", "mcut", "rele"]
    numbfeatsele = len(listnamefeatsele)
    listnamecnfgextn = ["nomi"]
    dictargsvari = {"nomi": {}}
    matrcutf = nump.empty((numbitermacr, numbiterelem, numbfeatsele))
    liststrgvarboutp = []
    for strgvarbelem in listnamefeatsele:
        liststrgvarboutp += ["truehist" + strgvarbelem]
        for namesele in listnamesele:
            liststrgvarboutp += ["true" + strgvarbelem + namesele]
            liststrgvarboutp += ["truehist" + strgvarbelem + namesele]
    for indxiter in range(numbitermacr):
        listgdat, dictglob = pcat.main.initarry(
            dictargsvari,
            dictargs,
            listnamecnfgextn,
            seedtypeelem="rand",
            liststrgvarboutp=liststrgvarboutp,
            strgcnfgextnexec=strgcnfgextnexec,
        )
        gdat = listgdat[0]
        if indxiter == 0:
            hist = nump.zeros((numbiterelem, gdat.numbbinsplot))
            histsele = nump.zeros((numbiterelem, gdat.numbbinsplot))
            histfitt = nump.zeros((numbiterelem, gdat.numbbinsplot))
            factsele = nump.zeros((numbiterelem, gdat.numbbinsplot))
            for namesele in listnamesele:
                pathimagsele = pathvisu + namesele + "/"
                oper.system("mkdir -p %s" % pathimagsele)
            truefixp = nump.empty((numbitermacr, gdat.truenumbfixp))
            corrfixpcutf = nump.empty(gdat.truenumbfixp)
            pvalfixpcutf = nump.empty(gdat.truenumbfixp)
        for namefixp in gdat.truenamefixp:
            truefixp[indxiter, :] = gdat.truefixp
        for indxfeat, namefeat in enumerate(listnamefeatsele):
            lablvarbtotl = getattr(gdat, "labl" + namefeat + "totl")
            meanvarb = getattr(gdat, "mean" + namefeat)
            deltvarb = getattr(gdat, "delt" + namefeat)
            factvarbplot = getattr(gdat, "fact" + namefeat + "plot")
            limtvarb = [
                factvarbplot * getattr(gdat, "minm" + namefeat),
                factvarbplot * getattr(gdat, "maxm" + namefeat),
            ]
            for indxelem in range(numbiterelem):
                hist[indxelem, :] = dictglob["truehist" + namefeat][indxelem][0, :]
            for namesele in listnamesele:
                pathimagsele = pathvisu + namesele + "/"
                for indxelem in range(numbiterelem):
                    histsele[indxelem, :] = dictglob["truehist" + namefeat + namesele][indxelem][
                        0, :
                    ]
                    factsele[indxelem, :] = nump.divide(
                        histsele[indxelem, :],
                        hist[indxelem, :],
                        out=nump.zeros_like(histsele[indxelem, :], dtype=float),
                        where=hist[indxelem, :] != 0.0,
                    )
                    alph, loca, cutf = scip.stats.invgamma.fit(
                        dictglob["true" + namefeat + namesele][indxelem][0], floc=0.0, f0=1.9
                    )
                    histfitt[indxelem, :] = (
                        nump.sum(histsele[indxelem, :])
                        * scip.stats.invgamma.pdf(meanvarb, alph, loc=loca, scale=cutf)
                        * deltvarb
                    )
                    matrcutf[indxiter, indxelem, indxfeat] = cutf
                meanfactsele = nump.mean(factsele, 0)
                meanmatrcutf = nump.mean(matrcutf, axis=1)
                figr, axis = pypl.subplots(figsize=(gdat.plotsize, gdat.plotsize))
                for indxelem in range(numbiterelem):
                    axis.loglog(
                        meanvarb * factvarbplot, hist[indxelem, :], color="g", alpha=0.1, ls="-."
                    )
                    axis.loglog(
                        meanvarb * factvarbplot,
                        histsele[indxelem, :],
                        color="g",
                        alpha=0.1,
                        ls="--",
                    )
                    axis.loglog(
                        meanvarb * factvarbplot, histfitt[indxelem, :], color="g", alpha=0.1, ls="-"
                    )
                    axis.axvline(
                        matrcutf[indxiter, indxelem, indxfeat] * factvarbplot, color="m", alpha=0.1
                    )
                axis.axvline(
                    nump.power(
                        nump.prod(matrcutf[indxiter, :, indxfeat]), nump.array([1.0 / numbiterelem])
                    )
                    * factvarbplot,
                    color="m",
                )
                axis.loglog(meanvarb * factvarbplot, nump.mean(hist, 0), color="g", ls="-.")
                axis.loglog(meanvarb * factvarbplot, nump.mean(histsele, 0), color="g", ls="--")
                axis.loglog(meanvarb * factvarbplot, nump.mean(histfitt, 0), color="g", ls="-")
                axis.set_xlabel(lablvarbtotl)
                axis.set_ylabel("$N$")
                axis.grid(False)
                axis.set_ylim([0.5, None])
                axis.set_xlim(limtvarb)
                path = pathimagsele + "hist" + namefeat + "%04d.png" % indxiter
                if not oper.path.exists(path):
                    pypl.tight_layout()
                    print("Writing to %s..." % path)
                    pypl.savefig(path, bbox_inches="tight")
                    pypl.close()
                else:
                    print("%s already exists..." % path)
                    pypl.close()
                figr, axis = pypl.subplots(figsize=(gdat.plotsize, gdat.plotsize))
                axis.loglog(meanvarb * factvarbplot, meanfactsele, color="black")
                for indxelem in range(numbiterelem):
                    axis.loglog(
                        meanvarb * factvarbplot, factsele[indxelem, :], alpha=0.1, color="g"
                    )
                    axis.axvline(
                        matrcutf[indxiter, indxelem, indxfeat] * factvarbplot, color="m", alpha=0.1
                    )
                axis.axvline(
                    nump.power(
                        nump.prod(matrcutf[indxiter, :, indxfeat]), nump.array([1.0 / numbiterelem])
                    )
                    * factvarbplot,
                    color="m",
                )
                axis.set_xlabel(lablvarbtotl)
                axis.set_ylabel("$f$")
                axis.grid(False)
                axis.set_xlim(limtvarb)
                path = pathimagsele + "fact" + namefeat + "%04d.png" % indxiter
                if not oper.path.exists(path):
                    pypl.tight_layout()
                    print("Writing to %s..." % path)
                    pypl.savefig(path, bbox_inches="tight")
                    pypl.close()
                else:
                    print("%s already exists..." % path)
                    pypl.close()
    for namesele in listnamesele:
        pathimagsele = pathvisu + namesele + "/"
        for indxfeat, namefeat in enumerate(listnamefeatsele):
            for indxfixp, namefixp in enumerate(gdat.truenamefixp):
                corrfixpcutf[indxfixp], pvalfixpcutf[indxfixp] = scip.stats.pearsonr(
                    truefixp[:, indxfixp], meanmatrcutf[:, indxfeat]
                )
            indx = nump.where(nump.isfinite(corrfixpcutf) & (pvalfixpcutf < 0.1))[0]
            numb = indx.size
            figr, axis = pypl.subplots(figsize=(2 * gdat.plotsize, gdat.plotsize))
            for indxiter in range(numb):
                size = 100.0 * (0.1 - pvalfixpcutf[indx[indxiter]]) + 5.0
                axis.plot(
                    indxiter + 0.5,
                    corrfixpcutf[indx][indxiter],
                    ls="",
                    marker="o",
                    markersize=size,
                    color="black",
                )
            axis.set_xticks(nump.arange(numb) + 0.5)
            axis.set_xticklabels(gdat.truelablfixp[indx])
            axis.set_xlim([0.0, numb])
            path = pathimagsele + "corr" + namefeat + namesele + ".png"
            if not oper.path.exists(path):
                pypl.tight_layout()
                print("Writing to %s..." % path)
                pypl.savefig(path, bbox_inches="tight")
                pypl.close()
            else:
                print("%s already exists..." % path)
                pypl.close()
    dictoutp = dictglob

    return dictoutp


def test_lensmocktmpr(strgcnfgextnexec=None):

    '''
    Test burn-in temperature behavior for an HST strong-lens model without subhalos.

    Reference and perturbed initializations are compared with the nominal setup to diagnose whether
    tempering removes dependence on the starting state.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["burntmpr"] = True
    dictargs["maxmnumbelempop0"] = 0
    dictargs["numbelempop0"] = 0
    listnamecnfgextn = ["nomi", "refr", "pert"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def exec_lensmockmany(strgcnfgextnexec=None):

    '''
    Test HST strong-lens sampling with different numbers of image regions.

    Single-, three-, and five-region configurations are compared with no active subhalos to validate
    regional bookkeeping, backgrounds, and sampler behavior.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["burntmpr"] = True
    for indxiter in range(5):
        dictargs["maxmnumbelempop0reg%d" % indxiter] = 0
        dictargs["numbelempop0reg%d" % indxiter] = 0
    dictargs["inittype"] = "pert"
    dictargs["typeelem"] = ["lens"]
    dictargs["numbregi"] = 3
    dictargs["backtype"] = [1.0, 1.0, 1.0]
    dictargs["numbswep"] = 10000
    dictargs["numbsamp"] = 100
    listnamecnfgextn = ["nomi", "regising", "regimany"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["regising"]["numbregi"] = 1
    dictargsvari["regising"]["backtype"] = [1.0]
    dictargsvari["regimany"]["numbregi"] = 5
    dictargsvari["regimany"]["backtype"] = [1.0] * 5
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def exec_lensmockspmr(strgcnfgextnexec=None):

    '''
    Test split-merge and transdimensional proposal behavior for a single HST lens perturber.

    Proposal probabilities, prior strength, true deflection amplitude, and transition disabling are
    varied to diagnose acceptance and mixing across representative sampler regimes.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["inittype"] = "refr"
    dictargs["numbelempop0"] = 1
    dictargs["truelgalpop00000"] = 1.0 / anglfact
    dictargs["truebgalpop00000"] = 0.5 / anglfact
    dictargs["truedefspop00000"] = 0.01 / anglfact
    dictargs["probtran"] = 1.0
    dictargs["typeelem"] = ["lens"]
    dictargs["probspmr"] = 1.0
    dictargs["indxenerincl"] = nump.array([0])
    listnamecnfgextn = ["nomi", "tranboth", "parshigh", "masshigh", "massloww", "trannone"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["tranboth"]["inittype"] = "pert"
    dictargsvari["tranboth"]["probtran"] = 0.4
    dictargsvari["tranboth"]["probspmr"] = 0.3
    dictargsvari["parshigh"]["priofactdoff"] = 1.0
    dictargsvari["masshigh"]["truedefspop00000"] = 0.03 / anglfact
    dictargsvari["massloww"]["truedefspop00000"] = 0.003 / anglfact
    dictargsvari["trannone"]["probtran"] = 0.0
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def writ_data():

    '''
    Prepare the observed SLACS strong-lens image products used by the input-data analysis.

    The function validates the FITS tables, selects high-quality grade-A lenses, records the
    download list, and converts downloaded imaging products into the expected PCAT data layout.
    '''

    dictoutp = None
    liststrgrade = []
    listrade = [[], []]
    pathbase = oper.environ["TDGU_DATA_PATH"] + "/pcat_lens_inpt/"
    pathdata = pathbase + "data/"
    pathimag = pathbase + "imag/"
    print("Reading SLACS tables...")
    pathslacpara = pathbase + "data/slacpara.fits"
    pathslacfull = pathbase + "data/slacfull.fits"
    hdun = narr_fits_open(pathslacfull)
    numbhead = len(hdun)
    print("%s extensions found." % numbhead)
    for indxiter in range(numbhead):
        print("Extension %d" % indxiter)
        head = hdun[indxiter].header
        data = hdun[indxiter].data
        if data is None:
            print("Data is None, skipping...")
            continue
        else:
            pass
        arry = nump.array(nump.stack((head.keys(), head.values()), 1))
        listtype = []
        for indxsubb in range(arry.shape[0]):
            if arry[indxsubb, 0].startswith("TTYPE"):
                listtype.append(arry[indxsubb, 1])
        if len(listtype) != len(data[0]):
            raise Exception("Number of types does not match the number of fields.")
    indxgold = nump.where((data["Mph"] == "E") & (data["Mul"] == "S") & (data["Lens"] == "A"))[0]
    numbslac = indxgold.size
    path = pathdata + "slacdownlist.txt"
    fileobjt = narr_open(path, "w")
    for indxiter in indxgold:
        strgrade = "%s %s %s %s %s %s" % (
            data["SDSS"][indxiter][:2],
            data["SDSS"][indxiter][2:4],
            data["SDSS"][indxiter][4:9],
            data["SDSS"][indxiter][9:12],
            data["SDSS"][indxiter][12:14],
            data["SDSS"][indxiter][14:],
        )
        liststrgrade.append(strgrade)
        listrade[0].append(data["_RA"][indxiter])
        listrade[1].append(data["_DE"][indxiter])
        strgline = strgrade + " \n"
        fileobjt.write(strgline)
    fileobjt.close()
    for indxiter in range(len(indxgold)):
        print(
            "%20s %20s %20g %20g"
            % (
                data["SDSS"][indxgold[indxiter]],
                data["Name"][indxgold][indxiter],
                data["_RA"][indxgold][indxiter],
                data["_DE"][indxgold][indxiter],
            )
        )
    numbside = 400
    numbsidehalf = numbside // 2
    pathdatapcat = oper.environ["PCAT_DATA_PATH"] + "/data/inpt/"
    strgradestar = "00 29 06.79 -00 54 07.5"
    liststrgrade.append(strgradestar)
    coorstar = astr.coordinates.SkyCoord(strgradestar, unit=(astr.units.hourangle, astr.units.deg))
    listrade[0].append(coorstar.ra.degree)
    listrade[1].append(coorstar.dec.degree)
    numbrade = len(listrade[0])
    print("%d coordinates found." % numbrade)
    listnamefile = ["hst_10886_02_acs_wfc_f814w_drz.fits"]
    numbfile = len(listnamefile)
    print("%d files found." % numbfile)
    for indxiter, namefile in enumerate(listnamefile):
        print("File number %d" % indxiter)
        pathfile = pathdata + namefile
        listdata = tdpy.util.read_fits(pathfile, verbtype=0)
        listhdun = narr_astr_fits_open(pathfile)
        wcso = astr.wcs.WCS(listhdun[2].header)
        strgrade = liststrgrade[indxiter]
        for indxsubb in range(numbrade):
            strgrade = liststrgrade[indxsubb]
            indxyaxi, indxxaxi = wcso.wcs_world2pix(listrade[0][indxsubb], listrade[1][indxsubb], 0)
            if (
                not nump.isfinite(indxyaxi)
                or not nump.isfinite(indxxaxi)
                or indxxaxi - numbsidehalf < 0
                or (indxyaxi - numbsidehalf < 0)
                or (indxxaxi + numbsidehalf > listdata[1].shape[1])
                or (indxyaxi + numbsidehalf > listdata[1].shape[0])
            ):
                continue
            path = pathdatapcat + "lens%s%s%s%s_%04d.fits" % (
                liststrgrade[indxsubb][3:5],
                liststrgrade[indxsubb][6:8],
                liststrgrade[indxsubb][16:18],
                liststrgrade[indxsubb][19:21],
                numbside,
            )
            indxxaxi = int(indxxaxi)
            indxyaxi = int(indxyaxi)
            rate = listdata[1][
                indxxaxi - numbsidehalf : indxxaxi + numbsidehalf,
                indxyaxi - numbsidehalf : indxyaxi + numbsidehalf,
            ]
            rate = rate[None, :, :, None]
            effa = 1.0 / listdata[4]["PHOTFLAM"][0]
            timeobsv = listdata[4]["EXPTIME"][0]
            apix = (0.05 * nump.pi / 3600.0 / 180.0) ** 2
            expo = effa * timeobsv
            sbrt = rate / effa / apix
            cntp = sbrt * expo * apix
            print("Writing to %s..." % path)
            fits.writeto(path, sbrt, overwrite=True)

    return dictoutp


def exec_lensinpt(strgcnfgextnexec=None):

    '''
    Run HST WFC3/IR inference on a prepared observed strong-lens image.

    The function defines the image scale, masks, exposure, background prior, lens and source
    starting values, and alternative crop or model configurations for the selected SLACS-like
    target.
    '''

    dictoutp = None
    anglfact = 3600.0 * 180.0 / nump.pi
    sizepixl = 0.05 / anglfact
    namedatasets = "lens29075550"
    strgexpo = 7.37487548893e21
    maxmgangdata = 100 * 0.5 * sizepixl
    maxmgangdatalarg = 400 * 0.5 * sizepixl
    strgexprsbrt = namedatasets + "_0100.fits"
    strgexprsbrtlarg = namedatasets + "_0400.fits"
    if namedatasets == "lens29075550":
        initlgalsour = -0.1 / anglfact
        initbgalsour = 0.1 / anglfact
        initbacpbac0en00 = 1e-07
        fittmeanbacpbac0en00 = 1.115e-07
        fittstdvbacpbac0en00 = fittmeanbacpbac0en00 * 0.001
        fittscalbacpbac0en00 = "gaus"
    else:
        initlgalsour = None
        initbgalsour = None
        initbacpbac0en00 = None
        fittmeanbacpbac0en00 = None
        fittstdvbacpbac0en00 = None
        fittscalbacpbac0en00 = None
    listmask = [["sqre", -0.3, 0.1, -0.1, 0.2], ["circ", -9, 8, 1]]
    for indxiter, mask in enumerate(listmask):
        for indxsubb, valu in enumerate(mask):
            if not isinstance(valu, str):
                listmask[indxiter][indxsubb] = valu / anglfact
    dictargs = {}
    dictargs["typeelem"] = ["lens"]
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["strgexpo"] = strgexpo
    dictargs["indxenerincl"] = nump.array([0])
    dictargs["savestat"] = True
    dictargs["sqzeprop"] = True
    dictargs["numbswep"] = 10000
    dictargs["numbswepplot"] = 1000
    dictargs["serstype"] = "intp"
    dictargs["numbsamp"] = 100
    dictargs["inittype"] = "reco"
    listnamecnfgextn = [
        "largrofi",
        "largrofimask",
        "nomi",
        "mask",
        "sour",
        "dsrcexpo",
        "sourmask",
        "hostmult",
    ]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargs["initlgalhostisf0"] = -0.1 / anglfact
    dictargs["initbgalhostisf0"] = 0.0
    dictargs["initlgalhostisf1"] = -0.1 / anglfact
    dictargs["initbgalhostisf1"] = 0.0
    dictargs["initlgalsour"] = -0.2 / anglfact
    dictargs["initbgalsour"] = 0.2 / anglfact
    dictargsvari["largrofi"]["maxmnumbelempop0"] = 0
    dictargsvari["largrofi"]["maxmnumbelempop1"] = 0
    dictargsvari["largrofi"]["maxmnumbelempop2"] = 0
    dictargsvari["largrofi"]["strgexprsbrt"] = strgexprsbrtlarg
    dictargsvari["largrofi"]["maxmgangdata"] = maxmgangdatalarg
    dictargsvari["largrofimask"]["maxmnumbelempop0"] = 0
    dictargsvari["largrofimask"]["maxmnumbelempop1"] = 0
    dictargsvari["largrofimask"]["maxmnumbelempop2"] = 0
    dictargsvari["largrofimask"]["listmask"] = listmask
    dictargsvari["largrofimask"]["strgexprsbrt"] = strgexprsbrtlarg
    dictargsvari["largrofimask"]["maxmgangdata"] = maxmgangdatalarg
    dictargsvari["dsrcexpo"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["dsrcexpo"]["maxmnumbelempop0"] = 2
    dictargsvari["dsrcexpo"]["typeelem"] = ["lghtgausbgrd"]
    dictargsvari["dsrcexpo"]["spatdisttype"] = ["dsrcexpo"]
    dictargsvari["dsrcexpo"]["dsrcdisttype"] = ["expo"]
    dictargsvari["dsrcexpo"]["dsrcdistsexppop0"] = 0.01 / anglfact
    dictargsvari["sour"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["sour"]["maxmnumbelempop0"] = 1
    dictargsvari["sour"]["typeelem"] = ["lghtgausbgrd"]
    dictargsvari["hostmult"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["hostmult"]["numbsersfgrd"] = nump.array([2])
    dictargsvari["hostmult"]["maxmnumbelempop0"] = 0
    dictargsvari["hostmult"]["maxmnumbelempop1"] = 0
    dictargsvari["hostmult"]["typeelem"] = ["lens", "lghtgausbgrd"]
    dictargsvari["sourmask"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["sourmask"]["maxmnumbelempop0"] = 0
    dictargsvari["sourmask"]["maxmnumbelempop1"] = 0
    dictargsvari["sourmask"]["typeelem"] = ["lens", "lghtgausbgrd"]
    dictargsvari["sourmask"]["listmask"] = listmask
    dictargsvari["nomi"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["nomi"]["maxmgangdata"] = maxmgangdata
    dictargsvari["nomi"]["maxmnumbelempop0"] = 0
    dictargsvari["mask"]["listmask"] = listmask
    dictargsvari["mask"]["strgexprsbrt"] = strgexprsbrt
    dictargsvari["mask"]["maxmgangdata"] = maxmgangdata
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def test_lensmockpsfn(strgcnfgextnexec=None):

    '''
    Quantify strong-lens substructure sensitivity to PSF uncertainty and misspecification.

    The PSF width is either fitted or fixed, using true or incorrect values, so biases from an
    inaccurate instrumental response can be isolated.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["truenumbelempop0"] = 25
    dictargs["typeelem"] = ["lens"]
    dictargs["truesigcen00evt0"] = 4e-07
    anglfact = 3600.0 * 180.0 / nump.pi
    listnamecnfgextn = ["psfnfrwr", "psfnfxwr", "psfnfrtr", "psfnfxtr"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["psfnfrwr"]["truesigcen00evt0"] = 1e-07
    dictargsvari["psfnfrwr"]["fittsigcen00evt0"] = 2e-07
    dictargsvari["psfnfxwr"]["proppsfp"] = False
    dictargsvari["psfnfxwr"]["initsigcen00evt0"] = 2e-07
    dictargsvari["psfnfrtr"]["truesigcen00evt0"] = 4e-07
    dictargsvari["psfnfxtr"]["proppsfp"] = False
    dictargsvari["psfnfxtr"]["initsigcen00evt0"] = 4e-07
    listtickxaxi = ["Free, Wrong PSF", "Fixed, Wrong PSF", "Free, True PSF", "Fixed, True PSF"]
    dictglob = pcat.main.initarry(
        dictargsvari,
        dictargs,
        listnamecnfgextn,
        namexaxivari="psfnvari",
        tickxaxivari=listtickxaxi,
        strgcnfgextnexec=strgcnfgextnexec,
    )
    dictoutp = dictglob

    return dictoutp


def exec_lensmockpapr(strgcnfgextnexec=None):

    '''
    Reproduce the strong-lens mock configurations used to compare fitted deflection thresholds.

    Three minimum fitted deflections are tested against a no-subhalo control for a 25-perturber
    mock, supporting paper-level comparisons of threshold-dependent recovery.
    '''

    dictoutp = None
    dictargs = {}
    dictargs["typeexpr"] = "HST_WFC3_IR"
    dictargs["typeelem"] = ["lens"]
    dictargs["seedtype"] = 4
    dictargs["priofactdoff"] = 0.0
    dictargs["limtydathistfeat"] = [0.5, 10.0]
    dictargs["truemaxmnumbelempop0"] = 25
    dictargs["truenumbelempop0"] = 25
    anglfact = 3600.0 * 180.0 / nump.pi
    numbelem = int(25.0 * 10.0**0.9)
    listnamecnfgextn = ["fittlhig", "fitthigh", "fittvhig", "truenone"]
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    dictargsvari["fittlhig"]["fittminmdefs"] = 0.005 / anglfact
    dictargsvari["fitthigh"]["fittminmdefs"] = 0.01 / anglfact
    dictargsvari["fittvhig"]["fittminmdefs"] = 0.02 / anglfact
    dictargsvari["truenone"]["fittminmdefs"] = 0.01 / anglfact
    dictargsvari["truenone"]["truenumbelempop0"] = 0
    dictglob = pcat.main.initarry(
        dictargsvari, dictargs, listnamecnfgextn, strgcnfgextnexec=strgcnfgextnexec
    )
    dictoutp = dictglob

    return dictoutp


def retr_namecnfg():

    '''
    Return the names of all independently runnable configurations in this module.

    Configuration functions are identified by their exec_, eval_, or test_ prefixes. The dispatcher
    and the all-configuration runner are excluded to prevent recursion.
    '''

    listnamecnfg = []
    for namefunc, funcvalu in globals().items():
        boolpref = namefunc.startswith(("exec_", "eval_", "test_"))
        boolexcl = namefunc in {"exec_main", "exec_allcnfg"}
        if boolpref and not boolexcl and callable(funcvalu):
            listnamecnfg.append(namefunc)
    listnamecnfg = sorted(listnamecnfg)

    return listnamecnfg


def exec_allcnfg(strgcnfgincl=None):

    '''
    Execute every configuration in a fresh Python process and summarize all failures.

    Running configurations in separate processes prevents global state left by one configuration from
    contaminating later checks. An optional substring limits execution to matching configuration names.
    The full suite continues after individual failures and raises only after printing the final summary.
    '''

    dictoutp = None
    listnamecnfg = retr_namecnfg()
    if strgcnfgincl is not None:
        listnamecnfg = [
            namecnfg for namecnfg in listnamecnfg if strgcnfgincl in namecnfg
        ]
    if len(listnamecnfg) == 0:
        raise RuntimeError("No configurations matched the requested selection.")

    dictstatcnfg = {}
    for indxcnfg, namecnfg in enumerate(listnamecnfg):
        print("")
        print("[%d/%d] Executing %s..." % (indxcnfg + 1, len(listnamecnfg), namecnfg))
        listcomm = [syst.executable, oper.path.abspath(__file__), namecnfg]
        procvalu = subp.run(listcomm, check=False)
        dictstatcnfg[namecnfg] = procvalu.returncode
        if procvalu.returncode == 0:
            print("PASS: %s" % namecnfg)
        else:
            print("FAIL: %s (exit status %d)" % (namecnfg, procvalu.returncode))

    listnamepass = [
        namecnfg for namecnfg, codestat in dictstatcnfg.items() if codestat == 0
    ]
    listnamefail = [
        namecnfg for namecnfg, codestat in dictstatcnfg.items() if codestat != 0
    ]
    print("")
    print("Configuration execution summary")
    print("Passed: %d" % len(listnamepass))
    print("Failed: %d" % len(listnamefail))
    if len(listnamefail) > 0:
        print("Failed configurations:")
        for namecnfg in listnamefail:
            print("  %s" % namecnfg)

    dictoutp = {
        "listnamepass": listnamepass,
        "listnamefail": listnamefail,
        "dictstatcnfg": dictstatcnfg,
    }
    if len(listnamefail) > 0:
        raise RuntimeError(
            "%d of %d configurations failed." % (len(listnamefail), len(listnamecnfg))
        )

    return dictoutp




def retr_docstrl_frst(funcobja):
    '''
    
    Extract first line of docstring from function object.
    
    '''
    
    strdocs = funcobja.__doc__
    if strdocs:
        strln = strdocs.strip().split('\n')[0]
        return strln if strln else None
    return None


def retr_listnamfunc_test():
    '''
    
    Retrieve list of all available test functions in module.
    
    '''
    
    listnam = [name for name in globals() 
               if (name.startswith('exec_') or name.startswith('eval_'))
               and callable(globals()[name])]
    listnam.sort()
    return listnam


def exec_main():

    '''
    
    Dispatch a configuration function named on command line and execute it.
    
    First command-line argument identifies callable in module; remaining arguments
    are forwarded to that function.
    
    '''

    dictoutp = None
    
    # Check if function name provided as first argument
    if len(syst.argv) < 2:
        print('Collecting available test functions...')
        listnam = retr_listnamfunc_test()
        
        # Build error message with function list
        errmess = "A function name must be provided.\n\nAvailable tests:\n"
        for nam in listnam:
            funcobja = globals()[nam]
            strdocs = retr_docstrl_frst(funcobja)
            if strdocs:
                errmess += "  %-40s  %s\n" % (nam, strdocs)
            else:
                errmess += "  " + nam + "\n"
        raise RuntimeError(errmess)
    
    # Get function name from command line
    namfunc = syst.argv[1]
    funcmain = globals().get(namfunc)
    
    # Verify function exists and is callable
    if funcmain is None or not callable(funcmain):
        listnam = retr_listnamfunc_test()
        
        # Build error message with function list
        errmess = "Unknown function: %s\n\nAvailable tests:\n" % namfunc
        for nam in listnam:
            funcobja = globals()[nam]
            strdocs = retr_docstrl_frst(funcobja)
            if strdocs:
                errmess += "  %-40s  %s\n" % (nam, strdocs)
            else:
                errmess += "  " + nam + "\n"
        raise RuntimeError(errmess)
    
    # Execute the selected function with remaining arguments
    print('Executing %s...' % namfunc)
    dictoutp = funcmain(*syst.argv[2:])
    print('Completed %s successfully.' % namfunc)

    return dictoutp


if __name__ == "__main__":
    try:
        dictoutp = exec_main()
        syst.exit(0)
    except Exception as excp:
        trbk.print_exc()
        syst.exit(1)
