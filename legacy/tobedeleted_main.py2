# plotting
import matplotlib as mpl

import matplotlib.pyplot as plt
#import seaborn as sns

# numpy
import random as randommod
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

# scipy
import scipy as sp
import scipy.interpolate
from scipy.special import erfinv, erf
from scipy.stats import poisson as pss
import scipy.ndimage

import scipy.fftpack

import scipy.ndimage.filters
import scipy.sparse

# jit
from numba import jit
import threading

#import pyximport; pyximport.install()
import ctypes

import subprocess as subp, psutil

import astropy
import astropy as ap
from astropy.convolution import convolve_fft, AiryDisk2DKernel

# multiprocessing
import multiprocessing as mp

from copy import deepcopy

# FITS files
import h5py

# utilities
import os, time, sys, getpass, glob, fnmatch, inspect, traceback, shelve
import pickle as cPickle
import functools

# tdpy
import tdpy.util
from tdpy.util import summgene
import tdpy.mcmc

# HealPix
#import healpy as hp
#from healpy import ang2pix

# ignore warnings if not in diagnostic mode
import warnings
    
#from skimage.feature import blob_doh

# defualt options
#seterr(divide='raise', over='raise', invalid='raise')
#seterr(all='raise')
#seterr(under='ignore')
warnings.simplefilter('ignore')
np.set_printoptions(linewidth=180)
#sns.set(context='poster', style='ticks', color_codes=True)

# secondaries
## Symbolic Jacobian calculation
#import sympy

## Probabilistic Graphical Model generation
import networkx as nx

from .util import *


def narr_open(path, mode='r'):

    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return open(pathnorm, mode)


def narr_h5(path, mode):

    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return h5py.File(pathnorm, mode)


def narr_fits_open(path):

    pathnorm = os.path.normpath(path)
    print 'Reading %s...' % pathnorm
    return ap.io.fits.open(pathnorm)


def init( \
         # user interaction
         ## type of verbosity
         typeverb=1, \

         ## path in which PCAT data lives
         pathbase=os.environ["PCAT_DATA_PATH"], \
        
         # miscelleneaous
         ## type of PDF to sample from
         strgpdfn='post', \

         # type of data
         typeexpr=None, \
         
         # diagnostics
         ## Boolean to enter the diagnostic mode
         booldiagmode=True, \
         ## squeeze exposure to check the low sample limit         
         boolsqzeexpo=False, \
         ### explode exposure to check the large sample limit         
         boolexplexpo=False, \
         ## squeeze proposal scale to check the acceptance ratio
         boolsqzeprop=False, \
         ## explode proposal scale to check the acceptance ratio
         boolexplprop=False, \
         ## Boolean to thin down the data
         boolthindata=False, \
         ## factor by which to thin down the data
         factthin=None, \
    
         # reference catalog
         ## Boolean to use the reference catalogs to associate
         boolasscrefr=None, \
        
         # sampling
         ## Boolean flag to make burn-in tempered
         boolburntmpr=False, \
         ## number of sweeps
         numbswep=100000, \
         ## number of samples
         numbsamp=None, \
         ## number of initial sweeps to be burned
         numbburn=None, \
        
         # output
         ## Boolean to make condensed catalog
         boolcondcatl=True, \
        
         refrlegd=None, \
         refrlegdpopl=None, \
         fittlegdpopl=None, \
    
         # numpy RNG seed
         seedtype=0, \
         seedchan=True, \
         seedelemtype=None, \
         
         indxevttincl=None, \
         indxenerincl=None, \
        
         listmask=None, \
        
         # number of samples for Bootstrap
         numbsampboot=None, \

         listnamefeatsele=None, \
         
         # type of mask for the exposure map
         typemaskexpo='ignr', \
         
         # maximum spatial distance out to which element kernel will be evaluated
         maxmangleval=None, \
         
         # initial state
         initpsfprefr=False, \
         initpsfp=None, \
        
         # evaluate the likelihood inside circles around elements
         elemspatevaltype=None, \
        
         namestattrue=None, \
        
         # plotting
         ## Boolean flag to make the frame plots short
         boolshrtfram=True, \
        
         boolrefeforc=False, \
         indxrefrforc=None, \

         suprelem=True, \
         plotelemcorr=True, \
         
         ## Boolean flag to vary the PSF
         fittboolmodipsfn=False, \

         # name of the configuration
         strgcnfg=None, \
        
         # model
         ## number of spatial dimensions
         numbspatdims=2, \
         # hyperparameters
         fittampldisttype=None, \
         # metamodel settings
         
         ## PSF evaluation type
         ## kernel evaluation type
         kernevaltype='ulip', \

         # photometric model
        
         ## base parameters
         ### Sersic type
         typesers='vauc', \
         ## transdimensional parameters (elements)
         ### vary projected scale radius
         variasca=True, \
         ### vary projected cutoff radius
         variacut=True, \

         # prior
         penalpridiff=False, \
         priotype='logt', \
         priofactdoff=None, \

         # initialization
         ## initialization type
         inittype=None, \
        
         loadvaripara=False, \
         
         # save the state of the MCMC
         savestat=False, \
         namesavestat=None, \
         # recover the state from a previous run
         namerecostat=None, \
         forcsavestat=False, \

         # proposals
         ## Boolean flag to turn on proposals on element parameters
         boolpropcomp=True, \
         boolpropcova=True, \
         propwithsing=True, \
         # type of covariance estimation
         typeopti='hess', \
         
         # modes of operation
         ## interactive
         intrevalcntpmodl=False, \
         intrevalcntpresi=False, \
         ## only generate and plot mock data
         boolmockonly=False, \
         ## perform an additional run sampling from the prior
         checprio=False, \

         strgexprsbrt=None, \
         anglassc=None, \
         nameexpr=None, \
         
         lgalprio=None, \
         bgalprio=None, \
         minmcntpdata=None, \
         strgexpo=None, \
         
         # number of processors
         numbproc=None, \
         
         # likelihood function
         liketype='pois', \
         # user-defined likelihood function
         retr_llik=None, \
         
         anlytype=None, \
         
         lgalcntr=0., \
         bgalcntr=0., \
         
         maxmangl=None, \
         
         # spatial grid
         ## type of spatial pixelization
         typepixl=None, \
         ## Boolean flag to force Cartesian spatial grid
         boolforccart=False, \
         # number of pixels on a side in the Cartesian grid
         numbsidecart=None, \
         # Nside in Healpix
         numbsideheal=256, \
         
         allwfixdtrue=True, \
         asscmetrtype='dist', \

         # plotting
         numbswepplot=None, \
         # Boolean flagt to make the frame plots only for the central energy and PSF bin
         boolmakeframcent=True, \
         makeplot=True, \
         makeplotinit=True, \
         makeplotfram=True, \
         makeplotfinlprio=True, \
         makeplotfinlpost=True, \
         
         makeplotintr=False, \
         scalmaps='asnh', \
         makeanim=True, \
         strgenerfull=None, \
         strgexprname=None, \
         strganglunit=None, \
         strganglunittext=None, \
         anglfact=None, \
         fluxfactplot=None, \
            
         limtydathistfeat=None, \
            
         typedata=None, \

         # model
         ## PSF
         specfraceval=None, \
         numbangl=1000, \
         binsangltype='logt', \
         numbsidepntsprob=100, \
    
         listprefsbrtsbrt=None, \
         listprefsbrtener=None, \
         listprefsbrtlabl=None, \

         lablgangunit=None, \
         labllgal=None, \
         lablbgal=None, \
         lablfluxunit=None, \
         lablflux=None, \
         strgenerunit=None, \
         indxenerfull=None, \
         indxevttfull=None, \
         binsenerfull=None, \
         asymfluxprop=False, \
         psfninfoprio=True, \
         ## spectral

         # lensing
         fittrelnpowr=0., \

         # temp
         margfactmodl=1., \
         maxmgangdata=None, \
        
         # proposals
         stdvprophypr=0.01, \
         stdvproppsfp=0.1, \
         stdvpropbacp=0.01, \
         stdvproplenp=1e-4, \
         stdvlgal=0.001, \
         stdvbgal=0.001, \
         stdvflux=0.001, \
         stdvspep=0.001, \
         stdvspmrsind=0.2, \
         varistdvlbhl=True, \
        
         rtagmock=None, \
        
         ## transdimensional proposal probabilities
         probtran=None, \
         probspmr=None, \
         # when proposing from the covariance, fracproprand should be very small!
         fracproprand=0., \
         # standard deviation of the Gaussian from which the angular splitting will be drawn for splits and merges
         radispmr=None, \
        
         defa=False, \
         **args \
        ):

    # preliminary setup
    # construct the global object 
    gdat = tdpy.util.gdatstrt()
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)
    
    # copy all provided inputs to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # PCAT folders
    if gdat.pathbase[-1] != '/':
        gdat.pathbase += '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathdataopti = gdat.pathdata + 'opti/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathoutp = gdat.pathdata + 'outp/'

    # run tag
    gdat.strgswep = '%d' % (gdat.numbswep)
    
    ## time stamp
    gdat.strgtimestmp = tdpy.util.retr_strgtimestmp()
    
    ## name of the configuration function
    if gdat.strgcnfg is None:
        gdat.strgcnfg = inspect.stack()[1][3]
   
    gdat.strgvers = 'v0.3'
    if gdat.typeverb > 0:
        print('PCAT %s started at %s.' % (gdat.strgvers, gdat.strgtimestmp))
        print('Configuration %s' % gdat.strgcnfg)
    
    # string describing the number of sweeps
    gdat.strgnumbswep = '%d' % gdat.numbswep
    
    # output paths
    gdat.rtag = retr_rtag(gdat.strgcnfg, gdat.strgnumbswep)
    gdat.pathoutprtag = retr_pathoutprtag(gdat.rtag)

    # start the timer
    gdat.timerealtotl = time.time()
    gdat.timeproctotl = time.clock()
   
    booltemp = chec_statfile(gdat.rtag, 'gdatmodi')
    if booltemp:
        print('gdatmodi already exists. Skipping...')
    else:
    
        # create output folder for the run
        os.system('mkdir -p %s' % gdat.pathoutprtag)

        # write the list of arguments to file
        fram = inspect.currentframe()
        listargs, temp, temp, listargsvals = inspect.getargvalues(fram)
        fileargs = narr_open(gdat.pathoutprtag + 'cmndargs.txt', 'w')
        fileargs.write('PCAT call arguments\n')
        for args in listargs:
            fileargs.write('%s = %s\n' % (args, listargsvals[args]))
        fileargs.close()
        
        # write the list of arguments to file
        fileargs = narr_open(gdat.pathoutprtag + 'args.txt', 'w')
        fileargs.write('PCAT call arguments\n')
        for args in listargs:
            fileargs.write('%20s %s\n' % (args, listargsvals[args]))
        fileargs.close()
        
        # defaults
        if gdat.typedata is None:
            if gdat.strgexprsbrt is None:
                gdat.typedata = 'mock'
            else:
                gdat.typedata = 'inpt'
        
        # list of models
        gdat.liststrgmodl = ['fitt']
        gdat.listlegdmodl = ['Fitting']
        if gdat.typedata == 'mock':
            gdat.liststrgmodl += ['true']
            gdat.listlegdmodl += ['True']
    
        ## number of processes
        gdat.strgproc = os.uname()[1]
        if gdat.numbproc is None:
            if gdat.strgproc == 'fink1.rc.fas.harvard.edu' or gdat.strgproc == 'fink2.rc.fas.harvard.edu' or gdat.strgproc == 'wise':
                gdat.numbproc = 1
            else:
                gdat.numbproc = 1
    
        if gdat.typedata == 'inpt' and gdat.rtagmock is not None:
            print('Will use %s to account for selection effects.' % gdat.rtagmock)
            gdat.pathoutprtagmock = retr_pathoutprtag(gdat.rtagmock)

        ## number of burned sweeps
        if gdat.numbburn is None:
            print('gdat.numbswep')
            print(gdat.numbswep)
            gdat.numbburn = int(gdat.numbswep / 10)
            print('gdat.numbburn')
            print(gdat.numbburn)
    
        if (gdat.boolsqzeprop or gdat.boolexplprop) and gdat.typeopti == 'hess':
            raise Exception('')

        ## factor by which to thin the sweeps to get samples
        
        if gdat.factthin is not None and gdat.numbsamp is not None:
            raise Exception('Both factthin and numbsamp cannot be provided at the same time.')
        elif gdat.factthin is None and gdat.numbsamp is None:
            gdat.factthin = int(np.ceil(1e-3 * (gdat.numbswep - gdat.numbburn)))
            gdat.numbsamp = int((gdat.numbswep - gdat.numbburn) / gdat.factthin)
        elif gdat.numbsamp is not None:
            gdat.factthin = int((gdat.numbswep - gdat.numbburn) / gdat.numbsamp)
        elif gdat.factthin is not None:
            gdat.numbsamp = int((gdat.numbswep - gdat.numbburn) / gdat.factthin)
        if not isinstance(gdat.numbsamp, int) or not isinstance(gdat.factthin, int) or \
                                        not isinstance(gdat.numbburn, int) or not isinstance(gdat.numbswep, int):
            print('gdat.numbsamp')
            print(gdat.numbsamp)
            print('gdat.factthin')
            print(gdat.factthin)
            print('gdat.numbburn')
            print(gdat.numbburn)
            print('gdat.numbswep')
            print(gdat.numbswep)
            raise Exception('Number of samples is not an integer.')

        # samples to be saved
        gdat.indxsamp = np.arange(gdat.numbsamp)
        
        # samples to be saved from all chains
        gdat.numbsamptotl = gdat.numbsamp * gdat.numbproc
        gdat.indxsamptotl = np.arange(gdat.numbsamptotl)
        gdat.numbsweptotl = gdat.numbswep * gdat.numbproc
        
        if gdat.typeverb > 0:
            print('%d samples will be taken, discarding the first %d. The chain will be thinned by a factor of %d.' % \
                                                                                                    (gdat.numbswep, gdat.numbburn, gdat.factthin))
            print('The resulting chain will contain %d samples per chain and %d samples in total.' % (gdat.numbsamp, gdat.numbsamptotl))
        
        if gdat.typeexpr is None:
            gdat.typeexpr = 'gene'

        if gdat.anlytype is None:
            if gdat.typeexpr == 'chan':
                gdat.anlytype = 'home'
            elif gdat.typeexpr == 'ferm':
                gdat.anlytype = 'rec8pnts'
            else:
                gdat.anlytype = 'nomi'
        
        if gdat.priofactdoff is None:
            gdat.priofactdoff = 1.
        
        # feature correlated with the significance of elements
        gdat.nameparaelemsign = 'deltllik'
        if gdat.typedata == 'mock':
            gdat.nameparaelemsignrefr = 'deltllik'
        
        # experiment defaults
        if gdat.typeexpr == 'ferm':
            gdat.lablenerunit = 'GeV'
        if gdat.typeexpr == 'chan':
            gdat.lablenerunit = 'keV'
        if gdat.typeexpr == 'sdyn':
            gdat.lablenerunit = ''
        if gdat.typeexpr == 'fire':
            gdat.lablenerunit = '$\mu$m^{-1}'
        
        if gdat.binsenerfull is None:
            if gdat.typeexpr == 'ferm':
                if gdat.anlytype[4:8] == 'pnts':
                    gdat.binsenerfull = np.logspace(np.log10(0.3), np.log10(10.), 4)
                if gdat.anlytype[4:8] == 'back':
                    gdat.binsenerfull = np.logspace(np.log10(0.3), np.log10(300.), 31)
            if gdat.typeexpr == 'chan':
                if gdat.anlytype.startswith('home'):
                    gdat.binsenerfull = np.array([0.5, 0.91, 1.66, 3.02, 5.49, 10.])
                if gdat.anlytype.startswith('extr'):
                    gdat.binsenerfull = np.array([0.5, 2., 8.])
                if gdat.anlytype.startswith('spec'):
                    gdat.binsenerfull = np.logspace(np.log10(0.5), np.log10(10.), 21)
            if gdat.typeexpr == 'fire':
                gdat.binsenerfull = np.logspace(np.log10(1. / 2.5e-6), np.log10(1. / 0.8e-6), 31)
            if gdat.typeexpr == 'hubb':
                # temp
                #gdat.binsenerfull = np.array([500., 750, 1000.])
                gdat.binsenerfull = np.array([750, 1000.])
        
        # energy band string
        if gdat.strgenerfull is None:
            if gdat.typeexpr == 'tess':
                gdat.strgenerfull = ['T']
            if gdat.typeexpr == 'sdss':
                gdat.strgenerfull = ['z-band', 'i-band', 'r-band', 'g-band', 'u-band']
            if gdat.typeexpr == 'hubb':
                #gdat.strgenerfull = ['F606W', 'F814W']
                gdat.strgenerfull = ['F814W']
            if gdat.typeexpr == 'ferm' or gdat.typeexpr == 'chan' or gdat.typeexpr == 'fire': 
                gdat.strgenerfull = []
                for i in range(len(gdat.binsenerfull) - 1):
                    gdat.strgenerfull.append('%.3g %s - %.3g %s' % (gdat.binsenerfull[i], gdat.lablenerunit, gdat.binsenerfull[i+1], gdat.lablenerunit))
            if gdat.typeexpr == 'sdyn':
                gdat.strgenerfull = ['']
        
        ## PSF class
        if gdat.indxevttfull is None:
            if gdat.typeexpr == 'ferm':
                gdat.indxevttfull = np.arange(2)
            else:
                gdat.indxevttfull = np.arange(1)
        
        if gdat.indxevttincl is None:
            if gdat.typeexpr == 'ferm':
                gdat.indxevttincl = np.array([0, 1])
            else:
                gdat.indxevttincl = np.arange(1)
        
        if gdat.indxevttincl is not None:
            gdat.evttbins = True
        else:
            gdat.evttbins = False
        if gdat.evttbins:
            gdat.numbevtt = gdat.indxevttincl.size
            gdat.numbevttfull = gdat.indxevttfull.size
        else:
            gdat.numbevtt = 1
            gdat.numbevttfull = 1
            gdat.indxevttincl = np.array([0])
        gdat.indxevtt = np.arange(gdat.numbevtt)
        
        # Boolean flag to indicate that the data are binned in energy
        if gdat.binsenerfull is None:
            gdat.boolbinsener = False
        else:
            gdat.boolbinsener = True
        
        if gdat.boolbinsener:
            gdat.numbenerfull = len(gdat.strgenerfull)
        else:
            gdat.numbenerfull = 1
        gdat.indxenerfull = np.arange(gdat.numbenerfull)

        gdat.pathinpt = gdat.pathdata + 'inpt/'
        
        if gdat.typepixl is None:
            if gdat.typeexpr == 'ferm':
                gdat.typepixl = 'heal'
            else:
                gdat.typepixl = 'cart'
        
        if gdat.boolbinsener:
            gdat.meanenerfull = np.sqrt(gdat.binsenerfull[1:] * gdat.binsenerfull[:-1])
        
        # default values for model types
        print('Starting to determine the default values for model types using setp_varbvalu()...')
        if gdat.typeexpr == 'hubb':
            typeemishost = 'sers'
        else:
            typeemishost = 'none'
        setp_varbvalu(gdat, 'typeemishost', typeemishost)

        ### background type
        #### template
        if gdat.typeexpr == 'ferm':
            if gdat.anlytype == 'bfun':
                gdat.ordrexpa = 10
                gdat.numbexpasing = gdat.ordrexpa**2
                gdat.numbexpa = gdat.numbexpasing * 4
                gdat.indxexpa = np.arange(gdat.numbexpa)
                backtype = ['bfun%04d' % k for k in gdat.indxexpa]
            else:
                backtype = [1., 'sbrtfdfmsmthrec8pntsnorm.fits']
        if gdat.typeexpr == 'chan':
            if True:
                backtype = [1.]
            else:
                # particle background
                if gdat.anlytype.startswith('spec'):
                    # temp -- this is fake!
                    sbrtparttemp = np.array([70.04, 70.04, 12.12, 15.98, 10.79, 73.59, 73.59])
                    binsenerpart = np.logspace(np.log10(0.5), np.log10(10.), 6)
                    meanenerpart = np.sqrt(binsenerpart[:-1] * binsenerpart[1:])
                    meanenerparttemp = np.concatenate((np.array([0.5]), meanenerpart, np.array([10.])))
                    backtypetemp = interp(gdat.meanenerfull, meanenerparttemp, sbrtparttemp)
                if gdat.anlytype.startswith('home') :
                    backtypetemp = 1.
                    #backtypetemp = np.array([70.04, 12.12, 15.98, 10.79, 73.59]) / 70.04
                if gdat.anlytype.startswith('extr'):
                    #backtypetemp = 'sbrtchanback' + gdat.anlytype + '.fits'
                    backtypetemp = 1.
                
                if gdat.anlytype.startswith('spec'):
                    backtype = [[1e2, 2.], backtypetemp]
                else:
                    backtype = [1., backtypetemp]
        
        if gdat.typeexpr == 'hubb':
            backtype = [1.]
        if gdat.typeexpr == 'tess':
            backtype = [1.]
        if gdat.typeexpr == 'sdyn':
            backtype = [1.]
        if gdat.typeexpr == 'fire':
            backtype = [1.]
        if gdat.typeexpr == 'gene':
            backtype = [1.]
        setp_varbvalu(gdat, 'backtype', backtype)
        
        numbsersfgrd = 1
        setp_varbvalu(gdat, 'numbsersfgrd', numbsersfgrd)
        
        if gdat.typeexpr == 'gene':
            elemtype = ['clus']
        if gdat.typeexpr == 'ferm':
            elemtype = ['lghtpnts']
        if gdat.typeexpr == 'tess':
            elemtype = ['lghtpnts']
        if gdat.typeexpr == 'chan':
            elemtype = ['lghtpnts']
        if gdat.typeexpr == 'hubb':
            elemtype = ['lghtpnts', 'lens', 'lghtgausbgrd']
        if gdat.typeexpr == 'sdyn':
            elemtype = ['clus']
        if gdat.typeexpr == 'fire':
            elemtype = ['lghtlineabso']
        setp_varbvalu(gdat, 'elemtype', elemtype)
        
        setp_prem(gdat)
        
        gdat.commelemtype = []
        for strgmodl in gdat.liststrgmodl:
            elemtype = getattr(gdat, strgmodl + 'elemtype')
            for elemtypetemp in elemtype:
                if not elemtypetemp in gdat.commelemtype:
                    gdat.commelemtype.append(elemtypetemp)
                
        if gdat.typeexpr == 'hubb':
            gdat.hubbexpofact = 1.63050e-19
        
        if gdat.strgexpo is None:
            if gdat.typeexpr == 'ferm':
                gdat.strgexpo = 'expofermrec8pntsigal0256.fits'
            elif gdat.typeexpr == 'hubb':
                gdat.strgexpo = 1000. / gdat.hubbexpofact
            else:
                gdat.strgexpo = 1.
        
        ## generative model
        # the factor to convert radians (i.e., internal angular unit of PCAT) to the angular unit that will be used in the output (i.e., plots and tables)
        if gdat.anglfact is None:
            if gdat.typeexpr == 'ferm':
                gdat.anglfact = 180. / np.pi
            if gdat.typeexpr == 'tess':
                gdat.anglfact = 60 * 180. / np.pi
            if gdat.typeexpr == 'sdss' or gdat.typeexpr == 'chan' or gdat.typeexpr == 'hubb':
                gdat.anglfact = 3600 * 180. / np.pi
            if gdat.typeexpr == 'sche' or gdat.typeexpr == 'sdyn' or gdat.typeexpr == 'gene':
                gdat.anglfact = 1.
        
        if gdat.numbsidecart is not None and gdat.typepixl == 'cart' and not gdat.boolforccart and not isinstance(strgexpo, np.float):
            raise Exception('numbsidecart argument should not be provided when strgexpo is a file name and pixelization is Cartesian.')
                    
        if gdat.typepixl == 'heal' or gdat.typepixl == 'cart' and gdat.boolforccart:
            if gdat.numbsidecart is None:
                gdat.numbsidecart = 100
        
        # exposure
        # temp
        gdat.correxpo = True
        if gdat.correxpo:
            if isinstance(gdat.strgexpo, float):
                if gdat.typedata == 'mock':
                    if gdat.numbsidecart is None:
                        gdat.numbsidecart = 100
                if gdat.typedata == 'mock':
                    if gdat.typepixl == 'heal':
                        gdat.expo = gdat.strgexpo * np.ones((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbevttfull))
                    if gdat.typepixl == 'cart':
                        gdat.expo = gdat.strgexpo * np.ones((gdat.numbenerfull, gdat.numbsidecart**2, gdat.numbevttfull))
                if gdat.typedata == 'inpt':
                    gdat.expo = gdat.strgexpo * np.ones((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbevttfull))
            else: 
                if isinstance(gdat.strgexpo, list):
                
                    path = gdat.pathinpt + gdat.strgexpo
                    if gdat.typeverb > 0:
                        print('Reading %s...' % path)
                    gdat.expo = astropy.io.fits.getdata(path)
                else:
                    path = gdat.pathinpt + gdat.strgexpo
                    if gdat.typeverb > 0:
                        print('Reading %s...' % path)
                    gdat.expo = astropy.io.fits.getdata(path)
                
                    if gdat.typepixl == 'cart':
                        gdat.expo = gdat.expo.reshape((gdat.expo.shape[0], -1, gdat.expo.shape[-1]))
        
                if gdat.numbsidecart is None:
                    # temp -- gdat.numbsidecart takes the value of the region 0
                    if np.sqrt(gdat.expo.shape[1]) % 1. != 0.:
                        raise Exception('')
                    gdat.numbsidecart = int(np.sqrt(gdat.expo.shape[1]))
        
        if gdat.typedata == 'mock':
            if gdat.typepixl == 'cart':
                gdat.numbpixlfull = gdat.numbsidecart**2
            if gdat.typepixl == 'heal':
                gdat.numbpixlfull = 12 * gdat.numbsideheal**2
        
        if gdat.typepixl == 'cart' and isinstance(gdat.strgexpo, float) and gdat.typedata == 'inpt':
            if np.sqrt(gdat.sbrtdata.shape[1]) % 1. != 0.:
                raise Exception('')
            gdat.numbsidecart = int(np.sqrt(gdat.sbrtdata.shape[1]))
        
        gdat.numbsidecarthalf = int(gdat.numbsidecart / 2)

        if gdat.typepixl == 'cart':
            gdat.numbpixlcart = gdat.numbsidecart**2
        
        if np.amin(gdat.expo) == np.amax(gdat.expo) and not isinstance(gdat.strgexpo, float):
            raise Exception('Bad input exposure map.')
        
        ### spatial extent of the data
        if gdat.maxmgangdata is None:
            if gdat.typeexpr == 'chan':
                gdat.maxmgangdata = 0.492 / gdat.anglfact * gdat.numbsidecarthalf
            if gdat.typeexpr == 'ferm':
                gdat.maxmgangdata = 15. / gdat.anglfact
            if gdat.typeexpr == 'tess':
                gdat.maxmgangdata = 20. / gdat.anglfact
            if gdat.typeexpr == 'sdyn':
                gdat.maxmgangdata = 1. / gdat.anglfact
            if gdat.typeexpr == 'hubb':
                gdat.maxmgangdata = 2. / gdat.anglfact
            if gdat.typeexpr == 'gene':
                gdat.maxmgangdata = 1. / gdat.anglfact
        
        print('gdat.numbsidecart')
        print(gdat.numbsidecart)
        print('gdat.maxmgangdata')
        print(gdat.maxmgangdata)
        
        # pixelization
        if gdat.numbpixlfull > 1:
            if gdat.typepixl == 'cart':
                gdat.apix = (2. * gdat.maxmgangdata / gdat.numbsidecart)**2
            if gdat.typepixl == 'heal':
                temp, temp, temp, gdat.apix = tdpy.util.retr_healgrid(gdat.numbsideheal)
            gdat.sizepixl = np.sqrt(gdat.apix)
        
        # factor by which to multiply the y axis limits of the surface brightness plot
        if gdat.numbpixlfull == 1:
            gdat.factylimsbrt = [1e-4, 1e7]
        else:
            gdat.factylimsbrt = [1e-4, 1e3]

        # axes
        gdat.minmlgaldata = -gdat.maxmgangdata
        gdat.maxmlgaldata = gdat.maxmgangdata
        gdat.minmbgaldata = -gdat.maxmgangdata
        gdat.maxmbgaldata = gdat.maxmgangdata
        
        if gdat.typepixl == 'cart' and gdat.boolforccart:
            if gdat.typedata == 'inpt':
                sbrtdatatemp = np.empty((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbevttfull))
                for i in gdat.indxenerfull:
                    for m in gdat.indxevttfull:
                        sbrtdatatemp[i, :, m] = tdpy.util.retr_cart(gdat.sbrtdata[i, :, m], \
                                                        numbsidelgal=gdat.numbsidecart, numbsidebgal=gdat.numbsidecart, \
                                                        minmlgal=gdat.anglfact*gdat.minmlgaldata, maxmlgal=gdat.anglfact*gdat.maxmlgaldata, \
                                                        minmbgal=gdat.anglfact*gdat.minmbgaldata, maxmbgal=gdat.anglfact*gdat.maxmbgaldata).flatten()
                gdat.sbrtdata = sbrtdatatemp

            if gdat.correxpo:
                expotemp = np.empty((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbevttfull))
                for i in gdat.indxenerfull:
                    for m in gdat.indxevttfull:
                        expotemp[i, :, m] = tdpy.util.retr_cart(gdat.expo[i, :, m], \
                                                        numbsidelgal=gdat.numbsidecart, numbsidebgal=gdat.numbsidecart, \
                                                        minmlgal=gdat.anglfact*gdat.minmlgaldata, maxmlgal=gdat.anglfact*gdat.maxmlgaldata, \
                                                        minmbgal=gdat.anglfact*gdat.minmbgaldata, maxmbgal=gdat.anglfact*gdat.maxmbgaldata).flatten()
                gdat.expo = expotemp
        
        # initialization type
        if gdat.inittype is None:
            gdat.inittype = 'rand'

        gdat.sdenunit = 'degr'

        gdat.factergskevv = 1.6e-9
        if gdat.typeexpr == 'ferm':
            gdat.listspecconvunit = [['en02', 'gevv']]
        if gdat.typeexpr == 'chan':
            gdat.listspecconvunit = [['en00', 'kevv'], ['en02', 'kevv'], ['en02', 'ergs'], ['en03', 'ergs', '0520', 0.5,  2.], \
                                                                                           ['en03', 'ergs', '0210',  2., 10.], \
                                                                                           ['en03', 'ergs', '0510', 0.5, 10.], \
                                                                                           ['en03', 'ergs', '0208',  2.,  8.], \
                                                                                           ['en03', 'ergs', '0508', 0.5,  8.], \
                                                                                           ['en03', 'ergs', '0207',  2.,  7.], \
                                                                                           ['en03', 'ergs', '0507', 0.5,  7.]]
        if gdat.typeexpr == 'hubb':
            gdat.listspecconvunit = [['en03', 'ergs']]
        if gdat.typeexpr == 'fire':
            gdat.listspecconvunit = [['en00', 'imum']]
        
        # temp
        #if gdat.typeexpr == 'chan' and (gdat.anlytype.startswith('home') or gdat.anlytype.startswith('extr')):
        #    gdat.truelegdpopl = ['AGN', 'Galaxy']

        ### background parameters
        if gdat.typeexpr == 'chan':
            if gdat.anlytype.startswith('extr'):
                meanbacpbac1 = 1.
            else:
                meanbacpbac1 = 70.04
            setp_varbvalu(gdat, 'scalbacp', 'gaus', back=1)
            stdvbacpbac1 = 1e-5 * meanbacpbac1
            setp_varblimt(gdat, 'bacp', [meanbacpbac1, stdvbacpbac1], back=1, typelimt='meanstdv')

        if gdat.typeexpr == 'ferm' or gdat.typeexpr == 'chan' or gdat.typeexpr == 'fire':
            gdat.enerdiff = True
        if gdat.typeexpr == 'hubb' or gdat.typeexpr == 'sdyn' or gdat.typeexpr == 'tess':
            gdat.enerdiff = False
        
        if gdat.indxenerincl is None:
            
            # default
            if gdat.binsenerfull is not None:
                gdat.indxenerincl = np.arange(gdat.binsenerfull.size - 1)
            
            if gdat.typeexpr == 'ferm':
                if gdat.anlytype[4:8] == 'pnts':
                    gdat.indxenerincl = np.arange(3)
                if gdat.anlytype[4:8] == 'back':
                    gdat.indxenerincl = np.arange(30)
            if gdat.typeexpr == 'chan':
                if gdat.anlytype.startswith('home'):
                    gdat.indxenerincl = np.arange(5)
                if gdat.anlytype.startswith('extr'):
                    gdat.indxenerincl = np.arange(2)
            if gdat.typeexpr == 'hubb':
                gdat.indxenerincl = np.array([0])
                #gdat.indxenerincl = np.array([1])
                #gdat.indxenerincl = np.array([0, 1])
            if gdat.typeexpr == 'sdyn':
                gdat.indxenerincl = np.array([0])
        
        if gdat.indxenerincl is None:
            gdat.numbener = 1
        else:
            gdat.numbener = gdat.indxenerincl.size
        gdat.indxener = np.arange(gdat.numbener, dtype=int)
        
        if gdat.indxenerincl is None:
            gdat.indxenerincl = gdat.indxener
        
        if gdat.boolbinsener:
            gdat.indxenerinclbins = np.empty(gdat.numbener+1, dtype=int)
            gdat.indxenerinclbins[0:-1] = gdat.indxenerincl
            gdat.indxenerinclbins[-1] = gdat.indxenerincl[-1] + 1
            gdat.indxenerpivt = 0
            gdat.numbenerplot = 100
            gdat.strgener = [gdat.strgenerfull[k] for k in gdat.indxenerincl]
            gdat.binsener = gdat.binsenerfull[gdat.indxenerinclbins]
            gdat.meanener = np.sqrt(gdat.binsener[1:] * gdat.binsener[:-1])
            gdat.deltener = gdat.binsener[1:] - gdat.binsener[:-1]
            gdat.minmener = gdat.binsener[0]
            gdat.maxmener = gdat.binsener[-1]
            for strg in ['plot']:
                if strg == '':
                    numbbins = gdat.numbener
                else:
                    numbbins = gdat.numbenerplot
                retr_axis(gdat, 'ener' + strg, gdat.minmener, gdat.maxmener, numbbins)

            gdat.limtener = [np.amin(gdat.binsener), np.amax(gdat.binsener)] 
        if gdat.boolbinsener: 
            if gdat.numbener > 1:
                gdat.enerpivt = gdat.meanener[gdat.indxenerpivt]
            # energy bin indices other than that of the pivot bin
            gdat.indxenerinde = np.setdiff1d(gdat.indxener, gdat.indxenerpivt)
        
            # temp
            if gdat.typeexpr == 'chan':
                gdat.edis = 0.3 * np.sqrt(gdat.binsener) / 2.35
                gdat.edisintp = sp.interpolate.interp1d(gdat.binsener, gdat.edis, fill_value='extrapolate')
            else:
                gdat.edis = None
                gdat.edisintp = None

        # number of elements
        setp_varblimt(gdat, 'numbelem', [0, 400], popl='full')
        
        print('gdat.fittminmnumbelempop0')
        print(gdat.fittminmnumbelempop0)

        # define maximum and minimum number of elements as lists of np.arrays
        for strgmodl in gdat.liststrgmodl:
            for strglimt in gdat.liststrglimt:
                numbpopl = getattr(gdat, strgmodl + 'numbpopl')
                indxpopl = getattr(gdat, strgmodl + 'indxpopl')
                limtnumbelem = [[] for l in indxpopl]
                for l in indxpopl:
                    limtnumbelem[l] = np.zeros(1, dtype=int)
                    for d in np.arange(limtnumbelem[l].size):
                        limtnumbelem[l] = getattr(gdat, strgmodl + strglimt + 'numbelempop%d' % l)
                setattr(gdat, strgmodl + strglimt + 'numbelem', limtnumbelem)
        
        ## hyperparameters
        for strgmodl in gdat.liststrgmodl:
            
            setp_varbvalu(gdat, 'typemodltran', 'drct')
            
            typemodltran = getattr(gdat, strgmodl + 'typemodltran')
            if typemodltran == 'pois':
                limtmeanelem = [0.1, 1000.]
                for strgmodl in gdat.liststrgmodl:
                    setp_varblimt(gdat, 'meanelem', limtmeanelem, popl='full', strgmodl=strgmodl)
        
        #### boolean flag background
        if gdat.typeexpr == 'chan':
            if gdat.numbpixlfull == 1:
                specback = [True, True]
            else:
                specback = [False, False]
            setp_varbvalu(gdat, 'specback', specback)
        else:
            for strgmodl in gdat.liststrgmodl:
                backtype = getattr(gdat, strgmodl + 'backtype')
                specback = [False for k in range(len(backtype))]
                setp_varbvalu(gdat, 'specback', specback, strgmodl=strgmodl)
        
        if gdat.strgexprname is None:
            if gdat.typeexpr == 'chan':
                gdat.strgexprname = 'Chandra'
            if gdat.typeexpr == 'ferm':
                gdat.strgexprname = 'Fermi-LAT'
            if gdat.typeexpr == 'hubb':
                gdat.strgexprname = 'HST'
            if gdat.typeexpr == 'sche':
                gdat.strgexprname = 'XXXXX'
            if gdat.typeexpr == 'sdyn':
                gdat.strgexprname = 'TGAS-RAVE'
        
        if gdat.lablgangunit is None:
            if gdat.typeexpr == 'ferm':
                gdat.lablgangunit = '$^o$'
            if gdat.typeexpr == 'sdyn':
                gdat.lablgangunit = ''
            if gdat.typeexpr == 'sdss' or gdat.typeexpr == 'chan' or gdat.typeexpr == 'hubb':
                gdat.lablgangunit = '$^{\prime\prime}$'
        
        if gdat.labllgal is None:
            if gdat.typeexpr == 'sdyn':
                gdat.labllgal = r'L_{z}'
            else:
                if gdat.typeexpr == 'ferm' and gdat.lgalcntr == 0 and gdat.bgalcntr == 0:
                    gdat.labllgal = r'l'
                else:
                    gdat.labllgal = r'\theta_1'
        if gdat.lablbgal is None:
            if gdat.typeexpr == 'sdyn':
                gdat.lablbgal = r'E_k'
            else:
                if gdat.typeexpr == 'ferm' and gdat.lgalcntr == 0 and gdat.bgalcntr == 0:
                    gdat.lablbgal = r'b'
                else:
                    gdat.lablbgal = r'\theta_2'

        if gdat.strgenerunit is None:
            if gdat.typeexpr == 'ferm':
                gdat.strgenerunit = 'GeV'
                gdat.nameenerunit = 'gevv'
            if gdat.typeexpr == 'chan':
                gdat.strgenerunit = 'keV'
                gdat.nameenerunit = 'kevv'
            if gdat.typeexpr == 'sdyn':
                gdat.strgenerunit = ''
                gdat.nameenerunit = ''
            if gdat.typeexpr == 'hubb':
                gdat.strgenerunit = 'erg'
                gdat.nameenerunit = 'ergs'
            if gdat.typeexpr == 'fire':
                gdat.strgenerunit = '$\mu$ m$^{-1}$'
                gdat.nameenerunit = 'imum'

        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            elemspatevaltype = [[] for l in indxpopl]
            for l in indxpopl:
                # these element types slow down execution!
                if elemtype[l] == 'lens' or elemtype[l].startswith('lghtline') or elemtype[l] == 'clusvari' or elemtype[l] == 'lghtgausbgrd':
                    elemspatevaltype[l] = 'full'
                else:
                    elemspatevaltype[l] = 'locl'
            setp_varbvalu(gdat, 'elemspatevaltype', elemspatevaltype, strgmodl=strgmodl)

        gdat.commelemspatevaltype = []
        for strgmodl in gdat.liststrgmodl:
            elemspatevaltype = getattr(gdat, strgmodl + 'elemspatevaltype')
            for elemspatevaltypetemp in elemspatevaltype:
                if not elemspatevaltypetemp in gdat.commelemspatevaltype:
                    gdat.commelemspatevaltype.append(elemspatevaltypetemp)
        
        if gdat.nameexpr is None:
            if gdat.typeexpr == 'ferm':
                gdat.nameexpr = 'Fermi-LAT'
            if gdat.typeexpr == 'sdss':
                gdat.nameexpr = 'SDSS'
            if gdat.typeexpr == 'chan':
                gdat.nameexpr = 'Chandra'
            if gdat.typeexpr == 'hubb':
                gdat.nameexpr = 'HST'
            if gdat.typeexpr == 'gaia':
                gdat.nameexpr = 'Gaia'
        
        ## Lensing
        if gdat.radispmr is None:
            if gdat.typeexpr == 'ferm':
                gdat.radispmr = 0.6 / gdat.anglfact
            if gdat.typeexpr == 'hubb':
                gdat.radispmr = 0.15 / gdat.anglfact
            if gdat.typeexpr == 'tess':
                gdat.radispmr = 1. / gdat.anglfact
            if gdat.typeexpr == 'chan':
                if gdat.anlytype == 'spec':
                    gdat.radispmr = 0.1
                else:
                    gdat.radispmr = 0.2 / gdat.anglfact
            if gdat.typeexpr == 'sdss':
                gdat.radispmr = 0.5 / gdat.anglfact
            if gdat.typeexpr == 'sdyn':
                gdat.radispmr = 0.2
            if gdat.typeexpr == 'gene':
                gdat.radispmr = 0.2
        
        print('gdat.radispmr')
        print(gdat.radispmr)

        if gdat.anglassc is None:
            gdat.anglassc = 5. * gdat.radispmr
    
        print('gdat.anglassc')
        print(gdat.anglassc)

        ### experimental PSFs
        if gdat.typeexpr == 'ferm':
            pass
            #retr_psfnferm(gdat)
            #angltemp = pi * np.linspace(0., 10., 100) / 180.
            #psfn = retr_psfnferm(meanener, angltemp)
            #fwhm = retr_fwhm(psfn, angl) 

        # temp -- this should depend on gdat.commboolelemkernanyy
        if gdat.numbpixlfull > 1:
            if gdat.typeexpr == 'chan' or gdat.typeexpr == 'sdss':
                numbpsfpform = 0
                numbpsfptotl = 0
                for strgmodl in gdat.liststrgmodl:
                    getattr(gdat, strgmodl + 'numbpsfpform', numbpsfpform)
                    getattr(gdat, strgmodl + 'numbpsfptotl', numbpsfptotl)
            if gdat.typeexpr == 'chan':
                retr_psfpchan(gdat)
            if gdat.typeexpr == 'ferm':
                retr_psfpferm(gdat)
            if gdat.typeexpr == 'sdss':
                retr_psfpsdss(gdat)
            if gdat.typeexpr == 'hubb':
                retr_psfphubb(gdat)
            if gdat.typeexpr == 'tess':
                retr_psfptess(gdat)
            if gdat.typeexpr == 'sdyn':
                retr_psfpsdyn(gdat)
            if gdat.typeexpr == 'gene':
                retr_psfpsdyn(gdat)
   
        gdat.factburntmpr = 0.75
        gdat.numbburntmpr = gdat.factburntmpr * gdat.numbburn

        # model evaluation approximation error tolerance in units of the fraction of the lowest PS flux
        if gdat.specfraceval is None:
            if gdat.typeexpr == 'ferm':
                gdat.specfraceval = 0.5
            else:
                gdat.specfraceval = 0.1

        ### element parameter distributions
        
        gdat.binslgalcart = np.linspace(gdat.minmlgaldata, gdat.maxmlgaldata, gdat.numbsidecart + 1)
        gdat.binsbgalcart = np.linspace(gdat.minmbgaldata, gdat.maxmbgaldata, gdat.numbsidecart + 1)
        gdat.meanlgalcart = (gdat.binslgalcart[0:-1] + gdat.binslgalcart[1:]) / 2.
        gdat.meanbgalcart = (gdat.binsbgalcart[0:-1] + gdat.binsbgalcart[1:]) / 2.
        
        ### PSF model
        #### angular profile
        if gdat.typeexpr == 'ferm':
            gdat.typepsfnexpr = 'doubking'
        if gdat.typeexpr == 'chan':
            gdat.typepsfnexpr = 'singking'
        if gdat.typeexpr == 'sdss':
            gdat.typepsfnexpr = 'singgaus'
        if gdat.typeexpr == 'hubb':
            gdat.typepsfnexpr = 'singgaus'
        if gdat.typeexpr == 'tess':
            gdat.typepsfnexpr = 'singgaus'
        if gdat.typeexpr == 'sdyn':
            gdat.typepsfnexpr = 'singgaus'
        if gdat.typeexpr == 'gene':
            gdat.typepsfnexpr = 'singgaus'
        if gdat.typeexpr == 'fire':
            gdat.typepsfnexpr = None
        print('gdat.typepsfnexpr')
        print(gdat.typepsfnexpr)

        psfntype = gdat.typepsfnexpr
        setp_varbvalu(gdat, 'psfntype', psfntype)
        
        #### background names
        listnameback = ['isot']
        if gdat.typeexpr == 'ferm':
            listnameback.append('fdfm')
        #if gdat.typeexpr == 'chan':
        #    listnameback.append('part')
        setp_varbvalu(gdat, 'listnameback', listnameback)
        
        # reference elements
        gdat.numbrefr = 0
        if gdat.typedata == 'mock':
            gdat.numbrefr = gdat.truenumbpopl
        if gdat.typedata == 'inpt':
            if gdat.typeexpr == 'ferm':
                gdat.numbrefr = 2
            if gdat.typeexpr == 'chan':
                gdat.numbrefr = 2
        
        for strgmodl in gdat.liststrgmodl:
            maxmnumbelem = getattr(gdat, strgmodl + 'maxmnumbelem')
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            maxmnumbelempopl = [[] for l in indxpopl]
            for l in indxpopl:
                maxmnumbelempopl[l] = np.sum(maxmnumbelem[l])
            setattr(gdat, strgmodl + 'maxmnumbelempopl', maxmnumbelempopl)
            
        gdat.indxrefr = np.arange(gdat.numbrefr)
        if gdat.boolasscrefr is None:
            gdat.boolasscrefr = [True for q in gdat.indxrefr]
        gdat.refrlistnameparaelemgenrampl = [[] for q in gdat.indxrefr]
        gdat.listnamerefr = [] 
        gdat.refrlistnameparaelem = [[] for q in gdat.indxrefr]
        gdat.refrlistnameparaelemodim = [[] for q in gdat.indxrefr]
        gdat.refrinfo = False
        gdat.listpathwcss = []
        gdat.numbpixllgalshft = []
        gdat.numbpixlbgalshft = []
        gdat.refrindxpoplassc = [[] for q in gdat.indxrefr] 
        
        gdat.factsindplot = 1.
        gdat.factmagtplot = 1.
        gdat.factotypplot = 1.
        
        # temp -- this allows up to 3 reference populations
        gdat.refrcolrelem = ['darkgreen', 'olivedrab', 'mediumspringgreen']
        # temp -- this allows up to 3 reference populations
        gdat.fittcolrelem = ['royalblue', 'dodgerblue', 'navy']
        if gdat.typedata == 'mock':
            gdat.refrinfo = True
            gdat.numbrefr = gdat.truenumbpopl
            gdat.listnamerefr = ['moc%d' % l for l in gdat.trueindxpopl] 
            gdat.indxrefr = np.arange(gdat.numbrefr)
        if gdat.typedata == 'inpt':
            if gdat.typeexpr == 'ferm':
                gdat.refrinfo = True
                retr_refrferminit(gdat)
                for q in gdat.indxrefr:
                    gdat.refrindxpoplassc[q] = gdat.fittindxpopl
            if gdat.typeexpr == 'chan':
                gdat.refrinfo = True
                retr_refrchaninit(gdat)
                for q in gdat.indxrefr:
                    gdat.refrindxpoplassc[q] = gdat.fittindxpopl
            
            for q in gdat.indxrefr:
                if 'lgal' in gdat.refrlistnameparaelem[q] and 'bgal' in gdat.refrlistnameparaelem[q]:
                    gdat.refrlistnameparaelem[q] += ['gang', 'aang']
                for strgfeat in gdat.refrlistnameparaelem[q]:
                    setattr(gdat, 'refr' + strgfeat, [[] for q in gdat.indxrefr])
            gdat.refrlistnameparaelemtotl = retr_listconc(gdat.refrlistnameparaelem)
            
            if gdat.typeexpr == 'ferm':
                retr_refrfermfinl(gdat)
            if gdat.typeexpr == 'chan':
                retr_refrchanfinl(gdat)
        
        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            spatdisttype = [[] for l in indxpopl]
            fluxdisttype = [[] for l in indxpopl]
            for l in indxpopl:
                spatdisttype[l] = 'unif'
               
                # temp -- this can assign powrslop to populations whose flux is not drawn from a power law!
                if elemtype[l].startswith('lght'):
                    fluxdisttype[l] = 'powrslop'
                else:
                    fluxdisttype[l] = None

            setp_varbvalu(gdat, 'spatdisttype', spatdisttype, strgmodl=strgmodl)
            setp_varbvalu(gdat, 'fluxdisttype', fluxdisttype, strgmodl=strgmodl)
        
        gdat.prsccmtr = 3.086e18
        gdat.ergsgevv = 624.151
        gdat.factnewtlght = 2.09e13 # Msun / pc
        
        # the adis in the file is kpc
        fileh5py = narr_h5(gdat.pathdata + 'inpt/adis.h5', 'r')
        
        gdat.redsintp = fileh5py['reds'][()]
        gdat.adisintp = fileh5py['adis'][()] * 1e6 # [pc]

        gdat.adisobjt = sp.interpolate.interp1d(gdat.redsintp, gdat.adisintp, fill_value='extrapolate')

        gdat.redsfromdlosobjt = sp.interpolate.interp1d(gdat.adisintp * gdat.redsintp, gdat.redsintp, fill_value='extrapolate')
        fileh5py.close()
        
        if gdat.typedata == 'mock':
            if gdat.trueboollenshost:
                setp_varbvalu(gdat, 'redshost', 0.2, strgmodl='true')
                setp_varbvalu(gdat, 'redssour', 1., strgmodl='true')
        
            for l in gdat.trueindxpopl:
                if gdat.trueelemtype[l] == 'lens':
                    numbelem = 25
                else:
                    numbelem = 100
                setp_varbvalu(gdat, 'numbelem', numbelem, popl=l, strgmodl='true')
        
            setp_indxpara(gdat, 'medi', strgmodl='true')
            gdat.refrlistnameparaelemtotl = gdat.truelistnameparaelemtotl
            for l in gdat.trueindxpopl:
                gdat.refrlistnameparaelemgenrampl[l] = gdat.truenameparaelemgenrampl[l]
                for strgfeat in gdat.truelistnameparaelemodim[l]:
                    gdat.refrlistnameparaelem[l].append(strgfeat)
        
        ### background template normalizations
        if gdat.typeexpr == 'ferm':
            if 'ferm_bubb' in gdat.strgcnfg:
                setp_varblimt(gdat, 'bacp', [1e-10, 1e10], ener='full', back='full')
            else:
                # isotropic + unresolved
                setp_varblimt(gdat, 'bacp', [1e-7, 1e-2], ener=0, back=0)
                setp_varblimt(gdat, 'bacp', [1e-9, 1e-3], ener=1, back=0)
                setp_varblimt(gdat, 'bacp', [1e-10, 1e-4], ener=2, back=0)
                # diffuse
                setp_varblimt(gdat, 'bacp', [1e-6, 1e-2], ener=0, back=1)
                setp_varblimt(gdat, 'bacp', [1e-7, 1e-3], ener=1, back=1)
                setp_varblimt(gdat, 'bacp', [1e-8, 1e-4], ener=2, back=1)
                # dark
                setp_varblimt(gdat, 'bacp', [1e-11, 1e-4], ener=0, back=2)
                setp_varblimt(gdat, 'bacp', [1e-11, 1e-4], ener=1, back=2)
                setp_varblimt(gdat, 'bacp', [1e-11, 1e-4], ener=2, back=2)

            # Fourier basis
            for strgmodl in gdat.liststrgmodl:
                backtype = getattr(gdat, strgmodl + 'backtype')
                indxback = getattr(gdat, strgmodl + 'indxback')
                for c in indxback:
                    if isinstance(backtype[c], str):
                        if 'bfun' in backtype[c]:
                            setp_varblimt(gdat, 'bacp', [1e-10, 1e10], ener='full', back=c)

        else:
            back = 0
            # sky background + unresolved
            if gdat.typeexpr == 'chan':
                if gdat.numbpixlfull == 1:
                    bacp = [1e0, 1e2]
                    setp_varblimt(gdat, 'bacp', bacp, back=0)
                else:
                    bacp = [1e-1, 1e3]
                    setp_varblimt(gdat, 'bacp', bacp, ener='full', back=0)
            else:
                if gdat.typeexpr == 'hubb':
                    bacp = [1e-10, 1e-6]
                # background
                if gdat.typeexpr == 'sdyn' or gdat.typeexpr == 'gene':
                    bacp = [1e-1, 1e1]
                if gdat.typeexpr == 'fire':
                    bacp = [1e-1, 1e1]
                if gdat.typeexpr == 'tess':
                    bacp = [1e-1, 1e1]
                setp_varblimt(gdat, 'bacp', bacp, ener='full', back=0)
        
            # particle background
            #if gdat.typeexpr == 'chan':
            #    if gdat.anlytype == 'spec':
            #        bacp = [1e-8, 1e-6]
            #    else:
            #        bacp = [1e-1, 1e2]
            #    setp_varblimt(gdat, 'bacp', bacp, back=1)
            
        ### element parameter boundaries
        #### spatial
        if gdat.numbpixlfull != 1:
            if gdat.typeexpr == 'ferm':
                minmgang = 1e-1 / gdat.anglfact
            else:
                minmgang = 1e-2 / gdat.anglfact
            setp_varbvalu(gdat, 'minmgang', minmgang)
        
        booltemp = False
        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            for l in indxpopl:
                if elemtype[l].startswith('lghtline'):
                    enertemp = np.sqrt(gdat.limtener[0] * gdat.limtener[1])
                    # temp -- these should depend on population index
                    setp_varblimt(gdat, 'elin', gdat.limtener, strgmodl=strgmodl)
                    setp_varblimt(gdat, 'sigm', np.array([1e-1, 1e0]) * enertemp, strgmodl=strgmodl)
                    setp_varblimt(gdat, 'gamm', np.array([1e-1, 1e0]) * enertemp, strgmodl=strgmodl)
            
        if gdat.numbpixlfull != 1:
            minmdefs = 0.003 / gdat.anglfact
            setp_varbvalu(gdat, 'minmdefs', minmdefs)
        
        minmnobj = 1e0
        setp_varbvalu(gdat, 'minmnobj', minmnobj)
        
        setp_varblimt(gdat, 'curv', [-1., 1.])

        maxmnobj = 1e3
        setp_varbvalu(gdat, 'maxmnobj', maxmnobj)
        
        if gdat.numbpixlfull != 1:
            maxmdefs = 1. / gdat.anglfact
            setp_varbvalu(gdat, 'maxmdefs', maxmdefs)
        
        # parameter defaults
        ## distribution
        ### flux
        if 'lens' in gdat.commelemtype:
            ### projected scale radius
            limtasca = np.array([0., 0.1]) / gdat.anglfact
            setp_varblimt(gdat, 'asca', limtasca)
            ### projected cutoff radius
            limtacut = np.array([0., 2.]) / gdat.anglfact
            setp_varblimt(gdat, 'acut', limtacut)
   
        # true model parameters
        if gdat.typedata == 'mock':
            gdat.truenumbelem = np.zeros(gdat.truenumbpopl, dtype=int)
            if gdat.truetypemodltran == 'pois':
                for l in gdat.trueindxpopl:
                    setattr(gdat, 'truemeanelempop%d' % l, getattr(gdat, 'truenumbelempop%d' % l))
                    gdat.truenumbelem[l] = getattr(gdat, 'truenumbelempop%d' % l)
        
                    if gdat.truenumbelem[l] > gdat.truemaxmnumbelem[l]:
                        raise Exception('True number of elements is larger than np.maximum.')

        for strgmodl in gdat.liststrgmodl:
            typemodltran = getattr(gdat, strgmodl + 'typemodltran')
            if typemodltran == 'pois':
                setp_varbvalu(gdat, 'scalmeanelem', 'logt')
        
        #setp_varbvalu(gdat, 'gangdisttype', ['self'])
        
        for strgmodl in gdat.liststrgmodl:
            setp_varblimt(gdat, 'lgal', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl=strgmodl)
            setp_varblimt(gdat, 'bgal', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl=strgmodl)
        
            spatdisttype = getattr(gdat, strgmodl + 'spatdisttype')
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')

            for l in indxpopl:
                if spatdisttype[l] == 'gangexpo':
                    setp_varbvalu(gdat, 'maxmgang', getattr(gdat, strgmodl + 'maxmlgal'), strgmodl=strgmodl)
                    if gdat.typeexpr == 'ferm':
                        gangdistsexp = 5. / gdat.anglfact
                    setp_varbvalu(gdat, 'gangdistsexp', gangdistsexp, strgmodl=strgmodl, popl=l)
                if spatdisttype[l] == 'dsrcexpo':
                    if gdat.typeexpr == 'hubb':
                        dsrcdistsexp = 0.5 / gdat.anglfact
                    setp_varbvalu(gdat, 'dsrcdistsexp', dsrcdistsexp, strgmodl=strgmodl, popl=l)
        
            if gdat.fittboollenshost or gdat.fittboolemishost:
                setp_varblimt(gdat, 'lgalhost', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl='fitt', isfr='full')
                setp_varblimt(gdat, 'bgalhost', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl='fitt', isfr='full')
        setp_varblimt(gdat, 'lgalsour', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl='fitt')
        setp_varblimt(gdat, 'bgalsour', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl='fitt')
        
        if gdat.numbpixlfull != 1:
            gdat.stdvhostsour = 0.04 / gdat.anglfact
            if gdat.typedata == 'mock':
                setp_varblimt(gdat, 'lgalsour', [0., gdat.stdvhostsour], strgmodl='true', typelimt='meanstdv')
                setp_varblimt(gdat, 'bgalsour', [0., gdat.stdvhostsour], strgmodl='true', typelimt='meanstdv')
                if gdat.trueboollenshost or gdat.trueboolemishost:
                    setp_varblimt(gdat, 'lgalhost', [0., gdat.stdvhostsour], strgmodl='true', typelimt='meanstdv', isfr='full')
                    setp_varblimt(gdat, 'bgalhost', [0., gdat.stdvhostsour], strgmodl='true', typelimt='meanstdv', isfr='full')
            
            setp_varblimt(gdat, 'redshost', [0., 0.4])
            setp_varblimt(gdat, 'redssour', [0.5, 1.5])
            setp_varblimt(gdat, 'fluxsour', np.array([1e-22, 1e-17]))
            setp_varblimt(gdat, 'sindsour', np.array([0., 4.]))
            setp_varblimt(gdat, 'sizesour', [0.1 / gdat.anglfact, 2. / gdat.anglfact])
            setp_varblimt(gdat, 'ellpsour', [0., 0.5])
            
            setp_varbvalu(gdat, 'redshost', 0.2, strgmodl='fitt')
            setp_varbvalu(gdat, 'redssour', 1., strgmodl='fitt')
            
            for strgmodl in gdat.liststrgmodl:
                boollenshost = getattr(gdat, strgmodl + 'boollenshost')
                boolemishost = getattr(gdat, strgmodl + 'boolemishost')
                if boollenshost or boolemishost:
                    setp_varblimt(gdat, 'fluxhost', np.array([1e-20, 1e-15]), isfr='full', strgmodl=strgmodl)
                    setp_varblimt(gdat, 'sindhost', np.array([0., 4.]), isfr='full', strgmodl=strgmodl)
                    setp_varblimt(gdat, 'sizehost', [0.1 / gdat.anglfact, 4. / gdat.anglfact], isfr='full', strgmodl=strgmodl)
                    setp_varblimt(gdat, 'beinhost', [0.5 / gdat.anglfact, 2. / gdat.anglfact], isfr='full', strgmodl=strgmodl)
                    setp_varblimt(gdat, 'ellphost', [0., 0.5], isfr='full', strgmodl=strgmodl)
                    setp_varblimt(gdat, 'anglhost', [0., np.pi], isfr='full', strgmodl=strgmodl)
                    if strgmodl == 'fitt':
                        setp_varblimt(gdat, 'serihost', [1., 8.], isfr='full', strgmodl=strgmodl)
                    else:
                        setp_varbvalu(gdat, 'serihost', 4., isfr='full', strgmodl=strgmodl)
                        setp_varblimt(gdat, 'serihost', [1., 8.], isfr='full', strgmodl=strgmodl)
            
                #print('gdat.fittminmfluxhost')
                #print(gdat.fittminmfluxhost)
                #raise Exception('')
                setp_varblimt(gdat, 'sherextr', [0., 0.1], strgmodl=strgmodl)
                setp_varblimt(gdat, 'anglsour', [0., np.pi], strgmodl=strgmodl)
                setp_varblimt(gdat, 'sangextr', [0., np.pi], strgmodl=strgmodl)
            
            # temp -- to be removed
            #gdat.truefactlgal = gdat.truemaxmlgal - gdat.trueminmlgal
            #gdat.truefactbgal = gdat.truemaxmbgal - gdat.trueminmbgal
            #gdat.trueminmaang = -np.pi
            #gdat.truemaxmaang = pi
            
            setp_varblimt(gdat, 'aang', [-np.pi, np.pi])
   
        # copy the true model to the inference model if the inference model parameter has not been specified
        #temp = deepcopy(gdat.__dict__)
        #for strg, valu in temp.items():
        #    if strg.startswith('true') and not strg[4:].startswith('indx'):
        #        try:
        #            valumodl = getattr(gdat, 'fitt' + strg[4:])
        #            if valumodl is None:
        #                raise
        #            if gdat.typeverb > 1:
        #                print 'Received custom input for ' + strg[4:]
        #        except:
        #            setattr(gdat, 'fitt' + strg[4:], getattr(gdat, strg))
        
        # check inputs
        if gdat.numbburn > gdat.numbswep:
            raise Exception('Bad number of burn-in sweeps.')
        if gdat.factthin > gdat.numbswep - gdat.numbburn or gdat.factthin < 1:
            raise Exception('Bad thinning factor.')
        if gdat.typepixl == 'heal' and gdat.numbspatdims > 2:
            raise Exception('More than 2 spatial dimensions require Cartesian binning.')
        
        if gdat.allwfixdtrue and gdat.typedata == 'mock':
            
            for l in gdat.trueindxpopl:
                if gdat.numbpixlfull != 1:
                    if gdat.trueboolelemspat[l]:
                        setp_varblimt(gdat, 'lgal', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl=strgmodl, popl=l)
                        setp_varblimt(gdat, 'bgal', [-gdat.maxmgangdata, gdat.maxmgangdata], strgmodl=strgmodl, popl=l)
                        setp_varbvalu(gdat, 'spatdistcons', 1e-3, popl=l)
                        setp_varbvalu(gdat, 'gangdistslop', 1.1, popl=l)
                        setp_varbvalu(gdat, 'bgaldistscal', 2. / gdat.anglfact, popl=l)
                if gdat.typeexpr == 'ferm':
                    setp_varbvalu(gdat, 'fluxdistsloplowr', 1.5, popl=l)
                    setp_varbvalu(gdat, 'fluxdistslopuppr', 2.5, popl=l)
                    setp_varbvalu(gdat, 'fluxdistbrek', 1e-9, popl=l)
                if gdat.trueelemtype[l] == 'lghtpnts':
                    setp_varbvalu(gdat, 'fluxdistslop', 2.2, popl=l)
                if gdat.trueelemtype[l].startswith('lghtline'):
                    setp_varbvalu(gdat, 'fluxdistslop', 2., popl=l)
                if gdat.trueelemtype[l] == 'lens':
                    setp_varbvalu(gdat, 'defsdistslop', 1.9, popl=l)
                if gdat.trueelemtype[l].startswith('clus'):
                    setp_varbvalu(gdat, 'nobjdistslop', 2., popl=l)

                if gdat.trueelemtype[l] == 'lens':
                    setp_varbvalu(gdat, 'ascadistmean', 0.05 / gdat.anglfact, popl=l)
                    setp_varbvalu(gdat, 'ascadiststdv', 0.04 / gdat.anglfact, popl=l)
                    setp_varbvalu(gdat, 'acutdistmean', 1. / gdat.anglfact, popl=l)
                    setp_varbvalu(gdat, 'acutdiststdv', 0.04 / gdat.anglfact, popl=l)
                
                if gdat.trueelemtype[l] == 'lghtgausbgrd' or gdat.trueelemtype[l] == 'clusvari':
                    setp_varbvalu(gdat, 'gwdtdistslop', 2., popl=l)
                
                if gdat.trueboolelemlght[l]:
                    if gdat.typeexpr == 'ferm':
                        sinddistmean = 2.15
                    if gdat.typeexpr == 'chan':
                        sinddistmean = 1.
                    if gdat.typeexpr == 'hubb':
                        sinddistmean = 1.
                    if gdat.typeexpr != 'fire':
                        setp_varbvalu(gdat, 'sinddistmean', sinddistmean, popl=l)
                        setp_varbvalu(gdat, 'sinddiststdv', 0.5, popl=l)
                        
                        setp_varbvalu(gdat, 'curvdistmean', 2., popl=l)
                        setp_varbvalu(gdat, 'curvdiststdv', 0.2, popl=l)
                        
                        setp_varbvalu(gdat, 'expcdistmean', 2., popl=l)
                        setp_varbvalu(gdat, 'expcdiststdv', 0.2, popl=l)
            
                if gdat.trueelemtype[l] == 'lghtpntspuls':
                    setp_varbvalu(gdat, 'per0distmean', 3e-3, popl=l)
                    setp_varbvalu(gdat, 'per0diststdv', 0.3, popl=l)
                    setp_varbvalu(gdat, 'magfdistmean', 10**8.5, popl=l)
                    setp_varbvalu(gdat, 'magfdiststdv', 0.7, popl=l)
                    setp_varbvalu(gdat, 'dglcdistslop', 2., popl=l)
                elif gdat.trueelemtype[l] == 'lghtpntsagnntrue':
                    setp_varbvalu(gdat, 'dlosdistslop', -2., popl=l)
                    setp_varbvalu(gdat, 'scaldlosdistslop', 'self', popl=l)
                    setp_varbvalu(gdat, 'lum0distsloplowr', 0.5, popl=l)
                    setp_varbvalu(gdat, 'lum0distslopuppr', 1.5, popl=l)
                
            if gdat.typeexpr == 'ferm':

                #setp_varbvalu(gdat, 'bacp', 5e-6, ener=0, back=0)
                setp_varbvalu(gdat, 'bacp', 5e-6, ener=0, back=0)
                setp_varbvalu(gdat, 'bacp', 2e-8, ener=1, back=0)
                setp_varbvalu(gdat, 'bacp', 2e-9, ener=2, back=0)
                #setp_varbvalu(gdat, 'bacp', 1e-5, ener=4, back=0)
                #setp_varbvalu(gdat, 'bacp', 7e-7, ener=0, back=1)
                setp_varbvalu(gdat, 'bacp', 1e-4, ener=0, back=1)
                setp_varbvalu(gdat, 'bacp', 1e-5, ener=1, back=1)
                setp_varbvalu(gdat, 'bacp', 7e-7, ener=2, back=1)
                #setp_varbvalu(gdat, 'bacp', 3e-8, ener=4, back=1)

            else:
                # sky background
                if gdat.typeexpr == 'chan':
                    if gdat.numbpixlfull == 1:
                        bacp = 10.
                        setp_varbvalu(gdat, 'bacp', bacp)
                    else:
                        setp_varbvalu(gdat, 'bacp', 170., back=0, ener=0)
                        setp_varbvalu(gdat, 'bacp', 17.4, back=0, ener=1)
                        setp_varbvalu(gdat, 'bacp', 27., back=0, ener=2)
                        setp_varbvalu(gdat, 'bacp', 11.8, back=0, ener=3)
                        setp_varbvalu(gdat, 'bacp', 101., back=0, ener=4)
                else:
                    if gdat.typeexpr == 'hubb':
                        bacp = 2e-7
                    if gdat.typeexpr == 'sdyn':
                        bacp = 1.
                    if gdat.numbpixlfull == 1:
                        setp_varbvalu(gdat, 'bacp', bacp, back=0)
                    else:
                        setp_varbvalu(gdat, 'bacp', bacp, ener='full', back=0)

                # particle background
                if gdat.typeexpr == 'chan':
                    bacp = 70.04
                    setp_varbvalu(gdat, 'bacp', bacp, back=1)
            
            if gdat.trueboollenshost:
                setp_varbvalu(gdat, 'beinhost', 1.5 / gdat.anglfact)
                setp_varbvalu(gdat, 'sizesour', 0.3 / gdat.anglfact)
                setp_varbvalu(gdat, 'sizehost', 1. / gdat.anglfact)
                setp_varbvalu(gdat, 'ellpsour', 0.2)
                setp_varbvalu(gdat, 'fluxsour', 1e-18)
                setp_varbvalu(gdat, 'sindsour', 1.5)
                setp_varbvalu(gdat, 'fluxhost', 1e-16)
                setp_varbvalu(gdat, 'sindhost', 2.5)
                setp_varbvalu(gdat, 'ellphost', 0.2)
                setp_varbvalu(gdat, 'sangextr', np.pi / 2.)
                setp_varbvalu(gdat, 'serihost', 4.)
                
        if gdat.defa:
            return gdat
        
        if gdat.typeverb > 0:
            if gdat.boolburntmpr:
                print('Warning: Tempered burn-in.')

        if gdat.typedata == 'inpt':
            gdat.minmsind = -1.
            gdat.maxmsind = 2.
            gdat.minmcurv = -1.
            gdat.maxmcurv = 1.
            gdat.minmexpc = 0.1
            gdat.maxmexpc = 10.

            for q in gdat.indxrefr:
                for strgfeat in gdat.refrlistnameparaelem[q]:
                    if strgfeat == 'etag' or strgfeat == 'gang' or strgfeat == 'aang':
                        continue
                    refrfeat = getattr(gdat, 'refr' + strgfeat)
                        
                    if len(refrfeat[q]) == 0 or refrfeat[q].ndim < 2:
                        raise Exception('')
            
        gdat.refrnumbelem = np.zeros(gdat.numbrefr, dtype=int)
        gdat.refrnumbelempopl = np.zeros(gdat.numbrefr, dtype=int)
        for q in gdat.indxrefr:
            if gdat.typedata == 'mock':
                gdat.refrnumbelem[q] = gdat.truenumbelem[q]
            else:
                # temp -- this treats gdat.refrlgal[q] as a 1D np.array. gdat.refrlgal gets modified later and gdat.refrnumbelem gets redefined
                gdat.refrnumbelem[q] = gdat.refrlgal[q].shape[0]
            gdat.refrnumbelempopl[q] = np.sum(gdat.refrnumbelem[q])
            
        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            nameparaelemgenrampl = getattr(gdat, strgmodl + 'nameparaelemgenrampl')
            for l in indxpopl:
                if elemtype[l] == 'lens':
                    meandistslop = 1.9
                    stdvdistslop = 0.5
                    scal = 'gaus'
                else:
                    minmdistslop = 0.5
                    maxmdistslop = 3.
                    scal = 'logt'
                if scal == 'gaus':
                    typelimt = 'meanstdv'
                    limt = [meandistslop, stdvdistslop]
                else:
                    typelimt = 'minmmaxm'
                    limt = [minmdistslop, maxmdistslop]
                
                setp_varblimt(gdat, nameparaelemgenrampl[l] + 'distslop', limt, popl=l, typelimt=typelimt, strgmodl=strgmodl)
                setp_varbvalu(gdat, 'scal' + nameparaelemgenrampl[l] + 'distslop', scal, popl=l, strgmodl=strgmodl)

                if elemtype[l] == 'lghtgausbgrd' or elemtype[l] == 'clusvari':
                    setp_varblimt(gdat, 'gwdtdistslop', [0.5, 4.], popl=l, strgmodl=strgmodl)
                    setp_varbvalu(gdat, 'scalgwdtdistslop', 'logt', popl=l, strgmodl=strgmodl)

        # initial setup
        setpinit(gdat) 
        
        # if there is any user-provided PSF template
        if gdat.numbpixlfull > 1:
            if gdat.commboolelempsfnanyy:
                gdat.psfnexpr = retr_psfn(gdat, gdat.psfpexpr, gdat.indxener, gdat.binsangl, gdat.typepsfnexpr, strgmodl)
        
        if gdat.fittmaxmnumbparaelem > 0 and gdat.commboolelempsfnanyy:
            gdat.limsangl = [[[] for m in gdat.indxevtt] for i in gdat.indxener]
            gdat.limspsfn = [[[] for m in gdat.indxevtt] for i in gdat.indxener]
            for i in gdat.indxener:
                for m in gdat.indxevtt:
                    psfn = gdat.psfnexpr[i, :, m]
                    maxmpsfn = np.amax(psfn)
                    gdat.limsangl[i][m] = [0., gdat.binsangl[np.amax(np.where(psfn > 1e-6 * maxmpsfn)[0])] * gdat.anglfact]
                    gdat.limspsfn[i][m] = [maxmpsfn * 1e-6, maxmpsfn]
           
        if gdat.commboolelempsfnanyy and gdat.fittmaxmnumbparaelem > 0:
            gdat.exprfwhm = 2. * retr_psfnwdth(gdat, gdat.psfnexpr, 0.5)
            
        gdat.minmgang = 1e-3 * np.sqrt(2.) * gdat.maxmgangdata
        gdat.maxmgang = np.sqrt(2.) * gdat.maxmgangdata
        
        # define minima and maxima for reference-only features or features derived from them
        for l in gdat.fittindxpopl:
            for strgfeat in gdat.fittlistnameparaelemextr[l]:
                # when the reference elements are from the true metamodel, define element feature limits based on element feature limits of the true metamodel
                #if gdat.typedata == 'mock':
                #    setattr(gdat, 'minm' + strgfeat, getattr(gdat, 'trueminm' + strgfeat[:-4]))
                #    setattr(gdat, 'maxm' + strgfeat, getattr(gdat, 'truemaxm' + strgfeat[:-4]))
                if strgfeat[:-4] == 'etag':
                    continue
                setattr(gdat, 'minm' + strgfeat, getattr(gdat, 'minm' + strgfeat[:-4]))
                setattr(gdat, 'maxm' + strgfeat, getattr(gdat, 'maxm' + strgfeat[:-4]))
        
        # try to pass true metamodel minima and maxima to common minima and maxima when that feature does not exist in the fitting metamodel
        if gdat.typedata == 'mock':
            for q in gdat.indxrefr:
                for strgfeat in gdat.truelistnameparaelemgenr[q]:
                    booltemp = False
                    for l in gdat.fittindxpopl:
                        if strgfeat in gdat.fittlistnameparaelem[l]:
                            booltemp = True
                    if not booltemp:
                        try:
                            setattr(gdat, 'minm' + strgfeat + gdat.listnamerefr[q], getattr(gdat, 'trueminm' + strgfeat))
                            setattr(gdat, 'maxm' + strgfeat + gdat.listnamerefr[q], getattr(gdat, 'truemaxm' + strgfeat))
                        except:
                            pass

        # element features
        ## plot limits for element parameters
        # define minima and maxima
        for namevarb in gdat.fittlistnameparaelemtotl + list(gdat.fittlistnameparabase):
            for strglimt in ['minm', 'maxm']:
                
                if namevarb[:-4] == 'etag':
                    continue

                if not hasattr(gdat, 'fitt' + strglimt + namevarb):
                    try:
                        setattr(gdat, 'fitt' + strglimt + namevarb, getattr(gdat, strglimt + namevarb))
                    except:
                        pass
                try:
                    limt = getattr(gdat, strglimt + namevarb)
                except:
                    try:
                        if strglimt == 'minm':
                            limt = np.minimum(getattr(gdat, 'fittminm' + namevarb), getattr(gdat, 'trueminm' + namevarb))
                        else:
                            limt = np.maximum(getattr(gdat, 'fittmaxm' + namevarb), getattr(gdat, 'truemaxm' + namevarb))
                    except:
                        try:
                            limt = getattr(gdat, 'fitt' + strglimt + namevarb)
                        except:
                            pass
                    try:
                        setattr(gdat, strglimt + namevarb, limt)
                    except:
                        pass
        # temp
        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            listnameparaelemprio = getattr(gdat, strgmodl + 'listnameparaelemprio')
            liststrgpdfnprio = getattr(gdat, strgmodl + 'liststrgpdfnprio')
            for l in indxpopl:
                for strgfeat, strgpdfn in zip(listnameparaelemprio[l], liststrgpdfnprio[l]):
                    if strgpdfn == 'self' or strgpdfn == 'logt':
                        minm = getattr(gdat, strgmodl + 'minm' + strgfeat)
                        maxm = getattr(gdat, strgmodl + 'maxm' + strgfeat)
                        if strgpdfn == 'self':
                            fact = maxm - minm
                        if strgpdfn == 'logt':
                            fact = log(maxm / minm)
                        setattr(gdat, strgmodl + 'fact' + strgfeat, fact)
        
        ## reference spectra
        if gdat.listprefsbrtlabl is None:
            if gdat.typeexpr == 'chan' and gdat.numbpixlfull != 1:
                gdat.listprefsbrtener = [[[] for k in range(3)]]
                gdat.listprefsbrtsbrt = [[[] for k in range(3)]]
                gdat.listprefsbrtlabl = ['Moretti+(2012)']
                gdat.listprefsbrttype = ['shad']
                
                for k, strgextn in enumerate(['', '_lower', '_higher']):
                    path = gdat.pathinpt + 'Moretti2012%s.csv' % strgextn
                    enerrefrplot = np.loadtxt(path, delimiter=',')[:, 0]
                    sbrtrefrplot = np.loadtxt(path, delimiter=',')[:, 1] / gdat.factergskevv / enerrefrplot**2 * (180. / np.pi)**2
                    gdat.listprefsbrtener[0][k] = enerrefrplot
                    gdat.listprefsbrtsbrt[0][k] = sbrtrefrplot

        # temp
        if gdat.numbener > 1:
            if gdat.enerpivt == 0.:
                raise Exception('Pivot energy cannot be zero.')
            #if gdat.typeexpr != 'fire':
            #    gdat.enerexpcfact = gdat.enerpivt - gdat.meanener
            #if maxmnumbparaelem > 0 and gdat.numbener > 1:
            #    minmsinddistmeanpop0 = getattr(gdat, strgmodl + 'minmsinddistmeanpop0')
            #    factspecener = (gdat.meanener / gdat.enerpivt)**(-np.sqrt(np.amin(minmsinddistmeanpop0) * np.amax(maxmsinddistmeanpop0)))
        else:
            pass
            #gdat.factspecener = np.array([1.])

        # temp -- this assumes square ROI
        if gdat.numbpixlfull > 1:
            gdat.frambndrmodl = gdat.maxmlgaldata * gdat.anglfact
        
        if gdat.fittboollenshost or gdat.typedata == 'mock' and gdat.trueboollenshost:
            
            if gdat.typesers == 'intp':
                # construct pixel-convolved Sersic surface brightness template
                gdat.factsersusam = 10
                maxmlgal = 4. * np.sqrt(2.) * gdat.maxmlgal
                gdat.numblgalsers = int(np.ceil(maxmlgal / gdat.sizepixl))
                gdat.numblgalsersusam = (1 + gdat.numblgalsers) * gdat.factsersusam
                retr_axis(gdat, 'lgalsers', 0., maxmlgal, gdat.numblgalsers)
                retr_axis(gdat, 'lgalsersusam', -gdat.sizepixl / 2., maxmlgal + gdat.sizepixl, gdat.numblgalsersusam)
                retr_axis(gdat, 'bgalsersusam', -gdat.sizepixl / 2., gdat.sizepixl / 2., gdat.factsersusam)
                
                gdat.numbhalfsers = 20
                gdat.numbindxsers = 20
                    
                retr_axis(gdat, 'halfsers', gdat.sizepixl, 5. / gdat.anglfact, gdat.numbhalfsers)
                retr_axis(gdat, 'indxsers', 0.5, 10., gdat.numbindxsers)
                
                gdat.binslgalsersusammesh, gdat.binsbgalsersusammesh = np.meshgrid(gdat.binslgalsersusam, gdat.binsbgalsersusam, indexing='ij')
                gdat.binsradisersusam = np.sqrt(gdat.binslgalsersusammesh**2 + gdat.binsbgalsersusammesh**2)
                 
                gdat.sersprofcntr = np.empty((gdat.numblgalsers + 1, gdat.numbhalfsers + 1, gdat.numbindxsers + 1))
                gdat.sersprof = np.empty((gdat.numblgalsers + 1, gdat.numbhalfsers + 1, gdat.numbindxsers + 1))
                
                for n in range(gdat.numbindxsers + 1):
                    for k in range(gdat.numbhalfsers + 1):
                        
                        profusam = retr_sbrtsersnorm(gdat.binsradisersusam, gdat.binshalfsers[k], indxsers=gdat.binsindxsers[n])
        
                        ## take the pixel average
                        indxbgallowr = gdat.factsersusam * (gdat.numblgalsers + 1) / 2
                        indxbgaluppr = gdat.factsersusam * (gdat.numblgalsers + 3) / 2
                        for a in range(gdat.numblgalsers):
                            indxlgallowr = gdat.factsersusam * a
                            indxlgaluppr = gdat.factsersusam * (a + 1) + 1
                            gdat.sersprofcntr[a, k, n] = profusam[(indxlgallowr+indxlgaluppr)/2, 0]
                            gdat.sersprof[a, k, n] = np.mean(profusam[indxlgallowr:indxlgaluppr, :])
                
                temp, indx = unique(gdat.binslgalsers, return_index=True)
                gdat.binslgalsers = gdat.binslgalsers[indx]
                gdat.sersprof = gdat.sersprof[indx, :, :]
                gdat.sersprofcntr = gdat.sersprofcntr[indx, :, :]
        
                indx = np.argsort(gdat.binslgalsers)
                gdat.binslgalsers = gdat.binslgalsers[indx]
                gdat.sersprof = gdat.sersprof[indx, :, :]
                gdat.sersprofcntr = gdat.sersprofcntr[indx, :, :]

        gdatdictcopy = deepcopy(gdat.__dict__)
        for strg, valu in gdatdictcopy.items():
            if strg.startswith('cmap') and strg[4:] != 'cntpdata' and strg[4:] != 'cntpresi' and strg[4:] != 'cntpmodl':
                retr_ticklabl(gdat, strg[4:])
                
        #liststrgcbar = ['llikmaps', 'perc', 'percresi', 'expo', 'lpdfspatpriointp', 'conv', 'magn', 'deflcomp', 'resiconvelem', 'resimagn']
        #for strgcbar in liststrgcbar:
        #    retr_ticklabl(gdat, strgcbar)
        
        # temp
        #for strgmodl in gdat.liststrgmodl:
        #    for namesele in gdat.listnamesele:
        #        for namefeat in gdat.listnamefeatsele:
        #            for strglimt in gdat.liststrglimt:
        #                try:
        #                    getattr(gdat, strglimt + namefeat + namesele)
        #                except:
        #                    setattr(gdat, strglimt + namefeat + namesele, getattr(gdat, strglimt + namefeat))

        # construct the bins for element features
        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            maxmnumbelempopl = getattr(gdat, strgmodl + 'maxmnumbelempopl')
            
            # list of names for element parameters, concatenated across all populations
            listnameparaelemtotl = getattr(gdat, strgmodl + 'listnameparaelemtotl')
            
            listnameparaelempriototl = getattr(gdat, strgmodl + 'listnameparaelempriototl')
            for l in indxpopl:
                if maxmnumbelempopl[l] > 0:
                    for strgfeat in listnameparaelemtotl:
                        if strgfeat[:-4] == 'etag':
                            continue
                        scal = getattr(gdat, 'scal' + strgfeat + 'plot')
                        try:
                            maxm = getattr(gdat, 'maxm' + strgfeat)
                            minm = getattr(gdat, 'minm' + strgfeat)
                        except:
                            maxm = getattr(gdat, strgmodl + 'maxm' + strgfeat)
                            minm = getattr(gdat, strgmodl + 'minm' + strgfeat)
                        retr_axis(gdat, strgfeat, minm, maxm, gdat.numbbinsplot, scal=scal)
                        if strgfeat in listnameparaelempriototl:
                            maxm = getattr(gdat, strgmodl + 'maxm' + strgfeat)
                            minm = getattr(gdat, strgmodl + 'minm' + strgfeat)
                            retr_axis(gdat, strgfeat + 'prio', minm, maxm, gdat.numbbinsplotprio, scal=scal, strginit=strgmodl)

        gdat.numbbinspdfn = 50
                
        # scalar variable setup continued
        for strgbins in ['lowr', 'higr']:
            for strgecom in ['dfnc', 'dfncsubt']:
                setattr(gdat, 'scalhistcntp' + strgbins + strgecom + 'en00evt0', 'self')
                setattr(gdat, 'minmhistcntp' + strgbins + strgecom + 'en00evt0', 0.)
                setattr(gdat, 'maxmhistcntp' + strgbins + strgecom + 'en00evt0', gdat.numbpixl)
                setattr(gdat, 'facthistcntp' + strgbins + strgecom + 'en00evt0', 1.)
        for i in gdat.indxener:
            setattr(gdat, 'scalfracsdenmeandarkdfncsubten%02d' % i, 'self')
            setattr(gdat, 'minmfracsdenmeandarkdfncsubten%02d' % i, 0.)
            setattr(gdat, 'maxmfracsdenmeandarkdfncsubten%02d' % i, 1.)
            setattr(gdat, 'factfracsdenmeandarkdfncsubten%02d' % i, 1.)
        
        gdat.scalbooldfncsubt = 'self'
        gdat.minmbooldfncsubt = -0.5
        gdat.maxmbooldfncsubt = 1.5
        gdat.factbooldfncsubt = 1.

        #sys.stdout = logg(gdat)
        #gdat.log.close()

        if gdat.typeverb > 0:
            sizetotl = 0.
            for root, dirs, listfile in os.walk(gdat.pathoutp):
                for thisfile in listfile:
                    sizetotl += os.path.getsize(root + '/' + thisfile) / 2**30
            if sizetotl > 10.:
                print('Warning: PCAT data path size is %d GB' % sizetotl)

        if gdat.typedata == 'inpt':
        
            ## rotate element coordinates to the ROI center
            if gdat.typepixl == 'heal' and (gdat.lgalcntr != 0. or gdat.bgalcntr != 0.):
                for q in gdat.indxrefr:
                    for l in gdat.fittindxpopl:
                        rttr = hp.rotator.Rotator(rot=[rad2deg(gdat.lgalcntr), rad2deg(gdat.bgalcntr), 0.], deg=True, eulertype='ZYX')
                        gdat.refrbgal[q][0, :], gdat.refrlgal[0, :] = rttr(pi / 2. - gdat.refrbgal[0, :], gdat.refrlgal[0, :])
                        gdat.refrbgal[q][0, :] = pi / 2. - gdat.refrbgal[0, :]

            ## assign zero to nonspecified uncertainties for the reference element features
            for q in gdat.indxrefr:
                for strgfeat in gdat.refrlistnameparaelem[q]:
                    if strgfeat == 'gang' or strgfeat == 'aang':
                        continue
                    if strgfeat == 'etag':
                        continue
                    refrfeat = getattr(gdat, 'refr' + strgfeat)
                    if refrfeat[q].ndim == 1:
                        refrfeat[q] = np.tile(refrfeat[q], (3, 1)) 
            
        # temp
        #if gdat.refrnumbelem > 0:
        #    gdat.refrfluxbrgt, gdat.refrfluxbrgtassc = retr_fluxbrgt(gdat, gdat.refrlgal, gdat.refrbgal, gdat.refrflux[0, :])
        
        print('gdat.liketype')
        print(gdat.liketype)

        print('Data settings')
        print('gdat.numbener')
        print(gdat.numbener)
        print('gdat.numbevtt')
        print(gdat.numbevtt)

        print('Model settings')
        print('gdat.fittnumbpopl')
        print(gdat.fittnumbpopl)
        print('gdat.fittnumbparabase')
        print(gdat.fittnumbparabase)
        
        # generate true data
        if gdat.typedata == 'mock':
            
            if gdat.typeverb > 0:
                print('Generating mock data...')

            if gdat.seedtype == 'rand':
                np.random.seed()
            else:
                if gdat.typeverb > 0:
                    print('Setting the seed for the RNG to %d...' % gdat.seedtype)
                np.random.seed(gdat.seedtype)
        
        if gdat.typedata == 'mock':
            ## unit sample vector
            gdat.truesampunit = np.random.rand(gdat.truemaxmnumbpara)
            gdat.truesamp = np.zeros(gdat.truemaxmnumbpara)
            
            if gdat.truemaxmnumbparaelem > 0:
                for l in gdat.trueindxpopl:
                    gdat.truesampunit[gdat.trueindxparabasenumbelem[l]] = gdat.truenumbelem[l]
                #gdat.truesamp[gdat.trueindxparabasemeanelem] = np.mean(gdat.truenumbelem, axis=0)
            
                gdat.trueindxelemfull = [[] for l in gdat.trueindxpopl]
                for l in gdat.trueindxpopl:
                    gdat.trueindxelemfull[l] = list(range(gdat.truenumbelem[l]))
                gdat.trueindxsampcomp = retr_indxparacomp(gdat, gdat.trueindxelemfull, 'true')
            else:
                gdat.trueindxsampcomp = None
                gdat.trueindxelemfull = []

            if gdat.truemaxmnumbparaelem > 0:
                if gdat.seedelemtype is not None:
                    if gdat.seedelemtype == 'rand':
                        np.random.seed()
                    else:
                        np.random.seed(gdat.seedelemtype)
                    gdat.truesampunit[gdat.indxparatrap] = np.random.rand(gdat.truemaxmnumbparaelem)
            
            gdat.truesamp = icdf_samp(gdat, 'true', gdat.truesampunit, gdat.trueindxsampcomp)

            for k in gdat.trueindxpara:
                
                if gdat.truemaxmnumbparaelem > 0 and (k in gdat.trueindxparabasenumbelem or \
                                            gdat.truetypemodltran == 'pois' and k in gdat.trueindxparabasemeanelem):
                        continue
        
                # assume the true PSF
                if gdat.truetypeevalpsfn != 'none' and gdat.numbpixl > 1 and k in gdat.trueindxparabasepsfp:
                    gdat.truesamp[k] = gdat.psfpexpr[k-gdat.trueindxparabasepsfp[0]]
                else:
                    ## read input mock model parameters
                    try:
                        # impose user-defined true parameter
                        gdat.truesamp[k] = getattr(gdat, 'true' + gdat.truenamepara[k])
                    except:
                        pass
        
            if gdat.typeverb > 1:
                show_samp(gdat, None, strgmodl='true')

        for strgmodl in gdat.liststrgmodl:
            indxpopl = getattr(gdat, strgmodl + 'indxpopl')
            for l in indxpopl:
                listnameparaelemmodu = getattr(gdat, strgmodl + 'listnameparaelemmodu')
                liststrgpdfnmodu = getattr(gdat, strgmodl + 'liststrgpdfnmodu')
                for strgfeat, strgpdfn in zip(listnameparaelemmodu[l], liststrgpdfnmodu[l]):
                    if strgpdfn == 'tmpl':
                        if gdat.lgalprio is None or gdat.bgalprio is None:
                            gdat.lgalprio = np.concatenate((gdat.truelgal))
                            gdat.bgalprio = np.concatenate((gdat.truebgal))
                        gdat.numbspatprio = gdat.lgalprio.size
        
                        # spatial template for the catalog prior
                        # temp -- this should move outside the if
                        gdat.pdfnspatpriotemp = np.zeros((gdat.numbsidecart + 1, gdat.numbsidecart + 1))
                        for k in range(gdat.numbspatprio):
                            gdat.pdfnspatpriotemp[:] += 1. / np.sqrt(2. * np.pi) / gdat.stdvspatprio * \
                                                                exp(-0.5 * (gdat.binslgalcartmesh - gdat.lgalprio[k])**2 / gdat.stdvspatprio**2) * \
                                                                exp(-0.5 * (gdat.binsbgalcartmesh - gdat.bgalprio[k])**2 / gdat.stdvspatprio**2)
                        gdat.pdfnspatpriotemp /= np.amax(gdat.pdfnspatpriotemp)
        
        if gdat.typedata == 'inpt':

            # rotate reference elements to the spatial coordinate system of PCAT
            # temp -- this does not rotate the uncertainties!

            if gdat.typeverb > 0:
                print('Rotating the reference elements...')
            for q in gdat.indxrefr:
                # temp -- this should depend on q
                if len(gdat.listpathwcss) > 0:
                    listhdun = narr_fits_open(gdat.listpathwcss)
                    wcso = ap.wcs.WCS(listhdun[0].header)
                    skycobjt = ap.coordinates.SkyCoord("galactic", l=gdat.refrlgal[q][0, :] * 180. / pi, b=gdat.refrbgal[q][0, :] * 180. / pi, unit='deg')
                    rasc = skycobjt.fk5.ra.degree
                    decl = skycobjt.fk5.dec.degree
                    lgal, bgal = wcso.wcs_world2pix(rasc, decl, 0)
                    lgal -= gdat.numbpixllgalshft + gdat.numbsidecarthalf
                    bgal -= gdat.numbpixlbgalshft + gdat.numbsidecarthalf
                    lgal *= gdat.sizepixl
                    bgal *= gdat.sizepixl
                    gdat.refrlgal[q][0, :] = bgal
                    gdat.refrbgal[q][0, :] = lgal

            ## preprocess reference element features
            for q in gdat.indxrefr:
                # temp -- this should depend on q
                # temp -- this does not properly calculate uncertainties
                gdat.refrgang[q] = np.zeros((3, gdat.refrlgal[q].shape[1]))
                gdat.refraang[q] = np.zeros((3, gdat.refrlgal[q].shape[1]))
                gdat.refrgang[q][:, :] = retr_gang(gdat.refrlgal[q][0, :], gdat.refrbgal[q][0, :])[None, :]
                gdat.refraang[q][:, :] = retr_aang(gdat.refrlgal[q][0, :], gdat.refrbgal[q][0, :])[None, :]

            # save all reference element features
            for strgfeat in gdat.refrlistnameparaelemtotl:
                refrfeattotl = [[] for q in gdat.indxrefr]
                for q in gdat.indxrefr:
                    for strgfeat in gdat.refrlistnameparaelem[q]:
                        refrfeat = getattr(gdat, 'refr' + strgfeat)
                        for l in gdat.fittindxpopl:
                            if len(refrfeat[q]) > 0:
                                refrfeattotl[q] = refrfeat[q]
                setattr(gdat, 'refr' + strgfeat + 'totl', refrfeattotl)
            
            # find the reference elements inside the ROI
            gdat.indxrefrpntsrofi = [[] for q in gdat.indxrefr]
            for q in gdat.indxrefr:
                gdat.indxrefrpntsrofi[q] = np.where((np.fabs(gdat.refrlgal[q][0, :]) < gdat.maxmgangdata) & \
                                                                        (np.fabs(gdat.refrbgal[q][0, :]) < gdat.maxmgangdata))[0]
            for strgfeat in gdat.refrlistnameparaelemtotl:
                refrfeat = getattr(gdat, 'refr' + strgfeat)
                refrfeatrofi = [[] for q in gdat.indxrefr]
                for q in gdat.indxrefr:
                    if len(refrfeat[q]) > 0:
                        refrfeatrofi[q] = refrfeat[q][..., gdat.indxrefrpntsrofi[q]]
                setattr(gdat, 'refr' + strgfeat, refrfeatrofi)
            
            # temp -- gdat.refrnumbelem is defined twice, one before and one after the filter. The initial definition is needed for strgfeat definitions.
            gdat.refrnumbelem = [[] for q in gdat.indxrefr]
            gdat.refrnumbelemtotl = 0
            for q in gdat.indxrefr:
                gdat.refrnumbelem[q] = 0
                gdat.refrnumbelem[q] = gdat.refrlgal[q].shape[1]
                gdat.refrnumbelempopl[q] = np.sum(gdat.refrnumbelem[q])
                gdat.refrnumbelemtotl += np.sum(gdat.refrnumbelem[q]) 
            
            ## check that all reference element features are finite
            for q in gdat.indxrefr:
                for strgfeat in gdat.refrlistnameparaelem[q]:
                    if strgfeat == 'etag':
                        continue
                    refrfeat = getattr(gdat, 'refr' + strgfeat)
                    if len(refrfeat[q]) > 0:
                        indxbadd = np.where(np.logical_not(np.isfinite(refrfeat[q])))
                        if indxbadd[0].size > 0:
                            refrfeat[q][indxbadd] = 0.
                            if gdat.typeverb > 0:
                                print('Warning: Provided reference element feature is not finite. Defaulting to 0...')
                        
                        if refrfeat[q].size == 0:
                            print('Warning! A reference element feature has length zero!')
                            print('strgfeat')
                            print(strgfeat)
                        else:
                            if np.amin(refrfeat[q]) == 0. and np.amax(refrfeat[q]) == 0.:
                                print('Warning! A reference element feature is all np.zeros!')
                                raise Exception('')
            
            ## element feature indices ordered with respect to the amplitude variable
            refrfeatsort = [[] for q in gdat.indxrefr]
            if not (gdat.typedata == 'mock' and gdat.truemaxmnumbparaelem == 0):
                for q in gdat.indxrefr:
                    refrparaelemgenrampl = getattr(gdat, 'refr' + gdat.refrlistnameparaelemgenrampl[q])
                    if len(refrparaelemgenrampl[q]) > 0:
                        indxelem = np.argsort(refrparaelemgenrampl[q][0, :])[::-1]
                        for strgfeat in gdat.refrlistnameparaelem[q]:
                            refrfeat = getattr(gdat, 'refr' + strgfeat)
                            if len(refrfeat[q]) > 0:
                                refrfeatsort[q] = refrfeat[q][..., indxelem]
                setattr(gdat, 'refr' + strgfeat, refrfeatsort)
            
            # bin reference element features
            for q0 in gdat.indxrefr:
                for strgfeatfrst in gdat.refrlistnameparaelem[q0]:
                    if strgfeatfrst.startswith('etag'):
                        continue
                    refrfeatfrst = getattr(gdat, 'refr' + strgfeatfrst)
                    if len(refrfeatfrst[q0]) > 0:
                        binsfrst = getattr(gdat, 'bins' + strgfeatfrst)
                        hist = np.histogram(refrfeatfrst[q0][0, :], binsfrst)[0]
                        setattr(gdat, 'refrhist' + strgfeatfrst + 'pop%d' % q0, hist)
                        for strgfeatseco in gdat.refrlistnameparaelem[q0]:
                            if strgfeatseco.startswith('etag'):
                                continue
                            refrfeatseco = getattr(gdat, 'refr' + strgfeatseco)
                            
                            strgfeattdim = strgfeatfrst + strgfeatseco + 'pop%d' % q0
                            
                            if not checstrgfeat(strgfeatfrst, strgfeatseco):
                                continue
                            
                            if len(refrfeatseco[q0]) > 0:
                                binsseco = getattr(gdat, 'bins' + strgfeatseco)
                                hist = np.histogram2d(refrfeatfrst[q0][0, :], refrfeatseco[q0][0, :], bins=(binsfrst, binsseco))[0]
                                setattr(gdat, 'refrhist' + strgfeattdim, hist)
            
        if gdat.fittmaxmnumbparaelem > 0:
            # plot settings
            ## upper limit of histograms
            if gdat.limtydathistfeat is None:
                gdat.limtydathistfeat = [0.5, max(100., 10**np.ceil(np.log10(gdat.refrnumbelemtotl)))]
                #gdat.limtydathistfeat = [0.5, max(100., 10**np.ceil(np.log10(gdat.fittmaxmnumbelemtotl)))]

        # initial plots
        if gdat.makeplot and gdat.makeplotinit:
            # problem-specific plots
            if gdat.makeplotintr:
                plot_intr(gdat)
                #plot_pert()
                #plot_king(gdat)
                plot_lens(gdat)
                #plot_3fgl_thrs(gdat)
                #if gdat.typeexpr == 'ferm':
                #    plot_fgl3(gdat)
        
        if gdat.typedata == 'mock':
            if boollenshost:
                proc_samp(gdat, None, 'this', 'true', raww=True, boolinit=True)
            proc_samp(gdat, None, 'this', 'true', boolinit=True)
        
        if not gdat.boolsqzeexpo and np.amax(gdat.cntpdata) < 1.:
            raise Exception('Data counts per pixel is less than 1.')
        
        # check the data
        if (np.fabs(gdat.cntpdata - np.round(gdat.cntpdata)) > 1e-3).any():
            raise Exception('')
        if np.amin(gdat.cntpdata) < 0.:
            raise Exception('')
    
        # list of variables for which the posterior is calculated at each sweep
        gdat.liststrgvarbarryswep = ['memoresi', 'accpprob', 'boolpropfilt', 'boolpropaccp', 'indxproptype', 'amplpert']
        for namechro in gdat.listnamechro:
            gdat.liststrgvarbarryswep += ['chro' + namechro]

        gdat.liststrgvarbarryswep += ['ltrp']
        
        if gdat.probtran > 0.:
            for l in gdat.fittindxpopl:
                gdat.liststrgvarbarryswep += ['auxiparapop%d' % l]
        
        gdat.liststrgvarbarryswep += ['ljcb']
    
        # write the numpy RNG state to file
        with narr_open(gdat.pathoutprtag + 'stat.p', 'wb') as thisfile:
        	cPickle.dump(np.random.get_state(), thisfile)
        
        # process lock for simultaneous plotting
        lock = mp.Manager().Lock()
        
        if gdat.typeverb > 0:
            print('Writing the global state to the disc before spawning workers...')
        path = gdat.pathoutprtag + 'gdatinit'
        writfile(gdat, path) 
        gdat.filestat = narr_open(gdat.pathoutprtag + 'stat.txt', 'w')
        gdat.filestat.write('gdatinit written.\n')
        gdat.filestat.close()
        
        # exit before running the sampler
        if gdat.boolmockonly:
            if gdat.typeverb > 0:
                print('Mock dataset is generated. Quitting...')
            return gdat.rtag
        
        # perform an initial run, sampling from the prior
        if gdat.checprio:
            
            if gdat.typeverb > 0:
                print('Sampling from the prior...')
            
            ## perform sampling
            worksamp(gdat, lock, strgpdfn='prio')
            
            ## post process the samples
            proc_finl(gdat=gdat, strgpdfn='prio')
            
        if gdat.typeverb > 0:
            print('Sampling from the posterior...')
        
        # run the sampler
        worksamp(gdat, lock)
    
    # post process the samples
    proc_finl(gdat=gdat)
    
    # make animations
    if gdat.makeanim and gdat.numbplotfram > 1:
        proc_anim(gdat.rtag)

    if gdat.typeverb > 0:
        print('The output is at ' + gdat.pathoutprtag)
        if gdat.makeplot:
            print('The plots are at ' + gdat.pathplotrtag)
        print('PCAT has run successfully. Returning to the OS...')

    return gdat.rtag


def initarry( \
             dictvarbvari, \
             dictvarb, \
             listnamecnfgextn, \
             forcneww=False, \
             forcprev=False, \
             strgpara=False, \
             
             # Boolean flag to execute the runs in parallel
             boolexecpara=True, \
             
             strgcnfgextnexec=None, \
             listnamevarbcomp=[], \
             listscalvarbcomp=[], \
             listlablvarbcomp=[], \
             listtypevarbcomp=[], \
             listpdfnvarbcomp=[], \
             listgdatvarbcomp=[], \
             
             # parameter name, axis label, tick values and scaling of the input variable changed across PCAT runs
             namexaxivari=None, \
             lablxaxivari=None, \
             tickxaxivari=None, \
             scalxaxivari=None, \
            ):
    
    print('Running PCAT in array mode...')
    
    numbiter = len(dictvarbvari)
    indxiter = np.arange(numbiter) 
    
    cntrcomp = 0
    
    if boolexecpara:
        cntrproc = 0

    listrtag = []
    listpridchld = []
    for k, strgcnfgextn in enumerate(listnamecnfgextn):
        
        if strgcnfgextnexec is not None:
            if strgcnfgextn != strgcnfgextnexec:
                continue
        
        strgcnfg = inspect.stack()[1][3] + '_' + strgcnfgextn
    
        #for proc in psutil.process_iter():
        #    print proc.open_files()
            
        #proc = psutil.Process(os.getpid())
        #print proc.get_open_files()

        dictvarbtemp = deepcopy(dictvarb)
        for strgvarb, valu in dictvarbvari[strgcnfgextn].items():
            dictvarbtemp[strgvarb] = valu
        dictvarbtemp['strgcnfg'] = strgcnfg
    
        listrtagprev = retr_listrtagprev(strgcnfg)
        cntrcomp += 1

        if (not forcneww and strgcnfgextnexec is None or forcprev and strgcnfgextnexec is not None) and len(listrtagprev) > 0:
            print('Found at least one previous run with the configuration %s' % strgcnfg)
            print('Skipping...')
            listrtag.append(listrtagprev[-1])
        else:
            if len(listrtagprev) > 0:
                print('Found at least one previous run. But, repeating the run anways...')
            else:
                print('Did not find any previous run.')
            if boolexecpara and strgcnfgextnexec is None:
                cntrproc += 1
                prid = os.fork()
                if prid > 0:
                    listpridchld.append(prid)
                else:
                    print('Forking a child process to run the configuration extension...')
                    rtag = init(**dictvarbtemp)
                    os._exit(0)
            else:
                print('Calling the main PCAT function without forking a child...')
                listrtag.append(init(**dictvarbtemp))
    
    if boolexecpara and strgcnfgextnexec is None:
        for prid in listpridchld:
            os.waitpid(prid, 0)
        if cntrproc > 0:
            print('Exiting before comparion plots because of parallel execution...')
            return
    
    if cntrcomp == 0:
        print('Found no runs...')

    print('Final-processing run outputs...')
    for rtag in listrtag:
        print(rtag)
        proc_finl(rtag=rtag, strgpdfn='post')
        proc_anim(rtag)
    
    strgtimestmp = tdpy.util.retr_strgtimestmp()
    
    if strgcnfgextnexec is not None or namexaxivari is None: 
        return
    
    print('Making plots to compare the output of different PCAT runs...')
     
    if 'boolmockonly' in dictvarb and dictvarb['boolmockonly']:
        listgdat = retr_listgdat(listrtag, typegdat='init')
    else:
        listgdat = retr_listgdat(listrtag)
    
    numbgdat = len(listgdat)

    for namevarbscal in listgdat[0].listnamevarbscal:
        booltemp = True
        for k in range(1, numbgdat - 1):
            if not namevarbscal in listgdat[k].listnamevarbscal:
                booltemp = False
        if booltemp:
            if namevarbscal in listnamevarbcomp:
                raise Exception('')
            listnamevarbcomp += [namevarbscal]
            listscalvarbcomp += [getattr(listgdat[0], 'scal' + namevarbscal)]
            listlablvarbcomp += [getattr(listgdat[0], 'labl' + namevarbscal + 'totl')]
            listtypevarbcomp += ['pctl']
            listpdfnvarbcomp += ['post']
            listgdatvarbcomp += ['post']
    
    # add others to the variable list
    listnamevarbcomp += ['lliktotl', 'lliktotl', 'infopost', 'bcom', 'lliktotl', 'lliktotl', 'lliktotl', 'levipost']
    listscalvarbcomp += ['self', 'self', 'self', 'self', 'self', 'self', 'self', 'self']
    listlablvarbcomp += ['$\ln P(D|M_{min})$', '$\ln P(D|M_{max})$', '$D_{KL}$', '$\eta_B$', '$\sigma_{P(D|M)}$', r'$\gamma_{P(D|M)}$', \
                                                                                                                    r'$\kappa_{P(D|M)}$', '$\ln P_H(D)$']
    listtypevarbcomp += ['minm', 'maxm', '', '', 'stdv', 'skew', 'kurt', '']
    listpdfnvarbcomp += ['post', 'post', 'post', 'post', 'post', 'post', 'post', 'post']
    listgdatvarbcomp += ['post', 'post', 'post', 'post', 'post', 'post', 'post', 'post']
    
    arrytemp = np.array([len(listnamevarbcomp), len(listscalvarbcomp), len(listlablvarbcomp), len(listtypevarbcomp), len(listpdfnvarbcomp), len(listgdatvarbcomp)])
    if (arrytemp - np.mean(arrytemp) != 0.).all():
        raise Exception('')

    # add log-evidence to the variable list, if prior is also sampled
    booltemp = True
    for k in range(numbgdat):
        if not listgdat[k].checprio:
            booltemp = False
    
    pathbase = '%s/imag/%s_%s/' % (os.environ["PCAT_DATA_PATH"], strgtimestmp, inspect.stack()[1][3])
    if booltemp:
        listgdatprio = retr_listgdat(listrtag, typegdat='finlprio')
        
        listnamevarbcomp += ['leviprio']
        listscalvarbcomp += ['self']
        listlablvarbcomp += ['$\ln P_{pr}(D)$']
        listtypevarbcomp += ['']
        listpdfnvarbcomp += ['prio']
        listgdatvarbcomp += ['prio']
    
    # time stamp
    strgtimestmp = tdpy.util.retr_strgtimestmp()
    
    dictoutp = dict()
    liststrgvarbtotl = []
    for (typevarbcomp, pdfnvarbcomp, namevarbcomp) in zip(listtypevarbcomp, listpdfnvarbcomp, listnamevarbcomp):
        strgtemp = typevarbcomp + pdfnvarbcomp + namevarbcomp
        liststrgvarbtotl.append(strgtemp)
        dictoutp[strgtemp] = [[] for k in range(numbiter)]
    
    for k in indxiter:
        for a, strgvarbtotl in enumerate(liststrgvarbtotl):
            if listgdatvarbcomp[a] == 'prio':
                gdattemp = listgdatprio[k]
            else:
                gdattemp = listgdat[k]
            dictoutp[strgvarbtotl][k] = getattr(gdattemp, strgvarbtotl)

    cmnd = 'mkdir -p %s' % pathbase 
    os.system(cmnd)
    
    cmnd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=%smrgd.pdf' % pathbase
    for strgvarbtotl, varboutp in dictoutp.items():
        
        figr, axis = plt.subplots(figsize=(6, 6))
        ydat = np.empty(numbiter)
        yerr = np.zeros((2, numbiter))
            
        indxlist = liststrgvarbtotl.index(strgvarbtotl)
        
        if listscalvarbcomp is None:
            scalyaxi = getattr(listgdat[0], 'scal' + listnamevarbcomp[indxlist])
        else:
            scalyaxi = listscalvarbcomp[indxlist]
        
        lablyaxi = listlablvarbcomp[indxlist]
        
        try:
            factplot = getattr(listgdat[0], 'fact' + listnamevarbcomp[indxlist] + 'plot')
        except:
            factplot = 1.
        
        try:
            if listtypevarbcomp[indxlist] == 'pctl':
                trueyaxi = getattr(listgdat[0], 'true' + listnamevarbcomp[indxlist])
            else:
                trueyaxi = getattr(listgdat[0], 'true' + listtypevarbcomp[indxlist] + listnamevarbcomp[indxlist])
        except:
            trueyaxi = None
        
        for k in indxiter:
            
            if isinstance(varboutp[k], list) or isinstance(varboutp[k], np.ndarray) and varboutp[k].ndim > 2:
                raise Exception('')
            elif isinstance(varboutp[k], float):
                ydat[k] = varboutp[k]
            else:
                if listtypevarbcomp[indxlist] != 'pctl':
                    yerr[:, k] = 0.
                if varboutp[k].ndim == 2:
                    if varboutp[k].shape[1] != 1:
                        raise Exception('varboutp format is wrong.')
                    varboutp[k] = varboutp[k][:, 0]
                    if listtypevarbcomp[indxlist] == 'pctl':
                        yerr[:, k] = getattr(listgdat[k], 'errr' + listpdfnvarbcomp[indxlist] + listnamevarbcomp[indxlist])[:, 0]
                else:
                    if listtypevarbcomp[indxlist] == 'pctl':
                        yerr[:, k] = getattr(listgdat[k], 'errr' + listpdfnvarbcomp[indxlist] + listnamevarbcomp[indxlist])
                ydat[k] = varboutp[k][0]
        
        axis.errorbar(indxiter+1., ydat * factplot, yerr=yerr * factplot, color='b', ls='', markersize=15, marker='o', lw=3)
        indxrtagyerr = np.where((yerr[0, :] > 0.) | (yerr[1, :] > 0.))[0]
        if indxrtagyerr.size > 0:
            temp, listcaps, temp = axis.errorbar(indxiter[indxrtagyerr]+1., ydat[indxrtagyerr] * factplot, yerr=yerr[:, indxrtagyerr] * factplot, \
                                                                                color='b', ls='', capsize=15, markersize=15, marker='o', lw=3)
            for caps in listcaps:
                caps.set_markeredgewidth(3)
        
        if trueyaxi is not None:
            axis.axhline(trueyaxi, ls='--', color='g')
        
        if lablxaxivari is None:
            lablxaxivari = getattr(listgdat[0], 'labl' + namexaxivari + 'totl')
        
        if scalxaxivari is None:
            scalxaxivari = getattr(listgdat[0], 'scal' + namexaxivari)
        
        axis.set_xlabel(lablxaxivari)
        axis.set_xticks(indxiter+1.)
        axis.set_xticklabels(tickxaxivari)
        
        axis.set_ylabel(lablyaxi)
        if scalyaxi == 'logt':
            axis.set_yscale('log')
        plt.tight_layout()
        
        pathfull = '%s%s_%s_%s.pdf' % (pathbase, strgtimestmp, inspect.stack()[1][3], liststrgvarbtotl[indxlist])
        print('Writing to %s...' % pathfull)
        plt.savefig(pathfull)
        plt.close(figr)
    
        cmnd += ' %s' % pathfull

    print(cmnd)
    os.system(cmnd)

    print('Making animations...')
    for rtag in listrtag:
        print('Working on %s...' % rtag)
        proc_anim(rtag=rtag)
    
    print('Compiling run plots...')
    cmnd = 'python comp_rtag.py'
    for rtag in listrtag: 
        cmnd += ' %s' % rtag
    os.system(cmnd)

    return listrtag


def retr_rtag(strgcnfg, strgnumbswep):
    
    rtag = strgcnfg + '_' + strgnumbswep
    
    return rtag


class logg(object):
    
    def __init__(self, gdat):
        self.terminal = sys.stdout
        gdat.pathstdo = gdat.pathoutprtag + 'stdo.txt'
        self.log = narr_open(gdat.pathstdo, 'a')
        pathlink = gdat.pathplotrtag + 'stdo.txt'
        os.system('ln -s %s %s' % (gdat.pathstdo, pathlink))
    
    def write(self, strg):
        self.terminal.write(strg)
        self.log.write(strg)  

    def flush(self):
        pass


def worktrac(pathoutprtag, lock, strgpdfn, indxprocwork):
	
    try:
        return work(pathoutprtag, lock, strgpdfn, indxprocwork)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def opti_hess(gdat, gdatmodi):
    
    if gdat.fittmaxmnumbparaelem > 0:
        cntr = 0
        for l in gdat.fittindxpopl:
            for k, strgcomp in enumerate(gdat.fittlistnameparaelemgenr[l]):
                gdatmodi.indxparastdp[gdat.fittnumbparabase-gdat.fittnumbpopl+cntr] = np.concatenate(gdatmodi.thisindxsampcomp[strgcomp])
                cntr += 1
    
    if gdat.fittmaxmnumbparaelem > 0:
        gdatmodi.nextindxelemfull = gdatmodi.thisindxelemfull
        gdatmodi.nextindxsampcomp = gdatmodi.thisindxsampcomp
    else:
        gdatmodi.nextindxsampcomp = None

    gdatmodi.stdpmatr = np.zeros((gdat.numbstdp, gdat.numbstdp)) 
    gdatmodi.hess = np.zeros((gdat.numbstdp, gdat.numbstdp)) 
    deltlpos = np.zeros((3, 3))
    diffpara = np.empty(gdat.numbstdp)
    for k, indxparatemp in enumerate(gdatmodi.indxparastdp):
        if len(indxparatemp) == 0:
            diffpara[k] = 0.
        else:
            diffpara[k] = min(min(np.amin(gdatmodi.thissampunit[indxparatemp]) * 0.9, np.amin(1. - gdatmodi.thissampunit[indxparatemp]) * 0.9), 1e-5)

    #gdatmodi.thissampunitsave = np.copy(gdatmodi.thissampunit)
    
    #if gdat.fittmaxmnumbparaelem > 0:
    #    gdatmodi.dictmodi = [[] for l in gdat.fittindxpopl]
    #    for l in gdat.fittindxpopl:
    #        gdatmodi.dictmodi[l] = dict()
    #        gdatmodi.dictmodi[l][gdat.fittnameparaelemgenrampl[l] + 'indv'] = gdatmodi.thissamp[gdatmodi.thisindxsampcomp[gdat.fittnameparaelemgenrampl[l]][l]]
    #        for strgcomp in gdat.fittlistnameparaelemgenr[l]:
    #            gdatmodi.dictmodi[l]['stdv' + strgcomp + 'indv'] = gdatmodi.thissamp[gdatmodi.thisindxsampcomp[strgcomp][l]]
    #if gdat.fittmaxmnumbparaelem > 0:
    #    gdatmodi.thisindxsampcompconc = np.concatenate([gdatmodi.thisindxsampcomp['comp'][l] for l in gdat.fittindxpopl])
    #if gdat.boolpropcomp:
    #    indxsamptranprop = gdatmodi.thisindxsampcompconc
    #else:
    #    indxsamptranprop = []
    
    deltlpos[1, 1] = gdatmodi.thislliktotl
    for indxstdpfrst in gdat.indxstdpprop:
        for indxstdpseco in gdat.indxstdpprop:
            
            if indxstdpfrst > indxstdpseco:
                continue
            
            if indxstdpfrst == indxstdpseco:
                
                #if gdat.fittmaxmnumbparaelem > 0:
                #    if k in gdatmodi.thisindxsampcompconc:
                #        indxtrapmoditemp = k - gdat.fittindxsampcompinit
                #        indxpoplmoditemp = np.array([np.amin(np.where(indxtrapmoditemp // gdat.fittmaxmnumbparaelempoplcumr == 0))])
                #        numbparapoplinittemp = indxtrapmoditemp - gdat.fittmaxmnumbparaelempoplcuml[indxpoplmoditemp[0]]
                #        indxelemmoditemp = [numbparapoplinittemp // gdat.fittmaxmnumbparaelemgenr[indxpoplmoditemp[0]]]
                #        indxcompmoditemp = numbparapoplinittemp % gdat.fittmaxmnumbparaelemgenr[indxpoplmoditemp[0]]
                #        strgcomp = gdat.fittlistnameparaelemgenr[indxpoplmoditemp[0]][indxcompmoditemp] 
                #        indxsampampltemp = k - indxcompmoditemp + gdat.fittindxcompampl[indxpoplmoditemp[0]]
                #        #amplfact = gdatmodi.thissamp[indxsampampltemp] / getattr(gdat, 'minm' + gdat.fittnameparaelemgenrampl[indxpoplmoditemp[0]])
                #        stdv = 1. / np.sqrt(gdatmodi.hess[indxstdpfrst, indxstdpseco])
                #        gdatmodi.stdpmatr[indxstdpfrst, indxstdpseco] += stdv
                #        gdatmodi.dictmodi[indxpoplmoditemp[0]]['stdv' + strgcomp + 'indv'][indxelemmoditemp[0]] = stdv
                #        gdatmodi.dictmodi[indxpoplmoditemp[0]][gdat.fittnameparaelemgenrampl[indxpoplmoditemp[0]] + 'indv'][indxelemmoditemp[0]] = \
                #                                                                                             gdatmodi.thissamp[indxsampampltemp]
                
                if len(gdatmodi.indxparastdp[indxstdpseco]) == 0:
                    continue
                
                for a in range(2):
                    gdatmodi.nextsampunit = np.copy(gdatmodi.thissampunit)
                    if a == 0:
                        gdatmodi.nextsampunit[gdatmodi.indxparastdp[indxstdpseco]] -= diffpara[indxstdpseco]
                    if a == 1:
                        gdatmodi.nextsampunit[gdatmodi.indxparastdp[indxstdpseco]] += diffpara[indxstdpseco]
                    
                    gdatmodi.nextsamp = icdf_samp(gdat, 'fitt', gdatmodi.nextsampunit, gdatmodi.nextindxsampcomp)
                    
                    proc_samp(gdat, gdatmodi, 'next', 'fitt')
                    if a == 0:
                        deltlpos[0, 1] = gdatmodi.nextlliktotl
                    if a == 1:
                        deltlpos[2, 1] = gdatmodi.nextlliktotl
                
                gdatmodi.hess[indxstdpfrst, indxstdpseco] = 1. / 4. / diffpara[indxstdpseco]**2 * np.fabs(deltlpos[0, 1] + \
                                                                                                        deltlpos[2, 1] - 2. * deltlpos[1, 1])
            else:
                # temp
                continue

                for a in range(4):
                    gdatmodi.thissampunit = np.copy(gdatmodi.thissampunitsave)
                    if a == 0:
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpfrst]] -= diffpara
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpseco]] -= diffpara
                    if a == 1:
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpfrst]] += diffpara
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpseco]] += diffpara
                    if a == 2:
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpfrst]] -= diffpara
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpseco]] += diffpara
                    if a == 3:
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpfrst]] += diffpara
                        gdatmodi.thissampunit[gdatmodi.indxparastdp[indxstdpseco]] -= diffpara
                    proc_samp(gdat, gdatmodi, 'this', 'fitt')
                    if a == 0:
                        deltlpos[0, 0] = gdatmodi.thislpostotl
                    if a == 1:
                        deltlpos[2, 2] = gdatmodi.thislpostotl
                    if a == 2:
                        deltlpos[1, 2] = gdatmodi.thislpostotl
                    if a == 3:
                        deltlpos[2, 1] = gdatmodi.thislpostotl
                gdatmodi.hess[indxstdpfrst, indxstdpseco] = 1. / 4. / diffpara**2 * \
                                                                                (deltlpos[2, 2] + deltlpos[0, 0] - deltlpos[1, 2] - deltlpos[2, 1])
            
            if not np.isfinite(gdatmodi.hess[indxstdpfrst, indxstdpseco]):
                raise Exception('')
            if gdat.booldiagmode and not np.isfinite(gdatmodi.nextsamp).all():
                raise Exception('')
            if gdatmodi.hess[indxstdpfrst, indxstdpseco] == 0.:
                raise Exception('')

    gdatmodi.hess[np.where(gdatmodi.hess == 0)] = 10.

    # temp
    #gdatmodi.stdpmatr = np.sqrt(linalg.inv(gdatmodi.hess))
    numbdoffefff = gdat.fittnumbparabase
    if gdat.fittmaxmnumbparaelem > 0:
        numbdoffefff += gdat.fittmaxmnumbparaelemgenr * 10
    gdatmodi.stdpmatr = np.sqrt(1. / gdatmodi.hess) / np.sqrt(numbdoffefff)
    
    if (gdatmodi.stdpmatr == 0).any():
        raise Exception('')
    
    gdatmodi.stdp = gdatmodi.stdpmatr[gdat.indxstdp, gdat.indxstdp]
    

def worksamp(gdat, lock, strgpdfn='post'): 
    
    pathorig = gdat.pathoutprtag + 'stat.txt'
    pathlink = gdat.pathplotrtag + 'stat.txt'
    os.system('ln -s %s %s' % (pathorig, pathlink))
    
    if gdat.numbproc == 1:
        worktrac(gdat.pathoutprtag, lock, strgpdfn, 0)
    else:
        if gdat.typeverb > 0:
            print('Forking the sampler...')

        # process pool
        pool = mp.Pool(gdat.numbproc)
        
        # spawn the processes
        workpart = functools.partial(worktrac, gdat.pathoutprtag, lock, strgpdfn)
        pool.map(workpart, gdat.indxproc)

        pool.close()
        pool.join()
    
    gdat.filestat = narr_open(gdat.pathoutprtag + 'stat.txt', 'a')
    gdat.filestat.write('gdatmodi%s written.\n' % strgpdfn)
    gdat.filestat.close()


def work(pathoutprtag, lock, strgpdfn, indxprocwork):
    
    print('Worker #%d' % indxprocwork)
    
    # read the initial global object, gdatinit
    path = pathoutprtag + 'gdatinit'
    gdat = readfile(path) 
    
    # define time functions
    timereal = time.time()
    timeproc = time.clock()
    
    # re-seed the random number generator for this chain
    if gdat.seedchan:
        np.random.seed(indxprocwork + 1000)

    # construct a global object to hold chain-specific variables that will be modified by the worker
    gdatmodi = tdpy.util.gdatstrt()
    
    gdat.strgpdfn = strgpdfn
    
    if gdat.strgpdfn == 'prio':
        gdat.legdsampdist = 'Prior'
    if gdat.strgpdfn == 'post':
        gdat.legdsampdist = 'Posterior'

    # construct the initial state
    if gdat.typeverb > 0:
        print('Initializing the sampler state...')
        print('inittype')
        print(gdat.inittype)
        
    ## initialization
    gdatmodi.indxprocwork = indxprocwork
    init_stat(gdat, gdatmodi)
   
    # final setup
    setpfinl(gdat) 

    proc_samp(gdat, gdatmodi, 'this', 'fitt', boolinit=True)

    print('Erasing the fitting model parameter indices...')

    # make sure fitting model parameter indices are erased
    gdatdictcopy = deepcopy(gdat.__dict__)
    for strg, valu in gdatdictcopy.items():
        if strg.startswith('fittindxparabase'):
            delattr(gdat, strg)
    # reflush gdatmodi

    print('Determining the parameter indices of the fitting model with only the floating parameters...')

    # rerun setp_indxpara with only the perturbed parameters
    setp_indxpara(gdat, 'finl')
    setp_parabase(gdat, 'finl')
    # final setup
    setpfinl(gdat, boolfittflot=True) 
    gdatmodi = tdpy.util.gdatstrt()
    gdatmodi.booldone = False
    gdatmodi.lock = lock
    gdatmodi.indxprocwork = indxprocwork
    #gdatmodi = retr_gdatmodifidu(gdat)
    
    ## initialization
    print('Rerunning initialization...')
    init_stat(gdat, gdatmodi)
    
    # list of scalar variable names
    gdatmodi.listnamevarbscal = list(gdat.fittlistnameparabase) 
    gdatmodi.listnamevarbscal += ['lliktotl']
    
    for e in gdat.fittindxsersfgrd:
        gdatmodi.listnamevarbscal += ['masshost' + strgsersfgrd + 'bein']
        for strgcalcmasssubh in gdat.liststrgcalcmasssubh:
            gdatmodi.listnamevarbscal += ['masshost' + strgsersfgrd + strgcalcmasssubh + 'bein']
    if gdat.fittmaxmnumbparaelem > 0:
        if 'lens' in gdat.fittelemtype:
            for strgcalcmasssubh in gdat.liststrgcalcmasssubh:
                gdatmodi.listnamevarbscal += ['masssubh' + strgcalcmasssubh + 'bein', 'fracsubh' + strgcalcmasssubh + 'bein'] 
    if gdat.fittmaxmnumbparaelem > 0:
        gdatmodi.listnamevarbscal += ['lpripena']
    if gdat.fittboolelemsbrtdfncanyy:
        for strgbins in ['lowr', 'higr']:
            gdatmodi.listnamevarbscal += ['histcntp%sdfncen00evt0' % strgbins]
            gdatmodi.listnamevarbscal += ['histcntp%sdfncsubten00evt0' % strgbins]
        for i in gdat.indxener:
            gdatmodi.listnamevarbscal += ['fracsdenmeandarkdfncsubten%02d' % i]
        gdatmodi.listnamevarbscal += ['booldfncsubt']
    if gdat.fittmaxmnumbparaelem > 0:
        for q in gdat.indxrefr:
            if gdat.boolasscrefr[q]:
                for l in gdat.fittindxpopl:
                    gdatmodi.listnamevarbscal += ['cmplpop%dpop%d' % (l, q)]
                    gdatmodi.listnamevarbscal += ['fdispop%dpop%d' % (q, l)]
    
    gdat.numbvarbscal = len(gdatmodi.listnamevarbscal)
    gdat.indxvarbscal = np.arange(gdat.numbvarbscal)
        
    # for each parameter in the fitting model, determine if there is a corresponding parameter in the generative model
    for k in gdat.indxvarbscal:
        try:
            temp = getattr(gdat, 'true' + gdatmodi.listnamevarbscal[k])
        except:
            temp = None
        setattr(gdat, 'corr' + gdatmodi.listnamevarbscal[k], temp)
    
    gdat.fittcorrparabase = np.empty(gdat.fittnumbparabase)
    for k in gdat.fittindxparabase:
        try:
            gdat.fittcorrparabase[k] = getattr(gdat, 'true' + gdat.fittlistnameparabase[k])
        except:
            gdat.fittcorrparabase[k] = None

    # construct bins for the scalar variables
    for namevarbscal in gdatmodi.listnamevarbscal:
        # temp
        if namevarbscal == 'lliktotl' or namevarbscal == 'lpripena':
            continue
        retr_axis_wrap(gdat, namevarbscal)

    # plotting factors for scalar variables
    for name in gdatmodi.listnamevarbscal:
        if name in gdat.fittlistnameparabase:
            indxparabasetemp = np.where(gdat.fittlistnameparabase == name)[0]
            factparabaseplot = gdat.fittfactparabaseplot[indxparabasetemp]
            setattr(gdat, 'fact' + name + 'plot', factparabaseplot)
        else:
            try:    
                getattr(gdat, 'fact' + name + 'plot')
            except:
                setattr(gdat, 'fact' + name + 'plot', 1.)

    
    # find the list of variables for which the posterior will be calculated
    if not gdat.boolmockonly:
        
        if gdat.typeverb > 1:
            show_samp(gdat, gdatmodi)
        proc_samp(gdat, gdatmodi, 'this', 'fitt')
        gdatmodi.liststrgvarbarrysamp = []
        gdatmodi.liststrgvarblistsamp = []

        for strg, valu in gdatmodi.__dict__.items():
            if strg.startswith('this') and not strg[4:] in gdat.liststrgvarbarryswep:
                if isinstance(valu, np.ndarray) or isinstance(valu, float):
                    gdatmodi.liststrgvarbarrysamp.append(strg[4:])
                elif isinstance(valu, list) and strg != 'thisindxsampcomp' and strg != 'thispsfnconv' and \
                                                                     strg != 'thistrueindxelemasscmiss' and strg != 'thistrueindxelemasschits':
                    gdatmodi.liststrgvarblistsamp.append(strg[4:])
        
        gdatmodi.liststrgvarbarry = gdatmodi.liststrgvarbarrysamp + gdat.liststrgvarbarryswep
        gdatmodi.liststrgvarbarry = gdatmodi.liststrgvarbarrysamp + gdat.liststrgvarbarryswep
        gdatmodi.liststrgchan = gdatmodi.liststrgvarbarry + ['parabase'] + gdatmodi.liststrgvarblistsamp + gdatmodi.listnamevarbscal
        
        # temp
        # get a unique list
        gdatmodi.liststrgchantemp = []
        for strgchan in gdatmodi.liststrgchan:
            if not strgchan in gdatmodi.liststrgchantemp:
                gdatmodi.liststrgchantemp.append(strgchan)
        gdatmodi.liststrgchan = gdatmodi.liststrgchantemp
        print('gdatmodi.liststrgvarbarrysamp')
        print(gdatmodi.liststrgvarbarrysamp)
        print('gdatmodi.liststrgchan')
        print(gdatmodi.liststrgchan)
        print('gdatmodi.listnamevarbscal')
        print(gdatmodi.listnamevarbscal)
        if len(set(gdatmodi.liststrgchan)) != len(gdatmodi.liststrgchan):
            print('len(gdatmodi.liststrgchan)')
            print(len(gdatmodi.liststrgchan))
            print('len(set(gdatmodi.liststrgchan))')
            print(len(set(gdatmodi.liststrgchan)))
            raise Exception('')
        #raise Exception('')

    ## sample index
    gdatmodi.cntrswep = 0
   
    if gdat.booldiagmode:
        if gdat.indxswepsave.size != gdat.numbsamp:
            raise Exception('Inappropriate number of samples.')

    # initialize the worker sampler
    
    # definitions required for the initial sample
    gdatmodi.thisboolpropfilt = True
    gdatmodi.thisboolpropaccp = True
    
    # dummy definitions required for logs
    gdatmodi.thisindxproptype = np.zeros(1, dtype=int)
    for l in gdat.fittindxpopl:
        setattr(gdatmodi, 'thisauxiparapop%d' % l, np.zeros(gdat.fittmaxmnumbparaelemgenr[l]))
    gdatmodi.thislpri = np.zeros(gdat.fittnumblpri)
    
    gdatmodi.thislpau = np.zeros(1)
    gdatmodi.thisltrp = np.zeros(1)
    gdatmodi.thisljcb = np.zeros(1)
    
    gdatmodi.thisaccpprob = np.zeros(1)
    gdatmodi.thismemoresi = np.zeros(1)
    gdatmodi.thisamplpert = np.zeros(1)
    
    # make sure the first sample derived variables are generated on gdatmodi
    proc_samp(gdat, gdatmodi, 'this', 'fitt')
    
    # log the initial state
    if gdat.typeverb > 1:
        tdpy.util.show_memo(gdatmodi, 'gdatmodi')
    
    # enter interactive mode
    if gdat.intrevalcntpmodl:
        plot_genemaps(gdat, gdatmodi, 'this', 'fitt', 'cntpmodl', 0, 0, strgcbar='cntpdata', intreval=True)
    if gdat.intrevalcntpresi:
        plot_genemaps(gdat, gdatmodi, 'this', 'fitt', 'cntpresi', 0, 0, intreval=True)

    for k, name in enumerate(gdat.listnamechro):
        setattr(gdatmodi, 'thischro' + name, 0.)
    
    gdatmodi.stdp = np.copy(gdat.stdp)
    
    # indices of parameters corresping to each proposal scale
    gdatmodi.indxparastdp = [[] for k in gdat.indxstdp]
    for k in gdat.fittindxparabase:
        if k < gdat.fittnumbpopl:
            continue
        gdatmodi.indxparastdp[k-gdat.fittnumbpopl] = [k]
    
    workdict = {}
    for strgvarb in gdatmodi.liststrgvarbarry:
        if strgvarb in gdat.liststrgvarbarryswep:
            valu = getattr(gdatmodi, 'this' + strgvarb)
            if isinstance(valu, dict):
                shap = [gdat.numbswep, len(valu.keys())]
            elif isinstance(valu, float) or isinstance(valu, bool):
                shap = [gdat.numbswep, 1]
            else:
                shap = [gdat.numbswep] + list(valu.shape)
        else:
            valu = getattr(gdatmodi, 'this' + strgvarb)
            shap = [gdat.numbsamp] + list(valu.shape)
        workdict['list' + gdat.strgpdfn + strgvarb] = np.zeros(shap)
   
    for strgvarb in gdatmodi.liststrgvarblistsamp:
        workdict['list' + gdat.strgpdfn + strgvarb] = []
    
    ## saved state of the sample index used for logging progress status
    gdatmodi.percswepsave = -1
   
    # store the initial sample as the best fit sample
    gdatmodi.maxmllikswep = np.sum(gdatmodi.thisllik)
    gdatmodi.indxswepmaxmllik = -1 
    gdatmodi.sampmaxmllik = np.copy(gdatmodi.thissamp)
    
    if gdat.typeverb > 0:
        print('Sampling...')
        print('gdat.stdp')
        for k in gdat.indxstdp:
            print('%04d %s %g' % (k, gdat.namestdp[k], gdat.stdp[k]))

    gdatmodi.thisstdp = np.copy(gdat.stdp)

    gdatmodi.optidone = False 
    
    while gdatmodi.cntrswep < gdat.numbswep:
        
        initchro(gdat, gdatmodi, 'totl')
        
        # Boolean flag to indicate burn-in phase
        gdatmodi.boolburn = gdatmodi.cntrswep < gdat.numbburn
        
        # temp
        if gdat.typeopti == 'hess' and gdatmodi.cntrswep % gdat.numbstdp * 4 == 0 and gdatmodi.cntrswep < gdat.numbburn:
            if gdat.typeverb > 0:
                print('Optimizing proposal scale...')
            opti_hess(gdat, gdatmodi)
            
            if (gdatmodi.stdpmatr[gdat.indxstdp, gdat.indxstdp] < 0.5).all():
                gdatmodi.optidone = True
           
        if gdat.typeopti == 'hess' and gdatmodi.cntrswep == gdat.numbburn:
            path = gdat.pathoutprtag + 'opti.h5'
            if gdat.typeverb > 0:
                print('Writing the estimated covariance matrix to %s...' % path)
            thisfile = narr_h5(path, 'w')
            thisfile.create_dataset('stdp', data=gdatmodi.stdp)
            thisfile.close()
            
            if False and gdat.makeplot:
                
                xdat = gdat.indxstdp
                ydat = gdatmodi.stdp
                
                pathopti = getattr(gdat, 'path' + gdat.strgpdfn + 'opti')
                path = pathopti + 'stdv%d.pdf' % gdatmodi.indxprocwork
                tdpy.util.plot_gene(path, xdat, ydat, scalydat='logt', \
                                lablxdat='$i_{stdp}$', lablydat=r'$\sigma$', plottype='hist', limtydat=[np.amin(ydat) / 2., 2. * np.amax(ydat)])
                
                # plot uncertainties of element parameters as a function of amplitude parameter
                if gdat.fittmaxmnumbparaelem > 0:
                    for l in gdat.fittindxpopl:
                        meanplot = getattr(gdat, 'mean' + gdat.fittnameparaelemgenrampl[l])
                        minm = getattr(gdat, 'minm' + gdat.fittnameparaelemgenrampl[l])
                        for strgcomp in gdat.fittlistnameparaelemgenr[l]:
                            path = pathopti + 'stdv' + strgcomp + 'pop%d.pdf' % l
                            factplot = getattr(gdat, 'fact' + strgcomp + 'plot')
                            factamplplot = getattr(gdat, 'fact' + gdat.fittnameparaelemgenrampl[l] + 'plot')
                            xdat = [gdatmodi.dictmodi[l][gdat.fittnameparaelemgenrampl[l] + 'indv'] * factamplplot, meanplot * factamplplot]
                            
                            if strgcomp == gdat.fittnameparaelemgenrampl[l]:
                                ydat = [gdatmodi.dictmodi[l]['stdv' + strgcomp + 'indv'], \
                                                                gdatmodi.stdp[getattr(gdat, 'indxstdp' + strgcomp)] / (meanplot / minm)**2.]
                            else:
                                ydat = [gdatmodi.dictmodi[l]['stdv' + strgcomp + 'indv'], \
                                                                gdatmodi.stdp[getattr(gdat, 'indxstdp' + strgcomp)] / (meanplot / minm)**0.5]
                            lablxdat = getattr(gdat, 'labl' + gdat.fittnameparaelemgenrampl[l] + 'totl')
                            scalxdat = getattr(gdat, 'scal' + gdat.fittnameparaelemgenrampl[l] + 'plot')
                            limtxdat = np.array(getattr(gdat, 'limt' + gdat.fittnameparaelemgenrampl[l])) * factamplplot
                            tdpy.util.plot_gene(path, xdat, ydat, scalxdat=scalxdat, scalydat='logt', lablxdat=lablxdat, limtxdat=limtxdat, \
                                                             lablydat=r'$\sigma_{%s}$' % getattr(gdat, 'labl' + strgcomp), plottype=['scat', 'lghtline'])
                            #tdpy.util.plot_gene(path, xdat, ydat, scalxdat=scalxdat, scalydat='logt', lablxdat=lablxdat, limtxdat=limtxdat, \
                            #                                 lablydat=r'$\sigma_{%s}$%s' % (getattr(gdat, 'labl' + strgcomp), getattr(gdat, 'labl' + strgcomp + 'unit')), plottype=['scat', 'lghtline'])
                            
                            tdpy.util.plot_gene(path, xdat, ydat, scalxdat=scalxdat, scalydat='logt', lablxdat=lablxdat, limtxdat=limtxdat, \
                                                             lablydat=r'$\sigma_{%s}$' % getattr(gdat, 'labl' + strgcomp), plottype=['scat', 'lghtline'])


        if gdat.typeverb > 1:
            print('-' * 10)
            print('Sweep %d' % gdatmodi.cntrswep)

        # decide whether to make a frame
        thismakefram = (gdatmodi.cntrswep % gdat.numbswepplot == 0) and \
                                                gdatmodi.indxprocwork == int(float(gdatmodi.cntrswep) / gdat.numbswep * gdat.numbproc) \
                                                and gdat.makeplotfram and gdat.makeplot
        
        # decide whether to make a log
        boollogg = False
        if gdat.typeverb > 0:
            gdatmodi.thispercswep = 5 * int(20. * gdatmodi.cntrswep / gdat.numbswep) 
            if gdatmodi.thispercswep > gdatmodi.percswepsave or thismakefram:
                gdatmodi.percswepsave = gdatmodi.thispercswep
                minmswepintv = max(0, gdatmodi.cntrswep - 1000)
                maxmswepintv = gdatmodi.cntrswep + 1
                if maxmswepintv > minmswepintv:
                    boollogg = True
        
        # propose the next sample
        if gdat.typeverb > 1:        
            print('-----')
            print('thislliktotl')
            print(gdatmodi.thislliktotl)
            print('thislpostotl')
            print(gdatmodi.thislpostotl)
            print('Proposing...')
        
        if gdat.boolburntmpr and gdatmodi.cntrswep < gdat.numbburntmpr:
            gdatmodi.thisfacttmpr = ((gdatmodi.cntrswep + 1.) / gdat.numbburntmpr)**4
            gdatmodi.thistmprfactstdv = 1. / gdatmodi.thisfacttmpr
            #gdatmodi.thistmprlposelem = -1000. * (1. - gdatmodi.thisfacttmpr) * np.concatenate(gdatmodi.thisindxsampcomp['comp']).size
            gdatmodi.thistmprlposelem = 0.
        else:
            gdatmodi.thistmprfactstdv = 1.
            gdatmodi.thistmprlposelem = 0. 
        
        # temp -- this can be faster
        for l in gdat.fittindxpopl:
            setattr(gdatmodi, 'thisauxiparapop%d' % l, np.empty(gdat.fittmaxmnumbparaelemgenr[l]))

        if gdat.typeverb > 1:
            show_samp(gdat, gdatmodi)
        
        # make a proposal
        initchro(gdat, gdatmodi, 'prop')
        prop_stat(gdat, gdatmodi, 'fitt')
        stopchro(gdat, gdatmodi, 'prop')

        if gdat.booldiagmode:
        
            for k in gdat.fittindxparabase:
                if gdat.fittscalparabase[k] == 'logt' and gdatmodi.thissamp[k] < 0.:
                    raise Exception('')

            if not np.isfinite(gdatmodi.nextsamp).all():
                raise Exception('')
        
        if gdat.typeverb > 1:
            show_samp(gdat, gdatmodi, strgstat='next')
    
        if (thismakefram or gdat.boolsave[gdatmodi.cntrswep] or boollogg):
            # preprocess the current sample to calculate variables that are not updated
            proc_samp(gdat, gdatmodi, 'this', 'fitt')
        
        # diagnostics
        if gdat.booldiagmode:
            
            initchro(gdat, gdatmodi, 'diag')
            
            indxsampbadd = np.where((gdatmodi.thissampunit[gdat.fittnumbpopl:] > 1.) | (gdatmodi.thissampunit[gdat.fittnumbpopl:] < 0.))[0] + 1
            if indxsampbadd.size > 0:
                raise Exception('Unit sample vector went outside [0,1].')
            
            if not np.isfinite(gdatmodi.thislliktotl):
                raise Exception('Log-likelihood is infinite!')
    
            #indxsampclos = np.where((gdatmodi.thissamp < 0.01) & (gdatmodi.thissamp % 1. != 0.))[0]
            #indxsampclos = list(indxsampclos)
            #for indxsamptemp in indxsampclos:
            #    for l in gdat.fittindxpopl:
            #        if not indxsamptemp in gdatmodi.thisindxsampcomp['comp'][l]:
            #            indxsampclos.remove(indxsamptemp)
            #indxsampclos = np.array(indxsampclos)
            #if indxsampclos.size > 0:
            #    print 'Warning! State is too close to 0!'
            #    print gdat.fittnamepara[indxsampclos]

            #indxsampclos = np.where((gdatmodi.thissamp > 0.99) & (gdatmodi.thissamp % 1. != 0.))[0]
            #indxsampclos = list(indxsampclos)
            #for indxsamptemp in indxsampclos:
            #    for l in gdat.fittindxpopl:
            #        if not indxsamptemp in gdatmodi.thisindxsampcomp['comp'][l]:
            #            indxsampclos.remove(indxsamptemp)
            #indxsampclos = np.array(indxsampclos)
            #if indxsampclos.size > 0:
            #    print 'Warning! State is too close to 1!'
            #    print gdat.fittnamepara[indxsampclos]

            if gdatmodi.cntrswep == 0:
                gdatmodi.thislliktotlprev = gdatmodi.thislliktotl
            
            lliktotldiff = gdatmodi.thislliktotl - gdatmodi.thislliktotlprev

            if gdatmodi.thislliktotl - gdatmodi.thislliktotlprev < -10.:
                raise Exception('loglikelihood drop is very unlikely!')
            gdatmodi.thislliktotlprev = gdatmodi.thislliktotl
       
            for strgstat in ['this', 'next']:
                for strgvarb in ['samp', 'sampunit']:
                    varb = getattr(gdatmodi, strgstat + strgvarb)
                    if not np.isfinite(varb).all():
                        raise Exception('Sample vector is not finite.')
            
            if gdat.fittmaxmnumbparaelem > 0:
                if gdat.fittboolelemsbrtdfncanyy:
                    thissbrtdfnc = getattr(gdatmodi, 'thissbrtdfnc')
                    frac = np.amin(thissbrtdfnc) / np.mean(thissbrtdfnc)
                    cntppntschec = retr_cntp(gdat, thissbrtdfnc)
                    if np.amin(cntppntschec) < -0.1 and frac < -1e-3:
                        raise Exception('thissbrtdfnc went negative by %.3g percent.' % (100. * frac))
                    
            # check the population index
            if (gdatmodi.thiscntpmodl <= 0.).any() or not (np.isfinite(gdatmodi.thiscntpmodl)).all():
                raise Exception('Current flux model is not positive')

            if gdat.fittmaxmnumbparaelem > 0:
                for l in gdat.fittindxpopl:
                    if gdatmodi.thissamp[gdat.fittindxparabasenumbelem[l]] != len(gdatmodi.thisindxelemfull[l]):
                        raise Exception('Number of elements is inconsistent with the element index list.')

                    if gdatmodi.thissamp[gdat.fittindxparabasenumbelem[l]] != len(gdatmodi.thisindxelemfull[l]):
                        raise Exception('Number of elements is inconsistent across data structures.')
                    
                    for k, strgcomp in enumerate(gdat.fittlistnameparaelemgenr[l]):
                        if gdat.fittlistscalparaelemgenr[l][k] == 'gaus' or gdat.fittlistscalparaelemgenr[l][k] == 'igam' \
                                                                                            or gdat.fittlistscalparaelemgenr[l][k] == 'expo':
                            continue
                        comp = gdatmodi.thissamp[gdatmodi.thisindxsampcomp[strgcomp][l]]
                        minm = getattr(gdat, 'fittminm' + strgcomp)
                        maxm = getattr(gdat, 'fittmaxm' + strgcomp)
                        indxtemp = np.where((comp < minm) | (comp > maxm))[0]
                        if indxtemp.size > 0:
                            raise Exception('A component of an element went outside the prior range.')
        
            stopchro(gdat, gdatmodi, 'diag')
    
        # save the sample
        if gdat.boolsave[gdatmodi.cntrswep]:
           
            initchro(gdat, gdatmodi, 'save')
        
            if gdat.savestat:
                
                if gdat.namesavestat is not None:
                    strgcnfg = gdat.namesavestat
                else:
                    strgcnfg = gdat.strgcnfg
                path = gdat.pathoutp + 'stat_' + strgcnfg + '.h5'
                
                booltemp = False
                if os.path.isfile(path) and gdatmodi.indxprocwork == 0:
                    thisfilechec = narr_h5(path, 'r')
                    if thisfilechec['lliktotl'][...] > gdatmodi.thislliktotl:
                        if gdat.typeverb > 0:
                            print('Not saving the state to %s because loglikelihood is lower...' % path)
                            print('Likelihood in the file:')
                            print(thisfilechec['lliktotl'][...])
                    else:
                        booltemp = True
                    thisfilechec.close()
                else:
                    booltemp = True
                if gdat.forcsavestat:
                    booltemp = True
                if booltemp:
                    if gdatmodi.indxprocwork > 0:
                        continue
                    if gdat.typeverb > 0:
                        print('Saving the state to %s...' % path)
        
                    thisfile = narr_h5(path, 'w')
                    thisfile.create_dataset('lliktotl', data=gdatmodi.thislliktotl)
                    for listnameparabase in gdat.fittlistnameparabase:
                        indxparabase = getattr(gdat, 'fittindxparabase' + listnameparabase)
                        valu = gdatmodi.thissamp[indxparabase]
                        thisfile.create_dataset(listnameparabase, data=valu)
                    if gdat.fittmaxmnumbparaelem > 0:
                        for l in gdat.fittindxpopl:
                            for strgcomp in gdat.fittlistnameparaelemgenr[l]:
                                comp = gdatmodi.thissamp[gdatmodi.thisindxsampcomp[strgcomp][l]]
                                for k in np.arange(comp.size):
                                    name = strgcomp + 'pop%d%04d' % (l, k)
                                    thisfile.create_dataset(name, data=comp[k])
                    thisfile.close()
            
            indxsampsave = gdat.indxsampsave[gdatmodi.cntrswep]
            
            # fill the sample lists
            for strgvarb in gdatmodi.liststrgvarbarrysamp:
                valu = getattr(gdatmodi, 'this' + strgvarb)
                workdict['list' + gdat.strgpdfn + strgvarb][indxsampsave, ...] = valu
            for strgvarb in gdatmodi.liststrgvarblistsamp:
                workdict['list' + gdat.strgpdfn + strgvarb].append(deepcopy(getattr(gdatmodi, 'this' + strgvarb)))
            stopchro(gdat, gdatmodi, 'save')

        # plot the current sample
        if thismakefram:
            
            initchro(gdat, gdatmodi, 'plot')
            
            writfile(gdatmodi, gdatmodi.pathgdatmodi) 

            if gdat.typeverb > 0:
                print('Process %d is in queue for making a frame.' % gdatmodi.indxprocwork)
            
            if gdat.numbproc > 1:
                gdatmodi.lock.acquire()
            
            if gdat.typeverb > 0:
                print('Process %d started making a frame.' % gdatmodi.indxprocwork)
            
            plot_samp(gdat, gdatmodi, 'this', 'fitt', 'fram')
            
            if gdat.typeverb > 0:
                print('Process %d finished making a frame.' % gdatmodi.indxprocwork)
        
            if gdat.numbproc > 1:
                gdatmodi.lock.release()
        
            stopchro(gdat, gdatmodi,  'plot')
    
        # determine the acceptance probability
        if gdatmodi.thisboolpropfilt:
            
            initchro(gdat, gdatmodi, 'proc')
            proc_samp(gdat, gdatmodi, 'next', 'fitt')
            stopchro(gdat, gdatmodi, 'proc')
        
            calc_probprop(gdat, gdatmodi)
            
            if gdat.booldiagmode:
                if not gdatmodi.thisindxproptype > 2 and gdatmodi.thisljcb != 0.:
                    raise Exception('log Jacobian can only be be nonzero when a split or merge is proposed.')
                if not gdatmodi.thisindxproptype > 2 and gdatmodi.thisltrp != 0.:
                    raise Exception('log ratio proposal probability can only be be nonzero when a split or merge is proposed.')
           
            # evaluate the acceptance probability
            gdatmodi.thisdeltlpostotl = gdatmodi.nextlpostotl - gdatmodi.thislpostotl
            gdatmodi.thisaccplprb = gdatmodi.thisdeltlpostotl + gdatmodi.thistmprlposelem - gdatmodi.thislpau + gdatmodi.thisltrp + gdatmodi.thisljcb
            gdatmodi.thisaccpprob[0] = np.exp(gdatmodi.thisaccplprb)
            if gdat.typeverb > 1:
                print('gdatmodi.thislpritotl')
                print(gdatmodi.thislpritotl)
                print('gdatmodi.nextlpritotl')
                print(gdatmodi.nextlpritotl)
                print('gdatmodi.thislliktotl')
                print(gdatmodi.thislliktotl)
                print('gdatmodi.nextlliktotl')
                print(gdatmodi.nextlliktotl)
                print('gdatmodi.thislpostotl')
                print(gdatmodi.thislpostotl)
                print('gdatmodi.nextlpostotl')
                print(gdatmodi.nextlpostotl)
                
                print('gdatmodi.thisdeltlpostotl')
                print(gdatmodi.thisdeltlpostotl)
                print('gdatmodi.thistmprlposelem')
                print(gdatmodi.thistmprlposelem)
                print('gdatmodi.thislpau')
                print(gdatmodi.thislpau)
                print('gdatmodi.thisltrp')
                print(gdatmodi.thisltrp)
                print('gdatmodi.thisljcb')
                print(gdatmodi.thisljcb)
            
                print('gdatmodi.thisaccplprb')
                print(gdatmodi.thisaccplprb)
        else:
            gdatmodi.thisaccpprob[0] = 0.
    
        # accept or reject the proposal
        booltemp = gdatmodi.thisaccpprob[0] >= np.random.rand()
        
        if gdat.booldiagmode:
            if gdatmodi.thisindxproptype == 0:
                if gdat.boolsqzeprop and not booltemp:
                    raise Exception('')

        if booltemp:
            if gdat.typeverb > 1:
                print('Accepted.')
            
            # update the current state
            updt_stat(gdat, gdatmodi)

            # check if the accepted sample has maximal likelihood
            if gdatmodi.thislliktotl > gdatmodi.maxmllikswep:
                gdatmodi.maxmllikswep = gdatmodi.thislliktotl
                gdatmodi.indxswepmaxmllik = gdatmodi.cntrswep
                gdatmodi.sampmaxmllik = np.copy(gdatmodi.thissamp)
            
            # register the sample as accepted
            gdatmodi.thisboolpropaccp = True

        # reject the sample
        else:

            if gdat.typeverb > 1:
                print('Rejected.')

            gdatmodi.thisboolpropaccp = False
            
        ## variables to be saved for each sweep
        for strg in gdat.liststrgvarbarryswep:
            workdict['list' + gdat.strgpdfn + strg][gdatmodi.cntrswep, ...] = getattr(gdatmodi, 'this' + strg)
        
        workdict['list' + gdat.strgpdfn + 'accpprob'][gdatmodi.cntrswep, 0] = gdatmodi.thisaccpprob[0]
        
        # log the progress
        if boollogg:
            
            print('--------------')
            print('Sweep number %d' % gdatmodi.cntrswep)
            print('%3d%% completed.' % gdatmodi.thispercswep)
            print('%30s %50s %10s' % ('Prop', 'Accp rate', 'Scale'))
            
            indxswepintv = np.arange(minmswepintv, maxmswepintv)
            for k in gdat.indxproptype:
                indxswepprop = indxswepintv[np.where(workdict['list' + gdat.strgpdfn + 'indxproptype'][indxswepintv, 0] == k)]
                boolproptype = workdict['list' + gdat.strgpdfn + 'indxproptype'][indxswepintv, 0] == k
                boolaccp = workdict['list' + gdat.strgpdfn + 'boolpropaccp'][indxswepintv, 0] == 1
                numbaccp = np.where(boolaccp & boolproptype)[0].size
                numbtotl = np.where(boolproptype)[0].size
                if numbtotl > 0:
                    percaccp = 100. * numbaccp / float(numbtotl)
                else:
                    percaccp = 0.
                if k in gdat.indxstdp:
                    strgstdp = '%.3g' % gdat.stdp[k]
                else:
                    strgstdp = ''
                print('%30s %50s' % (gdat.legdproptype[k], 'acceptance rate: %3d%% (%5d out of %5d)' % (percaccp, numbaccp, numbtotl)))
                
            if gdat.boolburntmpr and gdatmodi.cntrswep < gdat.numbburntmpr:
                print('Tempered burn-in')
                print('gdatmodi.thisfacttmpr')
                print(gdatmodi.thisfacttmpr)
            print 
            numbpara = gdat.fittnumbparabase
            if gdat.fittmaxmnumbparaelem > 0:
                for l in gdat.fittindxpopl:
                    numbpara += gdatmodi.thisindxsampcomp['comp'][l].size
            if gdat.fittmaxmnumbparaelem > 0:
                print('Number of elements:')
                for l in gdat.fittindxpopl:
                    print(gdatmodi.thissamp[gdat.fittindxparabasenumbelem[l]].astype(int))
            print('Current number of parameters:')
            print(numbpara)
            print('gdatmodi.thisnumbdoff')
            print(gdatmodi.thisnumbdoff)
            for attr, valu in gdatmodi.__dict__.items():
                if isinstance(valu, np.ndarray):
                    if 8 * valu.size * gdat.numbsamptotl > 1e9:
                        print('Warning! %s has total length %d and size %s' % (attr, valu.size * gdat.numbsamptotl, \
                                                                                        tdpy.util.retr_strgmemo(8 * valu.size * gdat.numbsamptotl)))
            if gdat.fittmaxmnumbparaelem > 0:
                print('Mean number of elements:')
                print(gdatmodi.thissamp[gdat.fittindxparabasemeanelem])
                for l in gdat.fittindxpopl:
                    if gdat.fittnameparaelemgenrampl[l] == 'flux' and gdat.fittfluxdisttype[l] == 'powrslop' or gdat.fittnameparaelemgenrampl[l] != 'flux':
                        print('Log-slope of the amplitude parameter distribution, population %d:' % l)
                        indxparabase = getattr(gdat, 'fittindxparabase' + gdat.fittnameparaelemgenrampl[l] + 'distsloppop%d' % l)
                        print(gdatmodi.thissamp[indxparabase])
                    else:
                        print('Flux distribution break:')
                        print(gdatmodi.thissamp[getattr(gdat, 'fittindxparabase' + gdat.fittnameparaelemgenrampl[l] + 'distbrekpop%d' % l)])
                        print('Flux distribution lower slope:')
                        print(gdatmodi.thissamp[getattr(gdat, 'fittindxparabase' + gdat.fittnameparaelemgenrampl[l] + 'distsloplowrpop%d' % l)])
                        print('Flux distribution upper slope:')
                        print(gdatmodi.thissamp[getattr(gdat, 'fittindxparabase' + gdat.fittnameparaelemgenrampl[l] + 'distslopupprpop%d' % l)])
            print('Backgrounds')
            print(gdatmodi.thissamp[gdat.fittindxparabasebacp])
            if gdat.fittmaxmnumbparaelem > 0:
                print('Log-prior penalization term: ')
                print(gdatmodi.thislpripena)
                for q in gdat.indxrefr:
                    if gdat.boolasscrefr[q]:
                        print('Completeness')
                        for l in gdat.fittindxpopl:
                            if gdat.refrnumbelem[q] == 0:
                                continue
                            namevarb = 'pop%dpop%d' % (l, q)
                            print('Reference %d, Population %d' % (q, l))
                            print('Total')
                            print(getattr(gdatmodi, 'thiscmpl' + namevarb))
                            refrfeat = getattr(gdat, 'refr' + gdat.nameparaelemsignrefr)
                            if len(refrfeat[q]) > 0:
                                print('Binned in significance feature')
                                print(getattr(gdatmodi, 'thiscmpl' + gdat.nameparaelemsignrefr + namevarb))
                        print('False discovery rate')
                        for l in gdat.fittindxpopl:
                            if gdat.refrnumbelem[q] == 0:
                                continue
                            namevarb = 'pop%dpop%d' % (q, l)
                            print('Reference %d, Population %d' % (q, l))
                            print('Total')
                            print(getattr(gdatmodi, 'thisfdis' + namevarb))
                            refrfeat = getattr(gdat, 'refr' + gdat.nameparaelemsignrefr)
                            if len(refrfeat[q]) > 0:
                                print('Binned in significance feature')
                                print(getattr(gdatmodi, 'thisfdis' + gdat.nameparaelemsignrefr + namevarb))
    
            print('gdatmodi.thislliktotl')
            print(gdatmodi.thislliktotl)
            print('Chi2 per degree of freedom')
            print(gdatmodi.thischi2doff)
        
        # save the execution time for the sweep
        stopchro(gdat, gdatmodi, 'totl')
        
        if boollogg:
            print('Chronometers: ')
            for k, name in enumerate(gdat.listnamechro):
                #for name, valu in gdat.indxchro.items():
                    #if valu == k:
                thischro = getattr(gdatmodi, 'thischro' + name)
                print('%s: %.3g msec' % (name, thischro * 1e3))
                booltemp = False
                for l in gdat.fittindxpopl:
                    if gdat.fittelemspatevaltype[l] == 'locl' and gdat.fittmaxmnumbelempopl[l] > 0:
                        booltemp = True
                if name == 'llik' and gdat.numbpixl > 1 and gdat.fittmaxmnumbparaelem > 0 and booltemp:
                    print('%.3g per pixel' % (thischro * 1e3 / np.amin(gdat.numbpixlprox)))
            print 

        if gdat.typeverb > 1:
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
        
        # update the sweep counter
        gdatmodi.cntrswep += 1
        
    for strgvarb in gdatmodi.liststrgvarbarry + gdatmodi.liststrgvarblistsamp:
        valu = workdict['list' + gdat.strgpdfn + strgvarb]
        setattr(gdatmodi, 'list' + gdat.strgpdfn + strgvarb, valu)

    gdatmodi.timereal = time.time() - timereal
    gdatmodi.timeproc = time.clock() - timeproc
    
    delattr(gdatmodi, 'lock')
    
    gdatmodi.booldone = True

    writfile(gdatmodi, gdatmodi.pathgdatmodi) 


