"""Multi-population corner, histogram, and pair plots of parameter samples.

Moved from ``tdpy.util.plot_grid``; ``pcat.plot_grid`` remains the compact posterior corner plot.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tdpy
from tdpy.util import (
    _is_integer_like_array,
    cdfn_atan,
    cdfn_logt,
    cdfn_self,
    retr_axis,
    retr_listlabltotl,
    setp_axislogt,
    summgene,
)


def _plot_population_diag(k, axis, listpara, truepara, listparadraw, boolplotquan, \
                            listlablpara, listtypeplottdim, indxpopl, listcolrpopl, listmrkrpopl, listlablpopl, boolmakelegd, listsizepopl, bins=None):
                    
    for u in indxpopl:
        if indxpopl.size > 1:
            labl = listlablpopl[u]
        else:
            labl = None
        axis.hist(listpara[u][:, k], bins=bins[k], label=labl, \
                                                               edgecolor=matplotlib.colors.to_rgba(listcolrpopl[u], 1.0), lw=2, ls='-', \
                                                               facecolor=matplotlib.colors.to_rgba(listcolrpopl[u], 0.2))
    if boolmakelegd and indxpopl.size > 1:
        axis.legend(framealpha=1.)
    
    if truepara is not None and truepara[k] is not None and not np.isnan(truepara[k]):
        axis.axvline(truepara[k], color='g', lw=4)
    
    # draw the provided reference values
    if listparadraw is not None:
        for m in indxdraw:
            axis.axvline(listparadraw[m][k], color='r', lw=3)
    
    if boolplotquan:
        quan = np.empty(4)
        quan[0] = np.nanpercentile(listpara[0][:, k], 2.5)
        quan[1] = np.nanpercentile(listpara[0][:, k], 16.)
        quan[2] = np.nanpercentile(listpara[0][:, k], 84.)
        quan[3] = np.nanpercentile(listpara[0][:, k], 97.5)
        axis.axvline(quan[0], color='r', ls='--', lw=2)
        axis.axvline(quan[1], color='r', ls='-.', lw=2)
        axis.axvline(quan[2], color='r', ls='-.', lw=2)
        axis.axvline(quan[3], color='r', ls='--', lw=2)
        medivarb = np.nanmedian(listpara[0][:, k])
    
        if listlablpara[k][1] != '':
            strgunit = ' ' + listlablpara[k][1]
        else:
            strgunit = ''
        axis.set_title(r'%s = %.3g $\substack{+%.2g \\\\ -%.2g}$ %s' % (listlablpara[k][0], medivarb, quan[2] - medivarb, medivarb - quan[1], strgunit))
    
    axis.set_yscale('log')
    
                

def _plot_population_pair(k, l, axis, limt, listmantlabl, listpara, truepara, listparadraw, boolplotquan, listlablpara, \
                                     listscalpara, boolsqua, listvectplot, listtypeplottdim, indxpopl, listcolrpopl, listmsizpopl, \
                                     typeannosamp, listlablannosamp, \
                                     listmrkrpopl, listcolrpopltdim, listlablpopl, boolmakelegd, \
                                     bins=None, midpgrid=None, \
                                     listlablsamp=None, boolcbar=True, \
                                    ):
    
    if not np.isfinite(limt).all():
        print('limt')
        print(limt)
        raise Exception('')
    
    if (listtypeplottdim == 'hist').any():
        binstemp = [bins[l], bins[k]]

    for u in indxpopl:
         
        if listpara[u][:, l].size == 0:
            continue
            
        if indxpopl.size > 1:
            labl = listlablpopl[u]
            alph = 1.
        else:
            labl = None
            alph = 1.

        if listtypeplottdim[u] == 'scat':
        
            if listpara[u][:, l].size >= 1e7:
                print('')
                print('')
                print('')
                print('Warning! Skipped the scatter plot because there are too many points to plot!')
                print('')
                print('')
                print('')
                return

            axis.scatter(listpara[u][:, l], listpara[u][:, k], s=listmsizpopl[u], color=listcolrpopl[u], label=labl, alpha=alph, marker=listmrkrpopl[u])
        
            # add text labels on outliers
            if listlablsamp is not None and indxpopl.size == 1:
                
                # remove infinite samples
                indx = np.where(np.isfinite(listpara[u][:, l]) & np.isfinite(listpara[u][:, k]))[0]
                listparapair = listpara[u][indx, :]
                listlablsamppair = listlablsamp[u][indx]
                listparapair = listparapair[:, np.array([l, k])]
                
                if listlablannosamp is not None:
                    listindxsamplouf = []
                    for lablannosamp in listlablannosamp:
                        indx = np.where(listlablsamppair == lablannosamp)[0]
                        if indx.size > 0:
                            if indx.size > 1:
                                raise Exception('')
                            listindxsamplouf.append(indx[0])
                    listindxsamplouf = np.array(listindxsamplouf, dtype=int)
                elif typeannosamp == 'LOF':
                    # transform from data to axis coordinate positions
                    ## display coordinate positions
                    posidisp = axis.transData.transform(listparapair)
                    ## axis coordinate positions
                    listparapairoutl = axis.transAxes.inverted().transform(posidisp)
                    
                    print('Determining the elements to be annotated...')
                    from sklearn.neighbors import LocalOutlierFactor
                    numbsamp = listparapairoutl.shape[0]
                    
                    numboutf = min(0, numbsamp)
                    if numboutf > 0:
                        if numbsamp > numboutf:
                            n_neighbors = min(numbsamp, 100)
                            objtfore = LocalOutlierFactor(n_neighbors=n_neighbors)
                            objtfore.fit(listparapairoutl)
                            louf = objtfore.negative_outlier_factor_
                        else:
                            louf = np.zeros(numbsamp)
                        listindxsamplouf = np.argsort(louf)[:numboutf]
                    else:
                        listindxsamplouf = np.array([], dtype=int)

                elif typeannosamp == 'minmax':
                    listindxsamplouf = np.array([np.argmin(listparapair[:, 0]), np.argmin(listparapair[:, 1]), np.argmax(listparapair[:, 0]), np.argmax(listparapair[:, 1])])
                    listindxsamplouf = np.unique(listindxsamplouf)
                else:
                    raise Exception('')

                numboutf = listindxsamplouf.size
                listparapair = listparapair[listindxsamplouf, :]
                listlablsamppair = listlablsamppair[listindxsamplouf]
            
                # place larger markers at the positions of the outliers
                axis.scatter(listparapair[:, 0], listparapair[:, 1], s=3, color=listcolrpopl[u], marker=listmrkrpopl[u])
                
                print('Automatically positioning the annotations...')
                # automatically position the annotations

                ## coordinates of the samples to be annotated
                if listscalpara[l] == 'self' or listscalpara[l] == 'gaus':
                    xpossamp = cdfn_self(listparapair[:, 0], limt[l][0], limt[l][1])
                elif listscalpara[l] == 'logt':
                    xpossamp = cdfn_logt(listparapair[:, 0], limt[l][0], limt[l][1])
                elif listscalpara[l] == 'atan':
                    xpossamp = cdfn_atan(listparapair[:, 0], limt[l][0], limt[l][1])
                else:
                    raise Exception('Unrecognized scaling: %s' % listscalpara[l])
                
                if listscalpara[k] == 'self' or listscalpara[k] == 'gaus':
                    ypossamp = cdfn_self(listparapair[:, 1], limt[k][0], limt[k][1])
                elif listscalpara[k] == 'logt':
                    ypossamp = cdfn_logt(listparapair[:, 1], limt[k][0], limt[k][1])
                elif listscalpara[k] == 'atan':
                    ypossamp = cdfn_atan(listparapair[:, 1], limt[k][0], limt[k][1])
                else:
                    raise Exception('Unrecognized scaling: %s' % listscalpara[k])
                
                # tunable parameters
                sizelablxpos = 0.3 
                sizelablypos = 0.1
                
                # minimum horizontal position of the label
                minmlablxpos = 0.#0.5 * sizelablxpos
                # minimum vertical position of the label
                minmlablypos = 0.#0.5 * sizelablypos

                # maximum horizontal position of the label
                maxmlablxpos = 1.# - 0.5 * sizelablxpos
                # maximum vertical position of the label
                maxmlablypos = 1.# - 0.5 * sizelablypos
                
                # list of trial maximum horizontal distances between the sample and label
                listdistxpos = np.array([1.5, 3., 100.]) * sizelablxpos
                # list of trial maximum vertical distances between the sample and label
                listdistypos = np.array([1.5, 3., 100.]) * sizelablypos
                for ll in range(len(listdistxpos)):
                    distxpos = listdistxpos[ll]
                    distypos = listdistypos[ll]
                    
                    numbtria = 0
                    while True:
                        
                        # trial coordinates of the annotations
                        listxposlabl = xpossamp + distxpos * (2. * np.random.rand(numboutf) - 1.)
                        listyposlabl = ypossamp + distypos * (2. * np.random.rand(numboutf) - 1.)
                        
                        indxxpos = np.where((listxposlabl > maxmlablxpos) | (listxposlabl < minmlablxpos))[0]
                        listxposlabl[indxxpos] = minmlablxpos + listxposlabl[indxxpos] % (maxmlablxpos - minmlablxpos)

                        indxypos = np.where((listyposlabl > maxmlablypos) | (listyposlabl < minmlablypos))[0]
                        listyposlabl[indxypos] = minmlablypos + listyposlabl[indxypos] % (maxmlablypos - minmlablypos)

                        # concatenated list of coordinates of samples and trial annotations
                        xpos = np.concatenate((listxposlabl, xpossamp))
                        ypos = np.concatenate((listyposlabl, ypossamp))
                        
                        # check if each trial annotation is sufficiantly distant from all samples and other trial annotations
                        boolgood = True
                        for n in range(numboutf):
                            xposdiff = abs(listxposlabl[n] - xpos)
                            yposdiff = abs(listyposlabl[n] - ypos)
                            if ((xposdiff < sizelablxpos) & (xposdiff > 0) & (yposdiff < sizelablypos) & (yposdiff > 0)).any():
                                boolgood = False
                        if boolgood or numbtria > 99999:
                            break
                        numbtria += 1
                    
                    if boolgood:
                        print('Automatically positioned the annotations in %d trials, ll=%d...' % (numbtria, ll))
                        break
                if not boolgood:
                    print('Failed to automatically positioned the annotations...')
                
                #indxoutf = np.arange(numboutf)
                #indxsampassi = []
                #indxlablassi = []
                #for n in indxoutf:
                #    indxlabllive = np.setdiff1d(indxoutf, np.array(indxlablassi))
                #    indxsamplive = np.setdiff1d(indxoutf, np.array(indxsampassi))
                #    distlabl = np.sqrt((listxposlabl[indxlabllive, None] - xpossamp[None, indxsamplive])**2 + (listyposlabl[indxlabllive, None] - ypossamp[None, indxsamplive])**2)
                #    
                #    indxlablminm, indxsampminm = np.unravel_index(distlabl.argmin(), distlabl.shape)
                #    indxlablassi.append(indxlabllive[indxlablminm])
                #    indxsampassi.append(indxsamplive[indxsampminm])
                #listxposlabl = listxposlabl[indxlablassi]
                #listyposlabl = listyposlabl[indxlablassi]
                
                for n in range(numboutf):
                    axis.annotate(listlablsamppair[n], \
                                  #xy=(xpossamp[n], ypossamp[n]), \
                                  xy=(listparapair[n, 0], listparapair[n, 1]), \
                                  xytext=(listxposlabl[n], listyposlabl[n]),
                                  textcoords=axis.transAxes, \
                                  xycoords=axis.transData, \
                                  ha='center', va='center', \
                                  color=listcolrpopl[u], \
                                  bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='black', alpha=1.), \
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='gray'), \
                                 )

        elif listtypeplottdim[u] == 'kdee':
            kdee = tdpy.retr_KDE(listpara[u][:, np.array([k, l])], midpgrid=midpgrid)
        
            if np.amax(kdee) / np.amin(kdee) < 1e2:
                norm = None
            else:
                norm = matplotlib.colors.LogNorm()
            
            objtaxispcol = axis.pcolor(midpgrid[l], midpgrid[k], kdee, cmap=listcolrpopltdim[u], label=labl, norm=norm)
        elif listtypeplottdim[u] == 'hist':
            hist = np.histogram2d(listpara[u][:, l], listpara[u][:, k], bins=binstemp)[0]
            
            if np.amax(hist) == 0:
                print('binstemp')
                print(binstemp)
                print('listpara[u][:, k]')
                summgene(listpara[u][:, k])
                raise Exception('')

            if np.amax(hist) / np.amin(hist[np.where(hist > 0)]) < 1e2:
                norm = None
            else:
                norm = matplotlib.colors.LogNorm()
            
            objtaxispcol = axis.pcolor(midpgrid[l], midpgrid[k], hist.T, cmap=listcolrpopltdim[u], label=labl, norm=norm)
        else:
            raise Exception('')

    if boolcbar and (listtypeplottdim == 'hist').any():
        cbar = plt.colorbar(objtaxispcol)
        
    if boolmakelegd and indxpopl.size > 1:
        axis.legend(framealpha=1.)
    
    if truepara is not None and truepara[l] is not None and not np.isnan(truepara[l]) and truepara[k] is not None and not np.isnan(truepara[k]):
        axis.scatter(truepara[l], truepara[k], color='g', marker='x', s=500)
    
    # draw the provided reference values
    if listparadraw is not None:
        for m in indxdraw:
            axis.scatter(listparadraw[m][l], listparadraw[m][k], color='r', marker='x', s=350)
    
    if listvectplot is not None:
        for vectplot in listvectplot:
            axis.arrow(vectplot[0], vectplot[1], vectplot[2], vectplot[3])
    
    if boolsqua:
        axis.set_aspect('equal')
    
    if listscalpara[l] == 'logt':
        setp_axislogt(axis, limt[l], 'x', listmantlabl)
    
    if listscalpara[k] == 'logt':
        setp_axislogt(axis, limt[k], 'y', listmantlabl)


def _plot_population_histodim(listmantlabl, listpara, k, listlablparatotl, indxpopl, listlablpopl, bins, \
                                    listcolrpopl, listparadraw, lablnumbsamp, lablsampgene, boolinte, \
                                    boolmakelegd, listscalpara, factulimyaxihist, titl, plotsize, limt, limtrims, boolcumu=False, path=None):
    
    figr, axis = plt.subplots(figsize=(plotsize, plotsize))
    for u in indxpopl:
        
        if listpara[u][:, k].size == 0:
            continue

        if indxpopl.size > 1:
            labl = listlablpopl[u]
            alph = 0.7
        else:
            labl = None
            alph = 1.
        axis.hist(listpara[u][:, k], bins=bins[k], label=labl, cumulative=boolcumu, \
                                                               edgecolor=matplotlib.colors.to_rgba(listcolrpopl[u], 1.0), lw=2, ls='-', \
                                                               facecolor=matplotlib.colors.to_rgba(listcolrpopl[u], 0.2))
    if listparadraw is not None:
        for m in indxdraw:
            axis.axvline(listparadraw[m][k], color='orange', lw=3)
    
    axis.set_xlabel(listlablparatotl[k])
    if lablnumbsamp is not None:
        axis.set_ylabel(lablnumbsamp)
    elif lablsampgene is not None:
        #axis.set_ylabel(r'$N_{\rm{%s}}$' % lablsampgene)
        axis.set_ylabel('Number of %s' % lablsampgene)
    
    if boolinte[k]:
        axis.xaxis.get_major_locator().set_params(integer=True)
    
    if listscalpara[k] == 'logt':
        setp_axislogt(axis, limtrims[k], 'x', listmantlabl)
    limtyaxi = axis.get_ylim()
    
    boolyaxilogt = limtyaxi[1] - limtyaxi[0] > 30.

    # rescale the upper limit of the vertical axis
    if factulimyaxihist != 1.:
        print('Scaling up the vertical axis upper limit for the histogram by a factor of %g...' % factulimyaxihist)
        limtyaxiprim = np.array(limtyaxi)
        if boolyaxilogt:
            # find the minimum value of the histogram for all populations
            llimyaxihist = 1e100
            for u in indxpopl:
                hist = np.histogram(listpara[u][:, k], bins=bins[k])[0]
                if np.amax(hist) > 0:
                    llimyaxihist = min(np.amin(hist[np.where(hist > 0)[0]]), llimyaxihist)
            llimyaxihist *= 0.5
        else:
            llimyaxihist = None
        limtyaxiprim = [llimyaxihist, limtyaxi[1] * factulimyaxihist]
        axis.set_ylim(limtyaxiprim)
        ncollegd = 1
    else:
        ncollegd = 1
    
    if boolmakelegd and indxpopl.size > 1:
        axis.legend(framealpha=1., ncol=ncollegd)

    if boolyaxilogt:
        axis.set_yscale('log')
    
    if titl is not None:
        axis.set_title(titl)
    axis.set_xlim(limt[k]) 
    plt.tight_layout()
    if path is not None:
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_population_grid(
              # a list with length equal to the number of parameters, 
              # Each element of the list should itself be list of two strings, where
              # the first string is the label for the parameter and the second string is the unit
              listlablpara, \
              
              # two dimensional numpy array of samples, where the first dimension is the sample and second dimension is the parameter
              listpara=None, \
              
              # dictionary of of samples, where the key should match the label roots and the values should be numpy arrays of samples
              dictpara=None, \
              
              # an optional string indicating the path of the folder in which to write the plot
              pathbase=None, \
              
              # an optional base string to include in the file name
              strgextn=None, \
              
              # the limits for the parameters
              limt=None, \

              # list of scalings for the parameters
              listscalpara=None, \
              
              # size of each subplot
              plotsize=3.5, \
              
              # type of the file for plots
              typefileplot='png', \
              
              # Boolean flag to generate the lower-triangle plot
              boolplottria=False, \

              # Boolean flag to generate individual histograms
              boolplothistodim=None, \
              
              # Boolean flag to generate pie plots
              boolplotpies=True, \
              
              # Boolean flag to generate individual pair-wise plots
              boolplotpair=None, \
              
              # list of base file names for the individual histograms
              listnamepara=None, \
                  
              # Boolean flag to make the two-dimensional plots square
              boolsqua=False, \
              
              # label to be used to denote the number of samples (takes priority over lablsampgene)
              lablnumbsamp=None, \

              # label to be used as subscript to denote the number of samples
              lablsampgene=None, \

              # list of markers for populations
              listmrkrpopl=None, \

              # list of marker sizes for populations
              listmsizpopl=None, \

              # list of colors for populations
              listcolrpopl=None, \

              # list of colors for populations to be used in two histograms
              listcolrpopltdim=None, \

              # list of vectors to overplot
              listvectplot=None, \
              
              # list of labels for each sample
              listlablsamp=None, \
              
              # list of feature names for which a cumulative histogram will be made
              listnamefeatcumu=None, \
              
              # type of grouping for populations
              ## 'together': populations are overplotted together
              ## 'individual': populations are separately plotted on common axes
              ## 'both': both
              typepgrp='together', \

              # list of pairs of feature names to be skipped
              listnamefeatskip=None, \
              
              # list of labels for populations
              listlablpopl=None, \
            
              # Boolean flag to overwrite
              boolwritover=False, \

              # Boolean flag to include a legend
              boolmakelegd=True, \
             
              # type of annotation
              ## 'minmax': minima and maxima
              typeannosamp='minmax', \
              
              # Boolean flag to force all parameters to be interpreted as floats
              boolforcflot=False, \

              # label of the sample to be annotated
              listlablannosamp=None, \

              # Boolean flag to indicate that the populations are mutually-exclusive
              boolpoplexcl=False, \
             
              # factor by which to multiply the upper limit of the y-axis
              factulimyaxihist=None, \

              # title for the plots
              titl=None, \
              
              # list of tick mantices (other than 1) to show in the label when the axis is log-streched
              listmantlabl=None, \
              
              # optional bins
              binsgridinpt=None, \

              # number of bins
              numbpntsgrid=None, \

              # Boolean flag to overplot quantiles
              boolplotquan=False, \
              
              # list of parameters to overplot
              listparadraw=None, \
              
              # true parameters to highlight
              truepara=None, \
              
              # a list of pairs of feature names, which enforces the given order (first item y-axis, second item y-axis)
              listnameordrpair=None, \

              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
              
              # type of the two-dimensional plots
              ## 'scat': all populations are scatter plots.
              ## 'hist': all populations are histogram plots.
              ## 'best': the largest population is histogram if there are too many samples, scatter otherwise. All other populations are scatter.
              typeplottdim='best', \
              
              # Boolean flag to diagnose the code
              booldiag=True, \
              
             ):
    
    '''
    Make a corner plot of a multivariate distribution.
    '''
    
    if typeverb > 1:
        print('pcat.plot_population_grid() initialized...')
    
    if pathbase is not None:
        if not os.path.exists(pathbase):
            os.system('mkdir -p %s' % pathbase)
    else:
        path = None

    if not (dictpara is None and listpara is not None or dictpara is not None and listpara is None):
        print('')
        print('')
        print('')
        print('dictpara')
        print(dictpara)
        print('listpara')
        print(listpara)
        raise Exception('Either dictpara or listpara should be defined.')
    elif listpara is None:
        listkeys = list(dictpara.keys())
        for keys in listkeys:
            if isinstance(dictpara[keys], list):
                dictpara[keys] = np.array(dictpara[keys])
        listpara = np.empty((dictpara[listkeys[0]].size, len(listkeys)))
        
        listlablpararoot = []
        for lablpara in listlablpara:
            listlablpararoot.append(lablpara[0])

        for k in range(len(listkeys)):
            indxparathis = listlablpararoot.index(listkeys[k])
            if isinstance(dictpara[listkeys[k]][0], str):
                dictparathisuniq = np.unique(dictpara[listkeys[k]])
                dictints = dict()
                for n in range(dictparathisuniq.size):
                    dictints[dictparathisuniq[n]] = n
                for strg in dictpara[listkeys[k]]:
                    listpara[:, indxparathis] = dictints[strg]
            else:
                listpara[:, indxparathis] = dictpara[listkeys[k]]

    print('pcat.plot_population_grid():')
    print('boolplotpies')
    print(boolplotpies)

    # check whether there is a single population or multiple populations
    if isinstance(listpara, list):
        boolmpop = True
    else:
        boolmpop = False
        listpara = [listpara]
    
    # preclude quantile lines if there are multiple populations
    if boolplotquan and boolmpop:
        raise Exception('')
    
    if listpara[0].ndim == 1:
        raise Exception('listpara should be a list of Nsamp by Nparam array')

    # temp: number of parameters should be able to be different for different populations
    numbpara = listpara[0].shape[1]
    indxpara = np.arange(numbpara)
    
    if len(listlablpara) != numbpara:
        print('')
        print('')
        print('')
        print('listlablpara')
        print(listlablpara)
        print('numbpara')
        print(numbpara)
        raise Exception('len(listlablpara) != numbpara')
    
    if listnamepara is None:
        listnamepara = []
        for k in indxpara:
            listnamepara.append('p%03d' % k)

    if boolplotpair is None:
        boolplotpair = not boolplottria and pathbase is not None
    
    if boolplothistodim is None:
        boolplothistodim = not boolplottria and pathbase is not None
    
    if (boolplotpair or boolplothistodim) and pathbase is None:
        raise Exception('If individual histograms or pairwise scatter plots are to be written to the disk, then pathbase must be provided.')
    
    if lablnumbsamp is None:
        lablnumbsamp = 'Number of samples'
    
    if booldiag:
        if isinstance(listlablpara[0][1], list):
            raise Exception('')

    if len(listlablpara[0]) == 2 and isinstance(listlablpara[0][0], str) and isinstance(listlablpara[0][1], str):
        listlablparatotl = retr_listlabltotl(listlablpara)
    else:
        print('hey')
        listlablparatotl = listlablpara
    
    if booldiag:
        if not isinstance(listlablparatotl[0], str):
            print('')
            print('')
            print('')
            print('listlablparatotl')
            print(listlablparatotl)
            raise Exception('')

    if listmrkrpopl is None:
        listmrkrpopl = np.array(['o', 'x', '+', 'D', '^', '*', '<', '>', 's', 'p'])
    
    if listcolrpopl is None:
        listcolrpopl = np.array(['g', 'r', 'b', 'purple', 'orange', 'pink', 'magenta', 'olive', 'cyan', 'teal'])
    
    if listmsizpopl is None:
        listmsizpopl = np.array([1] * 100)
    
    if listcolrpopltdim is None:
        listcolrpopltdim = np.array(['Greens', 'Blues', 'Purples', 'Oranges'])
    
    numbpopl = len(listpara)
    
    if numbpopl > 10:
        print('')
        print('')
        print('')
        print('Number of populations must be less than or equal to 10.')
        raise Exception('')

    if numbpopl > 1 and listlablpopl is None:
        print('listlablpopl should be defined when there are more than one populations.')
        raise Exception('')

    indxpopl = np.arange(numbpopl)
    listcolrpopl = listcolrpopl[:numbpopl]
    
    if listscalpara is None:
        listscalpara = ['self'] * numbpara
    
    if len(listscalpara) != len(listlablpara):
        print('')
        print('')
        print('')
        print('listscalpara')
        print(listscalpara)
        print('listlablpara')
        print(listlablpara)
        raise Exception('len(listscalpara) != len(listlablpara)')
    
    if len(listscalpara) != numbpara:
        print('listscalpara')
        print(listscalpara)
        print('len(listscalpara)')
        print(len(listscalpara))
        for u in indxpopl:
            print('listpara[u]')
            summgene(listpara[u])
        print('numbpara')
        print(numbpara)
        raise Exception('len(listscalpara) != numbpara')
    
    # number of samples in each population
    numbsamp = np.empty(numbpopl, dtype=int)
    for u in indxpopl:
        numbsamp[u] = listpara[u][:, 0].size

    # determine the type of 2-dimensional plots
    if typeplottdim == 'scat' or numbpopl > 1:
        listtypeplottdim = np.array(['scat' for u in indxpopl])
    elif typeplottdim == 'hist':
        listtypeplottdim = np.array(['hist' for u in indxpopl])
    elif typeplottdim == 'best':
        listtypeplottdim = np.empty(numbpopl, dtype=object)
        indxpoplmaxm = np.argmax(numbsamp)
        
        print('')
        print('')
        print('')
        print('numbsamp[indxpoplmaxm]')
        print(numbsamp[indxpoplmaxm])
        print('')
        print('')
        print('')

        if numbsamp[indxpoplmaxm] >= 1e3:
            listtypeplottdim[indxpoplmaxm] = 'hist'
        else:
            listtypeplottdim[indxpoplmaxm] = 'scat'
    
    strglisttypeplottdim = ''
    for temp in listtypeplottdim:
        strglisttypeplottdim += temp
    
    # sort the populations in decreasing order of size
    indxpopl = np.argsort(numbsamp)[::-1]

    for k in indxpara:
        for u in indxpopl:
            boolsampfini = np.isfinite(listpara[u][:, k])
            if listscalpara[k] == 'logt' and (listpara[u][boolsampfini, k] <= 0).any():
                print('Warning! Parameter %d (%s) has a log scaling but also nonpositive elements!' % (k, listlablpara[k][0]))
                print('Will reset its scaling to linear (self)...')
                listscalpara[k] = 'self'

    if listparadraw is not None:
        numbdraw = len(listparadraw)
        indxdraw = np.arange(numbdraw)
    
    # list of Booleans for each parameter indicating whether it is a list of integers
    boolinte = [[] for k in indxpara]
    for k in indxpara:
        boolinte[k] = True
        for u in indxpopl:
            if boolforcflot or not _is_integer_like_array(listpara[u][:, k], allow_nan=True):
                boolinte[k] = False
                break
    
    listsizepopl = []
    for u in indxpopl:
        listsizepopl.append(listpara[u][:, 0].size)
    
    if numbpopl > 1:
        for u in indxpopl:
            print('Number of samples in population %d (%s): %d' % (u, listlablpopl[u], listsizepopl[u]))
            if listlablsamp is not None and listsizepopl[u] < 100:
                print('Labels of these samples:')
                print(listlablsamp[u])
    
    for u in indxpopl:
        for k in indxpara:
            if not np.isfinite(listpara[u][:, k]).all():
                numbtotl = listpara[u][:, k].size
                numbfini = np.where(np.isfinite(listpara[u][:, k]))[0].size
                numbinfi = numbtotl - numbfini
                print('pcat.plot_population_grid(): %d out of %d samples (%.3g%%) are not finite for population %d (%s), parameter %d (%s)!' % \
                                                        (numbinfi, numbtotl, 100 * numbinfi / numbtotl, u, listlablpopl[u], k, listlablpara[k][0]))
    
    limtrims = [[] for k in indxpara]
    listindxgood = [[[] for k in indxpara] for u in indxpopl]
    if binsgridinpt is None:
        limt = [[] for k in indxpara]
        for k in indxpara:
            limt[k] = np.empty(2)
            limt[k][0] = 1e100
            limt[k][1] = -1e100
            have_valid = False
            for u in indxpopl:
                boolsampfini = np.isfinite(listpara[u][:, k])
                if listscalpara[k] == 'logt':
                    boolsampposi = listpara[u][:, k] > 0
                    listindxgood[u][k] = np.where(boolsampposi & boolsampfini)[0]
                    if (listpara[u][boolsampfini, k] <= 0).any():
                        print('')
                        print('')
                        print('')
                        print('listpara[u][:, k]')
                        summgene(listpara[u][:, k])
                        print('listindxgood[u][k]')
                        summgene(listindxgood[u][k])
                        print('np.where(boolsampfini)[0]')
                        summgene(np.where(boolsampfini)[0])
                        raise Exception('pcat.plot_population_grid(): Parameter %d (%s) has a log scaling but also nonpositive elements!' % (k, listlablpara[k]))
                else:
                    listindxgood[u][k] = np.where(boolsampfini)[0]

                if listindxgood[u][k].size > 0:
                    have_valid = True
                    valid_vals = listpara[u][listindxgood[u][k], k]
                    if valid_vals.size > 0:
                        limt[k][0] = min(limt[k][0], np.nanmin(valid_vals, 0))
                        limt[k][1] = max(limt[k][1], np.nanmax(valid_vals, 0))

            if not have_valid:
                limt[k][0] = 0.
                limt[k][1] = 1.

            if booldiag:
                if limt[k][0] == limt[k][1]:
                    print('')
                    print('')
                    print('')
                    print('')
                    print('listnamepara')
                    #print(listnamepara)
                    print('Warning! The lower and upper limits for parameters %s are the same: %g.' % (listlablpara[k][0], limt[k][0]))
                    print('listpara[u][:, k]')
                    summgene(listpara[u][:, k])
                    print('listpara[u][listindxgood[u][k], k]')
                    summgene(listpara[u][listindxgood[u][k], k])
                    print('')
                    #raise Exception('')

        # sanity checks
        if booldiag:
            for k in indxpara:
                #for u in indxpopl:
                #    if listpara[u][:, k].size == 0:
                #        print('')
                #        print('')
                #        print('')
                #        print('')
                #        print('')
                #        print('')
                #        print('')
                #        print('pcat.plot_population_grid(): size is 0 for parameter %d (%s)!' % (k, listlablpara[k][0]))
                #        print('k')
                #        print(k)
                #        print('listindxgood[u][k]')
                #        summgene(listindxgood[u][k])
                #        print('listpara[u][:, k]')
                #        summgene(listpara[u][:, k])
                #        print('listnamepara[k]')
                #        print(listnamepara[k])
                #        print('listlablpara[k]')
                #        print(listlablpara[k])
                #        print('listscalpara[k]')
                #        print(listscalpara[k])
                #        print('limt[k]')
                #        print(limt[k])
                #        raise Exception('')
                
                for u in indxpopl:
                    if listpara[u][:, k].size > 0:
                        minmtemp = np.amin(np.abs(listpara[u][:, k]))
                        if not np.isfinite(limt[k]).all() or minmtemp < 1e-100 and minmtemp > 0.:
                            print('')
                            print('')
                            print('')
                            print('pcat.plot_population_grid(): limit for parameter %d (%s) is infinite!' % (k, listlablpara[k][0]))
                            print('k')
                            print(k)
                            print('listindxgood[u][k]')
                            summgene(listindxgood[u][k])
                            print('listpara[u][:, k]')
                            summgene(listpara[u][:, k])
                            print('listlablpara[k]')
                            print(listlablpara[k])
                            print('listscalpara[k]')
                            print(listscalpara[k])
                            print('limt[k]')
                            print(limt[k])
                            print('minmtemp')
                            print(minmtemp)
                            #raise Exception('not np.isfinite(limt[k]).all() or minmtemp < 1e-100 and minmtemp > 0.')
                            print('Warning! Not np.isfinite(limt[k]).all() or minmtemp < 1e-100 and minmtemp > 0.')

        # sanity checks
        if booldiag:
            for k in indxpara:
                if not np.isfinite(limt[k]).all():
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('pcat.plot_population_grid(): limit for parameter %d (%s) is infinite!' % (k, listlablpara[k][0]))
                    print('k')
                    print(k)
                    print('numbpopl')
                    print(numbpopl)
                    for u in indxpopl:
                        print('listindxgood[u][k]')
                        summgene(listindxgood[u][k])
                        print('listpara[u][:, k]')
                        summgene(listpara[u][:, k])
                    print('listlablpara[k]')
                    print(listlablpara[k])
                    print('listscalpara[k]')
                    print(listscalpara[k])
                    print('limt[k]')
                    print(limt[k])
                    raise Exception('')

        if truepara is not None:
            for k in indxpara:
                if truepara[k] is not None:
                    if truepara[k] < limt[k][0]:
                        limt[k][0] = truepara[k] - 0.1 * (limt[k][1] - truepara[k]) 
                    if truepara[k] > limt[k][1]:
                        limt[k][1] = truepara[k] + 0.1 * (truepara[k] - limt[k][0])
    
        # limits that do not leave any room for white space which is good for histograms
        limtrims = np.copy(limt)

        if (listtypeplottdim == 'scat').any():
            # update limits to leave white space in the rims which is good for scatter plots
            for k in indxpara:
                
                if limt[k][0] == 1e100 and limt[k][1] == -1e100:
                    continue

                if listscalpara[k] == 'self':
                    if boolinte[k]:
                        delt = 0.5
                    else:
                        delt = 0.05 * (limt[k][1] - limt[k][0])
                    limt[k][0] -= delt
                    limt[k][1] += delt
                if listscalpara[k] == 'logt':
                    fact = np.exp(0.05 * np.log(limt[k][1] / limt[k][0]))
                    limt[k][0] /= fact
                    limt[k][1] *= fact
        
        ## if the limits are finite
        #for k in indxpara:
        #    if not np.isfinite(limt[k]).all():
        #        print('The limits for parameter %s were not finite. Fixing them to [0, 1]...' % listnamepara[k])
        #        limt[k][0] = 0.
        #        limt[k][1] = 1.
            
        # sanity checks
        if booldiag:
            for k in indxpara:
                if not np.isfinite(limt[k]).all():
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('pcat.plot_population_grid(): limit for parameter %d (%s) is infinite!' % (k, listlablpara[k][0]))
                    print('k')
                    print(k)
                    print('numbpopl')
                    print(numbpopl)
                    for u in indxpopl:
                        print('listindxgood[u][k]')
                        summgene(listindxgood[u][k])
                        print('listpara[u][:, k]')
                        summgene(listpara[u][:, k])
                    print('listlablpara[k]')
                    print(listlablpara[k])
                    print('listscalpara[k]')
                    print(listscalpara[k])
                    print('limt[k]')
                    print(limt[k])
                    raise Exception('')

        if False:
            for k in indxpara:
                if listlablpopl is not None:
                    for u in indxpopl:
                        if not np.isfinite(listpara[u][:, k]).all():
                            print('Warning! Population %d (%s), parameter %d (%s) has nonfinite samples.' % (u, listlablpopl[u], k, listlablpara[k][0]))
                            summgene(listpara[u][:, k])
    
        for k in indxpara:
            if limt[k][0] == limt[k][1]:
                print('')
                print('')
                print('')
                print('')
                print('pcat.plot_population_grid(): WARNING! Lower and upper limits are the same for the following parameter.')
                print('k')
                print(k)
                print('limt[k]')
                print(limt[k])
                print('listlablpara[k]')
                print(listlablpara[k])
                for u in indxpopl:
                    print('u')
                    print(u)
                    print('listpara[u]')
                    summgene(listpara[u])
                print('')
        
    if binsgridinpt is not None and numbpntsgrid is not None:
        raise Exception('')
    
    if binsgridinpt is None:
        binsgridinpt = []
        for k in indxpara:
            binsgridinpt.append(None)
        numbpntsgrid = 40
    else:
        numbpntsgrid = None
        limt = []
        for k in indxpara:
            limt.append(None)
    
    if boolplottria or boolplothistodim or boolplotpair and ((listtypeplottdim == 'hist').any() or (listtypeplottdim == 'kdee').any()):
        bins = [[] for k in indxpara]
        midpgrid = [[] for k in indxpara]
        
        for k in indxpara:
            
            if boolinte[k]:
                numbpntsgridtemp = None
            else:
                numbpntsgridtemp = numbpntsgrid
            
            bins[k], midpgrid[k], deltgrid, numbpntsgridtemp, indx = retr_axis(limt=limt[k], boolinte=boolinte[k], binsgrid=binsgridinpt[k], \
                                                                                                numbpntsgrid=numbpntsgridtemp, scalpara=listscalpara[k])
            
            if limt[k] is None:
                limt[k] = np.array([bins[k][0], bins[k][-1]])
                limtrims[k] = np.copy(limt[k])
            
            if booldiag:
                boolwarn = False
                boolstop = False
                if bins[k].size > 1e6:
                    boolwarn = True
                if bins[k][0] >= bins[k][-1]:
                    boolwarn = True
                if not np.isfinite(bins[k]).all():
                    boolstop = True
                    
                if boolstop or boolwarn:
                    print('')
                    print('')
                    print('')
                    print('k')
                    print(k)
                    print('listnamepara[k]')
                    print(listnamepara[k])
                    print('listlablpara[k]')
                    print(listlablpara[k])
                    print('listscalpara[k]')
                    print(listscalpara[k])
                    print('limt[k]')
                    print(limt[k])
                    print('binsgridinpt[k]')
                    summgene(binsgridinpt[k])
                    print('bins[k]')
                    summgene(bins[k])
                    for u in indxpopl:
                        print('listpara[u][:, k]')
                        summgene(listpara[u][:, k])
                        
                    if boolstop:
                        raise Exception('bins not good')
                    else:
                        print('Bins are not good, but skipping the exception...')
            
            if np.amin(bins[k]) == 0 and np.amax(bins[k]) == 0:
                print('')
                print('')
                print('')
                print('k')
                print(k)
                print('bins[k]')
                print(bins[k])
                print('limt[k]')
                print(limt[k])
                print('listscalpara[k]')
                print(listscalpara[k])
                print('listlablpara[k]')
                print(listlablpara[k])
                print('Lower and upper limits of the bins are the same for %s. Grid plot can fail, but skipping the exception...' % listlablpara[k][0])
    else:
        bins = None

    # list of Booleans indicating whether a parameter is good to plot
    boolparagood = np.ones(numbpara, dtype=bool)
    for k in indxpara:
        booltemp = False
        for u in indxpopl:
            if np.isfinite(listpara[u][:, k]).any():
                booltemp = True
        boolparagood[k] = booltemp
        if limt[k][0] == limt[k][1]:
            boolparagood[k] = False
        
    if boolplothistodim:
        
        if numbpopl == 1:
            factulimyaxihist = 1.
        else:
            factulimyaxihist = 1.2
            print('Multiple populations exist, which will require a legend. Will set factulimyaxihist to %g...' % factulimyaxihist)

        # one dimensional histograms
        for k in indxpara:
            if not boolparagood[k]:
                continue
            
            if pathbase is not None:
                path = pathbase + 'hist_%s_%s.%s' % (listnamepara[k], strgextn, typefileplot)
                if not os.path.exists(path) or boolwritover:
                    _plot_population_histodim(listmantlabl, listpara, k, listlablparatotl, indxpopl, listlablpopl, \
                                                bins, listcolrpopl, listparadraw, lablnumbsamp, lablsampgene, boolinte, \
                                                boolmakelegd, listscalpara, factulimyaxihist, titl, plotsize, limt, limtrims, boolcumu=False, path=path)
            if listnamefeatcumu is not None:
                if listnamepara[k] in listnamefeatcumu:
                    if pathbase is not None:
                        path = pathbase + 'histcumu_%s_%s.%s' % (listnamepara[k], strgextn, typefileplot)
                        if not os.path.exists(path) or boolwritover:
                            _plot_population_histodim(listmantlabl, listpara, k, listlablparatotl, indxpopl, listlablpopl, bins, \
                                                    listcolrpopl, listparadraw, lablnumbsamp, lablsampgene, boolinte, \
                                                    boolmakelegd, listscalpara, factulimyaxihist, titl, plotsize, limt, limtrims, boolcumu=True, path=path)
                    
    if boolplottria or boolplothistodim or boolplotpair:
        if not boolparagood[k]:
            print('Parameter %d is bad.')
            for u in indxpopl:
                print('listpara[u][:, k]')
                summgene(listpara[u][:, k])
    
    # make pie-chart of the populations if populations are mutually-exclusive
    if boolpoplexcl:
    
        def make_autopct(listsizepopl):
            
            def my_autopct(pct):
                
                total = sum(listsizepopl)
                val = int(round(pct*total/100.0))
                
                return '%.3g%%' % val
            
            return my_autopct
        
        if boolplotpies:
            
            def func(pct, allvals):
                absolute = int(pct/100.*np.sum(allvals))
                return "{:.1f}%\n({:d})".format(pct, absolute)

            figr, axis = plt.subplots(figsize=(plotsize, plotsize))
            objt = axis.pie(listsizepopl, labels=listlablpopl, \
                                                                    #autopct=make_autopct(listsizepopl), \
                                                                    colors=listcolrpopl, \
                                                                    autopct=lambda pct: func(pct, listsizepopl), \
                                                                    #autopct='%.3g%%', \
                                                                    )
            for kk in range(len(objt[0])):
                objt[0][kk].set_alpha(0.5)

            axis.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            if titl is not None:
                axis.set_title(titl)
            path = pathbase + 'pies_%s.%s' % (strgextn, typefileplot)
            print('Writing to %s...' % path)
            
            figr.savefig(path, bbox_inches='tight')
            plt.close(figr)
    
    if boolplotpair:
        if listlablsamp is None or numbpopl > 1 or typeplottdim == 'hist' or typeplottdim == 'kdee':
            liststrgtext = ['']
        else:
            liststrgtext = ['', '_anno']
        for strgtext in liststrgtext:
            if strgtext == '':
                listlablsamptemp = None
            else:
                listlablsamptemp = listlablsamp
            
            for k in indxpara:
                for l in indxpara:
                    if not boolparagood[k] or not boolparagood[l]:
                        print('Skipping the pair plot for parameter pair (%d, %d)...' % (k, l))
                        print('boolparagood[%d]' % k)
                        print(boolparagood[k])
                        print('boolparagood[%d]' % l)
                        print(boolparagood[l])
                        print('')
                        continue
                    if k <= l:
                        continue
                    
                    # skip the feature pair if specified by the user
                    if listnamefeatskip is not None:
                        boolskip = False
                        for gg in range(len(listnamefeatskip)):
                            if listnamepara[k] in listnamefeatskip[gg] and listnamepara[l] in listnamefeatskip[gg]:
                                boolskip = True
                        if boolskip:
                            continue
                    
                    if listnamepara is None or not (listnamepara[l] == 'rascstar' and listnamepara[k] == 'declstar'):
                        numbiter = 1
                    else:
                        numbiter = 2
                    
                    for e in range(numbiter):
                        
                        if e == 0:
                            projection = None
                            strgiter = ''
                        if e == 1:
                            projection = 'aitoff'
                            strgiter = '_aito'
                        if pathbase is not None:
                            path = pathbase + 'pmar_%s_%s_%s%s%s_%s.%s' % (listnamepara[k], \
                                                    listnamepara[l], strgextn, strgtext, strgiter, strglisttypeplottdim, typefileplot)
                        
                            if not os.path.exists(path) or boolwritover:
                            
                                figr = plt.figure(figsize=(plotsize, plotsize))
                                axis = figr.add_subplot(111, projection=projection)
                                
                                _plot_population_pair(k, l, axis, limt, listmantlabl, listpara, truepara, listparadraw, boolplotquan, listlablpara, \
                                                                 listscalpara, boolsqua, listvectplot, listtypeplottdim, indxpopl, listcolrpopl, listmsizpopl, \
                                                                 typeannosamp, listlablannosamp, \
                                                                 listmrkrpopl, listcolrpopltdim, listlablpopl, boolmakelegd, \
                                                                 bins=bins, midpgrid=midpgrid, \
                                                                 listlablsamp=listlablsamptemp, boolcbar=True)
                                
                                if e == 0:
                                    axis.set_xlim(limt[l])
                                    
                                    if booldiag:
                                        if limt[l][0] == limt[l][1] or limt[k][0] == limt[k][1]:
                                            print('')
                                            print('')
                                            print('')
                                            print('k, l')
                                            print(k, l)
                                            print('limt[k]')
                                            print(limt[k])
                                            print('limt[l]')
                                            print(limt[l])
                                            raise Exception('')
                                    
                                    axis.set_ylim(limt[k])
                
                                axis.set_xlabel(listlablparatotl[l])
                                axis.set_ylabel(listlablparatotl[k])
                                
                                if titl is not None:
                                    axis.set_title(titl)
                                
                                print('Writing to %s...' % path)
                                figr.savefig(path, bbox_inches='tight')
                                plt.close(figr)
    
    # number of population groups
    if typepgrp == 'together':
        numbpgrp = 1
    elif typepgrp == 'individual':
        numbpgrp = numbpopl
    elif typepgrp == 'both':
        numbpgrp = numbpopl + 1
    else:
        print('')
        print('')
        print('')
        raise Exception('typepgrp can only be "together", "individual", or "all".')
    indxpgrp = np.arange(numbpgrp)
    
    if boolplottria:
        
        for ou in indxpgrp:
            if typepgrp == 'together' or ou == numbpopl:
                strgpgrp = ''
                indxpopltemp = indxpopl
            elif typepgrp == 'individual':
                strgpgrp = '_%s' % listlablpopl[ou]
                indxpopltemp = np.array([ou])

            figr, axgr = plt.subplots(numbpara, numbpara, figsize=(0.6*plotsize*numbpara, 0.6*plotsize*numbpara))
            if numbpara == 1:
                axgr = [[axgr]]
            for k, axrw in enumerate(axgr):
                for l, axis in enumerate(axrw):
                    if not boolparagood[k] or not boolparagood[l]:
                        continue
                    if k < l:
                        axis.axis('off')
                        continue

                    if k == l:
                        _plot_population_diag(k, axis, listpara, truepara, listparadraw, boolplotquan, listlablpara, listtypeplottdim, indxpopltemp, \
                                                                                  listcolrpopl, listmrkrpopl, listlablpopl, boolmakelegd, listsizepopl, bins=bins)
                    else:
                        _plot_population_pair(k, l, axis, limt, listmantlabl, listpara, truepara, listparadraw, boolplotquan, listlablpara, \
                                                         listscalpara, boolsqua, listvectplot, listtypeplottdim, indxpopltemp, listcolrpopl, listmsizpopl, \
                                                         typeannosamp, listlablannosamp, \
                                                         listmrkrpopl, listcolrpopltdim, listlablpopl, boolmakelegd, \
                                                         bins=bins, midpgrid=midpgrid, \
                                                         boolcbar=False)
                        
                        axis.set_xlim(limt[l])
                        axis.set_ylim(limt[k])
                    
                        
                    if k == numbpara - 1:
                        axis.set_xlabel(listlablparatotl[l])
                    else:
                        axis.set_xticklabels([])
                    
                    if (l == 0 and k != 0):
                        axis.set_ylabel(listlablparatotl[k])
                    else:
                        if k != 0:
                            axis.set_yticklabels([])
            
            figr.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            if pathbase is not None:
                path = pathbase + 'pmar_%s%s_%s.%s' % (strgextn, strgpgrp, strglisttypeplottdim, typefileplot)
                print('Writing to %s...' % path)
                figr.savefig(path, dpi=300)
                plt.close(figr)
            else:
                plt.show()
