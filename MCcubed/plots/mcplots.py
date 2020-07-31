# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys, os
import six
import numpy as np
import scipy.interpolate as si
import matplotlib as mpl
import matplotlib.pyplot as plt

libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib')
sys.path.append(libdir)
import binarray as ba
mcdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mc')
sys.path.append(mcdir)
import credible_region as cr


__all__ = ["trace", "pairwise", "histogram", "RMS", "modelfit"]


def trace(allparams, title=None, parname=None, thinning=1,
          fignum=-10, savefile=None, fmt=".", sep=None, fs=24, 
          truepars=None):
  """
  Plot parameter trace MCMC sampling

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  fmt: String
     The format string for the line and marker.
  sep: Integer
     Number of samples per chain. If not None, draw a vertical line
     to mark the separation between the chains.

  Uncredited developers
  ---------------------
  Kevin Stevenson (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Ensure proper number of parameters for `truepars`
  if truepars is not None:
    if len(truepars) != npars:
      raise ValueError("True parameters passed to trace().\n" +              \
                       "But, it does not match the posterior dimensionality.")

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "|S%d"%namelen if six.PY2 else "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Get location for chains separations:
  xmax = len(allparams[0,0::thinning])
  if sep is not None:
    xsep = np.arange(sep/thinning, xmax, sep/thinning)

  # Make the trace plot:
  fig = plt.figure(fignum, figsize=(18, 1.2*npars))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=fs+4)

  plt.subplots_adjust(left=0.15, right=0.95, bottom=0.10, top=0.90,
                      hspace=0.4)

  for i in np.arange(npars):
    a = plt.subplot(npars, 1, i+1)
    #a.locator_params(tight=True, nbins=2)
    plt.plot(allparams[i, 0::thinning], fmt)
    yran = a.get_ylim()
    if sep is not None:
      plt.vlines(xsep, yran[0], yran[1], "0.3")
    plt.xlim(0, xmax)
    plt.ylabel(parname[i], size=fs, multialignment='center')
    plt.yticks(size=fs-6)
    if i == npars - 1:
      plt.xticks(size=fs-6)
      if thinning > 1:
        plt.xlabel('MCMC (thinned) iteration', size=fs)
      else:
        plt.xlabel('MCMC iteration', size=fs)
    else:
      plt.xticks(visible=False)
    plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    if truepars is not None:
      plt.gca().axhline(truepars[i], color='red', lw=3) # plot true param

  # Align labels
  fig.align_labels()

  if savefile is not None:
    plt.savefig(savefile, bbox_inches="tight")


def pairwise(allparams, title=None, parname=None, thinning=1,
             fignum=-11, savefile=None, style="hist", fs=34, nbins=20, 
             truepars=None, credreg=False, ptitle=False, ndec=None):
  """
  Plot parameter pairwise posterior distributions

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  style: String
     Choose between 'hist' to plot as histogram, or 'points' to plot
     the individual points.
  fs: Int
     Font size
  nbins: Int
     Number of bins for 2D histograms. 1D histograms will use 2*bins
  truepars: array.
     True parameters, if known.  Plots them in red.
  credreg: Boolean
     Determines whether or not to plot the credible regions.
  ptitle: Boolean.
     Controls subplot titles.
     If False, will not plot title. 
     If True, will plot the title as the 50% quantile +- the 68.27% region.
     If the 68.27% region consists of multiple disconnected regions, the 
     greatest extent will be used for the title.
     If `truepars` is given, the title also includes the true parameter value.
  ndec: None, or array of ints.
     If None, does nothing.
     If an array of ints, sets the number of decimal places to round each value
     in the title.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  Ryan Hardy  (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Ensure proper number of parameters for `truepars`
  if truepars is not None:
    if len(truepars) != npars:
      raise ValueError("True parameters passed to pairwise().\n" +           \
                       "But, it does not match the posterior dimensionality.")

  # Don't plot if there are no pairs:
  if npars == 1:
    return

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "|S%d"%namelen if six.PY2 else "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set palette color:
  palette = mpl.cm.get_cmap('YlOrRd', 256)
  palette.set_under(alpha=0.0)
  palette.set_bad(alpha=0.0)

  if credreg:
    hkw = {'histtype':'step', 'lw':0.0}
    cr_alpha = [1.0, 0.65, 0.4]
  else:
    hkw = {'edgecolor':'navy', 'color':'b'}

  fig = plt.figure(fignum, figsize=(18, 18))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=fs+4)

  h = 1 # Subplot index
  plt.subplots_adjust(left=0.15,   right=0.95, bottom=0.15, top=0.9,
                      hspace=0.20, wspace=0.20)

  for   j in np.arange(npars): # Rows
    for i in np.arange(npars): # Columns
      if j > i or j == i:
        a = plt.subplot(npars, npars, h)
        # Y labels:
        if i == 0 and j != 0:
          plt.yticks(size=fs-8)
          plt.ylabel(parname[j], size=fs, multialignment='center')
        elif i == 0 and j == 0:
          plt.yticks(visible=False)
          plt.ylabel(parname[j], size=fs, multialignment='center')
        else:
          a = plt.yticks(visible=False)
        # X labels:
        if j == npars-1:
          plt.xticks(size=fs-8, rotation=90)
          plt.xlabel(parname[i], size=fs)
        else:
          a = plt.xticks(visible=False)
        # The plot:
        if style=="hist":
          # 2D histogram
          if j > i:
            hist2d, xedges, yedges = np.histogram2d(allparams[i, 0::thinning],
                                                    allparams[j, 0::thinning], 
                                                    nbins, density=False)
            vmin = 0.0
            hist2d[np.where(hist2d == 0)] = np.nan
            a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], yedges[0],
                           yedges[-1]), cmap=palette, vmin=vmin, aspect='auto',
                           origin='lower', interpolation='bilinear')
            if truepars is not None: # plot true params
              plt.plot(truepars[i], truepars[j], '*', color='red', ms=20, 
                       markeredgecolor='black', markeredgewidth=1)
          # 1D histogram
          else:
            vals, bins, hh = plt.hist(allparams[i,0::thinning], 2*nbins, 
                                     density=False, **hkw)
        elif style=="points":
          if j > i:
            a = plt.plot(allparams[i], allparams[j], ",")
            if truepars is not None: # plot true params
              plt.plot(truepars[i], truepars[j], '*', color='red', ms=20, 
                       markeredgecolor='black', markeredgewidth=1)
          else:
            vals, bins, hh = plt.hist(allparams[i,0::thinning], 2*nbins, 
                                     density=False, **hkw)
        # Plotting credible regions, true params for 1D hist, and titles
        if j <= i:
          if credreg:
            pdf, xpdf, CRlo, CRhi = cr.credregion(allparams[i,0::thinning], 
                                         percentile=[0.68269, 0.95450, 0.99730])
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            f    = si.interp1d(bins+0.5 * (bins[1]-bins[0]), vals, kind='nearest')
            # Plot credible regions as shaded areas
            for k in range(len(CRlo)):
              for r in range(len(CRlo[k])):
                plt.gca().fill_between(xpdf, 0, f(xpdf),
                                       where=(xpdf>=CRlo[k][r]) * \
                                             (xpdf<=CRhi[k][r]), 
                                       facecolor=(0.1, 0.4, 0.75, cr_alpha[k]), 
                                       edgecolor='none', interpolate=False, 
                                       zorder=-2+2*k)
            if ptitle:
              med = np.median(allparams[i,0::thinning])
              if ndec is not None:
                lo  = np.around(CRlo[-1][ 0]-med, ndec[i])
                hi  = np.around(CRhi[-1][-1]-med, ndec[i])
                med = np.around(med, ndec[i])
              else:
                lo  = CRlo[-1][ 0]-med
                hi  = CRhi[-1][-1]-med
              titlestr = parname[i] + r' = $'+str(med)+'_{{{0:+g}}}^{{{1:+g}}}$'.format(lo, hi)
          if truepars is not None:
            plt.gca().axvline(truepars[i], color='red', lw=3) # plot true param
            if ptitle and credreg:
              if ndec is not None:
                titlestr = 'True value: '+str(np.around(truepars[i], ndec[i]))+\
                           '\n'+titlestr
              else:
                titlestr = 'True value: '+str(truepars[i])+'\n'+titlestr
          if ptitle and credreg:
            plt.title(titlestr, fontsize=fs-18)
          if i==0:
            # Add labels & legend
            sig1 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, cr_alpha[0]), 
                                     label='$68.27\%$ region')
            sig2 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, cr_alpha[1]), 
                                     label='$95.45\%$ region')
            sig3 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, cr_alpha[2]), 
                                     label='$99.73\%$ region')
            hndls = [sig1, sig2, sig3]
            if truepars is not None:
              hndls = hndls + [mpl.lines.Line2D([], [], color='red', lw=4, 
                                                marker='*', ms=20, 
                                                markeredgecolor='black', 
                                                markeredgewidth=1, 
                                                label='True value')]
            plt.legend(handles=hndls, prop={'size':fs-14}, 
                       bbox_to_anchor=(1, 1.05), ncol=2)

        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))

      h += 1

  # Align labels
  fig.align_labels()

  # The colorbar:
  if style == "hist":
    if npars > 2:
      a = plt.subplot(2, 6, 5, frameon=False)
      a.yaxis.set_visible(False)
      a.xaxis.set_visible(False)
    bounds = np.linspace(0, 1.0, 64)
    norm = mpl.colors.BoundaryNorm(bounds, palette.N)
    ax2 = fig.add_axes([0.8, 0.5, 0.025, 0.36])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
          spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Normalized point density", fontsize=fs)
    cb.set_ticks(np.linspace(0, 1, 5))
    cb.ax.tick_params(labelsize=fs-8)
    plt.draw()

  # Save file:
  if savefile is not None:
    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


def histogram(allparams, title=None, parname=None, thinning=1,
              fignum=-12, savefile=None, fs=34, nbins=40, 
              truepars=None, credreg=False, ptitle=False):
  """
  Plot parameter marginal posterior distributions

  Parameters
  ----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  nbins: Integer
     Number of bins for the histogram.
  credreg: Boolean
     Determines whether or not to plot the credible regions.
  ptitle: Boolean, or string.
     Controls the subplot titles.
     If False, will not plot title. Otherwise, puts `title` as the plot title.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Ensure proper number of parameters for `truepars`
  if truepars is not None:
    if len(truepars) != npars:
      raise ValueError("True parameters passed to histogram().\n" +          \
                       "But, it does not match the posterior dimensionality.")

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "|S%d"%namelen if six.PY2 else "<U%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set number of rows:
  if npars < 10:
    nperrow = 3
  else:
    nperrow = 4
  nrows = (npars - 1)//nperrow + 1
  # Set number of columns:
  if   npars > 9:
    ncolumns = 4
  elif npars > 4:
    ncolumns = 3
  else:
    ncolumns = (npars+2)//3 + (npars+2)%3  # (Trust me!)

  histheight = 4 + 4*(nrows)
  if nrows == 1:
    bottom = 0.25
  else:
    bottom = 0.15

  fig = plt.figure(fignum, figsize=(18, histheight))
  plt.clf()
  plt.subplots_adjust(left=0.1, right=0.95, bottom=bottom, top=0.9,
                      hspace=1.0, wspace=0.1)

  if title is not None:
    a = plt.suptitle(title, size=fs+4)

  if credreg:
    hkw = {'histtype':'step', 'lw':0.0}
    cr_alpha = [1.0, 0.65, 0.4]
  else:
    hkw = {'edgecolor':'navy', 'color':'b'}

  maxylim = 0  # Max Y limit
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    a  = plt.xticks(size=fs-6, rotation=90)
    if i%ncolumns == 0:
      a = plt.yticks(size=fs-6)
    else:
      a = plt.yticks(visible=False)
    plt.xlabel(parname[i], size=fs)
    vals, bins, h = ax.hist(allparams[i,0::thinning], nbins, density=False, 
                            **hkw)
    if credreg:
      pdf, xpdf, CRlo, CRhi = cr.credregion(allparams[i,0::thinning], 
                                   percentile=[0.68269, 0.95450, 0.99730])
      vals = np.r_[0, vals, 0]
      bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
      f    = si.interp1d(bins+0.5 * (bins[1]-bins[0]), vals, kind='nearest')
      # Plot credible regions as shaded areas
      for k in range(len(CRlo)):
        for r in range(len(CRlo[k])):
          ax.fill_between(xpdf, 0, f(xpdf),
                                 where=(xpdf>=CRlo[k][r]) * \
                                       (xpdf<=CRhi[k][r]), 
                                 facecolor=(0.1, 0.4, 0.75, cr_alpha[k]), 
                                 edgecolor='none', interpolate=False, zorder=-2+2*k)
    maxylim = np.amax((maxylim, ax.get_ylim()[1]))

  # Add labels
  sig1 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, 1.0), label='$68.27\%$ region')
  sig2 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, 0.65), label='$95.45\%$ region')
  sig3 = mpl.patches.Patch(color=(0.1, 0.4, 0.75, 0.4), label='$99.73\%$ region')
  hndls = [sig1, sig2, sig3]
  if truepars is not None:
    hndls = hndls + [mpl.lines.Line2D([], [], color='red', lw=4, label='True value')]
  plt.legend(handles=hndls, prop={'size':fs/1.5}, loc='upper left', 
             bbox_to_anchor=(1, 0.8))

  # Set uniform height:
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    ax.set_ylim(0, maxylim)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
    # Add true value if given
    if truepars is not None:
        ax.axvline(truepars[i], color='red', lw=4, zorder=20)
  fig.align_labels() #Align axis labels

  if savefile is not None:
    plt.savefig(savefile, bbox_inches='tight')


def RMS(binsz, rms, stderr, rmserr, cadence=None, binstep=1,
        timepoints=[], ratio=False, fignum=-20,
        yran=None, xran=None, savefile=None):
  """
  Plot the RMS vs binsize

  Parameters
  ----------
  binsz: 1D ndarray
     Array of bin sizes.
  rms: 1D ndarray
     RMS of dataset at given binsz.
  stderr: 1D ndarray
     Gaussian-noise rms Extrapolation
  rmserr: 1D ndarray
     RMS uncertainty
  cadence: Float
     Time between datapoints in seconds.
  binstep: Integer
     Plot every-binstep point.
  timepoints: List
     Plot a vertical line at each time-points.
  ratio: Boolean
     If True, plot rms/stderr, else, plot both curves.
  fignum: Integer
     Figure number
  yran: 2-elements tuple
     Minimum and Maximum y-axis ranges.
  xran: 2-elements tuple
     Minimum and Maximum x-axis ranges.
  savefile: String
     If not None, name of file to save the plot.

  Uncredited developers
  ---------------------
  Kevin Stevenson  (UCF)
  """

  if np.size(rms) <= 1:
    return

  # Set cadence:
  if cadence is None:
    cadence = 1.0
    xlabel = "Bin size"
  else:
    xlabel = "Bin size  (sec)"

  # Set plotting limits:
  if yran is None:
    #yran = np.amin(rms), np.amax(rms)
    yran = [np.amin(rms-rmserr), np.amax(rms+rmserr)]
    yran[0] = np.amin([yran[0],stderr[-1]])
    if ratio:
      yran = [0, np.amax(rms/stderr) + 1.0]
  if xran is None:
    xran = [cadence, np.amax(binsz*cadence)]

  fs = 14 # Font size
  if ratio:
    ylabel = r"$\beta =$ RMS / std. error"
  else:
    ylabel = "RMS"

  plt.figure(fignum, (8,6))
  plt.clf()
  ax = plt.subplot(111)

  if ratio: # Plot the residuals-to-Gaussian RMS ratio:
    a = plt.errorbar(binsz[::binstep]*cadence, (rms/stderr)[::binstep],
                     (rmserr/stderr)[::binstep], fmt='k-', ecolor='0.5',
                     capsize=0, label="__nolabel__")
    a = plt.semilogx(xran, [1,1], "r-", lw=2)
  else:     # Plot residuals and Gaussian RMS individually:
    # Residuals RMS:
    a = plt.errorbar(binsz[::binstep]*cadence, rms[::binstep],
                     rmserr[::binstep], fmt='k-', ecolor='0.5',
                     capsize=0, label="RMS")
    # Gaussian noise projection:
    a = plt.loglog(binsz*cadence, stderr, color='red', ls='-',
                   lw=2, label="Gaussian std.")
    a = plt.legend()
  for time in timepoints:
    a = plt.vlines(time, yran[0], yran[1], 'b', 'dashed', lw=2)

  a = plt.yticks(size=fs)
  a = plt.xticks(size=fs)
  a = plt.ylim(yran)
  a = plt.xlim(xran)
  a = plt.ylabel(ylabel, fontsize=fs)
  a = plt.xlabel(xlabel, fontsize=fs)

  if savefile is not None:
    plt.savefig(savefile)


def modelfit(data, uncert, indparams, model, nbins=75, title=None,
             fignum=-22, savefile=None):
  """
  Plot the model and (binned) data arrays, and their residuals.

  Parameters
  ----------
  data: 1D float ndarray
     The data array.
  uncert: 1D float ndarray
     Uncertainties of the data-array values.
  indparams: 1D float ndarray
     X-axis values of the data-array values.
  model: 1D ndarray
     The model of data (evaluated at indparams values).
  nbins: Integer
     Output number of data binned values.
  title: String
     Plot title.
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  """

  # Bin down array:
  binsize = (np.size(data)-1)/nbins + 1
  bindata, binuncert, binindp = ba.binarray(data, uncert, indparams, binsize)
  binmodel = ba.weightedbin(model, binsize)
  fs = 14 # Font-size

  p = plt.figure(fignum, figsize=(8,6))
  p = plt.clf()

  # Residuals:
  a = plt.axes([0.15, 0.1, 0.8, 0.2])
  p = plt.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
  p = plt.plot([indparams[0], indparams[-1]], [0,0],'k:',lw=1.5)
  p = plt.xticks(size=fs)
  p = plt.yticks(size=fs)
  p = plt.xlabel("x", size=fs)
  p = plt.ylabel('Residuals', size=fs)

  # Data and Model:
  a = plt.axes([0.15, 0.35, 0.8, 0.55])
  if title is not None:
    p = plt.title(title, size=fs)
  p = plt.errorbar(binindp, bindata, binuncert, fmt='ko', ms=4,
                   label='Binned Data')
  p = plt.plot(indparams, model, "b", lw=2, label='Best Fit')
  p = plt.setp(a.get_xticklabels(), visible = False)
  p = plt.yticks(size=13)
  p = plt.ylabel('y', size=fs)
  p = plt.legend(loc='best')

  if savefile is not None:
      p = plt.savefig(savefile)

