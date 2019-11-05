# -*- coding: utf-8 -*-
"""
This module implements plotting functions for NeuroChaT analyses.

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import itertools
import math
import logging
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from neurochat.nc_utils import find, angle_between_points, get_axona_colours

BLUE = '#1f77b4'
RED = '#d62728'

def scatterplot_matrix(_data, names=[], **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "_data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, _ = _data.shape
    fig, axs = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axs.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the _data.
    for i, j in zip(*np.triu_indices_from(axs, k=1)):
        for x, y in [(i, j), (j, i)]:
            axs[y, x].scatter(_data[x], _data[y], **kwargs)

    # Label the diagonal subplots...
    if len(names) == numvars:
        for i, label in enumerate(names):
            axs[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',\
               ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axs[j, i].xaxis.set_visible(True)
        axs[i, j].yaxis.set_visible(True)

def set_backend(backend):
    """
    Sets the  backend of Matplotlib

    Parameters
    ----------
    backend : str
        The new backend for Matplotlib

    Returns
    -------
    None

    See also
    --------
    matplotlib.pyplot.switch_backend()

    """

    if backend:
        plt.switch_backend(backend)

def wave_property(wave_data, plots=[2, 2]):
    """
    Plots mean +/-std of waveforms in electrode groups

    Parameters
    ----------
    wave_data : dict
        Graphical data from the Waveform analysis
    plots : list
        Subplot shape. [2, 2] for tetrode setup

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Matlab figure object

    """

    # Wave property analysis
    fig1, ax = plt.subplots(plots[0], plots[1])
    ax = ax.flatten()
    # Plot waves
    for i in np.arange(len(ax)):
        ax[i].plot(wave_data['Mean wave'][:, i], color='black', linewidth=2.0)
        ax[i].plot(wave_data['Mean wave'][:, i]+wave_data['Std wave'][:, i],\
          color='green', linestyle='dashed')
        ax[i].plot(wave_data['Mean wave'][:, i]-wave_data['Std wave'][:, i],\
          color='green', linestyle='dashed')

    return fig1

def largest_waveform(wave_data, ax=None):
    """
    Plot the largest waveform in electrode groups.

    Parameters
    ----------
    wave_data : dict
        Graphical data form the Waveform analysis
    ax : matplotlib.axes.Axes
        Optional axes to plot to
    Returns
    -------
    matplotlib.pyplot.Figure
        The figure plotted to, or None if an axes is provided
    """
    ax, fig = _make_ax_if_none(ax)

    mean_wave = wave_data['Mean wave'][:, wave_data["Max channel"]]
    std_wave = wave_data['Std wave'][:, wave_data["Max channel"]]
    ax.plot(mean_wave, color='black', linewidth=2.0)
    ax.plot(mean_wave+std_wave, color='green', linestyle='dashed')
    ax.plot(mean_wave-std_wave, color='green', linestyle='dashed')

    return fig

def isi(isi_data, axes=[None, None, None], **kwargs):
    """
    Plots Interspike interval histogram and scatter plots of interval-before
    vs interval-after.

    Parameters
    ----------
    isi_data : dict
        Graphical data from the ISI analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Histogram of ISI
    fig2 : matplotlib.pyplot.Figure
        Scatter plot of ISI-before vs ISI-after in loglog scale
    fig3 : matplotlib.pyplot.Figure
        2D histogram of the ISI-before vs ISI-after in log-log scale

    """

    # Plot ISI
    # histogram
    title = kwargs.get("title1", 'Distribution of inter-spike interval')
    xlabel = kwargs.get("xlabel1", 'ISI (ms)')
    ylabel = kwargs.get("ylabel1", 'Spike count')
    burst_ms = kwargs.get("burst_ms", 5)
    s_color = kwargs.get("should_color", True)
    raster = kwargs.get("raster", True)

    if s_color:
        color = "darkblue"
        edgecolor = "darkblue"
    else:
        color = "k"
        edgecolor = "k"
    ax, fig1 = _make_ax_if_none(axes[0])
    width = np.mean(np.diff(isi_data['isiBins']))
    ax.bar(isi_data['isiBins'], isi_data['isiHist'], color=color,
           edgecolor=edgecolor, rasterized=raster, align="edge", 
           width=width, linewidth=0)
    ax.plot(
        [burst_ms, burst_ms], [0, isi_data['maxCount']], 
        linestyle='dashed', linewidth=2, color='red')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    max_axis = isi_data['isiBins'].max()
    max_axisLog = np.ceil(np.log10(max_axis))

    ## ISI scatterplot
    ax, fig2 = _make_ax_if_none(axes[1])
    ax.loglog(isi_data['isiBefore'], isi_data['isiAfter'], axes=ax, \
            linestyle=' ', marker='o', markersize=1, \
            markeredgecolor='k', markerfacecolor=None, rasterized=raster)
#    ax.autoscale(enable= True, axis= 'both', tight= True)
    ax.plot(
        ax.get_xlim(), [burst_ms, burst_ms], 
        linestyle='dashed', linewidth=2, color='red')
    ax.set_aspect(1)
    #    ax.set_xlabel('Interval before (ms)')
    ax.set_ylabel('Interval after (ms)')
    ax.set_xlabel('Interval before (ms)')
    ax.set_title('Distribution of ISI \n (before and after spike)')

    #
    logBins = np.logspace(0, max_axisLog, max_axisLog*70)
    joint_count, xedges, yedges = np.histogram2d(isi_data['isiBefore'],\
                                isi_data['isiAfter'], bins=logBins)

    # Scatter colored
    _extent = [xedges[0], xedges[-2], yedges[0], yedges[-2]]

#        ax = fig2.add_subplot(212, aspect= 'equal')
    ax, fig3 = _make_ax_if_none(axes[2])
    c_map = plt.cm.jet
    c_map.set_under('white')
    ax.pcolormesh(xedges[0:-1], yedges[0:-1], joint_count,\
                  cmap=c_map, vmin=1, rasterized=raster)
    ax.plot(
        ax.get_xlim(), [burst_ms, burst_ms], 
        linestyle='dashed', linewidth=2, color='red')
    plt.axis(_extent)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')
    ax.set_xlabel('Interval before (ms)')
    ax.set_ylabel('Interval after (ms)')
    ax.set_title('Distribution of ISI \n (before and after spike)')

    return fig1, fig2, fig3

def isi_corr(isi_corr_data, ax=None, **kwargs):
    """
    Plots ISI correlation.

    Parameters
    ----------
    isi_corr_data : dict
        Graphical data from the ISI correlation
    ax : matplotlib.axes.Axes
        Optional axes object to plot to.
    kwargs :
        title : str

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        ISI correlation histogram

    """
    isi_time = abs(isi_corr_data['isiCorrBins'].min())
    default_title = (
        'Autocorrelation Histogram \n ({}ms)'.format(str(isi_time)))
    title = kwargs.get("title", default_title)
    xlabel = kwargs.get("xlabel", "Time (ms)")
    ylabel = kwargs.get("ylabel", "Counts")
    plot_theta = kwargs.get("plot_theta", False)
    raster = kwargs.get("raster", True)

    ax, fig = _make_ax_if_none(ax)

    # show_edges = False
    # line_width = 1 if show_edges else 0
    line_width = 0
    all_bins = isi_corr_data['isiAllCorrBins']
    width = np.mean(np.diff(all_bins))
    bin_centres = [
        (all_bins[i+1] + all_bins[i]) / 2 for i in range(len(all_bins) - 1)]
    ax.bar(bin_centres, isi_corr_data['isiCorr'],
           width=width, linewidth=line_width, color='darkblue',
           edgecolor='black', rasterized=raster, align='center')
    ax.tick_params(width=1.5)

    if plot_theta:
        ax.plot(
            bin_centres, isi_corr_data['corrFit'], 
            linewidth=2, color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig

def theta_cell(plot_data, ax=None, **kwargs):
    """
    Plots theta-modulated cell and theta-skipping cell analysis data

    Parameters
    ----------
    plot_data : dict
        Graphical data from the theta-modulated cell
    ax : matplotlib.axes.Axes
        Optional axes object to plot to.
    kwargs :
        title : str

    Returns
    -------
    matplotlib.pyplot.Figure
        ISI correlation histogram superimposed with fitted sinusoidal curve.

    """
    return isi_corr(plot_data, ax=ax, plot_theta=True, **kwargs)

def lfp_spectrum(plot_data):
    """
    Plots LFP spectrum analysis data

    Parameters
    ----------
    plot_data : dict
        Graphical data from the ISI correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Line plot of LFP spectrum using Welch's method

    """


    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(plot_data['f'], plot_data['Pxx'], linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    _extent = [0, plot_data['f'].max(), 0, plot_data['Pxx'].max()]
    plt.axis(_extent)

    return fig1

def lfp_spectrum_tr(plot_data):
    """
    Plots time-resolved LFP spectrum analysis data

    Parameters
    ----------
    plot_data : dict
        Graphical data from the ISI correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        3D plot of short-term FFT of the LFP signal

    """

    fig1 = plt.figure()
    ax = plt.gca()
    c_map = plt.cm.jet
    pcm = ax.pcolormesh(plot_data['t'], plot_data['f'], plot_data['Sxx'], cmap=c_map)
    _extent = [0, plot_data['t'].max(), 0, plot_data['f'].max()]
    plt.axis(_extent)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    fig1.colorbar(pcm)

    return fig1

def plv(plv_data):
    """
    Plots the analysis replay_data of Phase-locking value (PLV)

    Parameters
    ----------
    plv_data : dict
        Graphical data from the PLV analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Plot of spike-triggered average (STA)
    fig2 : matplotlib.pyplot.Figure
        Plot of FFT of STA (fSTA), average power spectrum of spike-triggered LFP signals (STP),
        spike-field coherence and PLV in four subplots

    """

    f = plv_data['f']
    t = plv_data['t']
    STA = plv_data['STA']
    fSTA = plv_data['fSTA']
    STP = plv_data['STP']
    SFC = plv_data['SFC']
    PLV = plv_data['PLV']

    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(t, STA, linewidth=2, color='darkblue')
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Spike-triggered average (STA)')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('STA (uV)')

    fig2 = plt.figure()
    ax = fig2.add_subplot(221)
    ax.plot(f, fSTA, linewidth=2, color='darkblue')
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('fft of STA')

    ax = fig2.add_subplot(222)
    ax.plot(f, STP, linewidth=2, color='darkblue')
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('STP')

    ax = fig2.add_subplot(223)
    ax.plot(f, SFC, linewidth=2, color='darkblue')
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('SFC')

    ax = fig2.add_subplot(224)
    ax.plot(f, PLV, linewidth=2, color='darkblue')
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('PLV')

    for ax in fig2.axes:
        ax.set_xlabel('Frequency (Hz)')

    fig2.suptitle('Frequency analysis of spike-triggered lfp metrics')

    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    return fig1, fig2

def plv_tr(plv_data):
    """
    Plots the analysis replay_data of time-resolved Phase-locking value (PLV)

    Parameters
    ----------
    plv_data : dict
        Graphical data from the time-resolved PLV analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Plot of fSTA
    fig2 : matplotlib.pyplot.Figure
        Plot of STP
    fig3 : matplotlib.pyplot.Figure
        Plot of SFC

    """

    offset = plv_data['offset']
    f = plv_data['f']
    fSTA = plv_data['fSTA']
#        STP= plv_data['STP']
    SFC = plv_data['SFC']
    PLV = plv_data['PLV']

    fig1 = plt.figure()
    ax = plt.gca()
    c_map = plt.cm.jet
    pcm = ax.pcolormesh(offset, f, fSTA, cmap=c_map, rasterized=True)
    plt.title('Time-resolved fSTA')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    fig1.colorbar(pcm)

    fig2 = plt.figure()
    ax = plt.gca()
    c_map = plt.cm.jet
    pcm = ax.pcolormesh(offset, f, SFC, cmap=c_map, rasterized=True)
    plt.title('Time-resolved SFC')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    fig2.colorbar(pcm)

    fig3 = plt.figure()
    ax = plt.gca()
    c_map = plt.cm.jet
    pcm = ax.pcolormesh(offset, f, PLV, cmap=c_map, rasterized=True)
    plt.title('Time-resolved PLV')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    fig3.colorbar(pcm)

    return fig1, fig2, fig3

def plv_bs(plv_data):
    """
    Plots the analysis replay_data of bootstrapped Phase-locking value (PLV)

    Parameters
    ----------
    plv_data : dict
        Graphical data from the time-resolved PLV analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Plot of fSTA
    fig2 : matplotlib.pyplot.Figure
        Plot of STP
    fig3 : matplotlib.pyplot.Figure
        Plot of SFC

    """
    f = plv_data['f']
    t = plv_data['t']
    STAm = plv_data['STAm']
    fSTAm = plv_data['fSTAm']
    STPm = plv_data['STPm']
    SFCm = plv_data['SFCm']
    PLVm = plv_data['PLVm']

    STAe = plv_data['STAe']
    fSTAe = plv_data['fSTAe']
    STPe = plv_data['STPe']
    SFCe = plv_data['SFCe']
    PLVe = plv_data['PLVe']

    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(t, STAm, linewidth=2, color='darkblue', marker='o', \
                 markerfacecolor='darkblue', markeredgecolor='none')
    ax.fill_between(t, STAm- STAe, STAm+ STAe, \
                 facecolor='cornflowerblue', alpha=0.5,\
                 edgecolor='none', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Spike-triggered average (STA)')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('STA (uV)')

    fig2 = plt.figure()
    ax = fig2.add_subplot(221)
    ax.plot(f, fSTAm, linewidth=2, color='darkblue', marker='.', rasterized=True)
    ax.fill_between(f, fSTAm- fSTAe, fSTAm+ fSTAe, \
                 facecolor='cornflowerblue', alpha=0.5,\
                 edgecolor='none', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('fft of STA')

    ax = fig2.add_subplot(222)
    ax.plot(f, STPm, linewidth=2, color='darkblue', marker='.', rasterized=True)
    ax.fill_between(f, STPm- STPe, STPm+ STPe, \
                 facecolor='cornflowerblue', alpha=0.5,\
                 edgecolor='none', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('STP')

    ax = fig2.add_subplot(223)
    ax.plot(f, SFCm, linewidth=2, color='darkblue', marker='.', rasterized=True)
    ax.fill_between(f, SFCm- SFCe, SFCm+ SFCe, \
                 facecolor='cornflowerblue', alpha=0.5,\
                 edgecolor='none', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('SFC')

    ax = fig2.add_subplot(224)
    ax.plot(f, PLVm, linewidth=2, color='darkblue', marker='.', rasterized=True)
    ax.fill_between(f, PLVm- PLVe, PLVm+ PLVe, \
                 facecolor='cornflowerblue', alpha=0.5,\
                 edgecolor='none', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('PLV')

    for ax in fig2.axes:
        ax.set_xlabel('Frequency (Hz)')

    fig2.suptitle('Frequency analysis of spike-triggered lfp metrics (bootstrap)')
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    return fig1, fig2

def spike_phase(phase_data):
    """
    Plots the analysis replay_data of spike-LFP phase locking

    Parameters
    ----------
    phase_data : dict
        Graphical data from the spike-LFP phase locking analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Phase histogram
    fig2 : matplotlib.pyplot.Figure
        Phase distribution in circular plot
    fig3 : matplotlib.pyplot.Figure
        Phase-raster in one subplot, phase histogram in another

    """
    phBins = phase_data['phBins']
    phCount = phase_data['phCount']

    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(np.append(phBins, phBins+ 360), np.append(phCount, phCount), \
           color='slateblue', width=np.diff(phBins).mean(),\
           alpha=0.6, align='center', rasterized=True)
    ax.plot(np.append(phBins, phBins+ 360), 0.5*np.max(phCount)*(np.cos(np.append(phBins, phBins+ 360)*np.pi/180)+ 1), \
            color='red', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('LFP phase distribution (red= reference cosine line)')
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Spike count')

    fig2 = plt.figure()
    ax = plt.gca(polar=True)
    ax.bar(phBins*np.pi/180, phCount, width=3*np.pi/180, color='blue',\
           alpha=0.6, bottom=np.max(phase_data['phCount'])/2, rasterized=True)
    ax.plot([0, phase_data['meanTheta']], [0, 1.5*np.max(phCount)], \
            linewidth=3, color='red', marker='.')
    plt.title('LFP phase distribution (red= mean direction)')

    fig3 = plt.figure()
    ax = fig3.add_subplot(211)
    #cdict= {'blue': (0, 0, 1),
    #       'white': (0, 0, 0)}
    #c_map = mcol.LinearSegmentedColormap('my_colormap', cdict, 256)
    ax.pcolormesh(phase_data['rasterbins'], np.arange(0, phase_data['raster'].shape[0]), \
                  phase_data['raster'], cmap=plt.cm.binary, rasterized=True)
    
    # Alternative idea for plotting, not currently working. 
    # rasters = phase_data['raster']
    # bin_length = np.mean(np.diff(phase_data['raster'], 0))

    # for idx, row in enumerate(rasters):
    #      rasters[idx] = [
    #          j_idx*(bin_length) +0.5*bin_length if j == 1 else 0 for 
    #             j_idx, j in enumerate(row)]
    # ax.eventplot(rasters)
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.title('Phase raster')
    ax.set_ylabel('Time')

    ax = fig3.add_subplot(212)
    ax.bar(phBins, phCount, color='slateblue', \
           width=np.diff(phBins).mean(), alpha=0.6, align='center', rasterized=True)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel('Phase(deg)')
    ax.set_ylabel('Spike count')

    return fig1, fig2, fig3

def speed(speed_data):
    """
    Plots the speed of the animal vs spike rate

    Parameters
    ----------
    speed_data : dict
        Graphical data from the unit firing to speed correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Scatter plot of speed vs spike-rate superimposed with fitted rate

    """

    ## Speed analysis
    fig1 = plt.figure()
    ax = plt.gca()
    ax.scatter(speed_data['bins'], speed_data['rate'], c=BLUE, zorder=1)
    ax.plot(speed_data['bins'], speed_data['fitRate'], color=RED, linewidth=1.5, zorder=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Speed vs Spiking Rate')
    ax.set_xlabel('Speed (cm/sec)')
    ax.set_ylabel('Spikes/sec')

    return fig1

def angular_velocity(angVel_data):
    """
    Plots the angular head velocity of the animal vs spike rate

    Parameters
    ----------
    angVel_data : dict
        Graphical data from the unit firing to angular head velocity correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Scatter plot of angular velocity vs spike-rate superimposed with fitted rate

    """

    ## Angular velocity analysis
    fig1 = plt.figure()
    ax = plt.gca()
    ax.scatter(angVel_data['leftBins'], angVel_data['leftRate'], c=BLUE, zorder=1)
    ax.plot(angVel_data['leftBins'], angVel_data['leftFitRate'], color=RED, linewidth=1.5, zorder=2)
    ax.scatter(angVel_data['rightBins'], angVel_data['rightRate'], c=BLUE, zorder=1)
    ax.plot(angVel_data['rightBins'], angVel_data['rightFitRate'], color=RED, linewidth=1.5, zorder=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Angular Velocity vs Spiking Rate')
    ax.set_xlabel('Angular velocity (deg/sec)')
    ax.set_ylabel('Spikes/sec')

    return fig1

def multiple_regression(mra_data):
    """
    Plots the replay_data of multiple regression analysis.

    Parameters
    ----------
    mra_data : dict
        Graphical data from multiple regression analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Bar plot of multiple regression replay_data

    """

    varOrder = ['Total', 'Loc', 'HD', 'Speed', 'Ang Vel', 'Dist Border']
    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(np.arange(6), mra_data['meanRsq'], color='royalblue', align='center')
    ax.errorbar(np.arange(6), mra_data['meanRsq'], fmt='ro',\
                yerr=mra_data['stdRsq'], ecolor='k', elinewidth=3)
    ax.set_title('Multiple regression scores')
    ax.set_ylabel('$R^2$')
    plt.xticks(np.arange(6), varOrder)
    plt.autoscale(enable=True, axis='both', tight=True)

    return fig1

def hd_rate(hd_data, ax=None, **kwargs):
    """
    Plot head direction vs spike rate.

    Parameters
    ----------
    hd_data : dict
        Graphical data from the unit firing to head-direction correlation
    ax : matplotlib.axes.Axes
        Polar Axes object. If specified, the figure is plotted in this axes.
    kwargs :

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes of the polar plot of head-direction vs spike-rate.

    """
    title = kwargs.get("title", "Head directional firing rate")
    if not ax:
        plt.figure()
        ax = plt.gca(polar=True)

    bins = np.append(hd_data['bins'], hd_data['bins'][0])
    rate = np.append(hd_data['smoothRate'], hd_data['smoothRate'][0])
    ax.plot(np.radians(bins), rate, color=BLUE)

    ax.set_title(title)
    ax.set_rticks([hd_data['hdRate'].max()])

    return ax

def hd_spike(hd_data, ax=None):
    """
    Plots the head-direction of the animal at the time of spiking-events in polar scatter plot.

    Parameters
    ----------
    hd_data : dict
        Graphical data from the unit firing to head-direction correlation
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the polar plot of head-direction during spiking events.

    """

    if not ax:
        plt.figure()
        ax = plt.gca(polar=True)

    ax.scatter(np.radians(hd_data['scatter_bins']), hd_data['scatter_radius'], \
             s=1, c=RED, alpha=0.75, edgecolors='none', rasterized=True)
    ax.set_rticks([])
    ax.spines['polar'].set_visible(False)

    return ax

def hd_firing(hd_data):
    """
    Plots the analysis replay_data of head directional correlation to spike-rate

    Parameters
    ----------
    hd_data : dict
        Graphical data from the unit firing to head-directional correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Polar plot of head-direction during spiking-events
    fig2 : matplotlib.pyplot.Figure
        Polar plot of head-direction vs spike-rate. Predicted firing rate is also plotted.

    """

    fig1 = plt.figure()
    hd_spike(hd_data, ax=plt.gca(polar=True))

    fig2 = plt.figure()
    ax2 = hd_rate(hd_data, ax=plt.gca(polar=True))
    bins = np.append(hd_data['bins'], hd_data['bins'][0])
    predRate = np.append(hd_data['hdPred'], hd_data['hdPred'][0])
    ax2.plot(np.radians(bins), predRate, color='green')
    ax2.set_rticks([hd_data['hdRate'].max(), hd_data['hdPred'].max()])

    return fig1, fig2

def hd_rate_ccw(hd_data):
    """
    Plots the analysis replay_data of head directional correlation to spike-rate
    but split into counterclockwise and clockwise head-movements.

    Parameters
    ----------
    hd_data : dict
        Graphical data from the unit firing to head-direction correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Polar plot of head-direction vs spike-rate in  clockwise and counterclockwise
        head movements

    """

    fig1 = plt.figure()
    ax = plt.gca(polar=True)
    ax.plot(np.radians(hd_data['bins']), hd_data['hdRateCW'], color=BLUE)
    ax.plot(np.radians(hd_data['bins']), hd_data['hdRateCCW'], color=RED)
    ax.set_title('Counter/clockwise firing rate')
    ax.set_rticks([hd_data['hdRateCW'].max(), hd_data['hdRateCCW'].max()])

    return fig1

def hd_shuffle(hd_shuffle_data):
    """
    Plots the analysis outcome of head directional shuffling analysis

    Parameters
    ----------
    hd_shuffle_data : dict
        Graphical data from head-directional shuffling anlaysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Distribution of Rayleigh Z statistics
    fig2 : matplotlib.pyplot.Figure
        Distribution of Von Mises concentration parameter Kapppa

    """

    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(hd_shuffle_data['raylZEdges'], hd_shuffle_data['raylZCount'],\
           color='slateblue', alpha=0.6,\
           width=np.diff(hd_shuffle_data['raylZEdges']).mean(), rasterized=True)
    ax.plot([hd_shuffle_data['raylZPer95'], hd_shuffle_data['raylZPer95']], \
            [0, hd_shuffle_data['raylZCount'].max()], color='red', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Rayleigh Z distribution for shuffled spikes (red= 95th percentile)')
    ax.set_xlabel('Rayleigh Z score')
    ax.set_ylabel('Count')

    fig2 = plt.figure()
    ax = plt.gca()
    ax.bar(hd_shuffle_data['vonMisesKEdges'], hd_shuffle_data['vonMisesKCount'],\
           color='slateblue', alpha=0.6, \
           width=np.diff(hd_shuffle_data['vonMisesKEdges']).mean(), rasterized=True)
    ax.plot([hd_shuffle_data['vonMisesKPer95'], hd_shuffle_data['vonMisesKPer95']], \
            [0, hd_shuffle_data['vonMisesKCount'].max()], color='red', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('von Mises kappa distribution for shuffled spikes \n (red= 95th percentile)')
    ax.set_xlabel('von Mises kappa')
    ax.set_ylabel('Count')

    return fig1, fig2

def hd_spike_time_lapse(hd_data):
    """
    Plots the analysis outcome of head directional time-lapse analysis

    Parameters
    ----------
    hd_data : dict
        Graphical data from head-directional time-lapsed anlaysis

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure
        Time-lapsed spike-plots

    """


    keys = [key[1] for key in list(enumerate(hd_data.keys()))]
    fig = []
    axs = []
    keys = list(hd_data.keys())
    nkey = len(keys)
    nfig = int(np.ceil(nkey/4))
    for _ in range(nfig):
        f, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
        plt.subplots_adjust(top=0.9, hspace=0.55)
        fig.append(f)
        axs.extend(list(ax.flatten()))
        
    for i, ax in enumerate(axs):
        if i < len(keys):
            key = keys[i]
            nice_key = _nice_lapse_key(key)
            ax = hd_spike(hd_data[key], ax=ax)
            ax.set_title(nice_key, y=1, fontsize=10)
        else:
            ax.set_visible(False)
    return fig

def hd_rate_time_lapse(hd_data):
    """
    Plots the analysis outcome of head directional time-lapse analysis

    Parameters
    ----------
    hd_data : dict
        Graphical data from head-directional time-lapsed anlaysis

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure
        Time-lapsed head-drectional firing rate plot

    """

    keys = [key[1] for key in list(enumerate(hd_data.keys()))]
    fig = []
    axs = []
    keys = list(hd_data.keys())
    nkey = len(keys)
    nfig = int(np.ceil(nkey/4))
    for _ in range(nfig):
        f, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
        plt.subplots_adjust(top=0.9, hspace=0.55)
        fig.append(f)
        axs.extend(list(ax.flatten()))
        
    for i, ax in enumerate(axs):
        if i < len(keys):
            key = keys[i]
            nice_key = _nice_lapse_key(key)
            ax = hd_rate(hd_data[key], ax=ax)
            ax.set_title(nice_key, y=1, fontsize=10)
        else:
            ax.set_visible(False)
    return fig

def _nice_lapse_key(key):
    parts = key.split("To")
    end = parts[1][-3:] if parts[1][-3:].lower() == "end" else (
        parts[1][0] + " " + parts[1][-3:])
    nice_key = parts[0] + " to " + end
    return nice_key

def hd_time_shift(hd_shift_data):
    """
    Plots the analysis outcome of head directional time-shift analysis

    Parameters
    ----------
    hd_shift_data : dict
        Graphical data from head-directional time-shift anlaysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Skaggs information content of head directional firing in shifted time of spiking events
    fig2 : matplotlib.pyplot.Figure
        Peak firing rate of head directional firing in shifted time of spiking events
    fig3 : matplotlib.pyplot.Figure
        Skaggs information content of head directional firing in shifted time of spiking events
    """

    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(hd_shift_data['shiftTime'], hd_shift_data['skaggs'],\
            marker='o', markerfacecolor=RED, linewidth=2)
    ax.set_xlabel('Shift time (ms)')
    ax.set_ylabel('Skaggs IC')
    ax.set_title('Skaggs IC of hd firing in shifted time of spiking events')

    fig2 = plt.figure()
    ax = plt.gca()
    ax.plot(hd_shift_data['shiftTime'], hd_shift_data['peakRate'],\
            marker='o', markerfacecolor=RED, linewidth=2)
    ax.set_xlabel('Shift time (ms)')
    ax.set_ylabel('Peak firing rate (spikes/sec)')
    ax.set_title('Peak FR of hd firing in shifted time of spiking events')

    fig3 = plt.figure()
    ax = plt.gca()
    ax.scatter(hd_shift_data['shiftTime'], hd_shift_data['delta'], c=RED, zorder=3)
    ax.plot(hd_shift_data['shiftTime'], hd_shift_data['deltaFit'], color=BLUE, linewidth=1.5, zorder=1)
    ax.plot(hd_shift_data['shiftTime'], np.zeros(hd_shift_data['shiftTime'].size), color='k', linestyle='--', linewidth=1.5, zorder=2)
    ax.set_xlabel('Shift time (ms)')
    ax.set_ylabel('Delta (degree)')
    ax.set_title('Delta of hd firing in shifted time of spiking events')

    return fig1, fig2, fig3

def loc_spike(place_data, ax=None, **kwargs):
    """
    Plots the location of spiking-events (spike-plot) along with the trace of animal in the enviroment.

    Parameters
    ----------
    place_data : dict
        Graphical data from the correlation of unit firing to location of the animal
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the spike-plot

    """
    # default_point_size = max(
    #     place_data['yedges'].max() - place_data['yedges'].min(),
    #     place_data['xedges'].max() - place_data['xedges'].min()
    # ) / 10
    default_point_size = 2

    color = kwargs.get("color", RED)
    point_size = kwargs.get("point_size", default_point_size)
    # spatial firing map
    if not ax:
        plt.figure()
        ax = plt.gca()

    ax.plot(place_data['posX'], place_data['posY'], color='black', zorder=1)
    ax.scatter(place_data['spikeLoc'][0], place_data['spikeLoc'][1], \
               s=point_size, marker='o', color=color, zorder=2)
    ax.set_ylim([0, place_data['yedges'].max()])
    ax.set_xlim([0, place_data['xedges'].max()])
    #asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    #ax.set_aspect(asp)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    return ax

def loc_rate(place_data, ax=None, smooth=True, **kwargs):
    """
    Plots location vs spike rate

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to locational correlation
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.
    kwargs :
        colormap : str
            viridis is used if not specified
            "default" uses the standard red green intensity colours
            but these are bad for colorblindness.
        style : str
            What kind of map to plot - can be
            "contour", "digitized" or "interpolated"
        levels : int
            Number of contour regions.
    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the firing rate map

    """
    colormap = kwargs.get("colormap", "viridis")
    style = kwargs.get("style", "contour")
    levels = kwargs.get("levels", 5)
    raster = kwargs.get("raster", True)
    splits = None

    if colormap is "default":
        clist = [(0.0, 0.0, 1.0),\
                (0.0, 1.0, 0.5),\
                (0.9, 1.0, 0.0),\
                (1.0, 0.75, 0.0),\
                (0.9, 0.0, 0.0)]
        colormap = mcol.ListedColormap(clist)

    ax, fig = _make_ax_if_none(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    if smooth:
        fmap = place_data['smoothMap']
    else:
        fmap = place_data['firingMap']

    if style == "digitized":
        res = ax.pcolormesh(
            place_data['xedges'], place_data['yedges'],
            np.ma.array(fmap, mask=np.isnan(fmap)),
            cmap=colormap, rasterized=raster)

    elif style == "interpolated":
        extent = (
            0, place_data['xedges'].max(),
            0, place_data['yedges'].max())
        tp = fmap[:-1, :-1]
        res = ax.imshow(
            tp, cmap=colormap,
            extent=extent, interpolation="bicubic",
            origin="lower")

    elif style == "contour":
        dx = np.mean(np.diff(place_data['xedges']))
        dy = np.mean(np.diff(place_data['yedges']))
        pad_map = np.pad(fmap[:-1, :-1], ((1, 1), (1, 1)), "edge")
        vmin, vmax = np.nanmin(pad_map), np.nanmax(pad_map)
        if vmin != vmax:
            splits = np.linspace(vmin, vmax, levels+1)
        else:
            splits = np.array([vmin, vmin*2])
        x_edges = np.append(
            place_data["xedges"] - dx/2,
            place_data["xedges"][-1] + dx/2)
        y_edges = np.append(
            place_data["yedges"] - dy/2,
            place_data["yedges"][-1] + dy/2)
        res = ax.contourf(
            x_edges, y_edges,
            np.ma.array(pad_map, mask=np.isnan(pad_map)),
            levels=splits, cmap=colormap, corner_mask=True)

        # This produces it with no padding
        # res = ax.contourf(
        #     place_data['xedges'][:-1] + dx / 2.,
        #     place_data['yedges'][:-1] + dy / 2.,
        #     np.ma.array(fmap[:-1, :-1], mask=np.isnan(fmap[:-1, :-1])),
        #     levels=15, cmap=colormap, corner_mask=True)

    else:
        logging.error("Unrecognised style passed to loc_rate")
        return

    ax.set_ylim([0, place_data['yedges'].max()])
    ax.set_xlim([0, place_data['xedges'].max()])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    cbar = plt.colorbar(res, cax=cax, orientation='vertical', use_gridspec=True)
    # cbar.ax.set_ticks(levels)
    # cbar.ax.set_yticklabels(np.around(levels, decimals=1))
    if splits is not None:
        split_text = np.around(splits, decimals=1)
        cbar.ax.set_yticklabels(split_text)

    return ax

def loc_firing(place_data):
    """
    Plots the analysis replay_data of locational correlation to spike-rate

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to head-directional correlation

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Spike-plot and firing rate map in two subplots respectively

    """
    fig = plt.figure()

    ax = loc_spike(place_data, ax=fig.add_subplot(121))
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')

    ax = loc_rate(place_data, ax=fig.add_subplot(122))
    ax.set_xlabel('cm')
    #ax.set_ylabel('YLoc')
#    fig.colorbar(cax)
    fig.set_tight_layout(True)
    return fig

# Created by Sean Martin: 14/02/2019
def loc_firing_and_place(place_data, smooth=True):
    """
    Plots the analysis replay_data of locational correlation to spike-rate
    with a place map

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to head-directional correlation

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Spike-plot and firing rate map and place field in three subplots respectively

    """
    fig = plt.figure()

    ax1 = loc_spike(place_data, ax=fig.add_subplot(131))
    ax1.set_xlabel('cm')
    ax1.set_ylabel('cm')

    ax2 = loc_rate(place_data, ax=fig.add_subplot(132, sharey=ax1), smooth=smooth)
    ax2.set_xlabel('cm')

    ax3 = loc_place_field(place_data, ax=fig.add_subplot(133, sharey=ax1))
    ax3.set_xlabel('cm')

    fig.set_tight_layout(True)
    plt.subplots_adjust(wspace=0.25)
    return fig

# Created by Sean Martin: 13/02/2019
def loc_place_field(place_data, ax=None):
    """
    Plots the location of the place field(s) of the unit.

    Parameters
    ----------
    place_data : dict
        Graphical data from the correlation of unit firing to location of the animal
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the spike-plot

    """

    # spatial place field
    ax, _ = _make_ax_if_none(ax)
    clist = [(0.0, 0.0, 1.0),\
            (0.0, 1.0, 0.5),\
            (0.9, 1.0, 0.0),\
            (1.0, 0.75, 0.0),\
            (0.9, 0.0, 0.0)]
    c_map = mcol.ListedColormap(clist)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    mask = (place_data['placeField'] == 0)
    pmap= ax.pcolormesh(place_data['xedges'], place_data['yedges'],
                        np.ma.array(place_data['placeField'], mask=mask),
                        cmap=c_map, rasterized=True)
    ax.set_ylim([0, place_data['yedges'].max()])
    ax.set_xlim([0, place_data['xedges'].max()])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    centroid = place_data['centroid']
    ax.plot([centroid[0]], [centroid[1]], 'gX')
    plt.colorbar(pmap, cax=cax, orientation='vertical', use_gridspec=True)
#        plt.autoscale(enable=True, axis='both', tight=True)
    return ax

# Created by Sean Martin: 13/02/2019
def loc_place_centroid(place_data, centroid):
    """
    Plots the analysis replay_data of locational correlation to spike-rate
    along with the centroid of the place field.

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to head-directional correlation
    centroid : ndarray
        The centroid of the place field

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Spike-plot and firing rate map in two subplots respectively

    """
    fig = plt.figure()

    ax = loc_spike(place_data, ax=fig.add_subplot(121))
    ax.plot([centroid[0]], [centroid[1]], 'gX')
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')

    ax = loc_rate(place_data, ax=fig.add_subplot(122))
    ax.plot([centroid[0]], [centroid[1]], 'gX')
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    plt.tight_layout()
    return fig

def loc_spike_time_lapse(place_data):
    """
    Plots the analysis outcome of locational time-lapse analysis

    Parameters
    ----------
    place_data : dict
        Graphical data from locational time-lapsed anlaysis

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure
        Time-lapsed spike-plots

    """

    keys = [key[1] for key in list(enumerate(place_data.keys()))]
    fig = []
    axs = []
    keys = list(place_data.keys())
    nkey = len(keys)
    nfig = int(np.ceil(nkey/4))
    for _ in range(nfig):
        f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        fig.append(f)
        axs.extend(list(ax.flatten()))

    for i, ax in enumerate(axs):
        if i < len(keys):
            key = keys[i]
            loc_spike(place_data[key], ax=ax)
            nice_key = _nice_lapse_key(key)
            ax.set_title(nice_key)
        else:
            ax.set_visible(False)

    return fig

def loc_rate_time_lapse(place_data):
    """
    Plots the analysis outcome of locational time-lapse analysis

    Parameters
    ----------
    place_data : dict
        Graphical data from locational time-lapsed anlaysis

    Returns
    -------
    fig : list of matplotlib.pyplot.Figure
        Time-lapsed firing-rate map

    """

    keys = [key[1] for key in list(enumerate(place_data.keys()))]

    fig = []
    axs = []
    keys = list(place_data.keys())
    nkey = len(keys)
    nfig = int(np.ceil(nkey/4))
    for _ in range(nfig):
        f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        fig.append(f)
        axs.extend(list(ax.flatten()))

    for i, ax in enumerate(axs):
        if i < len(keys):
            key = keys[i]
            loc_rate(place_data[key], ax=ax)
            nice_key = _nice_lapse_key(key)
            ax.set_title(nice_key)
        else:
            ax.set_visible(False)

    return fig

def loc_shuffle(loc_shuffle_data):
    """
    Plots the analysis outcome of locational shuffling analysis

    Parameters
    ----------
    loc_shuffle_data : dict
        Graphical data from head-directional shuffling anlaysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Distribution of Skaggs IC, sparsity and spatial coherecne in three subplots

    """

    # Loactional shuffling analysis
    fig1 = plt.figure()
#    ax= plt.gca()
    ax = fig1.add_subplot(221)
    ax.bar(loc_shuffle_data['skaggsEdges'][:-1], loc_shuffle_data['skaggsCount'], color='slateblue', alpha=0.6, \
           width=np.diff(loc_shuffle_data['skaggsEdges']).mean(), rasterized=True)
    ax.plot([loc_shuffle_data['skaggs95'], loc_shuffle_data['skaggs95']], \
            [0, loc_shuffle_data['skaggsCount'].max()], color='red', linewidth=2)
#        ax.plot([loc_shuffle_data['refSkaggs'], loc_shuffle_data['refSkaggs']], \
#                [0, loc_shuffle_data['skaggsCount'].max()], color='green', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel('Skaggs IC')
#    ax.set_ylabel('Count', fontsize=12)

    ax = fig1.add_subplot(222)
    ax.bar(loc_shuffle_data['sparsityEdges'][:-1], loc_shuffle_data['sparsityCount'], color='slateblue', alpha=0.6, \
                    width=np.diff(loc_shuffle_data['sparsityEdges']).mean(), rasterized=True)
    ax.plot([loc_shuffle_data['sparsity05'], loc_shuffle_data['sparsity05']], \
            [0, loc_shuffle_data['sparsityCount'].max()], color='red', linewidth=2)
#        ax.plot([loc_shuffle_data['refSparsity'], loc_shuffle_data['refSparsity']], \
#                [0, loc_shuffle_data['sparsityCount'].max()], color='green', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel('Sparsity')

    ax = fig1.add_subplot(223)
    ax.bar(loc_shuffle_data['coherenceEdges'][:-1], loc_shuffle_data['coherenceCount'], color='slateblue', alpha=0.6, \
                    width=np.diff(loc_shuffle_data['coherenceEdges']).mean(), rasterized=True)
    ax.plot([loc_shuffle_data['coherence95'], loc_shuffle_data['coherence95']], \
            [0, loc_shuffle_data['coherenceCount'].max()], color='red', linewidth=2)
#        ax.plot([loc_shuffle_data['refCoherence'], loc_shuffle_data['refCoherence']], \
#                [0, loc_shuffle_data['coherenceCount'].max()], color='green', linewidth=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel('Coherence')

#    i= 0
#    labels= ['Skaggs IC', 'Sparsity', 'Coherence']
    for ax in fig1.axes:
        ax.set_ylabel('Count')

    fig1.suptitle('Distribution of locational firing specificity indices')
    plt.subplots_adjust(hspace=0.35, wspace=0.23)

    return fig1

def loc_time_shift(loc_shift_data):
    """
    Plots the analysis outcome of locational time-shift analysis

    Parameters
    ----------
    loc_shift_data : dict
        Graphical data from head-directional time-shift anlaysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Skaggs information content of locational firing in shifted time of spiking events
    fig2 : matplotlib.pyplot.Figure
        Sparsity of locational firing in shifted time of spiking events
    fig3 : matplotlib.pyplot.Figure
        Coherence of locational firing in shifted time of spiking events

    """

    ## Locational time shift analysis

    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(loc_shift_data['shiftTime'], loc_shift_data['skaggs'], linewidth=2, zorder=1)
    ax.scatter(loc_shift_data['shiftTime'], loc_shift_data['skaggs'], marker='o', color=RED, zorder=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('Skaggs IC')
    ax.set_xlabel('Shift time (ms)')
    ax.set_title('Skaggs IC of place firing in shifted time of spiking events')

    fig2 = plt.figure()
    ax = plt.gca()
    ax.plot(loc_shift_data['shiftTime'], loc_shift_data['sparsity'], linewidth=2, zorder=1)
    ax.scatter(loc_shift_data['shiftTime'], loc_shift_data['sparsity'], marker='o', color=RED, zorder=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('Sparsity')
    ax.set_xlabel('Shift time (ms)')
    ax.set_title('Sparsity of place firing in shifted time of spiking events')

    fig3 = plt.figure()
    ax = plt.gca()
    ax.plot(loc_shift_data['shiftTime'], loc_shift_data['coherence'], linewidth=2, zorder=1)
    ax.scatter(loc_shift_data['shiftTime'], loc_shift_data['coherence'], marker='o', color=RED, zorder=2)
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylabel('Coherence')
    ax.set_xlabel('Shift time (ms)')
    ax.set_title('Coherence of place firing in shifted time of spiking events')

    #        for ax in fig1.axes:
    #            ax.set_xlabel('shift time')
    #        fig1.suptitle('Specifity indices in time shift')

    return fig1, fig2, fig3

def loc_auto_corr(locAuto_data):
    """
    Plots the analysis outcome of locational firing rate autocorrelation

    Parameters
    ----------
    locAuto_data : dict
        Graphical data from spatial correlation of firing map

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Spatial correlation map

    """
    # Locational firing map autocorrelation
    clist = [(1.0, 1.0, 1.0),\
            (0.0, 0.0, 0.5),\
            (0.0, 0.0, 1.0),\
            (0.0, 0.5, 1.0),\
            (0.0, 0.75, 1.0),\
            (0.5, 1.0, 0.0),\
            (0.9, 1.0, 0.0),\
            (1.0, 0.75, 0.0),\
            (1.0, 0.4, 0.0),\
            (1.0, 0.0, 0.0),\
            (0.5, 0.0, 0.0)]

    c_map = mcol.ListedColormap(clist)

    fig1 = plt.figure()
    ax = fig1.gca()
    pc = ax.pcolormesh(locAuto_data['xshift'], locAuto_data['yshift'], np.ma.array(locAuto_data['corrMap'], \
                    mask=np.isnan(locAuto_data['corrMap'])), cmap=c_map, rasterized=True)
    ax.set_title('Spatial correlation of firing intensity map)')
    ax.set_xlabel('X-lag')
    ax.set_ylabel('Y-lag')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.colorbar(pc)

    return fig1

def rot_corr(plot_data):
    """
    Plots the analysis outcome of rotational correlation of spatial autocorrelation map.

    Parameters
    ----------
    plot_data : dict
        Graphical data from spatial correlation of firing map

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Rotational correlation plot

    """

    # Location firing map rotational analysis
    fig1 = plt.figure()
    ax = fig1.gca()
    ax.plot(plot_data['rotAngle'], plot_data['rotCorr'], linewidth=2, zorder=1)
    ax.set_ylim([-1, 1])
    ax.set_xlim([0, 360])
    #plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Rotational correlation of spatial firing map')
    ax.set_xlabel('Rotation angle')
    ax.set_ylabel('Pearson correlation')

    return fig1

def dist_rate(dist_data):
    """
    Plots the firing rate vs distance from border

    Parameters
    ----------
    dist_data : dict
        Graphical data from border and gradient analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Distance from border vs spike rate

    """

    fig1 = plt.figure()
    ax = plt.gca()
    ax.plot(dist_data['distBins'], dist_data['smoothRate'], marker='o', markerfacecolor=RED, linewidth=2, label='Firing rate')
    if 'rateFit' in dist_data.keys():
        ax.plot(dist_data['distBins'], dist_data['rateFit'], 'go-', markerfacecolor='brown', linewidth=2, label='Fitted rate')
    ax.set_title('Distance from border vs spike rate')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Rate (spikes/sec)')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.legend(loc='lower right')

    return fig1

def stair_plot(dist_data):
    """
    Plots the stairs of mean distance vs firing-rate bands

    Parameters
    ----------
    dist_data : dict
        Graphical data from border and gradient analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Mean distance distance from border vs firing-rate percentage

    """

    perSteps = dist_data['perSteps']
    perDist = dist_data['perDist']
    stepsize = np.diff(perSteps).mean()

    fig1 = plt.figure()
    ax = plt.gca()
    for i, step in enumerate(perSteps):
        ax.plot([step, step+ stepsize], [perDist[i], perDist[i]], color='b', linestyle='--', marker='o', markerfacecolor=RED, linewidth=2)
        if i > 0: #perSteps.shape[0]:
            ax.plot([step, step], [perDist[i-1], perDist[i]], color='b', linestyle='--', linewidth=2)
    ax.set_xlabel('% firing rate (spikes/sec)')
    ax.set_ylabel('Mean distance (cm)')

    return fig1

def border(border_data):
    """
    Plots the analysis replay_data from border analysis

    Parameters
    ----------
    border_data : dict
        Graphical data from border analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Histogram of taxicab distance of active pixels
    fig2 : matplotlib.pyplot.Figure
        Angular distance of pixels vs active pixel count
    fig3 : matplotlib.pyplot.Figure
        Distance from border vs spike rate
    fig4 : matplotlib.pyplot.Figure
        Mean distance distance from border vs firing-rate percentage

    """

    fig1 = plt.figure()
    ax = fig1.add_subplot(211)
    ax.bar(border_data['distBins'], border_data['distCount'], color='slateblue', alpha=0.6, \
                    width=0.5*np.diff(border_data['distBins']).mean())
    ax.set_title('Histogram of taxicab distance of active pixels')
    ax.set_xlabel('Taxicab distance(cm)')
    ax.set_ylabel('Active pixel count')

    ax = fig1.add_subplot(212)
    ax.bar(border_data['circBins'], border_data['angDistCount'], color='slateblue', alpha=0.6, \
                    width=0.5*np.diff(border_data['circBins']).mean())
    ax.set_title('Angular distance vs Active pixel count')
    ax.set_xlabel('Angular distance')
    ax.set_ylabel('Active pixel count')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.subplots_adjust(hspace=0.5)
    fig2 = plt.figure()
    ax = plt.gca()
    pcm = ax.pcolormesh(border_data['cBinsInterp'], border_data['dBinsInterp'], \
                       border_data['circLinMap'], cmap='seismic', rasterized=True)
    ax.invert_yaxis()
    plt.autoscale(enable=True, axis='both', tight=True)
    ax.set_title('Histogram for angle vs distance from border of active pixels')
    ax.set_xlabel('Angular distance (Deg)')
    ax.set_ylabel('Taxicab distance (cm)')
    fig2.colorbar(pcm)

    fig3 = dist_rate(border_data)
    fig4 = stair_plot(border_data)

    return fig1, fig2, fig3, fig4

def gradient(gradient_data):
    """
    Plots the replay_data from gradient cell analysis

    Parameters
    ----------
    border_data : dict
        Graphical data from border analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Distance from border vs spike rate
    fig2 : matplotlib.pyplot.Figure
        Differential firing rate vs distance from border
    fig3 : matplotlib.pyplot.Figure
        Mean distance distance from border vs firing-rate percentage

    """

    fig1 = dist_rate(gradient_data)

    fig2 = plt.figure()
    ax = plt.gca()
    ax.plot(gradient_data['distBins'], gradient_data['diffRate'], color=BLUE, marker='o', markerfacecolor=RED, linewidth=2)
    ax.set_title('Differential firing rate (fitted)')
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Differential rate (spikes/sec)')
    plt.autoscale(enable=True, axis='both', tight=True)

    fig3 = stair_plot(gradient_data)

    return fig1, fig2, fig3

def grid(grid_data):
    """
    Plots the replay_data from grid analysis

    Parameters
    ----------
    grid_data : dict
        Graphical data from border analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Autocorrelation of firing rate map, superimposed with central peaks
    fig2 : matplotlib.pyplot.Figure
        Rotational correlation of autocorrelation map

    """

    fig1 = loc_auto_corr(grid_data)
    ax = fig1.axes[0]

    xmax = grid_data['xmax']
    ymax = grid_data['ymax']
    xshift = grid_data['xshift']

    ax.scatter(xmax, ymax, c='black', marker='s', zorder=2)
    for i in range(xmax.size):
        if i < xmax.size-1:
            ax.plot([xmax[i], xmax[i+1]], [ymax[i], ymax[i+1]], 'k', linewidth=2)
        else:
            ax.plot([xmax[i], xmax[0]], [ymax[i], ymax[0]], 'k', linewidth=2)
    ax.plot(xshift[xshift >= 0], np.zeros(find(xshift >= 0).size), 'k--', linewidth=2)
    ax.plot(xshift[xshift >= 0], xshift[xshift >= 0]*ymax[0]/ xmax[0], 'k--', linewidth=2)
    ax.set_title('Grid cell analysis')
    ax.set_xlim([grid_data['xshift'].min(), grid_data['xshift'].max()])
    ax.set_ylim([grid_data['yshift'].min(), grid_data['yshift'].max()])
    ax.invert_yaxis()

    fig2 = None
    if 'rotAngle' in grid_data.keys() and 'rotCorr' in grid_data.keys():
        fig2 = rot_corr(grid_data)
        ax = fig2.axes[0]
        rmax = grid_data['rotCorr'].max()
        rmin = grid_data['rotCorr'].min()
        for i, th in enumerate(grid_data['anglemax']):
            ax.plot([th, th], [rmin, rmax], 'r--', linewidth=1)
        for i, th in enumerate(grid_data['anglemin']):
            ax.plot([th, th], [rmin, rmax], 'g--', linewidth=1)

        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Rotational correlation of autocorrelation map')

    if fig2:
        return fig1, fig2
    else:
        return fig1

def spike_raster(events, xlim=None, colors=[0, 0, 0], ax=None, **kwargs):
    """
    Plots the spike raster for a number of units

    Parameters
    ----------
    events : list
        The positions of the events
    xlim : tuple
        Optional start and end of raster plot
    colors :
        Optional list of colours, or single colour - default black
    ax : matplotlib.axes.Axes
        Optional axis to plot into
    **kwargs :
        A set of keyword arguments to change graph appearance

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The spike raster
    """
    linewidths = kwargs.get("linewidths", 0.1)
    linelengths = kwargs.get("linelengths", 0.5)
    title = kwargs.get("title", "Spike raster")
    xlabel = kwargs.get("xlabel", "Time (seconds)")
    ylabel = kwargs.get("ylabel", "Cell ID")
    no_y_ticks = kwargs.get("no_y_ticks", False)
    orientation = kwargs.get("orientation", "horizontal")

    ax, fig = _make_ax_if_none(ax)

    ax.eventplot(
        events, colors=colors, linelengths=linelengths, linewidths=linewidths,
        orientation=orientation)

    # Be sure to only pick integer tick locations.
    if orientation == "horizontal":
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        if xlim is not None:
            ax.set_ylim(xlim[0], xlim[1])
        ax.invert_yaxis()

    ax.set_title(title)

    if no_y_ticks:
        ax.get_yaxis().set_visible(False)

    return fig

def replay_summary(replay_data):
    """
    Plot a replay data summary.

    Parameters
    ----------
    replay_data : dict
        Dictionary of graph data

    Returns
    -------
    fig : matplotlib.pyplot.Figure

    """
    lfp_times = replay_data["lfp times"]
    filtered_lfp = replay_data["lfp samples"]
    mua_hist = replay_data["mua hists"]
    swr_times = replay_data["swr times"]
    num_cells = replay_data["num cells"]
    spike_times = replay_data["spike times"]

    colors = get_axona_colours()[:num_cells]
    xlim = (lfp_times[0], lfp_times[-1])

    # SWR and filtered LFP
    fig, axes= plt.subplots(
        nrows=3, ncols=1, figsize=(12,6), sharex=True)
    spike_raster(
        swr_times, ax=axes[0], ylabel=None, xlabel=None,
        no_y_ticks=True, colors=('b'), linewidths=0.2, linelengths=0.5)
    axes[0].plot(lfp_times, filtered_lfp, color='k')
    axes[0].set_title("Filtered LFP and SWR Events")

    # MUA
    axes[1].plot(mua_hist[1], mua_hist[0], color='k')
    ticks = [i for i in range(num_cells + 1)]
    axes[1].set_yticks(ticks)
    axes[1].set_title("Number of Active Cells")

    # Raw spikes
    spike_raster(spike_times, linewidths=0.2, ax=axes[2], colors=colors)

    import matplotlib.ticker as ticker

    tick_spacing = 100
    for ax in axes:
        ax.set_xlim(xlim[0], xlim[1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.tight_layout()
    return fig

def plot_replay_sections(replay_data, spike_times, orientation="vertical"):
    """
    Plot zoomed in sections of the replay data spikes.

    Parameters
    ----------
    replay_data : dict
        Results from replay_summary
    spike_times : list
        A 3 tiered list, most commonly a list of nca.spike_times outputs
    orientation : str
        "vertical" or "horizontal" - the direction to plot rasters in

    Returns
    -------
    matplotlib.pyplot.Figure :
        Resulting multi Axes figure

    """
    num_plots = len(replay_data["overlap swr mua"])
    row_size = 6

    if num_plots <= row_size:
        num_cols = num_plots
        num_rows = 1
    else:
        num_cols = row_size
        num_rows = math.ceil(num_plots / row_size)

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        sharex='col', tight_layout=True, figsize=(num_rows*2, num_cols*2))

    for i, i_range in enumerate(replay_data["overlap swr mua"]):
        if num_plots == 1:
            ax = axes
        else:
            ax=axes.flatten()[i]
        # nca.spike_times(sleep_sample, ranges=[i_range])
        # can be used to get spike times
        spike_raster(
            spike_times[i],
            linewidths=1, ax=ax, orientation=orientation,
            colors=get_axona_colours()[:replay_data["num cells"]],
            #xlim=(round(i_range[0], 1), round(i_range[1], 1)),
            title=None, ylabel=None, xlabel=None)
    return fig

def plot_angle_between_points(points, xlim, ylim, ax=None):
    """
    Plots the angle between three points

    Parameters
    ----------
    points : list
        The list of points to plot the angle between
    xlim : float
        The upper xlimit of the graph
    ylim : float
        The upper ylimit of the graph
    ax : matplotlib.axes.Axes
        Optional axis to plot into

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The angle between the points
    """

    ax, fig = _make_ax_if_none(ax)
    arr = np.array(points)
    xdata = arr[:, 0]
    ydata = arr[:, 1]

    line_1 = Line2D(
        xdata[:2], ydata[:2], linewidth=1, linestyle = "-", color="green",
        marker=".", markersize=10, markeredgecolor='k', markerfacecolor='k'
    )
    line_2 = Line2D(
        xdata[1:], ydata[1:], linewidth=1, linestyle = "-", color="red",
        marker=".", markersize=10, markeredgecolor='k', markerfacecolor='k')

    ax.add_line(line_1)
    ax.add_line(line_2)

    angle_plot = _get_angle_plot(line_1, line_2, 0.2, 'b', [xdata[1], ydata[1]], xlim, ylim)
    ax.add_patch(angle_plot) # To display the angle arc

    ax.set_ylim([0, ylim])
    ax.set_xlim([0, xlim])
    ax.set_aspect('equal')
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    txt_list = ["P1", "P2", "P3"]
    for i, txt in enumerate(txt_list):
        ax.annotate(txt, (xdata[i], ydata[i] - (ylim * 0.02)))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.legend()
    return fig

def _get_angle_plot(
    line1, line2, offset = 1, color = None,
    origin = [0,0], len_x_axis = 1, len_y_axis = 1):
    """
    Internal helper function to get an arc between two lines
    Can be displayed as a patch

    Parameters
    ----------
    line1 : matplotlib.lines.Line2D
        The first line
    line2 : matplotlib.lines.Line2D
        The second line
    offset : float
        How far out the patch should be from the origin
    color : string
        The color of the patch
    origin : list
        Where the centre of the patch should be
    len_x_axis: float
        How long the x axis is in the plot
    len_y_axis: float
        How long the y axis is in the plot

    Returns
    -------
    matplotlib.patches.Arc
        The arc which represents the angle between the lines
    """

    l1xy = line1.get_xydata()
    further1 = l1xy[1] + [1, 0]
    # Angle between line1 and x-axis
    angle1 = angle_between_points(l1xy[0], l1xy[1], further1)
    if l1xy[0][1] < l1xy[1][1]:
        angle1 = 360 - angle1


    l2xy = line2.get_xydata()
    further2 = l2xy[0] + [1, 0]
    # Angle between line1 and x-axis
    angle2 = angle_between_points(l2xy[1], l2xy[0], further2)
    if l2xy[1][1] < l2xy[0][1]:
        angle2 = 360 - angle2

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color, label = "%0.2f"%float(angle)+u"\u00b0")

def _make_ax_if_none(ax, **kwargs):
    """
    Makes a figure and gets the axis from this if no ax exists

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Input axis

    Returns
    -------
    ax, fig
        The created figure and axis if ax is None, else
        the input ax and None
    """

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = plt.gca(**kwargs)
    return ax, fig


def print_place_cells(
        rows, cols=7, size_multiplier=4, wspace=0.3, hspace=0.3,
        placedata=None, wavedata=None, graphdata=None, isidata=None,
        headdata=None, thetadata=None, point_size=10, units=None, 
        output=["Wave", "Path", "Place", "HD", "LowISI", "Theta", "HighISI"],
        fixed_color=None, burst_ms=5, color_isi=True, one_by_one=False,
        raster=True):
    if one_by_one:
        figs = []
        width, height = cols * size_multiplier, (1 - 0.20) * size_multiplier
    else:
        width, height = cols * size_multiplier, (rows - 0.20) * size_multiplier
        fig = plt.figure(figsize=(width, height), tight_layout=False)
        gs = gridspec.GridSpec(
            rows, cols, figure=fig, wspace=wspace, hspace=hspace)

    def get_mapping_idx(name):
        if not name in output:
            return False
        return output.index(name)

    for i in range(rows):
        if one_by_one:
            fig = plt.figure(figsize=(width, height), tight_layout=False)
            gs = gridspec.GridSpec(
                1, cols, figure=fig, wspace=wspace, hspace=hspace)
            j = 0
        else:
            j = i

            
        # Plot the spike position
        if placedata is not None:
            place_data = placedata[i]
            if place_data is not None:

                # Plot the place map
                idx = get_mapping_idx("Path")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    if fixed_color:
                        color = fixed_color
                    elif units == None:
                        color = get_axona_colours(i)
                    else:
                        color = get_axona_colours(units[i] - 1)
                    loc_spike(
                        place_data, ax=ax, color=color,
                        point_size=point_size, raster=raster)
                    ax.set_aspect('auto')

                # Plot the rate map
                idx = get_mapping_idx("Place")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    loc_rate(place_data, ax=ax, smooth=True)
                    ax.set_aspect('auto')

        if headdata is not None:
            if headdata[i] is not None:
                idx = get_mapping_idx("HD")
                if idx is not None:
                    hd_data = headdata[i]
                    ax = fig.add_subplot(gs[j, idx], projection='polar')
                    hd_rate(hd_data, ax=ax, title=None)
                    bins = np.append(hd_data['bins'], hd_data['bins'][0])
                    predRate = np.append(
                        hd_data['hdPred'], hd_data['hdPred'][0])
                    ax.plot(np.radians(bins), predRate, color='green')
                    ax.set_rticks(
                        [hd_data['hdRate'].max(), hd_data['hdPred'].max()])

        
        if wavedata is not None:
            if wavedata[i] is not None:
                idx = get_mapping_idx("Wave")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    largest_waveform(wavedata[i], ax=ax)

        # Plot -10 to 10 autocorrelation
        if graphdata is not None:
            if graphdata[i] is not None:
                idx = get_mapping_idx("LowAC")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    isi_corr(
                        graphdata[i], ax=ax, title=None, xlabel=None, ylabel=None, raster=raster)

        if thetadata is not None:
            if thetadata[i] is not None:
                idx = get_mapping_idx("Theta")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    theta_cell(thetadata[i], ax=ax, title=None,
                            xlabel=None, ylabel=None, raster=raster)

        if isidata is not None:
            if isidata[i] is not None:
                idx = get_mapping_idx("HighISI")
                if idx is not None:
                    ax = fig.add_subplot(gs[j, idx])
                    temp_fig, (ax1, ax2) = plt.subplots(2)
                    isi(isidata[i], axes=[ax, ax1, ax2],
                        title1=None, xlabel1=None, ylabel1=None,
                        should_color=color_isi, burst_ms=burst_ms,
                        raster=raster)
                    plt.close(temp_fig)
        
        if one_by_one:
            figs.append(fig)

    if one_by_one:
        return figs
    else:
        return fig
