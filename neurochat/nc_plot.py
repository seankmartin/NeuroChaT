# -*- coding: utf-8 -*-
"""
This module implements plotting functions for NeuroChaT analyses.

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.gridspec import GridSpec #Added by matheus 06/03/19
import matplotlib.gridspec as gridspec

from neurochat.nc_utils import find

BLUE = '#1f77b4'
RED = '#d62728'

def plot_cells(lines):
    """ PLot the cells found in a recording"""
    #lines = spike.set_unit_no(unit)
    fig = plt.figure(figsize=(8, lines*2), constrained_layout=False)

    gs = gridspec.GridSpec(lines, 4, figure=fig)
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[3])
    ax = []
    for i in range(0, lines):
        for j in range(0,4):
            ax.append(fig.add_subplot(gs[i,j]))
        ax[i] = plt.Subplot(fig, gs0[0,0])
        ax[i].plot(wave_data['Mean wave'][0,0], color='black', linewidth=2.0)
        fig.add_subplot(ax[i])
        ax[i] = plt.Subplot(fig, gs0[0,1])
        ax[i].plot(wave_data['Mean wave'][0,1]+wave_data['Std wave'][:, i],\
                color='green', linestyle='dashed')
        fig.add_subplot(ax[i])
        ax[i] = plt.Subplot(fig, gs0[1,0])
        ax[i].plot(wave_data['Mean wave'][1, 0]-wave_data['Std wave'][:, i],\
                color='green', linestyle='dashed')
        fig.add_subplot(ax[i])
        ax[i] = plt.Subplot(fig, gs0[1,1])
        fig.add_subplot(ax[i])
        for i in np.arange(len(ax)):
           
            
             
     

    # recording = 'filename'
    # rat_number = 1
    # tetrode = 3
    # fig.suptitle("Rat: "+str(rat_number)+", tetrode:"+str(tetrode)+", recording: "+recording)


def scatterplot_matrix(_data, names=[], **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "_data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, num_data = _data.shape
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

    plt.show()

    return fig1

def isi(isi_data):
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
    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(isi_data['isiBins'], isi_data['isiHist'], color='darkblue', \
           edgecolor='darkblue', rasterized=True)
    ax.plot([5, 5,], [0, isi_data['maxCount']], linestyle='dashed',\
            linewidth=2, color='red')
    ax.set_title('Distribution of inter-spike interval')
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('Spike count')
    max_axis = isi_data['isiBins'].max()
    max_axisLog = np.ceil(np.log10(max_axis))

    ## ISI scatterplot
    fig2 = plt.figure()
#        fig2.suptitle('Distribution of ISI \n (before and after spike)')
    # Scatter
#        ax = fig2.add_subplot(211)
    ax = plt.gca()
    ax.loglog(isi_data['isiBefore'], isi_data['isiAfter'], axes=ax, \
            linestyle=' ', marker='o', markersize=1, \
            markeredgecolor='k', markerfacecolor=None, rasterized=True)
#    ax.autoscale(enable= True, axis= 'both', tight= True)
    ax.plot(ax.get_xlim(), [5, 5], linestyle='dashed', linewidth=2, color='red')
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
    fig3 = plt.figure()
    ax = plt.gca()
    c_map = plt.cm.jet
    c_map.set_under('white')
    ax.pcolormesh(xedges[0:-1], yedges[0:-1], joint_count,\
                  cmap=c_map, vmin=1, rasterized=True)
    ax.plot(ax.get_xlim(), [5, 5], linestyle='dashed', linewidth=2, color='red')
    plt.axis(_extent)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')
    ax.set_xlabel('Interval before (ms)')
    ax.set_ylabel('Interval after (ms)')
    ax.set_title('Distribution of ISI \n (before and after spike)')

    return fig1, fig2, fig3

def isi_corr(isi_corr_data):
    """
    Plots ISI correlation.

    Parameters
    ----------
    isi_corr_data : dict
        Graphical data from the ISI correlation

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        ISI correlation histogram

    """

    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(isi_corr_data['isiCorrBins'], isi_corr_data['isiCorr'],\
           color='darkblue', edgecolor='darkblue', rasterized=True)
    ax.set_title('Autocorrelation Histogram \n' + '('+ str(abs(isi_corr_data['isiCorrBins'].min()))+ 'ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Counts')

    return fig1

def theta_cell(plot_data):
    """
    Plots theta-modulated cell and theta-skipping cell analysis data

    Parameters
    ----------
    plot_data : dict
        Graphical data from the theta-modulated cell and theta skipping cell analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        ISI correlation histogram superimposed with fitted sinusoidal curve.

    """

    fig1 = plt.figure()
    ax = plt.gca()
    ax.bar(plot_data['isiCorrBins'], plot_data['isiCorr'],\
           color='darkblue', edgecolor='darkblue', rasterized=True)
    ax.plot(plot_data['isiCorrBins'], plot_data['corrFit'], linewidth=2, color='red')
    ax.set_title('Autocorrelation Histogram \n' + '('+ str(abs(plot_data['isiCorrBins'].min()))+ 'ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Counts')

    return fig1

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
    Plots the analysis results of Phase-locking value (PLV)

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

    return fig1, fig2

def plv_tr(plv_data):
    """
    Plots the analysis results of time-resolved Phase-locking value (PLV)

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
    Plots the analysis results of bootstrapped Phase-locking value (PLV)

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

    return fig1, fig2

def spike_phase(phase_data):
    """
    Plots the analysis results of spike-LFP phase locking

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
    Plots the results of multiple regression analysis.

    Parameters
    ----------
    mra_data : dict
        Graphical data from multiple regression analysis

    Returns
    -------
    fig1 : matplotlib.pyplot.Figure
        Bar plot of multiple regression results

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

def hd_rate(hd_data, ax=None):
    """
    Plots head direction vs spike rate

    Parameters
    ----------
    hd_data : dict
        Graphical data from the unit firing to head-direction correlation
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the polar plot of head-direction vs spike-rate.

    """

    if not ax:
        plt.figure()
        ax = plt.gca(polar=True)
    bins = np.append(hd_data['bins'], hd_data['bins'][0])
    rate = np.append(hd_data['smoothRate'], hd_data['smoothRate'][0])
    ax.plot(np.radians(bins), rate, color=BLUE)

    ax.set_title('Head directional firing rate')
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
    Plots the analysis results of head directional correlation to spike-rate

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
    Plots the analysis results of head directional correlation to spike-rate
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
    for n in range(nfig):
        f, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
        fig.append(f)
        axs.extend(list(ax.flatten()))

    kcount = 0
    for key in keys:
        axs[kcount] = hd_spike(hd_data[key], ax=axs[kcount])
        axs[kcount].set_title(key)
        kcount += 1

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
    for n in range(nfig):
        f, ax = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
        fig.append(f)
        axs.extend(list(ax.flatten()))

    kcount = 0
    for key in keys:
        axs[kcount] = hd_rate(hd_data[key], ax=axs[kcount])
        axs[kcount].set_title(key)
        kcount += 1
    return fig

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

def loc_spike(place_data, ax=None):
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


    # spatial firing map
    if not ax:
        plt.figure()
        ax = plt.gca()

    ax.plot(place_data['posX'], place_data['posY'], color='black', zorder=1)
    ax.scatter(place_data['spikeLoc'][0], place_data['spikeLoc'][1], \
               s=2, marker='x', color=RED, zorder=2)
    ax.set_ylim([0, place_data['yedges'].max()])
    ax.set_xlim([0, place_data['xedges'].max()])
    ax.set_aspect('equal')
    ax.invert_yaxis()
#        plt.autoscale(enable=True, axis='both', tight=True)
    return ax

def loc_rate(place_data, ax=None):
    """
    Plots location vs spike rate

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to locational correlation
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the firing rate map

    """

    if not ax:
        plt.figure()
        ax = plt.gca()
    clist = [(0.0, 0.0, 1.0),\
            (0.0, 1.0, 0.5),\
            (0.9, 1.0, 0.0),\
            (1.0, 0.75, 0.0),\
            (0.9, 0.0, 0.0)]
    c_map = mcol.ListedColormap(clist)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    pmap= ax.pcolormesh(place_data['xedges'], place_data['yedges'], np.ma.array(place_data['smoothMap'], \
                    mask=np.isnan(place_data['smoothMap'])), cmap=c_map, rasterized=True)
    ax.set_ylim([0, place_data['yedges'].max()])
    ax.set_xlim([0, place_data['xedges'].max()])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.colorbar(pmap, cax=cax, orientation='vertical')
#        plt.autoscale(enable=True, axis='both', tight=True)
    return ax

def loc_firing(place_data):
    """
    Plots the analysis results of locational correlation to spike-rate

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

    return fig

# Created by Sean Martin: 13/02/2019
def loc_place_field(place_data, centroid):
    """
    Plots the analysis results of locational correlation to spike-rate along with
    the centroid of the place field.

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
    ax.set_xlabel('XLoc')
    ax.set_ylabel('YLoc')


    ax = loc_rate(place_data, ax=fig.add_subplot(122))
    ax.plot([centroid[0]], [centroid[1]], 'gX')
    ax.set_xlabel('XLoc')
    ax.set_ylabel('YLoc')
#    fig.colorbar(cax)

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
    for n in range(nfig):
        f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        fig.append(f)
        axs.extend(list(ax.flatten()))

    kcount = 0
    for key in keys:
        loc_spike(place_data[key], ax=axs[kcount])
        axs[kcount].set_title(key)
        kcount += 1
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
    for n in range(nfig):
        f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        fig.append(f)
        axs.extend(list(ax.flatten()))

    kcount = 0
    for key in keys:
        loc_rate(place_data[key], ax=axs[kcount])
        axs[kcount].set_title(key)
        kcount += 1

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

# Locationa firing map rotational analysis
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
    Plots the analysis results from border analysis

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
    Plots the results from gradient cell analysis

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
    Plots the results from grid analysis

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
