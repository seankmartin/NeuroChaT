# -*- coding: utf-8 -*-
"""
This module implements utility functions and classes for NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import logging
import time
import datetime
import traceback
from collections import OrderedDict as oDict
import os
from os import listdir
from os.path import isfile, isdir, join
import re
import math

import pandas as pd
import numpy as np
import numpy.linalg as nalg

import scipy
import scipy.stats as stats
import scipy.signal as sg
from scipy.fftpack import fft


class NLog(logging.Handler):
    """
    Class for handling log information (messages, errors and warnings).

    It formats the incoming message in HTML and sends it to the log
    interface of NeuroChaT.

    """

    def __init__(self):
        super().__init__()
        self.setup()

    def setup(self):
        """
        Remove all the logging handlers and set up a logger in HTML format.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        log = logging.getLogger()
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        fmt = logging.Formatter(
            '%(asctime)s (%(filename)s)  %(levelname)s--  %(message)s', '%H:%M:%S')
        self.setFormatter(fmt)
        log.addHandler(self)
        # You can control the logging level
        log.setLevel(logging.DEBUG)
        logging.addLevelName(20, '')

    def emit(self, record):
        """
        Format the incoming record and display it.

        Parameters
        ----------
        record
            Log record to display or store

        Returns
        -------
        None

        """
        msg = self.format(record)
        level = record.levelname
        msg = level + ':' + msg
        print(msg)
        time.sleep(0.25)


class Singleton(object):
    """Create a Singleton object created from a subclass of this class."""

    def __new__(cls, *arg, **kwarg):
        """Create a Singleton object created from a subclass of this class."""
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *arg, **kwarg)
        return cls._instance


def bhatt(X1, X2):
    """
    Calculate Bhattacharyya coefficient and distance between distributions.

    Parameters
    ----------
    X1, X2 : ndarray
        Distributions under consideration

    Returns
    -------
    bc, d : float
        Bhattacharyya coefficient and Bhattacharyya distance

    """
    r1, c1 = X1.shape
    r2, c2 = X2.shape
    if c1 == c2:
        mu1 = X1.mean(axis=0)
        mu2 = X2.mean(axis=0)
        C1 = np.cov(X1.T)
        C2 = np.cov(X2.T)
        C = (C1 + C2) / 2
        chol = nalg.cholesky(C).T
        dmu = (mu1 - mu2) @ nalg.inv(chol)
        try:
            d = 0.125 * dmu @ (dmu.T) + 0.5 * np.log(
                nalg.det(C) / np.sqrt(nalg.det(C1) * nalg.det(C2)))
        except BaseException:
            d = 0.125 * dmu @ (dmu.T) + 0.5 * np.log(
                np.abs(nalg.det(C @ nalg.inv(scipy.linalg.sqrtm(C1 @ C2)))))
        bc = np.exp(-1 * d)

        return bc, d
    else:
        logging.error(
            'Cannot measure Bhattacharyya distance, column sizes do not match!')


def butter_filter(x, Fs, *args):
    """
    Filter using bidirectional zero-phase shift Butterworth filter.

    Parameters
    ----------
    x : ndarray
        Data or signal to filter
    Fs : Sampling frequency
    *kwargs
        Arguments with filter paramters

    Returns
    -------
    ndarray
        Filtered signal

    """
    gstop = 20  # minimum dB attenuation at stopabnd
    gpass = 3  # maximum dB loss during ripple
    for arg in args:
        if isinstance(arg, str):
            filttype = arg
    if filttype == 'lowpass' or filttype == 'highpass':
        wp = args[1] / (Fs / 2)
        if wp > 1:
            wp = 1
            if filttype == 'lowpass':
                logging.warning(
                    'Butterworth filter critical frequency Wp is capped at 1')
            else:
                logging.error('Cannot highpass filter over Nyquist frequency!')

    elif filttype == 'bandpass':
        if len(args) < 4:
            logging.error('Insufficient Butterworth filter arguments')
        else:
            wp = np.array(args[1:3]) / (Fs / 2)
            if wp[0] >= wp[1]:
                logging.error(
                    'Butterworth filter lower cutoff frequency must be smaller than upper cutoff frequency!')

            if wp[0] == 0 and wp[1] >= 1:
                logging.error(
                    'Invalid filter specifications, check cut off frequencies and sampling frequency!')
            elif wp[0] == 0:
                wp = wp[1]
                filttype = 'lowpass'
                logging.warning('Butterworth filter type selected: lowpass')
            elif wp[1] >= 1:
                wp = wp[0]
                filttype = 'highpass'
                logging.warning('Butterworth filter type selected: highpass')

    if filttype == 'lowpass':
        ws = min([wp + 0.1, 1])
    elif filttype == 'highpass':
        ws = max([wp - 0.1, 0.01 / (Fs / 2)])
    elif filttype == 'bandpass':
        ws = np.zeros_like(wp)
        ws[0] = max([wp[0] - 0.1, 0.01 / (Fs / 2)])
        ws[1] = min([wp[1] + 0.1, 1])

    min_order, min_wp = sg.buttord(wp, ws, gpass, gstop)

    b, a = sg.butter(min_order, min_wp, btype=filttype, output='ba')

    return sg.filtfilt(b, a, x)


def chop_edges(x, xlen, ylen):
    """
    Chop the edges of a firing rate map.

    They are considered to be the edges if they are not
    visited at all or with zero firing rate.

    Parameters
    ----------
    x : ndarray
        Matrix of firing rate
    xlen : int
        Maximum length of the x-axis
    ylen : int
        Maximum length of the y-axis

    Returns
    -------
    low_ind : list of int
        Index of low end of valid edges
    hig_end :
        Index of high end of valid edges
    y : ndarray
        Chopped firing map

    """
    y = np.copy(x)
    low_ind = [0, 0]
    high_ind = [x.shape[0], x.shape[1]]

    MOVEON = True
    while y.shape[1] > xlen and MOVEON:
        no_filled_bins1 = np.sum(y[:, 0] > 0)
        no_filled_bins2 = np.sum(y[:, -1] > 0)

        if no_filled_bins1 == 0:
            low_ind[1] += 1
            MOVEON = True
        else:
            MOVEON = False
        if no_filled_bins2 == 0:
            high_ind[1] -= 1
            MOVEON = True
        else:
            MOVEON = False

        y = x[low_ind[0]: high_ind[0], low_ind[1]:high_ind[1]]

    MOVEON = True
    while y.shape[0] > ylen and MOVEON:
        no_filled_bins1 = np.sum(y[0, :] > 0)
        no_filled_bins2 = np.sum(y[-1, :] > 0)

        if no_filled_bins1 == 0:
            low_ind[0] += 1
            MOVEON = True
        else:
            MOVEON = False
        if no_filled_bins2 == 0:
            high_ind[0] -= 1
            MOVEON = True
        else:
            MOVEON = False

        y = x[low_ind[0]: high_ind[0], low_ind[1]:high_ind[1]]

    return low_ind, high_ind, y


def corr_coeff(x1, x2):
    """
    Correlation coefficient between two numeric series or two signals.

    Parameters
    ----------
    x1, x2 : ndarray
        Input numeric array or signals

    Returns
    -------
    float
        Correlation coefficient of input arrays

    """
    try:
        return np.sum(np.multiply(x1 - x1.mean(), x2 - x2.mean())) / \
            np.sqrt(np.sum((x1 - x1.mean())**2) * np.sum((x2 - x2.mean())**2))
    except BaseException:
        return 0


def extrema(x, mincap=None, maxcap=None):
    """
    Find the extrema in a numeric array or a signal.

    Parameters
    ----------
    mincap
        Maximum value for the minima
    maxcap
        Minimum value for the maxima

    Returns
    -------
    xmax : ndarray
        Maxima values
    imax : ndarray
        Maxima indices
    xmin : ndarray
        Minima values
    imin : ndarray
        Minima indices

    """
    x = np.array(x)
    # Flat peaks at the end of the series are not considered yet
    dx = np.diff(x)
    if not np.any(dx):
        return [], [], [], []

    a = find(dx != 0)  # indices where x changes
    lm = find(np.diff(a) != 1) + 1  # indices where a is not sequential
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - np.floor(d // 2)

    xa = x[a]  # series without flat peaks
    d = np.sign(xa[1:-1] - xa[:-2]) - np.sign(xa[2:] - xa[1:-1])
    imax = a[find(d > 0) + 1]
    xmax = x[imax]
    imin = a[find(d < 0) + 1]
    xmin = x[imin]

    if mincap:
        imin = imin[xmin <= mincap]
        xmin = xmin[xmin <= mincap]
    if maxcap:
        imax = imax[xmax <= maxcap]
        xmax = xmax[xmax <= maxcap]

    return xmax, imax, xmin, imin


def fft_psd(x, Fs, nfft=None, side='one', ptype='psd'):
    """
    Calculate the Fast Fourier Transform (FFT) of a signal.

    Parameters
    ----------
    x : ndarray
        Input signal
    Fs
        Sampling frequency
    nfft : int
        Number of FFT points
    side : str
        'one'-sided or 'two'-sided FFT
    ptype : str
        Calculates power-spectral density if set to 'psd'

    Returns
    -------
    x_fft : ndarray
        FFT of input
    f : ndarray
        FFt frequency

    """
    if nfft is None:
        nfft = 2**(np.floor(np.log2(len(x))) + 1)

    if nfft < Fs:
        nfft = 2**(np.floor(np.log2(Fs)) + 1)
    nfft = int(nfft)
    dummy = np.zeros(nfft)
    if nfft > len(x):
        dummy[:len(x)] = x
        x = dummy

    winfun = np.hanning(nfft)
    xf = np.arange(0, Fs, Fs / nfft)
    f = xf[0: int(nfft / 2) + 1]

    if side == 'one':
        x_fft = fft(np.multiply(x, winfun), nfft)
        if ptype == 'psd':
            x_fft = np.absolute(x_fft[0: int(nfft / 2) + 1])**2 / nfft**2
            x_fft[1:-1] = 2 * x_fft[1:-1]

    return x_fft, f


def find(X, n=None, direction='all'):
    """
    Find the non-zero entries of a signal or array.

    Parameters
    ----------
    X : ndarray or list
        Array or list of numbers whose non-zero entries need to find out
    n : int
        Number of such entries
    direction : str
        If 'all', all entries of length n are returned.
        If 'first', first n entries are returned.
        If 'last', last n entries are returned.

    Returns
    -------
    ndarray
        Indices of non-zero entries.

    """
    if isinstance(X, list):
        X = np.array(X)
    X = X.flatten()
    if n is None:
        n = len(X)
    ind = np.where(X)[0]
    if ind.size:
        if direction == 'all' or direction == 'first':
            ind = ind[:n]
        elif direction == 'last':
            ind = ind[np.flipud(np.arange(-1, -(n + 1), - 1))]
    return np.array(ind)


def find2d(X, n=None):
    """
    Find the non-zero entries of a matrix.

    Parameters
    ----------
    X : ndarray
        Matrix whose non-zero entries need to find out
    n : int
        Number of such entries

    Returns
    -------
    ndarray
        x-indices of non-zero entries.
    ndarray
        y-indices of non-zero entries.

    """
    if len(X.shape) == 2:
        J = []
        I = []
        for r in np.arange(X.shape[0]):
            I.extend(find(X[r, ]))
            J.extend(r * np.ones((len(find(X[r, ])), ), dtype=int))
        if len(I):
            if n is not None and n < len(I):
                I = I[:n]
                J = J[:n]
        return np.array(J), np.array(I)

    else:
        logging.error('ndrray is not 2D. Check shape attributes of the input!')


def find_chunk(x):
    """
    Find size and indices of chunks of non-zero segments in an array.

    Parameters
    ----------
    x : ndarray
        Inout array whose non-zero chunks are to be explored

    Returns
    -------
    segsize : ndarray
        Lengths of non-zero chunks
    segind : ndarray
        Indices of non-zero chunks

    """
    # x is a binary array input i.e. x= data> 0.5 will find all the chunks in
    # data where data is greater than 0.5
    i = 0
    segsize = []
    segind = np.zeros(x.shape)
    while i < len(x):
        if x[i]:
            c = 0
            j = i
            while i < len(x):
                if x[i]:
                    c += 1
                    i += 1
                else:
                    break
            segsize.append(c)
            segind[j:i] = c  # indexing by size of the chunk
        i += 1
    return segsize, segind


def hellinger(X1, X2):
    """
    Calculate Hellinger distance between two distributions.

    Parameters
    ----------
    X1, X2 : ndarray
        Distributions under consideration

    Returns
    -------
    d : float
        Calculated Hellinger distance

    """
    if X1.shape[1] != X2.shape[1]:
        logging.error(
            'Hellinger distance cannot be computed, column sizes do not match!')
    else:
        return np.sqrt(1 - bhatt(X1, X2)[0])


def histogram(x, bins):
    """
    Calculate the histogram count of input array.

    This function is not a replacement of np.histogram;
    it is created for convenience of binned-based rate calculations
    and mimicking matlab histc that includes digitized indices

    Parameters
    ----------
    x : ndarray
        Array whose histogram needs to be calculated
    bins
        Number of histogram bins

    Returns
    -------
    ndarray
        Histogram count
    ndarray
        Histogram bins(lowers edges)

    """
    if isinstance(bins, int):
        bins = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x)) / bins)
    bins = np.append(bins, bins[-1] + np.mean(np.diff(bins)))
    return np.histogram(x, bins)[0], np.digitize(x, bins) - 1, bins[:- 1]


def histogram2d(y, x, ybins, xbins):
    """
    Calculate the joint histogram count of two arrays.

    This function is not a replacement of np.histogram2d;
    it is created for convenience of binned-based rate calculations
    and mimicking matlab histc that includes digitized indices

    Parameters
    ----------
    y, x : ndarray
        Arrays whose histogram needs to be calculated
    ybins
        Number of histogram bins in y-axis
    xbins
        Number of histogram bins in x-axis

    Returns
    -------
    ndarray
        Histogram count
    ndarray
        Histogram bins in x-axis (lowers edges)
    ndarray
        Histogram bins in y-axis (lowers edges)

    """
    if isinstance(xbins, int):
        xbins = np.arange(np.min(x), np.max(
            x), (np.max(x) - np.min(x)) / xbins)
    xbins = np.append(xbins, xbins[-1] + np.mean(np.diff(xbins)))
    if isinstance(ybins, int):
        ybins = np.arange(np.min(y), np.max(
            y), (np.max(y) - np.min(y)) / ybins)
    ybins = np.append(ybins, ybins[-1] + np.mean(np.diff(ybins)))

    return np.histogram2d(y, x, [ybins, xbins])[0], ybins[:-1], xbins[:-1]


def linfit(X, Y, getPartial=False):
    """
    Calculate the linear regression coefficients in least-square sense.

    Parameters
    ----------
    X : ndarray
        Matrix with input variables or factors (num_dim X num_obs)
    Y : ndarray
        Array of oservation data
    getPartial : bool
        Get the partial correlation coefficients if 'True'

    Returns
    -------
    _results : dict
        Dictionary with results of least-square optimization of linear regression

    """
    _results = oDict()
    if len(X.shape) == 2:
        Nd, Nobs = X.shape
    else:
        Nobs = X.shape[0]
        Nd = 1
    if Nobs == len(Y):
        A = np.vstack([X, np.ones(X.shape[0])]).T
        B = np.linalg.lstsq(A, Y, rcond=-1)[0]
        Y_fit = np.matmul(A, B)
        _results['coeff'] = B[:-1]
        _results['intercept'] = B[-1]
        _results['yfit'] = Y_fit
        _results.update(residual_stat(Y, Y_fit, 1))
    else:
        logging.error('linfit: Number of rows in X and Y does not match!')

    if Nd > 1 and getPartial:
        semiCorr = np.zeros(Nd)  # Semi partial correlation
        for d in np.arange(Nd):
            part_results = linfit(np.delete(X, 1, axis=0), Y, getPartial=False)
            semiCorr[d] = _results['Rsq'] - part_results['Rsq']
        _results['semiCorr'] = semiCorr

    return _results


def nxl_write(
        file_name, data_frame, sheet_name='Sheet1', startRow=0, startColumn=0):
    """
    Write Pandas DataFrame to excel file, wraps Pandas.ExcelWriter().

    Parameters
    ----------
    filename : str
        Name of the output file
    data_frame : pandas.DataFrame
        DataFrame to export
    sheet_name : str
        Sheet name of the Excel file where the data is written
    startRow : int
        Which row in the file the data writing should start
    startColumn : int
        Which column in the file the data writing should start

    Returns
    -------
    None

    """
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    data_frame.to_excel(writer, sheet_name)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def residual_stat(y, y_fit, p):
    """
    Calculate the goodness of fit and other residual statistics.

    These are calculated between observed and fitted values from a model.

    Parameters
    ----------
    y : ndarray
        Observed data
    y_fit : ndarray
        Fitted data to a linear model
    p : int
        Model order

    Returns
    -------
    _results : dict
        Dictionary of residual statistics

    """
    # p= total explanatory variables excluding constants
    _results = oDict()
    res = y - y_fit
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1 - ss_res / ss_tot
    adj_r_sq = 1 - (ss_res / ss_tot) * ((len(y) - 1) / (len(y) - p - 1))
    _results['Pearson R'], _results['Pearson P'] = stats.pearsonr(y, y_fit)

    _results['Rsq'] = r_sq
    _results['adj Rsq'] = adj_r_sq

    return _results


def rot_2d(x, theta):
    """
    Rotate a firing map by a specified angle.

    Parameters
    ----------
    x : ndarray
        Matrix of firing rate map
    theta
        Angle of rotation in theta

    Returns
    -------
    ndarray
        Rotated matrix

    """
    return scipy.ndimage.interpolation.rotate(
        x, theta, reshape=False, mode='constant', cval=np.min(x))


def angle_between_points(a, b, c):
    """
    Return the angle between the lines ab and bc, <abc.

    This function always returns an angle less than 180degrees.
    The orientation of the lines can be used to determine which
    side of the lines this angle is formed from.

    Returns np.nan if ab and bc are the same point.

    Parameters
    ----------
    a : ndarray
        The first point
    b : ndarray
        The second point
    c : the last point

    Returns
    -------
    float
        The angle in degrees

    """
    ba = a - b
    bc = c - b

    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)

    if length_bc != 0 and length_ba != 0:
        cosine_angle = np.dot(ba, bc) / (length_ba * length_bc)
        angle = np.arccos(cosine_angle)
    else:
        logging.error(
            "Angle between points: Two points are the same" +
            " can't measure angle as a result")
        angle = np.NAN

    return np.degrees(angle)


def centre_of_mass(co_ords, weights, axis=0):
    """
    Calculate the co-ordinate centre of mass for a 2D system of particles.

    The particles all have co-ords and weights.

    Parameters
    ----------
    co_ords : ndarray
        Array of co-ordinate positions,
        assumed to have co_ords.shape[axis] co-ordinates.
    weights : ndarray
        Array of corresponding weights
    axis : int, default 0
        The axis along which the co-ordinates are specified, expected 0 or 1

    Returns
    -------
    ndarray
        Co-ordinate of the centre of mass

    """
    shape = co_ords.shape
    if axis == 0:
        weighted = np.multiply(
            co_ords,
            np.repeat(weights, shape[1]).reshape(shape))
    elif axis == 1:
        weighted = np.multiply(
            co_ords,
            np.tile(weights, shape[0]).reshape(shape))
    else:
        logging.error("centre_of_mass: Expected axis to be 0 or 1")
    return np.sum(weighted, axis=axis) / np.sum(weights)


def smooth_1d(x, kernel_type='b', kernel_size=5, axis=0, **kwargs):
    """
    Filter a 1D array or signal.

    Parameters
    ----------
    x : ndarray
        Array or signal to be filtered.
        If matrix, each column or row is filtered
        individually depending on 'dir' parameter that takes either '0' for along-column
        and '1' for along-row filtering.
    kernel_type : str
        'b' for moving average or box filter. 'g' for Gaussian filter.
        'hs' for Heaviside filter
        'hg' for half-Gaussian filter
    kernel_size : int
        Box size for box filter and sigma for Gaussian filter
    axis : int
        Defaults to 0. The axis along which to smooth matrices.

    Returns
    -------
    ndarray
        Filtered data

    """
    def pad_and_convolve(xx, kernel):
        npad = len(kernel)
        xx = np.pad(xx, (npad, npad), 'edge')
        yy = np.convolve(xx, kernel, mode='same')
        return yy[npad:-npad]

    x = np.array(x)
    half_width = kernel_size / 2
    xx = np.arange(-half_width, half_width + 1, 1)
    if kernel_type == 'g':
        half_width = kernel_size / 2
        xx = np.arange(-half_width, half_width + 1, 1)
        sigma = kernel_size / (2 * 2.7)
        kernel = np.exp(-(xx**2) / (2 * sigma**2)) / \
            (np.sqrt(2 * np.pi) * sigma)

    elif kernel_type == 'b':
        kernel = np.ones(kernel_size) / kernel_size

    elif kernel_type == 'hs':
        half_width = kernel_size
        xx = np.arange(-half_width, half_width + 1, 1)
        sigma = kernel_size / 2 / np.sqrt(3)
        kernel = (0.5 / (np.sqrt(3) * sigma)) * \
            (xx < 2 * np.sqrt(3) * sigma and xx >= 0)

    elif kernel_type == 'hg':
        half_width = kernel_size
        xx = np.arange(-half_width, half_width + 1, 1)
        sigma = 2 * kernel_size / 2 / np.sqrt(3)
        kernel = np.exp(-(xx**2) / (2 * sigma**2)) / \
            (np.sqrt(2 * np.pi) * sigma)
        kernel[xx < 0] = 0
        kernel[xx > 0] = 2 * kernel[xx > 0]

    result = np.apply_along_axis(
        lambda xx: pad_and_convolve(xx, kernel), axis, x)

    return result


def smooth_2d(x, filttype='b', filtsize=5):
    """
    Filter a 2D array or signal.

    Parameters
    ----------
    x : ndarray
        Matrix to be filtered
    filttype : str
        'b' for moving average or box filter. 'g' for Gaussian filter.
    filtsize
        Box size for box filter and sigma for Gaussian filter

    Returns
    -------
    smoothX
        Filtered matrix

    """
    nanInd = np.isnan(x)
    x[nanInd] = 0
    if filttype == 'g':
        halfwid = np.round(3 * filtsize)
        xx, yy = np.meshgrid(np.arange(-halfwid, halfwid + 1, 1),
                             np.arange(-halfwid, halfwid + 1, 1), copy=False)
        # /(2*np.pi*filtsize**2) # This is the scaling used before;
        filt = np.exp(-(xx**2 + yy**2) / (2 * filtsize**2))
        # But tested with ones(50, 50); gives a hogher value
        filt = filt / np.sum(filt)
    elif filttype == 'b':
        filt = np.ones((filtsize, filtsize)) / filtsize**2

    smoothX = sg.convolve2d(x, filt, mode='same')
    smoothX[nanInd] = np.nan

    return smoothX


def find_true_ranges(arr, truth_arr, min_range, return_idxs=False):
    """
    Return a list of ranges where truth values occur in sorted array.

    Also return the corresponding values from the input array.

    Note
    ----
    The input array arr is assumed to be a sorted list.

    Parameters
    ----------
    arr : ndarray
        list of values to get ranges from, equal in length to truth_arr
    truth_arr : ndarray
        list of truth values to make the ranges
    min_range : int or float
        the minimum length of range

    Returns
    -------
    list
        A list of tuples, ranges in arr where truth values are truth_arr

    """
    in_range = False
    ranges = []
    range_idxs = []
    for idx, b in enumerate(truth_arr):
        if b and not in_range:
            in_range = True
            range_start = arr[idx]
            range_start_idx = idx
        if not b and in_range:
            in_range = False
            range_end = arr[idx - 1]
            range_end_idx = idx
            if range_end - range_start >= min_range:
                ranges.append((range_start, range_end))
                for i in range(range_start_idx, range_end_idx):
                    range_idxs.append(i)
    if not return_idxs:
        return ranges
    else:
        return ranges, range_idxs


def find_peaks(data, **kwargs):
    """
    Return the peaks in the data based on gradient calculations.

    Parameters
    ----------
    kwargs
        start : int
            Where to start looking for peaks in the data, default 0
        end : int
            Where to stop looking for peaks in the data, default data.size - 1
        thresh : float
            Don't consider any peaks with a value below this, default 0

    """
    data = np.array(data)
    slope = np.diff(data)
    start_at = kwargs.get('start', 0)
    end_at = kwargs.get('end', slope.size)
    thresh = kwargs.get('thresh', 0)

    peak_loc = [j for j in np.arange(start_at, end_at - 1)
                if slope[j] > 0 and slope[j + 1] <= 0]
    peak_val = [data[peak_loc[i]] for i in range(0, len(peak_loc))]

    valid_loc = [
        i for i in range(0, len(peak_loc)) if peak_val[i] >= thresh]
    if len(valid_loc) == 0:
        return []
    peak_val, peak_loc = zip(*((peak_val[i], peak_loc[i]) for i in valid_loc))
    return np.array(peak_val), np.array(peak_loc)


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info.

    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info : str, optional
        Additional string to log. Default is "".

    Returns
    -------
    None

    """
    default_loc = os.path.join(
        os.path.expanduser("~"), ".nc_saved", "nc_caught.txt")
    now = datetime.datetime.now()
    # tb = traceback.format_tb(ex.__traceback__)
    make_dir_if_not_exists(default_loc)
    with open(default_loc, "a+") as f:
        f.write("\n----------Caught Exception at {}----------\n".format(now))
        traceback.print_exc(file=f)
    logging.error(
        "{} failed with caught exception.\nSee {} for more information.".format(
            more_info, default_loc), exc_info=False)
    # template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    # message = template.format(more_info, type(ex).__name__, ex.args)
    # logging.error(message)


def window_rms(a, window_size, mode="same"):
    """
    Calculate the rms envelope, similar to matlab.

    Parameters
    ----------
    a : ndarray
        The input signal to envelope.
    window_size : int
        The length of the window to convolve the signal with.
    mode : str
        The mode determines how many points are output
        mode "valid" will have no border effects
        mode "same" will produce a value for each input
        See np.convolve for more information.

    Returns
    -------
    np.ndarray
        The RMS envelope of the signal

    """
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, mode))


def distinct_window_rms(a, N):
    """
    Calculate the rms of an array in windows of N data points.

    Parameters
    ----------
    a : np.ndarray
        The input array to compute the RMS of.
    N : int
        The length of the window to compute RMS in.

    Returns
    -------
    list
        The RMS in each window.

    """
    a = np.array(a)
    a = np.square(a) / float(N)
    rms_array = []
    rms = 0

    # For now, just throw away the last window if it does not fit
    for idx, point in enumerate(a):
        rms += point
        if idx % N == N - 1:
            rms_array.append(np.sqrt(rms))
            rms = 0
    return rms_array


def static_vars(**kwargs):
    """Return decorator to create a function with static variables."""
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(colorcells=[])
def get_axona_colours(index=None):
    """
    Create Axona cell colours.

    Parameters
    ----------
    index : int
        Optional integer to get colours at

    Returns
    -------
    list | tuple
        A list of colours as rgb tuples with values in 0 to 1.
        Or a single rgb tuple if index is specified.

    """
    if len(get_axona_colours.colorcells) == 0:
        # create Axona cell colours if don't exist
        get_axona_colours.colorcells.append((0, 0, 200 / 255))
        get_axona_colours.colorcells.append((80 / 255, 1, 80 / 255))
        get_axona_colours.colorcells.append((1, 0, 0))
        get_axona_colours.colorcells.append((245 / 255, 0, 1))
        get_axona_colours.colorcells.append((75 / 255, 200 / 255, 255 / 255))
        get_axona_colours.colorcells.append((0 / 255, 185 / 255, 0 / 255))
        get_axona_colours.colorcells.append((255 / 255, 185 / 255, 50 / 255))
        get_axona_colours.colorcells.append((0 / 255, 150 / 255, 175 / 255))
        get_axona_colours.colorcells.append((150 / 255, 0 / 255, 175 / 255))
        get_axona_colours.colorcells.append((170 / 255, 170 / 255, 0 / 255))
        get_axona_colours.colorcells.append((200 / 255, 0 / 255, 0 / 255))
        get_axona_colours.colorcells.append((255 / 255, 255 / 255, 0 / 255))
        get_axona_colours.colorcells.append((140 / 255, 140 / 255, 140 / 255))
        get_axona_colours.colorcells.append((0 / 255, 255 / 255, 255 / 255))
        get_axona_colours.colorcells.append((255 / 255, 0 / 255, 160 / 255))
        get_axona_colours.colorcells.append((175 / 255, 75 / 255, 75 / 255))
        get_axona_colours.colorcells.append((255 / 255, 155 / 255, 175 / 255))
        get_axona_colours.colorcells.append((190 / 255, 190 / 255, 190 / 255))
        get_axona_colours.colorcells.append((255 / 255, 255 / 255, 75 / 255))
        get_axona_colours.colorcells.append((154 / 255, 205 / 255, 50 / 255))
        get_axona_colours.colorcells.append((255 / 255, 99 / 255, 71 / 255))
        get_axona_colours.colorcells.append((0 / 255, 255 / 255, 127 / 255))
        get_axona_colours.colorcells.append((255 / 255, 140 / 255, 0 / 255))
        get_axona_colours.colorcells.append((32 / 255, 178 / 255, 170 / 255))
        get_axona_colours.colorcells.append((255 / 255, 69 / 255, 0 / 255))
        get_axona_colours.colorcells.append((240 / 255, 230 / 255, 140 / 255))
        get_axona_colours.colorcells.append((100 / 255, 145 / 255, 237 / 255))
        get_axona_colours.colorcells.append((255 / 255, 218 / 255, 185 / 255))
        get_axona_colours.colorcells.append((153 / 255, 50 / 255, 204 / 255))
        get_axona_colours.colorcells.append((250 / 255, 128 / 255, 114 / 255))

    if index is None:
        return get_axona_colours.colorcells
    else:
        if index >= len(get_axona_colours.colorcells):
            logging.error("Passed colour index out of range")
            return
        return get_axona_colours.colorcells[index]


def has_ext(filename, ext, case_sensitive_ext=False):
    """
    Check if the filename ends in the extension.

    Parameters
    ----------
    filename : str
        The name of the file
    ext : str
        The extension, may have leading dot (e.g txt == .txt)
    case_sensitive_ext: bool, optional. Defaults to False,
        Whether to match the case of the file extension

    Returns
    -------
    bool indicating if the filename has the extension

    """
    if ext is None:
        return True
    if ext[0] != ".":
        ext = "." + ext
    if case_sensitive_ext:
        return filename[-len(ext):] == ext
    else:
        return filename[-len(ext):].lower() == ext.lower()


def get_all_files_in_dir(
        in_dir, ext=None, return_absolute=True,
        recursive=False, verbose=False, re_filter=None,
        case_sensitive_ext=False):
    """
    Get all files in the directory with the given extension.

    Parameters
    ----------
    in_dir : str
        The absolute path to the directory
    ext : str, optional. Defaults to None.
        The extension of files to get.
    return_absolute : bool, optional. Defaults to True.
        Whether to return the absolute filename or not.
    recursive : bool, optional. Defaults to False.
        Whether to recurse through directories.
    verbose : bool, optional. Defaults to False.
        Whether to print the files found.
    re_filter : str, optional. Defaults to None
        a regular expression used to filter the results
    case_sensitive_ext : bool, optional. Defaults to False,
        Whether to match the case of the file extension

    Returns
    -------
    list 
        A list of filenames

    """
    if not isdir(in_dir):
        print("Non existant directory " + str(in_dir))
        return []

    def match_filter(f):
        if re_filter is None:
            return True
        search_res = re.search(re_filter, f)
        return search_res is not None

    def ok_file(root_dir, f):
        good_ext = has_ext(f, ext, case_sensitive_ext=case_sensitive_ext)
        good_file = isfile(join(root_dir, f))
        good_filter = match_filter(f)
        return good_ext and good_file and good_filter

    def convert_to_path(root_dir, f):
        return join(root_dir, f) if return_absolute else f

    if verbose:
        print("Adding following files from {}".format(in_dir))

    if recursive:
        onlyfiles = []
        for root, _, filenames in os.walk(in_dir):
            start_root = root[:len(in_dir)]

            if len(root) == len(start_root):
                end_root = ""
            else:
                end_root = root[len(in_dir + os.sep):]
            for filename in filenames:
                filename = join(end_root, filename)
                if ok_file(start_root, filename):
                    to_add = convert_to_path(start_root, filename)
                    if verbose:
                        print(to_add)
                    onlyfiles.append(to_add)

    else:
        onlyfiles = [
            convert_to_path(in_dir, f) for f in sorted(listdir(in_dir))
            if ok_file(in_dir, f)
        ]
        if verbose:
            for f in onlyfiles:
                print(f)

    if verbose:
        print()
    return onlyfiles


def make_dir_if_not_exists(location):
    """Make directory structure for given location."""
    os.makedirs(os.path.dirname(location), exist_ok=True)


def remove_extension(filename, keep_dot=True, return_ext=False):
    """
    Return the filename without the extension.

    Very similar to os.path.splitext()

    Parameters
    ----------
    filename : str
        The filename to remove extension from.
    keep_dot : bool
        Whether to return filename + ".".
    return_ext : bool
        Whether to return filename or filename, ext.

    Returns
    -------
    str | tuple
        str if return_ext is False, the filename with no ext
        (str, str) if return_ext is True, (filename, ext)

    """
    modifier = 0 if keep_dot else 1
    ext = filename.split(".")[-1]
    remove = len(ext) + modifier
    if return_ext:
        return filename[:-remove], ext
    else:
        return filename[:-remove]


class RecPos:
    """
    Read .pos file.

    Work in progress and does not support head direction.

    TODO
    ----
        Read different numbers of LEDs
        Verbose file reading (prints info like number of untracked points)

    Attributes
    ----------
    pos_file : str
        The path to the position file.
    x : np.ndarray
        The x position data
    y : np.ndarray
        The y position data
    speed : np.ndarray
        The speed in cm/s
    head_direction : np.ndarray
        The head direction data
    raw_position : dict
        The raw position data decoded from .pos file.

    Parameters
    ----------
    file_name : str
        The path to the .set file or .pos file to load the data from
    load : bool
        If file_name is passed, load the data from this

    """

    def __init__(self, file_name=None, load=True):
        """See help(RecPos)."""
        self.pos_file = ""
        self.x = np.array([])
        self.y = np.array([])
        self.speed = np.array([])
        self.head_direction = np.array([])
        self.raw_position = {}
        if file_name is not None:
            self.set_file(file_name)
            if load:
                self.load()

    def set_file(self, file_name):
        """Set the input file - can be .pos or .set"""
        file_directory, file_basename = os.path.split(file_name)
        file_tag, file_extension = os.path.splitext(file_basename)
        if file_extension != ".pos":
            self.pos_file = os.path.join(file_directory, file_tag + ".pos")
        else:
            self.pos_file = file_name

    def load(self, file_name=None):
        """Load data, optionally from given file name."""
        if file_name is not None:
            self.set_file(file_name)
        self.load_raw()
        self.calculate_position()
        self.calculate_speed()
        self.calculate_angular()

    def load_raw(self):
        """Load raw position data."""
        self.bytes_per_sample = 20  # Axona daqUSB manual

        if os.path.isfile(self.pos_file):
            with open(self.pos_file, "rb") as f:
                while True:
                    line = f.readline()
                    try:
                        line = line.decode("latin-1")
                    except BaseException:
                        break
                    if line == "":
                        break
                    if line.startswith("trial_date"):
                        # Blank pos file
                        if line.strip() == "trial_date":
                            logging.error("No position data.")
                            return
                        # date = " ".join(line.replace(",", " ").split()[1:])
                    # if line.startswith("num_colours"):
                    #     colors = int(line.split()[1])
                    if line.startswith("min_x"):
                        self.min_x = int(line.split()[1])
                    if line.startswith("max_x"):
                        self.max_x = int(line.split()[1])
                    if line.startswith("min_y"):
                        self.min_y = int(line.split()[1])
                    if line.startswith("max_y"):
                        self.max_y = int(line.split()[1])
                    if line.startswith("window_min_x"):
                        self.window_min_x = int(line.split()[1])
                    if line.startswith("window_max_x"):
                        self.window_max_x = int(line.split()[1])
                    if line.startswith("window_min_y"):
                        self.window_min_y = int(line.split()[1])
                    if line.startswith("window_max_y"):
                        self.window_max_y = int(line.split()[1])
                    if line.startswith("bytes_per_timestamp"):
                        self.bytes_per_tstamp = int(line.split()[1])
                    if line.startswith("bytes_per_coord"):
                        self.bytes_per_coord = int(line.split()[1])
                    if line.startswith("pixels_per_metre"):
                        self.pixels_per_metre = int(line.split()[1])
                        self.pixels_per_cm = self.pixels_per_metre / 100.0
                    if line.startswith("num_pos_samples"):
                        self.total_samples = int(line.split()[1])
                    if line.startswith("pos_format"):
                        info = line.split(" ")[-1]
                        if info[:-2] != "t,x1,y1,x2,y2,numpix1,numpix2":
                            logging.error(
                                ".pos reading only supports 2-spot mode currently")
                            print(info[:-2])
                            print("t,x1,y1,x2,y2,numpix1,numpix2")
                            return
                    if line.startswith("data_start"):
                        break

                f.seek(0, 0)
                header_offset = []
                while True:
                    try:
                        buff = f.read(10).decode("UTF-8")
                    except BaseException:
                        break
                    if buff == "data_start":
                        header_offset = f.tell()
                        break
                    else:
                        f.seek(-9, 1)

                if not header_offset:
                    print("Error: data_start marker not found!")
                else:
                    f.seek(header_offset, 0)
                    byte_buffer = np.fromfile(f, dtype="uint8")
                    big_spotx = np.zeros([self.total_samples, 1])
                    big_spoty = np.zeros([self.total_samples, 1])
                    little_spotx = np.zeros([self.total_samples, 1])
                    little_spoty = np.zeros([self.total_samples, 1])
                    # pos format: t,x1,y1,x2,y2,numpix1,numpix2 => 20 bytes
                    for i, k in enumerate(
                        np.arange(0, self.total_samples * 20, 20)
                    ):  # Extract bytes from 20 bytes words
                        big_spotx[i] = int(
                            256 * byte_buffer[k + 4] + byte_buffer[k + 5]
                        )  # 4,5 bytes for big LED x
                        big_spoty[i] = int(
                            256 * byte_buffer[k + 6] + byte_buffer[k + 7]
                        )  # 6,7 bytes for big LED x
                        little_spotx[i] = int(
                            256 * byte_buffer[k + 8] + byte_buffer[k + 9]
                        )
                        little_spoty[i] = int(
                            256 * byte_buffer[k + 10] + byte_buffer[k + 11]
                        )

                    self.raw_position = {
                        "big_spotx": big_spotx,
                        "big_spoty": big_spoty,
                        "little_spotx": little_spotx,
                        "little_spoty": little_spoty,
                    }

        else:
            print(f"No pos file found for file {self.pos_file}")

    # Methods
    def get_cam_view(self):
        self.cam_view = {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }
        return self.cam_view

    def get_window_view(self):
        try:
            self.windows_view = {
                "window_min_x": self.window_min_x,
                "window_max_x": self.window_max_x,
                "window_min_y": self.window_min_y,
                "window_max_y": self.window_max_y,
            }
            return self.windows_view
        except BaseException:
            print("No window view")

    def get_pixel_per_metre(self):
        return self.pixels_per_metre

    def get_raw_pos(self):
        bigx = [value[0] for value in self.raw_position["big_spotx"]]
        bigy = [value[0] for value in self.raw_position["big_spoty"]]
        smallx = [value[0] for value in self.raw_position["little_spotx"]]
        smally = [value[0] for value in self.raw_position["little_spoty"]]
        return bigx, bigy, smallx, smally

    def filter_max_speed(self, x, y, max_speed=4):
        tmp_x = x.copy()
        tmp_y = y.copy()
        # max speed * distance (m) /  50 samples (s)
        threshold = max_speed * self.pixels_per_metre * 50
        for i in range(1, len(tmp_x)):
            distance = math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
            if distance > threshold:
                tmp_x[i] = np.nan
                tmp_y[i] = np.nan

        return tmp_x, tmp_y

    def get_position(self, raw=False):
        if not raw:
            return self.x, self.y
        else:
            return self.get_raw_pos()

    def calculate_position(self, raw=False):
        try:
            count_missing = 0
            bxx, sxx = [], []
            byy, syy = [], []
            bigx = [value[0] for value in self.raw_position["big_spotx"]]
            bigy = [value[0] for value in self.raw_position["big_spoty"]]
            smallx = [value[0] for value in self.raw_position["little_spotx"]]
            smally = [value[0] for value in self.raw_position["little_spoty"]]
            for bx, sx in zip(bigx, smallx):  # Try to clean single blocked LED x
                if bx == 1023 and sx != 1023:
                    bx = sx
                elif bx != 1023 and sx == 1023:
                    sx = bx
                elif bx == 1023 and sx == 1023:
                    count_missing += 1
                    bx = np.nan
                    sx = np.nan
                bxx.append(bx)
                sxx.append(sx)

            for by, sy in zip(bigy, smally):  # Try to clean single blocked LED y
                if by == 1023 and sy != 1023:
                    by = sy
                elif by != 1023 and sy == 1023:
                    sy = by
                elif by == 1023 and sy == 1023:
                    by = np.nan
                    sy = np.nan
                byy.append(by)
                syy.append(sy)

            # Remove coordinates with max_speed > 4ms
            bxx, byy = self.filter_max_speed(bxx, byy)
            sxx, syy = self.filter_max_speed(sxx, syy)

            # Interpolate missing values
            bxx = (pd.Series(bxx).astype(float)).interpolate("linear")
            sxx = (pd.Series(sxx).astype(float)).interpolate("linear")
            byy = (pd.Series(byy).astype(float)).interpolate("linear")
            syy = (pd.Series(syy).astype(float)).interpolate("linear")
            if raw:
                return [(bxx, byy), (sxx, syy)]

            # Average both LEDs
            x = list((bxx + sxx) / 2)
            y = list((byy + syy) / 2)

            # Boxcar filter 400 ms
            # sample rate = 20 ms
            b = int(400 / 20)
            kernel = np.ones(b) / b

            def pad_and_convolve(xx, kernel):
                npad = len(kernel)
                xx = np.pad(xx, (npad, npad), "edge")
                yy = np.convolve(xx, kernel, mode="same")
                return yy[npad:-npad]

            x = pad_and_convolve(x, kernel)
            y = pad_and_convolve(y, kernel)

            if np.count_nonzero(np.isnan(x)) != 0:
                num_start_nan = 0
                for val in x:
                    if np.isnan(val):
                        num_start_nan += 1
                    else:
                        break
                if num_start_nan != 0:
                    np.put(x, np.arange(0, num_start_nan, 1), x[num_start_nan])
                    np.put(y, np.arange(0, num_start_nan, 1), y[num_start_nan])

            if np.count_nonzero(np.isnan(x)) != 0:
                num_end_nan = 0
                for val in x[::-1]:
                    if np.isnan(val):
                        num_end_nan += 1
                    else:
                        break
                from_end = len(x) - num_end_nan
                back = num_end_nan + 1
                if num_end_nan != 0:
                    np.put(x, np.arange(from_end, len(x), 1), x[-back])
                    np.put(y, np.arange(from_end, len(x), 1), y[-back])

            self.x = x
            self.y = y
            return x, y

        except BaseException:
            print(f"No position information found in {self.pos_file}")

    def get_speed(self):
        return self.speed

    def calculate_speed(self, num_samples=5, smooth_size=5, smooth=False):
        """
        Calculate the speed.

        Performs as follows:
        1. Get the box smoothed position data.
        2. Calculate the speed at 10Hz (real sample rate is 50Hz).
        2a. Do this calculating the speed at time x by using positions at
            time x + 0.1, and x - 0.1. Want the real time point in the middle.
        4. Interpolate these values to get speed at every time point(50Hz)
        5. Smooth the interpolated speeds to remove bumps around sample times.

        """
        x, y = self.get_position()

        def pad_and_convolve(xx, kernel):
            npad = len(kernel)
            xx = np.pad(xx, (npad, npad), "edge")
            yy = np.convolve(xx, kernel, mode="same")
            return yy[npad:-npad]

        speed = [0]
        s_rate = num_samples  # 50 Hz is too fine grained
        t_rate = 0.02 * s_rate
        duration = len(x) * 0.02
        for i in range(s_rate * 3 // 2, len(x), s_rate):
            pixel_dist = math.sqrt(
                (x[i] - x[i - s_rate]) ** 2 + (y[i] - y[i - s_rate]) ** 2
            )
            # (pixel/s) - 300 pixels per metre * 100 (cm/s)
            cms_speed = pixel_dist / (self.pixels_per_cm * t_rate)
            speed.append(cms_speed)
        xp = np.array(
            [0.0] + [0.02 * i for i in range(s_rate, len(x) - (s_rate // 2), s_rate)]
        )
        xs = np.arange(0, duration, 0.02)
        kernel_size = smooth_size
        interp_speed = np.interp(xs, xp, speed)

        if smooth:
            kernel = np.ones(kernel_size) / kernel_size
            interp_speed = pad_and_convolve(interp_speed, kernel)

        self.speed = interp_speed
        return interp_speed

    def calculate_angular(self):
        bigx, bigy, smallx, smally = self.get_position(raw=True)

        angles = np.zeros(len(bigx))
        # for i in range(len(bigx)):
        #     A = np.array([bigx[i], bigy[i]]) + np.array([1, 0])
        #     B = np.array([bigx[i], bigy[i]])
        #     C = np.array([smallx[i], smally[i]])
        #     try:
        #         angle = angle_between_points(A, B, C)
        #     angles[i] = angle

        self.angular = angles

    def get_angular_pos(self):
        return self.angular
