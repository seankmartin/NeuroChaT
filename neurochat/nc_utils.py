# -*- coding: utf-8 -*-
"""
This module implements utility functions and classes for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import logging
import time
from collections import OrderedDict as oDict
import os
from os import listdir
from os.path import isfile, isdir, join
import re

import pandas as pd
import numpy as np
import numpy.linalg as nalg

import scipy
import scipy.stats as stats
import scipy.signal as sg
from scipy.fftpack import fft

class NLog(logging.Handler):
    """
    Class for handling log information (messages, errors and warnings) for NeuroChaT.
    It formats the incoming message in HTML and sends it to the log interface of NeuroChaT.
    
    """
    def __init__(self):
        super().__init__()
        self.setup()
    def setup(self):
        """
        Removes all the logging handlers and sets up a new logger with HTML formatting.
        
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
        fmt = logging.Formatter('%(asctime)s (%(filename)s)  %(levelname)s--  %(message)s', '%H:%M:%S')
        self.setFormatter(fmt)
        log.addHandler(self)
        # You can control the logging level
        log.setLevel(logging.DEBUG)
        logging.addLevelName(20, '')
        
    def emit(self, record):
        """
        Formats the incoming record and 
        
        Parameters
        ----------
        record
            Log record to dispkay or store
        
        Returns
        -------
        None
        
        """
        
        msg = self.format(record)
        level = record.levelname
        msg = level+ ':'+ msg
        print(msg)
        time.sleep(0.25)
#        self.emit(QtCore.SIGNAL('update_log(QString)'), msg)

class Singleton(object):
    """
    Creates a Singleton object created from a subclass of this class
    
    """
    
    def __new__(cls, *arg, **kwarg):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *arg, **kwarg)
        return cls._instance

def bhatt(X1, X2):
    """
    Calculates Bhattacharyya coefficient and Bhattacharyya distance between two distributions
    
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
        C = (C1+ C2)/2
        chol = nalg.cholesky(C).T
        dmu = (mu1- mu2)@nalg.inv(chol)
        try:
            d = 0.125*dmu@(dmu.T)+ 0.5*np.log(nalg.det(C)/np.sqrt(nalg.det(C1)*nalg.det(C2)))
        except:
            d = 0.125*dmu@(dmu.T)+ 0.5*np.log(np.abs(nalg.det(C@nalg.inv(scipy.linalg.sqrtm(C1@C2)))))
        bc = np.exp(-1*d)

        return bc, d
    else:
        logging.error('Cannot measure Bhattacharyya distance, column sizes do not match!')

def butter_filter(x, Fs, *args):
    """
    Filtering function using bidirectional zero-phase shift Butterworth filter.
    
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
    
    gstop = 20 # minimum dB attenuation at stopabnd
    gpass = 3 # maximum dB loss during ripple
#    order= args[0]
    for arg in args:
        if isinstance(arg, str):
            filttype = arg
    if filttype == 'lowpass' or filttype == 'highpass':
        wp = args[1]/(Fs/2)
        if wp > 1:
            wp = 1
            if filttype == 'lowpass':
                logging.warning('Butterworth filter critical freqeuncy Wp is capped at 1')
            else:
                logging.error('Cannot highpass filter over Nyquist frequency!')

    elif filttype == 'bandpass':
        if len(args) < 4:
            logging.error('Insufficient Butterworth filter arguments')
        else:
            wp = np.array(args[1:3])/(Fs/2)
            if wp[0] >= wp[1]:
                logging.error('Butterworth filter lower cutoff frequency must be smaller than upper cutoff freqeuncy!')

            if wp[0] == 0 and wp[1] >= 1:
                logging.error('Invalid filter specifications, check cutt off frequencies and sampling frequency!')
            elif wp[0] == 0:
                wp = wp[1]
                filttype = 'lowpass'
                logging.warning('Butterworth filter type selected: lowpass')
            elif wp[1] >= 1:
                wp = wp[0]
                filttype = 'highpass'
                logging.warning('Butterworth filter type selected: highpass')

    if filttype == 'lowpass':
        ws = min([wp+ 0.1, 1])
    elif filttype == 'highpass':
        ws = max([wp- 0.1, 0.01/(Fs/2)])
    elif filttype == 'bandpass':
        ws = np.zeros_like(wp)
        ws[0] = max([wp[0]- 0.1, 0.01/(Fs/2)])
        ws[1] = min([wp[1]+ 0.1, 1])

    min_order, min_wp = sg.buttord(wp, ws, gpass, gstop)
#    if order<= min_order:
#        order= min_order
#        wp= min_wp

    b, a = sg.butter(min_order, min_wp, btype=filttype, output='ba')

    return sg.filtfilt(b, a, x)

def chop_edges(x, xlen, ylen):
    """
    Chope the edges of a firing rate map if they are not visited at ll or with zero firing rate
    
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

# Following is the old MATLAB logic, we have changed it to remove the edges with zero count
#        if no_filled_bins1< no_filled_bins2:
#            low_ind[1] += 1
#        else:
#            high_ind[1] -= 1
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

# Following is the old MATLAB logic, we have changed it to remove the edges with zero count
#        if no_filled_bins1< no_filled_bins2:
#            low_ind[0] += 1
#        else:
#            high_ind[0]-=1
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
        return np.sum(np.multiply(x1- x1.mean(), x2- x2.mean()))/ \
            np.sqrt(np.sum((x1- x1.mean())**2)*np.sum((x2- x2.mean())**2))
    except:
        return 0

def extrema(x, mincap=None, maxcap=None):
    """
    Finds the extrema in a numeric array or a signal
    
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

    a = find(dx != 0) # indices where x changes
    lm = find(np.diff(a) != 1)+1 # indices where a is not sequential
    d = a[lm]- a[lm-1]
    a[lm] = a[lm]- np.floor(d//2)

    xa = x[a] # series without flat peaks
    d = np.sign(xa[1:-1]- xa[:-2])- np.sign(xa[2:]- xa[1:-1])
    imax = a[find(d > 0)+1]
    xmax = x[imax]
    imin = a[find(d < 0)+1]
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
    Calculates the Fast Fourier Transform (FFT) of a signal.
    
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
        nfft = 2**(np.floor(np.log2(len(x)))+1)

    if nfft < Fs:
        nfft = 2**(np.floor(np.log2(Fs))+1)
    nfft = int(nfft)
    dummy = np.zeros(nfft)
    if nfft > len(x):
        dummy[:len(x)] = x
        x = dummy

    winfun = np.hanning(nfft)
    xf = np.arange(0, Fs, Fs/nfft)
    f = xf[0: int(nfft/2)+ 1]


    if side == 'one':
        x_fft = fft(np.multiply(x, winfun), nfft)
        if ptype == 'psd':
            x_fft = np.absolute(x_fft[0: int(nfft/2)+ 1])**2/nfft**2
            x_fft[1:-1] = 2*x_fft[1:-1]

    return x_fft, f

def find(X, n=None, direction='all'):
    """
    Finds the non-zero entries of a signal or array.
    
    Parameters
    ----------
    X : ndarray or list
        Array or list of numbers whose non-zero entries need to find out
    n : int
        Number of such entries
    direction : str
        If 'all', all entries of length n are returned. If 'first', first n entries
        are returned. If 'last', last n entrues are returned.
    
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
            ind = ind[np.flipud(np.arange(-1, -(n+1), - 1))]
    return np.array(ind)

def find2d(X, n=None):
    """
    Finds the non-zero entries of a matrix.
    
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
            J.extend(r*np.ones((len(find(X[r, ])), ), dtype=int))
        if len(I):
            if n is not None and n < len(I):
                I = I[:n]
                J = J[:n]
        return np.array(J), np.array(I)

    else:
        logging.error('ndrray is not 2D. Check shape attributes of the input!')

def find_chunk(x):
    """
    Finds size and indeices of chunks of non-zero segments in an array
    
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
    
    # x is a binary array input i.e. x= data> 0.5 will find all the chunks in data where data is greater than 0.5
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
            segind[j:i] = c # indexing by size of the chunk
        i += 1
    return segsize, segind

def hellinger(X1, X2):
    """
    Calculates Hellinger distance between two distributions.
    
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
        logging.error('Hellinger distance cannot be computed, column sizes do not match!')
    else:
        return np.sqrt(1- bhatt(X1, X2)[0])

def histogram(x, bins):    
    """
    Calculates the histogram count of input array
    
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
    # This function is not a replacement of np.histogram; it is created for convenience
    # of binned-based rate calculations and mimicking matlab histc that includes digitized indices
    if isinstance(bins, int):
        bins = np.arange(np.min(x), np.max(x), (np.max(x)- np.min(x))/bins)
    bins = np.append(bins, bins[-1]+ np.mean(np.diff(bins)))
    return np.histogram(x, bins)[0], np.digitize(x, bins)-1, bins[:- 1]

def histogram2d(y, x, ybins, xbins):
    """
    Calculates the joint histogram count of two arrays
    
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
    
    # This function is not a repalcement of np.histogram
    if isinstance(xbins, int):
        xbins = np.arange(np.min(x), np.max(x), (np.max(x)- np.min(x))/xbins)
    xbins = np.append(xbins, xbins[-1]+ np.mean(np.diff(xbins)))
    if isinstance(ybins, int):
        ybins = np.arange(np.min(y), np.max(y), (np.max(y)- np.min(y))/ybins)
    ybins = np.append(ybins, ybins[-1]+ np.mean(np.diff(ybins)))

    return np.histogram2d(y, x, [ybins, xbins])[0], ybins[:-1], xbins[:-1]

def linfit(X, Y, getPartial=False):
    """
    Calculates the linear regression coefficients in least-square sense.
    
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
        semiCorr = np.zeros(Nd) # Semi partial correlation
        for d in np.arange(Nd):
            part_results = linfit(np.delete(X, 1, axis=0), Y, getPartial=False)
            semiCorr[d] = _results['Rsq']- part_results['Rsq']
        _results['semiCorr'] = semiCorr

    return _results

def nxl_write(
    file_name, data_frame, sheet_name='Sheet1', startRow=0, startColumn=0):
    """
    Write Pandas DataFrame to excel file. It is a wrapper for Pandas.ExcelWriter()
    
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
    Calculates the goodness of fit and other residual statistics between observed
    and fitted values from a model
    
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
    res = y- y_fit
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y- np.mean(y))**2)
    r_sq = 1- ss_res/ss_tot
    adj_r_sq = 1- (ss_res/ ss_tot)* ((len(y)-1)/(len(y)- p-1))
    _results['Pearson R'], _results['Pearson P'] = stats.pearsonr(y, y_fit)

    _results['Rsq'] = r_sq
    _results['adj Rsq'] = adj_r_sq

    return _results

def rot_2d(x, theta):
    """
    Rotates a firing map by a specified angle
    
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
    
    return scipy.ndimage.interpolation.rotate(x, theta, reshape=False, mode='constant', cval=np.min(x))

# Created by Sean Martin 14/02/2019
def angle_between_points(a, b, c):
    """
    Returns the angle between the lines ab and bc, <abc
    
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
    Calculates the co-ordinate centre of mass for a system of particles with co ords and weights

    Parameters
    ----------
    co_ords : ndarray
        Array of co-ordinate positions, assumed to have co_ords.shape[axis] co-ordinates
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
    
def smooth_1d(x, filttype='b', filtsize=5, **kwargs):
    """
    Filters a 1D array or signal.
    
    Parameters
    ----------
    x : ndarray
        Array or signal to be filtered. If matrix, each column or row is filtered
        individually depending on 'dir' parameter that takes either '0' for along-column
        and '1' for along-row filtering.
    filttype : str
        'b' for moving average or box filter. 'g' for Gaussian filter
    filtsize
        Box size for box filter and sigma for Gaussian filter
        
    Returns
    -------
    ndarray
        Filtered data
    
    """
    
    x = np.array(x)
    direction = kwargs.get('dir', 0) # default along column
    if filttype == 'g':
        halfwid = np.round(3*filtsize)
        xx = np.arange(-halfwid, halfwid+1, 1)
        filt = np.exp(-(xx**2)/(2*filtsize**2))/(np.sqrt(2*np.pi)*filtsize)
    elif filttype == 'b':
        filt = np.ones(filtsize, )/filtsize

    if len(x.shape) == 1:
        result = np.convolve(x, filt, mode='same')
    elif len([x.shape]) == 2:
        result = np.zeros(x.shape)
        if direction:
            for i in np.arange(0, x.shape[0]):
                result[i, :] = np.convolve(x[i, :], filt, mode='same')
        else:
            for i in np.arange(0, x.shape[0]):
                result[:, i] = np.convolve(x[:, i], filt, mode='same')
    return result

def smooth_2d(x, filttype='b', filtsize=5):
    """
    Filters a 2D array or signal.
    
    Parameters
    ----------
    x : ndarray
        Matrix to be filtered
    filttype : str
        'b' for moving average or box filter. 'g' for Gaussian filter
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
        halfwid = np.round(3*filtsize)
        xx, yy = np.meshgrid(np.arange(-halfwid, halfwid+1, 1), np.arange(-halfwid, halfwid+1, 1), copy=False)
        filt = np.exp(-(xx**2+ yy**2)/(2*filtsize**2)) # /(2*np.pi*filtsize**2) # This is the scaling used before;
                                                        #But tested with ones(50, 50); gives a hogher value
        filt = filt/ np.sum(filt)
    elif filttype == 'b':
        filt = np.ones((filtsize, filtsize))/filtsize**2

    smoothX = sg.convolve2d(x, filt, mode='same')
    smoothX[nanInd] = np.nan

    return smoothX

def find_true_ranges(arr, truth_arr, min_range):
    """
    Returns a list of ranges where truth values occur and the corresponding
    values from arr, arr is assumed to be a sorted list

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
    for idx, b in enumerate(truth_arr):
        if b and not in_range:
            in_range = True
            range_start = arr[idx]
        if not b and in_range:
            in_range = False
            range_end = arr[idx - 1]
            if range_end - range_start > min_range:
                ranges.append((range_start, range_end))
    return ranges

def find_peaks(data, **kwargs):
    """
    Returns the peaks in the data based on gradient calculations

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

    peak_loc = [j for j in np.arange(start_at, end_at-1) \
                if slope[j] > 0 and slope[j+1] <= 0]
    peak_val = [data[peak_loc[i]] for i in range(0, len(peak_loc))]

    valid_loc = [
        i for i in range(0, len(peak_loc)) if peak_val[i] >= thresh]
    if len(valid_loc) == 0:
        return []
    peak_val, peak_loc= zip(*((peak_val[i], peak_loc[i]) for i in valid_loc))
    return np.array(peak_val), np.array(peak_loc)


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info

    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info : 
        Additional string to log
    
    Returns
    -------
    None
    
    """

    template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    message = template.format(more_info, type(ex).__name__, ex.args)
    logging.error(message)

def window_rms(a, window_size, mode="same"):
    """
    Calculate the rms envelope, similar to matlab.  
    
    mode determines how many points are output
    mode valid will have no border effects
    mode same will produce a value for each input
    """
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, mode))

def distinct_window_rms(a, N):
    """
    Calculate the rms of a in windows of N data points.
    """
    a = np.array(a)
    a = np.square(a) / float(N)
    rms_array = []
    rms = 0
    
    # For now, just throw away the last window if it does not fit
    for idx, point in enumerate(a):
        rms += point
        if idx % N == N-1:
            rms_array.append(np.sqrt(rms))
            rms = 0
    return rms_array

def static_vars(**kwargs):
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
    list :
        A list of colours as rgb tuples with values in 0 to 1
    """

    if len(get_axona_colours.colorcells) == 0:
    # crget_axona_colours.eate Axona cell colours if don't exist
        get_axona_colours.colorcells.append((0,0,200/255))
        get_axona_colours.colorcells.append((80/255,1,80/255))
        get_axona_colours.colorcells.append((1,0,0))
        get_axona_colours.colorcells.append((245/255,0,1))
        get_axona_colours.colorcells.append((75/255,200/255,255/255))
        get_axona_colours.colorcells.append((0/255, 185/255,0/255))
        get_axona_colours.colorcells.append((255/255, 185/255,50/255))
        get_axona_colours.colorcells.append((0/255, 150/255,175/255))
        get_axona_colours.colorcells.append((150/255, 0/255,175/255))
        get_axona_colours.colorcells.append((170/255, 170/255,0/255))
        get_axona_colours.colorcells.append((200/255, 0/255,0/255))
        get_axona_colours.colorcells.append((255/255, 255/255,0/255))
        get_axona_colours.colorcells.append((140/255, 140/255,140/255))
        get_axona_colours.colorcells.append((0/255, 255/255,255/255))
        get_axona_colours.colorcells.append((255/255, 0/255,160/255))
        get_axona_colours.colorcells.append((175/255, 75/255, 75/255))
        get_axona_colours.colorcells.append((255/255, 155/255, 175/255))
        get_axona_colours.colorcells.append((190/255, 190/255, 190/255))
        get_axona_colours.colorcells.append((255/255, 255/255, 75/255))
        get_axona_colours.colorcells.append((154/255, 205/255, 50/255))
        get_axona_colours.colorcells.append((255/255, 99/255, 71/255))
        get_axona_colours.colorcells.append((0/255, 255/255, 127/255))
        get_axona_colours.colorcells.append((255/255, 140/255, 0/255))
        get_axona_colours.colorcells.append((32/255, 178/255, 170/255))
        get_axona_colours.colorcells.append((255/255, 69/255, 0/255))
        get_axona_colours.colorcells.append((240/255, 230/255, 140/255))
        get_axona_colours.colorcells.append((100/255, 145/255, 237/255))
        get_axona_colours.colorcells.append((255/255, 218/255, 185/255))
        get_axona_colours.colorcells.append((153/255, 50/255, 204/255))
        get_axona_colours.colorcells.append((250/255, 128/255, 114/255))

    if index is None:
        return get_axona_colours.colorcells
    else:
        if index >= len(get_axona_colours.colorcells):
            logging.error("Passed colour index out of range")
            return
        return get_axona_colours.colorcells[index]

def has_ext(filename, ext):
    """
    Check if the filename ends in the extension
    
    Parameters
    ----------
    filename : str
        The name of the file
    ext : str
        The extension, may have leading dot (e.g txt == .txt)
    
    Returns
    -------
    bool indicating if the filename has the extension

    """
    if ext == None:
        return True
    if ext[0] != ".":
        ext = "." + ext
    return filename[-len(ext):].lower() == ext.lower()


def get_all_files_in_dir(
        in_dir, ext=None, return_absolute=True, 
        recursive=False, verbose=False, re_filter=None):
    """
    Get all files in the directory with the given extension.
    
    Parameters
    ----------
    in_dir : str
        The absolute path to the directory
    ext : str, optional. Defaults to True.
        The extension of files to get.
    return_absolute : bool, optional. Defaults to True.
        Whether to return the absolute filename or not.
    recursive: bool, optional. Defaults to False.
        Whether to recurse through directories.
    verbose: bool, optional. Defaults to False.
        Whether to print the files found.

    Returns
    -------
    List : A list of filenames
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
        return has_ext(f, ext) and isfile(join(root_dir, f)) and match_filter(f)

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
    """Makes directory structure for given location"""
    os.makedirs(os.path.dirname(location), exist_ok=True)

def remove_extension(filename, keep_dot=True, return_ext=False):
    modifier = 0 if keep_dot else 1
    ext = filename.split(".")[-1]
    remove = len(ext) + modifier
    if return_ext:
        return filename[:-remove], ext
    else:
        return filename[:-remove]
