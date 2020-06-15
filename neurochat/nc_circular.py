# -*- coding: utf-8 -*-
"""
This module implements CircStat Class for NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import logging
from collections import OrderedDict as oDict

import numpy as np

from neurochat.nc_utils import find


class CircStat(object):
    """
    This class is the placeholder for circular data.

    It provides functionalities for calculating circular statistics.
    For example, Rayleigh Z statistics of the circular data.

    Currently the class only supports calculations in degrees.
    As such, radians should be converted to degrees when entering theta.

    The initialisation function should be passed with keyword arguments.

    Keyword Arguments
    -----------------
        rho : ndarry
            Polar co-ordinate rho, the radii of points.
        theta : ndarray
            Polar co-ordinate theta, the angle of points.

    Attributes
    ----------
    _rho : ndarray
        Polar co-ordinate rho, the radii of points.
    _theta : ndarray
        Polar co-ordinate theta, the angle of points.
    _result : OrderedDict
        Holds the result of many circular statistics functions.

    """

    def __init__(self, **kwargs):
        """
        Create a circular statistics object.

        Parameters
        ----------
        **kwargs: keyword arguments
            rho : ndarry
                Polar co-ordinate rho, the radii of points.
            theta : ndarray
                Polar co-ordinate theta, the angle of points.

        Returns
        -------
        None

        """
        self._rho = kwargs.get('rho', None)
        self._theta = kwargs.get('theta', None)
        self._result = oDict()

    def set_rho(self, rho=None):
        """
        Set the radial coordinates (rho) of the circular data.

        Parameters
        ----------
        rho : ndarray
            Radial coordinates of the circular data

        Returns
        -------
        None

        """
        if rho is not None:
            self._rho = rho

    def get_rho(self):
        """
        Return the radial coordinates (rho) of the circular data.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Radial coordinates of the circular data

        """
        return self._rho

    def set_theta(self, theta=None):
        """
        Set the angular coordinates (theta) of the circular data in degrees.

        Parameters
        ----------
        theta : ndarray
            Angular coordinates of the circular data

        Returns
        -------
        None

        """
        if theta is not None:
            self._theta = theta

    def get_theta(self):
        """
        Return the angular coordinates (theta) of the circular data.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Angular coordinates of the circular data

        """
        return self._theta

    def get_mean_std(self):
        """
        Return the circular mean and standard deviation of the data.

        Also return the resultant vector length of the data.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary of mean, standard deviation and resultant vector length.

        """
        if self._rho is None or not len(self._rho):
            self._rho = np.ones(self._theta.shape)
        return self._calc_mean_std()

    def _calc_mean_std(self):
        """
        Calculate mean, standard deviation, resultant vector length.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary of mean, standard deviation and resultant vector length.

        """
        result = {}
        if self._rho.shape[0] == self._theta.shape[0]:
            xm = np.sum(np.multiply(
                self._rho, np.cos(self._theta * np.pi / 180)))
            ym = np.sum(np.multiply(
                self._rho, np.sin(self._theta * np.pi / 180)))
            meanTheta = np.arctan2(ym, xm) * 180 / np.pi
            if meanTheta < 0:
                meanTheta = meanTheta + 360
            meanRho = np.sqrt(xm**2 + ym**2)
            result['meanTheta'] = meanTheta
            result['meanRho'] = meanRho
            result['totalObs'] = np.sum(self._rho)
            result['resultant'] = meanRho / result['totalObs']
            try:
                x = -2 * np.log(result['resultant'])
                if x < 0:
                    result['stdRho'] = 0
                else:
                    result['stdRho'] = np.sqrt(x)
            except BaseException:
                # This except is to protect against -ve inside sqrt
                result['stdRho'] = 0

        else:
            logging.warning('Size of rho and theta must be equal')

        return result

    def get_rayl_stat(self):
        """
        Return the Rayleigh Z statistics of the circular data.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Rayleigh Z statistics

        """
        return self._rayl_stat()

    def _rayl_stat(self):
        """
        Compute the Rayleigh Z statistics of the circular data.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Rayleigh Z statistics

        """
        result = {}
        N = self._result['totalObs']
        Rn = self._result['resultant'] * N
        result['RaylZ'] = Rn**2 / N
        result['RaylP'] = np.exp(
            np.sqrt(1 + 4 * N + 4 * (N**2 - Rn**2)) - (1 + 2 * N))

        return result

    def get_vonmises_stat(self):
        """
        Return the von Mises concentration parameter kappa.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Returns the von Mises concentration parameter kappa

        """
        return self._vonmises_stat()

    def _vonmises_stat(self):
        """
        Calculate the von Mises concentration parameter kappa.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Returns the von Mises concentration parameter kappa

        """
        result = {}
        R = self._result['resultant']
        N = self._result['totalObs']

        if R < 0.53:
            kappa = 2 * R + R**3 + 5 * (R**5) / 6
        elif R <= 0.53 and R < 0.85:
            kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            kappa = 1 / (R**3 - 4 * R**2 + 3 * R)

        if N < 15 and N > 1:
            kappa = max(kappa - 2 * (N * kappa)**-1,
                        0) if kappa < 2 else kappa * (N - 1)**3 / (N**3 + N)

        result['vonMisesK'] = kappa
        return result

    def calc_stat(self):
        """
        Calculate and return all the circular statistics parameters.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Returns the mean, standard deviation, resultant vector length.
            The Rayleigh Z statistics.
            The von Mises concentration parameter Kappa.

        """
        result = self._calc_mean_std()
        self._update_result(result)
        result = self._rayl_stat()
        self._update_result(result)
        result = self._vonmises_stat()
        self._update_result(result)

        return self.get_result()

    @staticmethod
    # Example, x = [270, 340, 350, 20, 40], y = [270, 340, 350, 380, 400] etc.
    def circ_regroup(x):
        """
        Circular regrouping of the angles. It unwraps the angular coordinates.

        For example, if the input array x is
        x = np.ndarray([270, 340, 350, 20, 40]),
        the output will be
        y = np.ndarray([270, 340, 350, 380, 400])

        Parameters
        ----------
        x : ndarray
            Array containing the angular coordinates

        Returns
        -------
        y : ndarray
            Regrouped or unwrapped angular coordinates

        """
        y = np.copy(x)
        if (any(np.logical_and(x >= 0, x <= 90)) and
                any(np.logical_and(x >= 180, x <= 360))):
            y[np.logical_and(x >= 0, x <= 90)] = (
                x[np.logical_and(x >= 0, x <= 90)] + 360)

        return y

    def circ_histogram(self, bins=5):
        """
        Calculate the circular histogram of the angular coordinates.

        Parameters
        ----------
        bins : int or ndarray
            Angular binsize for the circular histogram if int.
            Angular bins if ndarray.

        Returns
        -------
        count : ndarray
            Histogram bin count
        ind : ndarray
            Indices of the bins to which each value in input array belongs.
            Similar to the return values of the numpy.digitize function.
        bins : ndarray
            Histogram bins

        """
        if isinstance(bins, int):
            bins = np.arange(0, 360, bins)

        nbins = bins.shape[0]
        count = np.zeros(bins.shape)
        ind = np.zeros(self._theta.shape, dtype=int)
        for i in np.arange(nbins):
            if i < nbins - 1:
                ind[np.logical_and(self._theta >= bins[i],
                                   self._theta < bins[i + 1])] = i
                count[i] = np.sum(np.logical_and(
                    self._theta >= bins[i], self._theta < bins[i + 1]))

            elif i == nbins - 1:
                ind[np.logical_or(self._theta >= bins[i],
                                  self._theta < bins[0])] = i
                count[i] = np.sum(np.logical_or(
                    self._theta >= bins[i], self._theta < bins[0]))

        return count, ind, bins

    def circ_smooth(self, filttype='b', filtsize=5):
        """
        Calculate the circular average of theta.

        Each sample is replaced by the circular mean of length 'filtsize'
        and weights determined by the type of filter.

        Parameters
        ----------
        filttype : str
            Type of smoothing filter.
            'b' for Box filter, or 'g' for Gaussian filter.
        filtsize : int
            Length of the averaging filter

        Returns
        -------
        smooth_theta : ndarray
            Theta values after the smoothing

        """
        if filttype == 'g':
            halfwid = np.round(3 * filtsize)
            xx = np.arange(-halfwid, halfwid + 1, 1)
            filt = np.exp(-(xx**2) / (2 * filtsize**2)) / \
                (np.sqrt(2 * np.pi) * filtsize)
        elif filttype == 'b':
            filt = np.ones(filtsize, ) / filtsize

        cs = CircStat()

        smooth_theta = np.zeros(self._theta.shape)
        N = self._theta.shape[0]
        L = filt.shape[0]
        l = int(np.floor(L / 2))
        for i in np.arange(l):
            cs.set_rho(filt[l - i:])
            cs.set_theta(self.circ_regroup(self._theta[:L - l + i]))
            csResult = cs.get_mean_std()
            smooth_theta[i] = csResult['meanTheta']
        for i in np.arange(l, N - l, 1):
            cs.set_rho(filt)
            cs.set_theta(self.circ_regroup(self._theta[i - l:i + l + 1]))
            csResult = cs.get_mean_std()
            smooth_theta[i] = csResult['meanTheta']
        for i in np.arange(N - l, N):
            cs.set_theta(self.circ_regroup(self._theta[i - l:]))
            cs.set_rho(filt[:len(self._theta[i - l:])])
            csResult = cs.get_mean_std()
            smooth_theta[i] = csResult['meanTheta']

        return smooth_theta

    def circ_scatter(self, bins=2, step=0.05, rmax=None):
        """
        Prepare data for circular scatter plot.

        For each theta in a bin, the radius is increased by 'step'
        The size of step is capped at 'rmax'.

        Parameters
        ----------
        bins : int
            Angular binsize for the circular scatter
        step : float
            Stepsize to increase the radius for each count of theta
        rmax : float
            Maximum value for the radius

        Returns
        -------
        radius : ndarray
            Radius for the theta values.
            For each new theta in a bin, the radius is increased by 'step'.
        theta : ndarray
            Binned theta samples

        """
        count, ind, bins = self.circ_histogram(bins=2)
        radius = np.ones(ind.shape)
        theta = np.zeros(ind.shape)
        for i, b in enumerate(bins):
            rad = (
                np.ones(find(ind == i).shape) +
                np.array(
                    list(step * j for j, loc in enumerate(find(ind == i)))
                )
            )
            if rmax:
                rad[rad > rmax] = rmax
            radius[ind == i] = rad
            theta[ind == i] = b

        return radius, theta

    def _update_result(self, new_result={}):
        """
        Update the statistical results with a new dict of results.

        Parameters
        ----------
        new_results : dict
            Dictionary of the circular statistics analyses

        Returns
        -------
        None

        """
        self._result.update(new_result)

    def get_result(self):
        """
        Return the results of the circular statistics analyses.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Results of the circular statistics analyses

        """
        return self._result
