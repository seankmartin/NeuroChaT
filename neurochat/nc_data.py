# -*- coding: utf-8 -*-
"""
This module implements NData Class for NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""
import logging
from collections import OrderedDict as oDict

import numpy as np

from neurochat.nc_hdf import Nhdf

from neurochat.nc_spike import NSpike
from neurochat.nc_spatial import NSpatial
from neurochat.nc_lfp import NLfp


class NData():
    """
    The NData class composes (NSpike(), NSpatial(), NLfp(), and Nhdf()).

    The NData class is built upon the composite structural object pattern.
    This data class is the main data element in NeuroChaT which
    delegates the analysis and other operations to respective objects.

    Attributes
    ----------
    spatial: NSpatial
        Spatial data object
    spike: NSpike
        Spike data object
    lfp: Nlfp
        LFP data object
    hdf: NHdf
        Object for manipulating HDF5 file
    data_format: str
        Recording system or format of the data file
    _results: OrderedDict
        The results of the analysis are written to this.
    """

    def __init__(self):
        """See NData class description."""
        super().__init__()
        self.spike = NSpike(name='C0')
        self.spatial = NSpatial(name='S0')
        self.lfp = NLfp(name='L0')
        self.data_format = 'Axona'
        self._results = oDict()
        self.hdf = Nhdf()

        self.__type = 'data'

    def subsample(self, sample_range):
        """
        Sample a time range in seconds from the overall data.

        Parameters
        ----------
        sample_range: tuple
            times in seconds to extract

        Returns
        -------
        NData
            subsampled version of initial ndata object

        """
        ndata = NData()
        if self.lfp.get_duration() != 0:
            ndata.lfp = self.lfp.subsample(
                sample_range)
        if self.spike.get_duration() != 0:
            ndata.spike = self.spike.subsample(
                sample_range)
        if self.spatial.get_duration() != 0:
            ndata.spatial = self.spatial.subsample(
                sample_range)

        return ndata

    def get_type(self):
        """
        Return the type of object. For NData, this is always `data` type.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self.__type

    def get_results(self, spaces_to_underscores=False):
        """
        Return the parametric results of the analyses.

        Parameters
        ----------
        spaces_to_underscores: bool
            If True, any keys in the results dicts with spaces
            will be converted to keys with underscores
            this is useful for compatibility with R, as an example.

        Returns
        -------
        OrderedDict

        """
        if spaces_to_underscores:
            results = oDict()
            for x, v in self._results.items():
                results[x.replace(' ', '_')] = v
            return results
        return self._results

    def update_results(self, results):
        """
        Add new parametric results of the analyses.

        Parameters
        ----------
        results : OrderedDict
            New analyses results (parametric)

        Returns
        -------
        None

        """
        self._results.update(results)

    def reset_results(self):
        """
        Reset the NData results to an empty OrderedDict.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._results = oDict()
        # self.spike.reset_results()
        # self.spatial.reset_results()
        # self.lfp.reset_results()

    def get_data_format(self):
        """
        Return the recording system or data format.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self.data_format

    def set_data_format(self, data_format=None):
        """
        Return the parametric results of the analyses.

        Parameters
        ----------
        data_format : str
            Recording system or format of the data

        Returns
        -------
        None

        """
        if data_format is None:
            data_format = self.get_data_format()
        self.data_format = data_format
        self.spike.set_system(data_format)
        self.spatial.set_system(data_format)
        self.lfp.set_system(data_format)

    def load(self):
        """
        Load the data from the filenames in each constituting object.

        This loads the spatial, spike and LFP objects.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.load_spike()
        self.load_spatial()
        self.load_lfp()

    def save_to_hdf5(self):
        """
        Store the spatial, spike and LFP datasets to HDF5 file.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        try:
            self.hdf.save_object(obj=self.spike)
        except BaseException:
            logging.warning(
                'Error in exporting NSpike data from NData object to the hdf5 file!')

        try:
            self.hdf.save_object(obj=self.spatial)
        except BaseException:
            logging.warning(
                'Error in exporting NSpatial data from NData object to the hdf5 file!')

        try:
            self.hdf.save_object(obj=self.lfp)
        except BaseException:
            logging.warning(
                'Error in exporting NLfp data from NData object to the hdf5 file!')

    def set_unit_no(self, unit_no):
        """
        Set the unit number of the spike dataset to analyse.

        Parameters
        ----------
        unit_no : int
            Unit or cell number to analyse

        Returns
        -------
        None

        """
        self.spike.set_unit_no(unit_no)

    def set_spike_name(self, name='C0'):
        """
        Set the name of the spike dataset.

        Parameters
        ----------
        name : str
            Name of the spike dataset

        Returns
        -------
        None

        """
        self.spike.set_name(name)

    def set_spike_file(self, filename):
        """
        Set the filename of the spike dataset.

        Parameters
        ----------
        filename : str
            Full file directory of the spike dataset

        Returns
        -------
        None

        """
        self.spike.set_filename(filename)

    def get_spike_file(self):
        """
        Get the filename of the spike dataset.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the spike dataset

        """
        return self.spike.get_filename()

    def load_spike(self):
        """
        Load spike dataset from the file to NSpike() object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.spike.load()

    def set_spatial_file(self, filename):
        """
        Set the filename of the spatial dataset.

        Parameters
        ----------
        filename : str
            Full file directory of the spike dataset

        Returns
        -------
        None

        """
        self.spatial.set_filename(filename)

    def get_spatial_file(self):
        """
        Get the filename of the spatial dataset.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the spatial dataset

        """
        return self.spatial.get_filename()

    def set_spatial_name(self, name):
        """
        Set the name of the spatial dataset.

        Parameters
        ----------
        name : str
            Name of the spatial dataset

        Returns
        -------
        None

        """
        self.spatial.set_name(name)

    def load_spatial(self):
        """
        Load spatial dataset from the file to NSpatial() object.

        Parameters
        ----------
        filename : str
            Full file directory of the spike dataset

        Returns
        -------
        None

        """
        self.spatial.load()

    def set_lfp_file(self, filename):
        """
        Set the filename of the LFP dataset.

        Parameters
        ----------
        filename : str
            Full file directory of the spike dataset

        Returns
        -------
        None

        """
        self.lfp.set_filename(filename)

    def get_lfp_file(self):
        """
        Get the filename of the LFP dataset.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the LFP dataset

        """
        return self.lfp.get_filename()

    def set_lfp_name(self, name):
        """
        Set the name of the NLfp() object.

        Parameters
        ----------
        name : str
            Name of the LFP dataset

        Returns
        -------
        None

        """
        self.lfp.set_name(name)

    def load_lfp(self):
        """
        Load LFP dataset to NLfp() object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.lfp.load()

    # Forwarding to analysis
    def wave_property(self):
        """
        Analysis of waveform characteristics of the spikes of a unit.

        Delegates to NSpike().wave_property()

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().wave_property

        """
        gdata = self.spike.wave_property()
        self.update_results(self.spike.get_results())

        return gdata

    def isi(self, bins='auto', bound=None, density=False,
            refractory_threshold=2):
        """
        Analysis of ISI histogram.

        Delegates to NSpike().isi()

        Parameters
        ----------
        bins : str or int
            Number of ISI histogram bins. If 'auto', NumPy default is used
        bound : int
            Length of the ISI histogram in msec
        density : bool
            If true, normalized historagm is calcultaed
        refractory_threshold : int
            Length of the refractory period in msec

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().isi

        """
        gdata = self.spike.isi(bins, bound, density, refractory_threshold)
        self.update_results(self.spike.get_results())

        return gdata

    def isi_auto_corr(self, spike=None, **kwargs):
        """
        Analysis of ISI autocrrelation histogram.

        Delegates to NSpike().isi_corr()

        Parameters
        ----------
        spike : NSpike()
            If specified, it calulates cross-correlation.

        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().isi_corr, nc_spike.NSpike().psth

        """
        gdata = self.spike.isi_corr(spike, **kwargs)

        return gdata

    def burst(self, burst_thresh=5, ibi_thresh=50):
        """
        Burst analysis of spike-train.

        Delegates to NSpike().burst()

        Parameters
        ----------
        burst_thresh : int
            Minimum ISI between consecutive spikes in a burst

        ibi_thresh : int
            Minimum inter-burst interval between two bursting groups of spikes

        Returns
        -------
        None

        See also
        --------
        nc_spike.NSpike().burst

        """
        self.spike.burst(burst_thresh, ibi_thresh=ibi_thresh)
        self.update_results(self.spike.get_results())

    def theta_index(self, **kwargs):
        """
        Theta-modulation of spike-train ISI autocorrelation histogram.

        Delegates to NSpike().theta_index()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().theta_index()

        """
        gdata = self.spike.theta_index(**kwargs)
        self.update_results(self.spike.get_results())

        return gdata

    def theta_skip_index(self, **kwargs):
        """
        Calculate theta-skipping of spike-train ISI autocorrelation histogram.

        Delegates to NSpike().theta_skip_index()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().theta_skip_index()

        """
        gdata = self.spike.theta_skip_index(**kwargs)
        self.update_results(self.spike.get_results())

        return gdata

    def bandpower_ratio(self, first_band, second_band, win_sec, **kwargs):
        """
        Calculate the ratio in power between two bandpass filtered signals.

        Delegates to NLfp.bandpower_ratio()
        For example [5, 11] and [1.5, 4 bands] for theta delta ratio.

        Parameters
        ----------
        first_band : list | tuple
            The frequency band of the ratio numerator [lower, upper].
        second_band : list | tuple
            The frequency band of the ratio denominator [lower, upper].
        win_sec : float
            The length of the window in seconds for power computation.
        **kwargs : keyword arguments
            Passed to nc_lfp.NLfp.bandpower_ratio()

        Returns
        -------
        float
            The ratio between the band powers first_band / second_band.

        See also
        --------
        nc_lfp.NLfp.bandpower_ratio()

        """
        bp = self.lfp.bandpower_ratio(
            first_band, second_band, win_sec, **kwargs)
        self.update_results(self.lfp.get_results())
        return bp

    def spectrum(self, **kwargs):
        """
        Analyses frequency spectrum of the LFP signal.

        Delegates to NLfp().spectrum()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_lfp.NLfp().spectrum()

        """
        gdata = self.lfp.spectrum(**kwargs)

        return gdata

    def phase_dist(self, **kwargs):
        """
        Analysis of spike to LFP phase distribution.

        Delegates to NLfp().phase_dist()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_lfp.NLfp().phase_dist()

        """
        gdata = self.lfp.phase_dist(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.lfp.get_results())

        return gdata

    def phase_at_spikes(self, **kwargs):
        """
        Analysis of spike to LFP phase distribution.

        Can use the spatial data to only consider
        the phase information within the place field.

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        phases, times, positions

        See also
        --------
        nc_lfp.NLfp().phase_at_events()

        """
        key = "keep_zero_idx"
        out_data = {}
        if key not in kwargs.keys():
            kwargs[key] = True
        should_filter = kwargs.get("should_filter", True)

        ftimes = self.spike.get_unit_stamp()
        phases = self.lfp.phase_at_events(ftimes, **kwargs)
        _, positions, directions = self.get_event_loc(ftimes, **kwargs)

        if should_filter:
            place_data = self.place(**kwargs)
            boundary = place_data["placeBoundary"]
            co_ords = place_data["indicesInPlaceField"]
            largest_group = place_data["largestPlaceGroup"]

            out_data["good_place"] = (largest_group != 0)
            out_data["phases"] = phases[co_ords]
            out_data["times"] = ftimes[co_ords]
            out_data["directions"] = directions[co_ords]
            out_data["positions"] = [
                positions[0][co_ords], positions[1][co_ords]]
            out_data["boundary"] = boundary

        else:
            out_data["phases"] = phases
            out_data["times"] = ftimes
            out_data["positions"] = positions
            out_data["directions"] = directions

        self.update_results(self.get_results())
        return out_data

    def plv(self, **kwargs):
        """
        Phase-locking value of the spike train to underlying LFP signal.

        Delegates to NLfp().plv()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_lfp.NLfp().plv()

        """
        gdata = self.lfp.plv(self.spike.get_unit_stamp(), **kwargs)

        return gdata

    # def sfc(self, **kwargs):
        # """
        # Calculates spike-field coherence of spike train with underlying LFP
        # signal.

        # Delegates to NLfp().sfc()

        # Parameters
        # ----------
        # **kwargs
        #     Keyword arguments

        # Returns
        # -------
        # dict
        #     Graphical data of the analysis

        # See also
        # --------
        # nc_lfp.NLfp().sfc()

        # """

        # gdata = self.lfp.plv(self.spike.get_unit_stamp(), **kwargs)

        # return gdata

    def event_trig_average(self, **kwargs):
        """
        Averaging event-triggered LFP signals based on spike times.

        Delegates to NLfp().event_trig_average()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_lfp.NLfp().event_trig_average()

        """
        gdata = self.lfp.event_trig_average(
            self.spike.get_unit_stamp(), **kwargs)

        return gdata

    def spike_lfp_causality(self, **kwargs):
        """
        Analyses spike to underlying LFP causality.

        Delegates to NLfp().spike_lfp_causality()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_lfp.NLfp().spike_lfp_causality()

        """
        gdata = self.lfp.spike_lfp_causality(
            self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.lfp.get_results())

        return gdata

    def speed(self, **kwargs):
        """
        Analysis of unit correlation with running speed.

        Delegates to NSpatial().speed()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().speed()

        """
        gdata = self.spatial.speed(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def angular_velocity(self, **kwargs):
        """
        Analyse unit correlation to angular head velocity (AHV) of the animal.

        Delegates to NSpatial().angular_velocity()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().angular_velocity()

        """
        gdata = self.spatial.angular_velocity(
            self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def place(self, **kwargs):
        """
        Analysis of place cell firing characteristics.

        Delegates to NSpatial().place()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().place()

        """
        gdata = self.spatial.place(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def loc_time_lapse(self, **kwargs):
        """
        Time-lapse firing properties of the unit with respect to location.

        This means that the recording is split into multiple continuous samples.
        Delegates to NSpatial().loc_time_lapse()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().loc_time_lapse()

        """
        gdata = self.spatial.loc_time_lapse(
            self.spike.get_unit_stamp(), **kwargs)

        return gdata

    def loc_shuffle(self, **kwargs):
        """
        Shuffling analysis of the unit for spatial properties.

        This should be performed to see if the locational firing specificity
        is by chance or actually correlated to the location of the animal.

        Delegates to NSpatial().loc_shuffle()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().loc_shuffle()

        """
        gdata = self.spatial.loc_shuffle(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def loc_shift(self, shift_ind=np.arange(-10, 11), **kwargs):
        """
        Linear time shifting analysis of the unit for spatial properties.

        Analysis of firing specificity of the unit with respect to animal's
        location to observe whether it represents past location of the animal
        or anticipates a future location.

        Delegates to NSpatial().loc_shift()

        Parameters
        ----------
        shift_ind : ndarray
            Index of spatial resolution shift for the spike event time.
            For example if shift_ind = np.array([-1, +2])
            Shift -1 implies shift to the past by 1 spatial time resolution,
            and +2 implies shift to the future by 2 spatial time resolution.
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().loc_shift()

        """
        gdata = self.spatial.loc_shift(
            self.spike.get_unit_stamp(), shift_ind=shift_ind, **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def loc_auto_corr(self, **kwargs):
        """
        Calculate the two-dimensional correlation of the firing map.

        The firing map is the map of the firing rate
        of the animal with respect to its location.

        Delegates to NSpatial().loc_auto_corr()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().loc_auto_corr()

        """
        gdata = self.spatial.loc_auto_corr(
            self.spike.get_unit_stamp(), **kwargs)

        return gdata

    def loc_rot_corr(self, **kwargs):
        """
        Calculate the rotational correlation of the firing map.

        The locational firing rate of the animal with respect to location,
        is what is meant by the firing map.

        Delegates to NSpatial().loc_rot_corr()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().loc_rot_corr()

        """
        gdata = self.spatial.loc_rot_corr(
            self.spike.get_unit_stamp(), **kwargs)

        return gdata

    def hd_rate(self, **kwargs):
        """
        Analysis the firing of a unit with respect to animal's head-direction.

        Delegates to NSpatial().hd_rate()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().hd_rate()

        """
        gdata = self.spatial.hd_rate(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def hd_rate_ccw(self, **kwargs):
        """
        Analysis of head directional tuning split into CW and CCW.

        Analysis of the firing characteristics of a unit with respect to
        animal's head-direction split into clockwise and counterclockwise
        directions.

        Delegates to NSpatial().hd_rate_ccw()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().hd_rate_ccw()

        """
        gdata = self.spatial.hd_rate_ccw(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def hd_time_lapse(self):
        """
        Time-lapse firing properties of the unit with respect to HD.

        HD stands for Head-Direction.

        Delegates to NSpatial().hd_time_lapse()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().hd_time_lapse()

        """
        gdata = self.spatial.hd_time_lapse(self.spike.get_unit_stamp())

        return gdata

    def hd_shuffle(self, **kwargs):
        """
        Shuffling analysis of the unit for head-directional firing.

        This should be performed to see if the head-directional firing
        specificity is by chance or actually correlated to the head-direction
        of the animal.

        Delegates to NSpatial().hd_shuffle()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().hd_shuffle()

        """
        gdata = self.spatial.hd_shuffle(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def hd_shift(self, shift_ind=np.arange(-10, 11), **kwargs):
        """
        Time shift analysis performed on head-directional firing.

        Analysis of firing specificity of the unit with respect to animal's
        head direction to observe whether it represents past direction or
        anticipates a future direction.

        Delegates to NSpatial().hd_shift()

        Parameters
        ----------
        shift_ind : ndarray
            Index of spatial resolution shift for the spike event time.
            Shift -1 implies shift to the past by 1 spatial time resolution
            and +2 implies shift to the future by 2 spatial time resolution.
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().speed()

        """
        gdata = self.spatial.hd_shift(
            self.spike.get_unit_stamp(), shift_ind=shift_ind)
        self.update_results(self.spatial.get_results())

        return gdata

    def border(self, **kwargs):
        """
        Analysis of the firing of a unit related to the environmental border.

        Delegates to NSpatial().border()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().border()

        """
        gdata = self.spatial.border(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def gradient(self, **kwargs):
        """
        Analysis of gradient cell.

        A gradient cell is a unit whose firing rate gradually increases
        as the animal traverses from the border to the center of the
        environment.

        Delegates to NSpatial().gradient()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().gradient()

        """
        gdata = self.spatial.gradient(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def grid(self, **kwargs):
        """
        Analysis of Grid cells.

        Grid cells are characterised by formation of grid-like pattern
        of high activity in the firing-rate map.

        Delegates to NSpatial().grid()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().grid()

        """
        gdata = self.spatial.grid(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def multiple_regression(self, **kwargs):
        """
        Multiple-regression analysis to predict firing rate of the unit.

        Specifically, the firing rate for each variable, namely
        location, head-direction, speed, AHV, and distance from border, are
        used to regress the instantaneous firing rate of the unit.

        Delegates to NSpatial().multiple_regression()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spatial.NSpatial().multiple-regression()

        """
        gdata = self.spatial.multiple_regression(
            self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

        return gdata

    def interdependence(self, **kwargs):
        """
        Interdependence analysis between variables.

        The firing rate of each variable is
        predicted from another variable and the distributive ratio is measured
        between the predicted firing rate and the calculated firing rate.

        Delegates to NSpatial().interdependence()

        Parameters
        ----------
        **kwargs
            Keyword arguments

        Returns
        -------
        None

        See also
        --------
        nc_spatial.NSpatial().interdependence()

        """
        self.spatial.interdependence(self.spike.get_unit_stamp(), **kwargs)
        self.update_results(self.spatial.get_results())

    def __getattr__(self, arg):
        """
        Set precedence for delegation with NSpike() > NLfp() > NSpatial().

        Parameters
        ----------
        arg : str
            Name of the function attributes to look for

        """
        if hasattr(self.spike, arg):
            return getattr(self.spike, arg)
        elif hasattr(self.lfp, arg):
            return getattr(self.lfp, arg)
        elif hasattr(self.spatial, arg):
            return getattr(self.spatial, arg)
        else:
            logging.warning(
                'No ' + arg + ' method or attribute in NeuroData' +
                'or in composing data class')

    def __str__(self):
        """Return a friendly string representation of the object."""
        return "{} object in {} format containing:\n\t{}\n\t{}\n\t{}".format(
            "NeuroChaT NData", self.data_format,
            self.spike,
            self.spatial,
            self.lfp
        )
