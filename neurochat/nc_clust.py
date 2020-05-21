# -*- coding: utf-8 -*-
"""
This module implements NClust Class for NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""
import logging

import numpy as np

from neurochat.nc_base import NBase
from neurochat.nc_spike import NSpike

from neurochat.nc_utils import bhatt, find, hellinger

import scipy as sc
from sklearn.decomposition import PCA


class NClust(NBase):
    """
    This class facilitates clustering-related operations.

    Although no clustering algorithm is implemented in this class,
    it can be subclassed to create such algorithms.

    Many of the functions in this class are delegated to the spike attr.

    Attributes
    ----------
    spike : NSpike
        An object of NSpike() class.

    """

    def __init__(self, **kwargs):
        """
        Create an NClust object.

        Parameters
        ----------
        **kwargs : Keyword arguments
            spike: NSpike object,
                If directly passed an NSpike object, this is stored.
            Otherwise if spike is not NSpike or spike is not a kwarg,
            self.spike = NSpike(**kwargs)

        Returns
        -------
        None

        """
        spike = kwargs.get('spike', None)
        self.wavetime = []
        self.UPSAMPLED = False
        self.ALLIGNED = False
        self.NULL_CHAN_REMOVED = False

        if isinstance(spike, NSpike):
            self.spike = spike
        else:
            self.spike = NSpike(**kwargs)
        super().__init__(**kwargs)

    def get_unit_tags(self):
        """
        Return tags of the spiking waveforms from clustering.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        return self.spike.get_unit_tags()

    def set_unit_tags(self, new_tags=None):
        """
        Return tags of the spiking waveforms from clustering.

        Parameters
        ----------
        new_tags : ndarray
            Array that contains the tags for spike-waveforms
            which is based on the cluster number.

        Returns
        -------
        None

        """
        self.spike.set_unit_tags(new_tags)

    def get_unit_list(self):
        """
        Return the list of units in a spike dataset.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of units

        """
        return self.spike.get_unit_list()

    def _set_unit_list(self):
        """
        Set the unit list.

        Delegates to NSpike._set_unit_list()

        Parameters
        ----------
        None

        Returns
        -------
        None

        See also
        --------
        nc_spike.NSPike()._set_unit_list

        """
        self.spike._set_unit_list()

    def get_timestamp(self, unit_no=None):
        """
        Return the timestamps of the spike-waveforms of specified unit.

        Parameters
        ----------
        unit_no : int
            Unit whose timestamps are to be returned

        Returns
        -------
        ndarray
            Timestamps of the spiking waveforms

        """
        self.spike.get_timestamp(unit_no=unit_no)

    def get_unit_spikes_count(self, unit_no=None):
        """
        Return the total number of spikes in a specified unit.

        Parameters
        ----------
        unit_no : int
            Unit whose count is returned

        Returns
        -------
        int
            Total number of spikes in the unit

        """
        return self.spike.get_unit_spikes_count(unit_no=unit_no)

    def get_waveform(self):
        """
        Return the waveforms in the spike dataset.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Each key represents one channel of the electrode group.
            Each value represents the waveforms of the spikes
            in a matrix form (no_samples x no_spikes)

        """
        return self.spike.get_waveform()

    def _set_waveform(self, spike_waves=[]):
        """
        Set the waveforms of the spike dataset.

        Parameters
        ----------
        spike_waves : dict
            Each key represents one channel of the electrode group.
            Each value represents the waveforms of the spikes
            in a matrix form (no_samples x no_spikes)

        Returns
        -------
        None

        """
        self.spike._set_waveform(spike_waves=spike_waves)

    def get_unit_waves(self, unit_no=None):
        """
        Return spike waveforms of a specific unit.

        Parameters
        ----------
        unit_no : int
            Unit whose waveforms are returned

        Returns
        -------
        dict
            Spike waveforms in each channel of the electrode group

        """
        return self.spike.get_unit_waves(unit_no=unit_no)

    # For multi-unit analysis,
    # {'SpikeName': cell_no} pairs should be used as function input

    def load(self, filename=None, system=None):
        """
        Load spike dataset from the file.

        Parameters
        ----------
        filename: str
            Name of the spike file
        system : str
            Name of the recording format or system.

        Returns
        -------
        None

        See Also
        --------
        nc_spike.NSpike().load()

        """
        self.spike.load(filename=filename, system=system)

    def add_spike(self, spike=None, **kwargs):
        """
        Add new spike node to current NSpike() object.

        Parameters
        ----------
        spike : NSpike
            NSPike object. If None, new object is created

        Returns
        -------
        `:obj:NSpike`
            A new NSpike() object

        """
        return self.spike.add_spike(spike=spike, **kwargs)

    def load_spike(self, names=None):
        """
        Load datasets of the spike nodes.

        The name of each node is used for obtaining the filenames.

        Parameters
        ----------
        names : list of str
            Names of the nodes to load.
            If None, current NSpike() object is loaded

        Returns
        -------
        None

        """
        self.spike.load_spike(names=names)

    def wave_property(self):
        """
        Calculate different waveform properties for currently set unit.

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
        NSpike().wave_property()

        """
        return self.spike.wave_property()

    def isi(self, bins='auto', bound=None, density=False):
        """
        Calculate the ISI histogram of the spike train.

        Delegates to NSpike().isi()

        Parameters
        ----------
        bins : str or int
            Number of ISI histogram bins. If 'auto', NumPy default is used
        bound : int
            Length of the ISI histogram in msec
        density : bool
            If true, normalized histogram is calculated

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        NSpike().isi()

        """
        return self.spike.isi(bins=bins, bound=bound, density=density)

    def isi_corr(self, **kwargs):
        """
        Analysis of ISI autocorrelation histogram.

        Delegates to NSpike().isi_auto_corr()

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
        nc_spike.NSpike().isi_corr

        """
        return self.spike.isi_corr(**kwargs)

    def psth(self, event_stamp, **kwargs):
        """
        Calculate peri-stimulus time histogram (PSTH).

        Delegates to NSpike().psth()

        Parameters
        ----------
        event_stamp : ndarray
            Event timestamps

        **kwargs
            Keyword arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        See also
        --------
        nc_spike.NSpike().psth()

        """
        return self.spike.psth(event_stamp, **kwargs)

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
        self.spike.burst(burst_thresh=burst_thresh, ibi_thresh=ibi_thresh)

    def get_total_spikes(self):
        """
        Return total number of spikes in the recording.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of spikes

        """
        return self.spike.get_total_spikes()

    def get_total_channels(self):
        """
        Return total number of electrode channels in the spike data file.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of electrode channels

        """
        return self.spike.get_total_channels()

    def get_channel_ids(self):
        """
        Return the identities of individual channels.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Identities of individual channels

        """
        return self.spike.get_channel_ids()

    def get_timebase(self):
        """
        Return the timebase for spike event timestamps.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Timebase for spike event timestamps

        """
        return self.spike.get_timebase()

    def get_sampling_rate(self):
        """
        Return the sampling rate of spike waveforms.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Sampling rate for spike waveforms

        """
        return self.spike.get_sampling_rate()

    def get_samples_per_spike(self):
        """
        Return the number of bytes to represent each timestamp.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of bytes to represent timestamps

        """
        return self.spike.get_samples_per_spike()

    def get_wave_timestamp(self):
        """
        Return the temporal resolution to represent samples of spike-waves.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of bytes to represent timestamps

        """
        # return as microsecond
        # fs downsampled so that the time is given in microsecond
        fs = self.spike.get_sampling_rate() / 10**6
        return 1 / fs

    def save_to_hdf5(self):
        """
        Store NSpike() object to HDF5 file.

        Delegates to NSPike().save_to_hdf5()

        Parameters
        ----------
        None

        Returns
        -------
        None

        Also see
        --------
        nc_hdf.Nhdf().save_spike()

        """
        self.spike.save_to_hdf5()

    def get_feat(self, npc=2):
        """
        Return the spike-waveform features.

        Parameters
        ----------
        npc : int
            Number of principle components in each channel.

        Returns
        -------
        feat : ndarray
            Matrix of size (number_spike X number_features)

        """
        if not self.NULL_CHAN_REMOVED:
            self.remove_null_chan()
        if not self.ALLIGNED:
            self.align_wave_peak()

        trough, trough_loc = self.get_min_wave_chan()
        peak, peak_chan, peak_loc = self.get_max_wave_chan()
        pc = self.get_wave_pc(npc=npc)
        shape = (self.get_total_spikes(), 1)
        feat = np.concatenate(
            (peak.reshape(shape), trough.reshape(shape), pc), axis=1)

        return feat

    def get_feat_by_unit(self, unit_no=None):
        """
        Return the spike-waveform features for a particular unit.

        Parameters
        ----------
        unit_no : int
            Unit of interest

        Returns
        -------
        feat : ndarray
            Matrix of size (number_spike X number_features)

        """
        if unit_no in self.get_unit_list():
            feat = self.get_feat()
            return feat[self.get_unit_tags() == unit_no, :]
        else:
            logging.error('Specified unit does not exist in the spike dataset')

    def get_wave_peaks(self):
        """
        Return the peaks of the spike-waveforms.

        Parameters
        ----------
        None

        Returns
        -------
        (peak, peak_loc) : (ndarray, ndarray)
            peak:
                Spike waveform peaks in all the electrode channels
                Shape is (num_waves X num_channels)
            peak_loc :
                Index of peak locations

        """
        wave = self.get_waveform()
        peak = np.zeros((self.get_total_spikes(), len(wave.keys())))
        peak_loc = np.zeros(
            (self.get_total_spikes(), len(wave.keys())), dtype=int)
        for i, key in enumerate(wave.keys()):
            peak[:, i] = np.amax(wave[key], axis=1)
            peak_loc[:, i] = np.argmax(wave[key], axis=1)

        return peak, peak_loc

    def get_max_wave_chan(self):
        """
        Return the maximum of waveform peaks among the electrode groups.

        Parameters
        ----------
        None

        Returns
        -------
        (max_wave_val, max_wave_chan, peak_loc) : (ndarray, ndarray, ndarray)
        max_wave_val : ndarray
            Maximum value of the peaks of the waveforms
        max_wave_chan : ndarray
            Channel of the electrode group where a spike waveform is strongest
        peak_loc : ndarray
            Peak location in the channel with strongest waveform

        """
        peak, peak_loc = self.get_wave_peaks()
        max_wave_chan = np.argmax(peak, axis=1)
        max_wave_val = np.amax(peak, axis=1)
        return (
            max_wave_val, max_wave_chan,
            peak_loc[np.arange(len(peak_loc)), max_wave_chan])

    def align_wave_peak(self, reach=300, factor=2):
        """
        Align the waves by their peaks.

        Parameters
        ----------
        reach : int
            Maximum allowed time-shift in microsecond
        factors : int
            Resampling factor

        Returns
        -------
        None

        """
        if not self.UPSAMPLED:
            self.resample_wave(factor=factor)
        if not self.ALLIGNED:
            # maximum 300microsecond allowed for shift
            shift = round(reach / self.get_wave_timestamp())
            # NC waves are stored in waves['ch1'], waves['ch2'] etc. ways
            wave = self.get_waveform()
            maxInd = shift + self.get_max_wave_chan()[2]
            shift_ind = int(np.median(maxInd)) - maxInd
            shift_ind[np.abs(shift_ind) > shift] = 0
            stacked_chan = np.empty((
                self.get_total_spikes(),
                self.get_samples_per_spike(),
                self.get_total_channels()))
            keys = []
            i = 0
            for key, val in wave.items():
                stacked_chan[:, :, i] = val
                keys.append(key)
                i += 1

            stacked_chan = np.lib.pad(
                stacked_chan, [(0, 0), (shift, shift), (0, 0)], 'edge')

            stacked_chan = np.array([
                np.roll(stacked_chan[i, :, :], shift_ind[i], axis=0)[
                    shift: shift + self.get_samples_per_spike()]
                for i in np.arange(shift_ind.size)])

            for i, key in enumerate(keys):
                wave[key] = stacked_chan[:, :, i]
            self._set_waveform(wave)
            self.ALLIGNED = True

    def get_wave_min(self):
        """
        Return the minimum values of the spike-waveforms.

        Parameters
        ----------
        None

        Returns
        -------
        (min_w, min_loc) : (ndarray, ndarray)
            min_w : ndarray
                Minimum value of the waveforms
            min_loc : ndarray
                Index of minimum value

        """
        wave = self.get_waveform()
        min_w = np.zeros((self.get_total_spikes(), len(wave.keys())))
        min_loc = np.zeros((self.get_total_spikes(), len(wave.keys())))
        for i, key in enumerate(wave.keys()):
            min_w[:, i] = np.amin(wave[key], axis=1)
            min_loc[:, i] = np.argmin(wave[key], axis=1)

        return min_w, min_loc

    def get_min_wave_chan(self):
        """
        Return the maximum of waveform peaks among the electrode groups.

        Parameters
        ----------
        None

        Returns
        -------
        (min_val, min_index) : (ndarray, ndarray)
            min_val : ndarray
                Minimum value of the waveform at channels with peak value
            min_index : ndarray
                Index of minimum values

        """
        max_wave_chan = self.get_max_wave_chan()[1]
        trough, trough_loc = self.get_wave_min()
        return (
            trough[np.arange(len(max_wave_chan)), max_wave_chan],
            trough_loc[np.arange(len(max_wave_chan)), max_wave_chan])

    def get_wave_pc(self, npc=2):
        """
        Return the Principle Components of the waveforms.

        Parameters
        ----------
        npc : int
            Number of principle components from waveforms of each channel

        Returns
        -------
        pc : ndarray
            Principle components (num_waves X npc*num_channels)

        """
        wave = self.get_waveform()
        pc = np.array([])
        for key, w in wave.items():
            pca = PCA(n_components=5)
            w_new = pca.fit_transform(w)
            pc_var = pca.explained_variance_ratio_

            if npc and npc < w_new.shape[1]:
                w_new = w_new[:, :npc]
            else:
                w_new = w_new[:, 0:(
                    find(np.cumsum(pc_var) >= 0.95, 1, 'first')[0] + 1)]
            if not len(pc):
                pc = w_new
            else:
                pc = np.append(pc, w_new, axis=1)
        return pc

    def get_wavetime(self):
        """
        Return the timestamps of the waveforms, not the spiking-event.

        Parameters
        ----------
        None

        Returns
        -------
            Timestamps of the spike-waveforms

        """
        # calculate the wavetime from the sampling rate and number of sample
        # returns in microsecond
        nsamp = self.spike.get_samples_per_spike()
        timestamp = self.get_wave_timestamp()
        return np.arange(0, (nsamp) * timestamp, timestamp)

    def resample_wavetime(self, factor=2):
        """
        Resample the timestamps of spike-waveforms.

        Parameters
        ----------
        factor : int
            Resampling factor

        Returns
        -------
            Resampled timestamps

        """
        wavetime = self.get_wavetime()
        timestamp = self.get_wave_timestamp()

        return np.arange(0, wavetime[-1], timestamp / factor)

    def resample_wave(self, factor=2):
        """
        Resample spike waveforms using spline interpolation.

        Parameters
        ----------
        factor : int
            Resampling factor

        Returns
        -------
        wave : dict
            Upsampled waveforms
        uptime  ndarray
            Upsampled wave timestamps

        """
        # resample wave using spline interpolation using the resampled_time
        if not self.UPSAMPLED:
            wavetime = self.get_wavetime()
            uptime = self.resample_wavetime(factor=factor)
            wave = self.get_waveform()
            for key, w in wave.items():
                f = sc.interpolate.interp1d(
                    wavetime, w, axis=1, kind='quadratic')
                wave[key] = f(uptime)

            self.spike._set_sampling_rate(self.get_sampling_rate() * factor)
            self.spike._set_samples_per_spike(uptime.size)
            self.UPSAMPLED = True

            return wave, uptime

        else:
            logging.warning(
                'You can upsample only once. ' +
                'Please reload data from source file ' +
                'for changing sampling factor!')

    def get_wave_energy(self):
        """
        Energy of the spike waveforms.

        This is measured as the summation of the square of samples.

        Parameters
        ----------
        None

        Returns
        -------
        energy : ndarray
            Energy of spikes (num_spike X num_channels)

        """
        wave = self.get_waveform()
        energy = np.zeros((self.get_total_spikes(), len(wave.keys())))
        for i, key in enumerate(wave.keys()):
            # taken the energy in mV2
            energy[:, i] = (np.sum(np.square(wave[key]), 1) / 10**6)
        return energy

    def get_max_energy_chan(self):
        """
        Return the maximum energy of the spike waveforms.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Maximum energy of the spikes

        """
        energy = self.get_wave_energy()
        return np.argmax(energy, axis=1)

    def remove_null_chan(self):
        """
        Remove the channel from the electrode group that has no spike in it.

        Parameters
        ----------
        None

        Returns
        -------
        off_chan : int
            Channel number that has been removed

        """
        # simply detect in which channel everything is zero,
        # which means it's a reference channel or nothing is recorded here
        wave = self.get_waveform()
        off_chan = []
        for key, w in wave.items():
            if np.abs(w).sum() == 0:
                off_chan.append(key)
        if off_chan:
            for key in off_chan:
                del wave[key]
            self._set_waveform(wave)
            self.NULL_CHAN_REMOVED = True

        return off_chan

    def cluster_separation(self, unit_no=0):
        """
        Measure the separation of a specific unit from other clusters.

        This is performed quantitatively using the following:
        1. Bhattacharyya coefficient
        2. Hellinger distance

        Parameters
        ----------
        unit_no : int
            Unit of interest.
            If '0', pairwise comparison of all units are returned.

        Returns
        -------
        (bc, dh) : (ndarray, ndarray)
        bc : ndarray
            Bhattacharyya coefficient
        dh : ndarray
            Hellinger distance

        """
        # if unit_no==0 all units, matrix output for pairwise comparison,
        # else maximum BC for the specified unit
        feat = self.get_feat()
        unit_list = self.get_unit_list()
        n_units = len(unit_list)

        if unit_no == 0:
            bc = np.zeros([n_units, n_units])
            dh = np.zeros([n_units, n_units])
            for c1 in np.arange(n_units):
                for c2 in np.arange(n_units):
                    X1 = feat[self.get_unit_tags() == unit_list[c1], :]
                    X2 = feat[self.get_unit_tags() == unit_list[c2], :]
                    bc[c1, c2] = bhatt(X1, X2)[0]
                    dh[c1, c2] = hellinger(X1, X2)
                    unit_list = self.get_unit_list()
            return bc, dh

        else:
            bc = np.zeros(n_units)
            dh = np.zeros(n_units)
            X1 = feat[self.get_unit_tags() == unit_no, :]
            for c2 in np.arange(n_units):
                if c2 == unit_no:
                    bc[c2] = 0
                    dh[c2] = 1
                else:
                    X2 = feat[self.get_unit_tags() == unit_list[c2], :]
                    bc[c2] = bhatt(X1, X2)[0]
                    dh[c2] = hellinger(X1, X2)
                idx = find(np.array(unit_list) != unit_no)

            return bc[idx], dh[idx]

    def cluster_similarity(self, nclust=None, unit_1=None, unit_2=None):
        """
        Measure the similarity or distance of units in a cluster.

        This is performed on one unit
        in a spike dataset to cluster of another unit in another dataset.

        This is performed quantitatively using the following:
        1. Bhattacharyya coefficient
        2. Hellinger distance

        Parameters
        ----------
        nclust : Nclust
            NClust object whose unit is under comparison
        unit_1 : int
            Unit of current Nclust object
        unit_2 : int
            Unit of another NClust object under comparison

        Returns
        -------
        (bc, dh) : (ndarray, ndarray)
        bc : ndarray
            Bhattacharyya coefficient
        dh : ndarray
            Hellinger distance

        """
        if isinstance(nclust, NClust):
            if ((unit_1 in self.get_unit_list()) and
                    (unit_2 in nclust.get_unit_list())):
                X1 = self.get_feat_by_unit(unit_no=unit_1)
                X2 = nclust.get_feat_by_unit(unit_no=unit_2)
                bc = bhatt(X1, X2)[0]
                dh = hellinger(X1, X2)
        return bc, dh
