# -*- coding: utf-8 -*-
"""
This module implements NLfp Class for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie
"""
import os

import re
import inspect
from functools import reduce

import logging
from collections import OrderedDict as oDict
from copy import deepcopy

from math import floor, ceil
from neurochat.nc_utils import window_rms
from neurochat.nc_utils import butter_filter
from neurochat.nc_utils import find_peaks

from neurochat.nc_utils import butter_filter, fft_psd, find

from neurochat.nc_circular import CircStat
from neurochat.nc_hdf import Nhdf
from neurochat.nc_base import NBase

import numpy as np


import scipy.stats as stats
import scipy.signal as sg
from scipy.fftpack import fft

class NLfp(NBase):
    """
    This data class is the placeholder for the dataset that contains information
    about the neural LFP signal. It decodes data from different formats and analyses
    LFP signal in the recording.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._file_tag = ''
        self._channel_id = 0
        self._samples = None
        self._timestamp = None
        self.set_record_info({'Total samples': 0})

        self.__type = 'lfp'

    def get_type(self):
        """
        Returns the type of object. For NLfp, this is always `lfp` type

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self.__type

    # For multi-unit analysis, {'SpikeName': cell_no} pairs should be used as function input

    def set_channel_id(self, channel_id=''):
        """
        Sets the electrode channels ID

        Parameters
        ----------
        channel_id : str
            Channel ID for the LFP data

        Returns
        -------
        None

        """
        self._channel_id = channel_id

    def get_channel_id(self):
        """
        Returns the electrode channels ID

        Parameters
        ----------
        None

        Returns
        -------
        str
            LFP channel ID

        """

        return self._channel_id

    def set_file_tag(self, file_tag):
        """
        Sets the file tag or extension for the LFP dataset. For example, Axona recordings usually
        have file tags like 'eeg' or 'eeg8' etc.

        Parameters
        ----------
        file_tag : str
            File tag or extension for the LFP dataset

        Returns
        -------
        None

        """

        self._file_tag = file_tag

    def get_file_tag(self):
        """
        Returns the file tag or extension for the LFP dataset. For example, Axona recordings usually
        have file tags like 'eeg' or 'eeg8' etc.

        Parameters
        ----------
        None

        Returns
        -------
        str
            File tag or extension for the LFP dataset
        """

        return self._file_tag


    def get_timestamp(self):
        """
        Returns the timestamps of the LFP waveform

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Timestamps of the LFP signal

        """

        return self._timestamp

    def _set_timestamp(self, timestamp=None):
        """
        Sets the timestamps for LFP samples

        Parameters
        ----------
        timestamp : list or ndarray
            Timestamps of LFP samples

        Returns
        -------
        None

        """

        if timestamp is not None:
            self._timestamp = timestamp

    def get_samples(self):
        """
        Returns LFP waveform samples

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Samples of the LFP signal

        """

        return self._samples

    def _set_samples(self, samples=[]):
        """
        Sets LFP samples

        Parameters
        ----------
        samples : list or ndarray
            LFP samples

        Returns
        -------
        None

        """

        self._samples = samples

    def _set_total_samples(self, tot_samples=0):
        """
        Sets the number of LFP samples as part of storing the recording information

        Parameters
        ----------
        tot_samples : int
            Total number of samples in the LFP signal

        Returns
        -------
        None

        """

        self._record_info['No of samples'] = tot_samples

    def _set_total_channel(self, tot_channels):
        """
        Sets the value of number of channels as part of storing the recording information

        Parameters
        ----------
        tot_channels : int
            Total number of channels

        Returns
        -------
        None

        """

        self._record_info['No of channels'] = tot_channels

    def _set_timestamp_bytes(self, bytes_per_timestamp):
        """
        Sets `bytes per timestamp` value as part of storing the recording information

        Parameters
        ----------
        bytes_per_timestamp : int
            Total number of bytes to represent timestamp in the binary file

        Returns
        -------
        None

        """

        self._record_info['Bytes per timestamp'] = bytes_per_timestamp

    def _set_sampling_rate(self, sampling_rate):
        """
        Sets the sampling rate of the LFP signal as part of storing the recording information

        Parameters
        ----------
        sampling_rate : int
            Sampling rate of the LFP waveform

        Returns
        -------
        None

        """

        self._record_info['Sampling rate'] = sampling_rate

    def _set_bytes_per_sample(self, bytes_per_sample):
        """
        Sets `bytes per sample` value as part of storing the recording information

        Parameters
        ----------
        bytes_per_sample : int
            Total number of bytes to represent each sample in the binary file

        Returns
        -------
        None

        """

        self._record_info['Bytes per sample'] = bytes_per_sample

    def _set_fullscale_mv(self, adc_fullscale_mv):
        """
        Sets fullscale value of ADC value in mV as part of storing the recording information

        Parameters
        ----------
        adc_fullscale_mv : int
            Fullscale voltage of ADC signal in mV

        Returns
        -------
        None

        """

        self._record_info['ADC Fullscale mv'] = adc_fullscale_mv

    def get_total_samples(self):
        """
        Returns total number of LFP samples

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Total number of LFP samples

        """
        return self._record_info['No of samples']

    def get_total_channel(self):
        """
        Returns total number of electrode channels in the LFP data file

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of electrode channels
        """

        return self._record_info['No of channels']

    def get_timestamp_bytes(self):
        """
        Returns the number of bytes to represent each timestamp in the binary file

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of bytes to represent timestamps

        """

        return self._record_info['Bytes per timestamp']

    def get_sampling_rate(self):
        """
        Returns the sampling rate of spike waveforms

        Parameters
        ----------
        None

        Returns
        -------
        int
            Sampling rate for spike waveforms

        """

        return self._record_info['Sampling rate']

    def get_bytes_per_sample(self):
        """
        Returns the number of bytes to represent each LFP waveform sample

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of bytes to represent each sample of the LFP waveform

        """

        return self._record_info['Bytes per sample']

    def get_fullscale_mv(self):
        """
        Returns the fullscale value of the ADC in mV

        Parameters
        ----------
        None

        Returns
        -------
        int
            Fullscale ADC value in mV

        """

        return self._record_info['ADC Fullscale mv']

    def get_recording_time(self):
        """
        Returns the recording time in seconds

        Parameters
        ----------
        None

        Returns
        -------
        int
            Recording time in seconds
        """

        return self.get_total_samples() / (self.get_sampling_rate())

    def load(self, filename=None, system=None):
        """
        Loads LFP datasets

        Parameters
        ----------
        filename : str
            Name of the spike datafile
        system : str
            Recording system or format of the spike data file

        Returns
        -------
        None

        See also
        --------
        load_lfp_axona(), load_lfp_NLX(), load_lfp_NWB()

        """
        if system is None:
            system = self._system
        else:
            self._system = system
        if filename is None:
            filename = self._filename
        else:
            self._filename = filename
        loader = getattr(self, 'load_lfp_' + system)
        loader(filename)

    def add_spike(self, spike=None, **kwargs):
        """
        Adds new spike node to current NLfp() object

        Parameters
        ----------
        spike : NSpikes
            NSPike object. If None, new object is created

        Returns
        -------
        `:obj:NSpike()`
            A new NSpike() object

        """

        cls= kwargs.get('cls', None)
        if not inspect.isclass(cls):
            try:
                data_type = spike.get_type()
                if data_type == 'spike':
                    cls = spike.__class__
            except:
                 logging.error('Data type cannot be determined!')
        if inspect.isclass(cls):
             new_spike = self._add_node(cls, spike, 'spike', **kwargs)
             return new_spike
        else:
            logging.error('Cannot add the spike data!')


    def load_spike(self, names='all'):
        """
        Loads datasets of the spike nodes. Name of each node is used for obtaining the
        filenames

        Parameters
        ----------
        names : list of str
            Names of the nodes to load. If None, current NSpike() object is loaded

        Returns
        -------
        None

        """


        if names == 'all':
            for spike in self._spikes:
                spike.load()
        else:
            logging.error("Spikes by name has yet to be implemented")
            # for name in names:
            #     spike = self.get_spikes_by_name(name)
            #     spike.load()

    def add_lfp(self, lfp=None, **kwargs):
        """
        Adds new LFP node to current NLfp() object

        Parameters
        ----------
        lfp : NLfp
            NLfp object. If None, new object is created

        Returns
        -------
        `:obj:Nlfp`
            A new NLfp() object

        """

        new_lfp = self._add_node(self.__class__, lfp, 'lfp', **kwargs)

        return new_lfp

    def load_lfp(self, names=None):
        """
        Loads datasets of the LFP nodes. Name of each node is used for obtaining the
        filenames

        Parameters
        ----------
        names : list of str
            Names of the nodes to load. If `all`, all LFP nodes are loaded

        Returns
        -------
        None
        """

        if names is None:
            self.load()
        elif names == 'all':
            for lfp in self._lfp:
                lfp.load()
        else:
            logging.error("Lfp by name has yet to be implemented")
            # for name in names:
            #     lfp = self.get_lfp_by_name(name)
            #     lfp.load()

    def spectrum(self, **kwargs):
        """
        Analyses frequency spectrum of the LFP signal

        Parameters
        ----------
        **kwargs
            Keywrod arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        """

        graph_data = oDict()

        Fs = self.get_sampling_rate()
        slc = kwargs.get('slice', None)
        if slc:
            lfp = self.get_samples()[slc]
        else:
            lfp = self.get_samples()

        window = kwargs.get('window', 1.0)
        window = sg.get_window('hann', int(window*Fs)) if isinstance(window, float)\
                or isinstance(window, int) else window

        win_sec = np.ceil(window.size/Fs)

        noverlap = kwargs.get('noverlap', 0.5*win_sec)
        noverlap = noverlap if noverlap < win_sec else 0.5*win_sec
        noverlap = np.ceil(noverlap*Fs)

        nfft = kwargs.get('nfft', 2*Fs)
        nfft = np.power(2, int(np.ceil(np.log2(nfft))))

        ptype = kwargs.get('ptype', 'psd')
        ptype = 'spectrum' if ptype == 'power' else 'density'

        prefilt = kwargs.get('prefilt', True)
        _filter = kwargs.get('filtset', [10, 1.5, 40, 'bandpass'])

        fmax = kwargs.get('fmax', Fs/2)

        if prefilt:
            lfp = butter_filter(lfp, Fs, *_filter)

        tr = kwargs.get('tr', False)
        db = kwargs.get('db', False)
        if tr:
            f, t, Sxx = sg.spectrogram(lfp, fs=Fs, \
                    window=window, nperseg=window.size, noverlap=noverlap, nfft=nfft, \
                    detrend='constant', return_onesided=True, scaling=ptype)

            graph_data['t'] = t
            graph_data['f'] = f[find(f <= fmax)]

            if db:
                Sxx = 10*np.log10(Sxx/np.amax(Sxx))
                Sxx = Sxx.flatten()
                Sxx[find(Sxx < -40)] = -40
                Sxx = np.reshape(Sxx, [f.size, t.size])

#            graph_data['Sxx'] = np.empty([find(f<= fmax).size, t.size])
#            graph_data['Sxx'] = np.array([Sxx[i, :] for i in find(f<= fmax)])
            graph_data['Sxx'] = Sxx[find(f <= fmax), :]
        else:
            f, Pxx = sg.welch(lfp, fs=Fs, \
                    window=window, nperseg=window.size, noverlap=noverlap, nfft=nfft, \
                    detrend='constant', return_onesided=True, scaling=ptype)

            graph_data['f'] = f[find(f <= fmax)]

            if db:
                Pxx = 10*np.log10(Pxx/Pxx.max())
                Pxx[find(Pxx < -40)] = -40
            graph_data['Pxx'] = Pxx[find(f <= fmax)]

        return graph_data

    def phase_dist(self, event_stamp, **kwargs):
        """
        Analysis of spike to LFP phase distribution

        Parameters
        ----------
        evnet_stamp : ndarray
            Timestamps of the events of spiking activities for measring the phase
            distribution
        **kwargs
            Keywrod arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        """

        _results= oDict()
        graph_data = oDict()

        cs = CircStat()

        lfp = self.get_samples()*1000
        Fs = self.get_sampling_rate()
        time = self.get_timestamp()

        # Input parameters
        bins = int(360/kwargs.get('binsize', 5))
        rbinsize = kwargs.get('rbinsize', 2) # raster binsize
        rbins = int(360/rbinsize)
        fwin = kwargs.get('fwin', [6, 12])
        pratio = kwargs.get('pratio', 0.2)
        aratio = kwargs.get('aratio', 0.15)

    # Filter
        fmax = fwin[1]
        fmin = fwin[0]
        _filter = [5, fmin, fmax, 'bandpass']
        _prefilt = kwargs.get('filtset', [10, 1.5, 40, 'bandpass'])

        b_lfp = butter_filter(lfp, Fs, *_filter) # band LFP
        lfp = butter_filter(lfp, Fs, *_prefilt)

    # Measure phase
        hilb = sg.hilbert(b_lfp)
#        self.hilb = hilb
#        phase = np.remainder(np.angle(hilb, deg=True)+ 360, 360)
        phase = np.angle(hilb, deg=True)
        phase[phase < 0] = phase[phase < 0] + 360
        mag = np.abs(hilb)

        ephase = np.interp(event_stamp, time, phase)

        p2p = np.abs(np.max(lfp) - np.min(lfp))
        xline = 0.5* np.mean(mag) # cross line

        # Detection algo
        # zero cross
        mag1 = mag[0:-3]
        mag2 = mag[1:-2]
        mag3 = mag[2:-1]

        xind = np.union1d(find(np.logical_and(mag1 < xline, mag2 > xline)), \
                find(np.logical_and(np.logical_and(mag1 < xline, mag2 == xline), mag3 > xline)))

        # Ignore segments <1/fmax
        i = 0
        rcount = np.empty([0,])
        bcount = np.empty([0, 0])

        phBins = np.arange(0, 360, 360/bins)
        rbins = np.arange(0, 360, 360/rbins)

        seg_count = 0
        while i < len(xind)-1:
            k = i+1
            while time[xind[k]]- time[xind[i]] < 1/fmin and k < len(xind)-1:
                k += 1
#            print(time[xind[i]], time[xind[k]])
            s_lfp = lfp[xind[i]: xind[k]]
            s_p2p = np.abs(np.max(s_lfp)- np.min(s_lfp))

            if s_p2p >= aratio*p2p:
                s_psd, f = fft_psd(s_lfp, Fs)
                if np.sum(s_psd[np.logical_and(f >= fmin, f <= fmax)]) > pratio* np.sum(s_psd):
                    # Phase distribution
                    s_phase = ephase[np.logical_and(event_stamp > time[xind[i]], event_stamp <= time[xind[k]])]
#                    print(s_phase.shape, s_phase.shape)

                    if not s_phase.shape[0]:
                        pass
                    else:
                        seg_count += 1
                        cs.set_theta(s_phase)
                        temp_count = cs.circ_histogram(bins=rbinsize)
#                        temp_count = np.histogram(s_phase, bins=rbins, range=[0, 360])
                        temp_count = temp_count[0]
                        if not rcount.size:
                            rcount = temp_count
                        else:
                            rcount = np.append(rcount, temp_count)

                        temp_count = np.histogram(s_phase, bins=bins, range=[0, 360])
                        temp_count = np.resize(temp_count[0], [1, bins])
                        if not len(bcount):
                            bcount = temp_count
                        else:
                            bcount = np.append(bcount, temp_count, axis=0)
            i = k

        rcount = rcount.reshape([seg_count, rbins.size])

        phCount = np.sum(bcount, axis=0)

        cs.set_rho(phCount)
        cs.set_theta(phBins)

        cs.calc_stat()
        result = cs.get_result()
        meanTheta = result['meanTheta']*np.pi/180

        _results['LFP Spike Mean Phase']= result['meanTheta']
        _results['LFP Spike Mean Phase Count']= result['meanRho']
        _results['LFP Spike Phase Res Vect']= result['resultant']

        graph_data['meanTheta'] = meanTheta
        graph_data['phCount'] = phCount
        graph_data['phBins'] = phBins
        graph_data['raster'] = rcount
        graph_data['rasterbins'] = rbins

        self.update_result(_results)

        return graph_data

    def phase_at_events(self, event_stamps, **kwargs):
        """
        Phase based on times.

        Parameters
        ----------
        event_stamps : array
            an array of event times
        **kwargs:
            keyword arguments

        Returns
        -------
            (array)
            Phase values for each position
        """
        lfp = self.get_samples() * 1000
        Fs = self.get_sampling_rate()
        time = self.get_timestamp()

        # Input parameters
        fwin = kwargs.get('fwin', [6, 12])

        # Filter
        fmax = fwin[1]
        fmin = fwin[0]
        _filter = [5, fmin, fmax, 'bandpass']
        _prefilt = kwargs.get('filtset', [10, 1.5, 40, 'bandpass'])

        b_lfp = butter_filter(lfp, Fs, *_filter)  # band LFP
        lfp = butter_filter(lfp, Fs, *_prefilt)

        # Measure phase
        hilb = sg.hilbert(b_lfp)
        phase = np.angle(hilb, deg=True)
        phase[phase < 0] = phase[phase < 0] + 360

        ephase = np.interp(event_stamps, time, phase)

        return ephase

    def plv(self, event_stamp, **kwargs):
        """
        Calculates phase-locking value of the spike train to underlying LFP signal.

        When 'mode'= None in the inpput kwargs, it calculates the PLV and SFC over
        the entire spike-train.

        If 'mode'= 'bs', it bootstraps the spike-timestamps
        and calculates the locking values for each set of new spike timestamps.

        If 'mode'= 'tr', a time-resilved phase-locking analysis is performed where
        the LFP signal is split into overlapped segments for each calculation.

        Parameters
        ----------
        evnet_stamp : ndarray
            Timestamps of the events or the spiking activities for measuring the phase
            locking
        **kwargs
            Keywrod arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        """
        graph_data = oDict()

        lfp = self.get_samples()*1000
        Fs = self.get_sampling_rate()
        time = self.get_timestamp()

        window = np.array(kwargs.get('window', [-0.5, 0.5]))
        win = np.ceil(window*Fs).astype(int)
        win = np.arange(win[0], win[1])
        slep_win = sg.hann(win.size, False)

        nfft = kwargs.get('nfft', 1024)
        mode = kwargs.get('mode', None) # None, 'bs', 'tr' bs=bootstrp, tr=time-resolved
        fwin = kwargs.get('fwin', [])

        xf = np.arange(0, Fs, Fs/nfft)
        f = xf[0: int(nfft/2)+ 1]

        ind = np.arange(f.size) if len(fwin) == 0 else find(np.logical_and(f >= fwin[0], f <= fwin[1]))

        if mode == 'bs':
            nsample = kwargs.get('nsample', 50)
            nrep = kwargs.get('nrep', 500)

            STA = np.empty([nrep, win.size])
            fSTA = np.empty([nrep, ind.size])
            STP = np.empty([nrep, ind.size])
            SFC = np.empty([nrep, ind.size])
            PLV = np.empty([nrep, ind.size])

            for i in np.arange(nrep):
                data = self.plv(np.random.choice(event_stamp, nsample, False), \
                        window=window, nfft=nfft, mode=None, fwin=fwin)
                t = data['t']
                STA[i, :] = data['STA']
                fSTA[i, :] = data['fSTA']
                STP[i, :] = data['STP']
                SFC[i, :] = data['SFC']
                PLV[i, :] = data['PLV']

            graph_data['t'] = t
            graph_data['f'] = f[ind]
            graph_data['STAm'] = STA.mean(0)
            graph_data['fSTAm'] = fSTA.mean(0)
            graph_data['STPm'] = STP.mean(0)
            graph_data['SFCm'] = SFC.mean(0)
            graph_data['PLVm'] = PLV.mean(0)

            graph_data['STAe'] = stats.sem(STA, 0)
            graph_data['fSTAe'] = stats.sem(fSTA, 0)
            graph_data['STPe'] = stats.sem(STP, 0)
            graph_data['SFCe'] = stats.sem(SFC, 0)
            graph_data['PLVe'] = stats.sem(PLV, 0)

        elif mode == 'tr':
            nsample = kwargs.get('nsample', None)

            slide = kwargs.get('slide', 25) # in ms
            slide = slide/1000 # convert to sec

            offset = np.arange(window[0], window[-1], slide)
            nwin = offset.size

            fSTA = np.empty([nwin, ind.size])
            STP = np.empty([nwin, ind.size])
            SFC = np.empty([nwin, ind.size])
            PLV = np.empty([nwin, ind.size])

            if nsample is None or nsample > event_stamp.size:
                stamp = event_stamp
            else:
                stamp = np.random.choice(event_stamp, nsample, False)

            for i in np.arange(nwin):
                data = self.plv(stamp + offset[i], \
                        nfft=nfft, mode=None, fwin=fwin, window=window)
                t = data['t']
                fSTA[i, :] = data['fSTA']
                STP[i, :] = data['STP']
                SFC[i, :] = data['SFC']
                PLV[i, :] = data['PLV']

            graph_data['offset'] = offset
            graph_data['f'] = f[ind]
            graph_data['fSTA'] = fSTA.transpose()
            graph_data['STP'] = STP.transpose()
            graph_data['SFC'] = SFC.transpose()
            graph_data['PLV'] = PLV.transpose()

        elif mode is None:
            center = time.searchsorted(event_stamp)
            # Keep windows within data
            center = np.array([center[i] for i in range(0, len(event_stamp)) \
                if center[i] + win[0] >= 0 and center[i] + win[-1] <= time.size])

            sta_data = self.event_trig_average(event_stamp, **kwargs)
            STA = sta_data['ETA']

            fSTA = fft(np.multiply(STA, slep_win), nfft)

            fSTA = np.absolute(fSTA[0: int(nfft/2)+ 1])**2/nfft**2
            fSTA[1:-1] = 2*fSTA[1:-1]

            fLFP = np.array([fft(np.multiply(lfp[x+ win], slep_win), nfft) \
                    for x in center])

            STP = np.absolute(fLFP[:, 0: int(nfft/2)+ 1])**2/nfft**2
            STP[:, 1:-1] = 2*STP[:, 1:-1]
            STP = STP.mean(0)

            SFC = np.divide(fSTA, STP)*100

            PLV = np.copy(fLFP)

            # Normalize
            PLV = np.divide(PLV, np.absolute(PLV))
            PLV[np.isnan(PLV)] = 0

            PLV = np.absolute(PLV.mean(0))[0: int(nfft/2)+ 1]
            PLV[1:-1] = 2*PLV[1:-1]

            graph_data['t'] = sta_data['t']
            graph_data['f'] = f[ind]
            graph_data['STA'] = STA
            graph_data['fSTA'] = fSTA[ind]
            graph_data['STP'] = STP[ind]
            graph_data['SFC'] = SFC[ind]
            graph_data['PLV'] = PLV[ind]

        return graph_data

    def event_trig_average(self, event_stamp=None, **kwargs):
        """
        Averaging event-triggered LFP signals

        Parameters
        ----------
        event_stamp : ndarray
            Timestamps of the events or the spiking activities for measuring the
            event triggered average of the LFP signal
        **kwargs
            Keywrod arguments

        Returns
        -------
        dict
            Graphical data of the analysis

        """

        graph_data = oDict()
        window = np.array(kwargs.get('window', [-0.5, 0.5]))

        if event_stamp is None:
            spike = kwargs.get('spike', None)

            try:
                data_type = spike.get_type()
            except:
                logging.error('The data type of the addes object cannot be determined!')

            if data_type == 'spike':
                event_stamp = spike.get_unit_stamp()
            elif spike in self.get_spike_names():
                event_stamp = self.get_spike(spike).get_unit_stamp()

        if event_stamp is None:
            logging.error('No valid event timestamp or spike is provided')
        else:
            lfp = self.get_samples()*1000
            Fs = self.get_sampling_rate()
            time = self.get_timestamp()
            center = time.searchsorted(event_stamp, side='left')
            win = np.ceil(window*Fs).astype(int)
            win = np.arange(win[0], win[1])

            # Keep windows within data
            center = np.array([center[i] for i in range(0, len(event_stamp)) \
                if center[i]+ win[0] >= 0 and center[i]+ win[-1] <= time.size])

            eta = reduce(lambda y, x: y+ lfp[x+ win], center)
            eta = eta/center.size

            graph_data['t'] = win/Fs
            graph_data['ETA'] = eta
            graph_data['center'] = center

        return graph_data

    def spike_lfp_causality(self, spike=None, **kwargs):
        """
        (Not implemented yet)

        Analyses spike to underlying LFP causality

        Parameters
        ----------
        spike : NSpike
            Spike dataset which is used for the causality analysis
        **kwargs
            Keywrod arguments

        Returns
        -------
        dict
            Should return graphical data of the analysis. The function is not
            implemented yet.

        """

        pass

    def subsample(self, sample_range=None):
        """
        Extract a time range from the lfp.

        Parameters
        ----------
        sample_range : tuple
            the time in seconds to extract from the lfp

        Returns
        -------
        NLfp
            subsampled version of initial lfp object
        """
        in_range = sample_range
        sample_rate = self.get_sampling_rate()
        if in_range is None:
            length = int(self.get_duration() * sample_rate)
            if (length != self.get_total_samples()):
                logging.warning(
                    "Unequal calculated and recorded total lfp samples" +
                    "Calculated {} and recorded {}".format(
                        length, self.get_total_samples()))
            return self
        else:
            new_lfp = deepcopy(self)
            lfp_samples = self.get_samples()[
                int(sample_rate * in_range[0]):int(sample_rate * in_range[1])]
            lfp_times = self.get_timestamp()[
                int(sample_rate * in_range[0]):int(sample_rate * in_range[1])]
            new_lfp._set_samples(lfp_samples)
            new_lfp._set_timestamp(lfp_times)
            new_lfp._set_total_samples(len(lfp_samples))
            new_lfp._set_duration(in_range[1] - in_range[0])
            return new_lfp

    def sharp_wave_ripples(self, in_range=None, **kwargs):
        """
        Detect SWR events in the lfp, optionally in a given range

        Parameters
        ----------
        in_range : tuple
            A range in seconds

        kwargs
        ------
        swr_lower : float
            Lower band in hz
        swr_upper : float
            Upper band in hz
        rms_window_size_ms : int
            Size of the rms window in ms
        percentile : float
            The percentile threshold for a peak

        Returns
        -------
        dict
            lfp times, lfp samples, swr times, lfp sample rate

        """
        swr_lower = kwargs.get("swr_lower", 100)
        swr_higher = kwargs.get("swr_upper", 250)
        rms_window_size_ms = kwargs.get("rms_window_size_ms", 7)
        percentile = kwargs.get("peak_percentile", 99.5)

        lfp = self.subsample(in_range)
        sample_rate = lfp.get_sampling_rate()
        # Estimate SWR events
        filtered_lfp = butter_filter(
            lfp.get_samples(), sample_rate, 10,
            swr_lower, swr_higher, 'bandpass')
        rms_window_size = floor((rms_window_size_ms / 1000) * sample_rate)
        rms_envelope = window_rms(filtered_lfp, rms_window_size, mode="same")
        p_val = np.percentile(rms_envelope, percentile)
        _, peaks = find_peaks(rms_envelope, thresh=p_val)
        peaks = lfp.get_timestamp()[0] + (peaks / sample_rate)

        """
        Alternative way to get SWR
        #rms_envelope = distinct_window_rms(filtered_lfp, rms_window_size)
        #peaks = (
        # longest_sleep_period[0] + peaks * rms_window_size) / sample_rate
        """

        return {
            "lfp times": lfp.get_timestamp(),
            "lfp samples": filtered_lfp,
            "swr times": peaks, "lfp sample rate": sample_rate}

    def bandpower(self, band, **kwargs):
        """Compute the average power of the signal x in a specific frequency band.

        Modified from excellent article at
        https://raphaelvallat.com/bandpower.html

        Parameters
        ----------
        band : list
        Lower and upper frequencies of the band of interest.

        kwargs:
            method : string
                Periodogram method: 'welch'
            window_sec : float
                Length of each window in seconds.
                If None, window_sec = (1 / min(band)) * 2.
            band_total : bool
                Whether to band the total power
                Default False
            total_band: List
                low and high frequency values for the filter
                Default [1.5, 40]

        Returns
        ------
        bp : Dict
            "bandpower", "total_power" and "relative_power".
        """
        from scipy.signal import welch
        from scipy.integrate import simps

        band = np.asarray(band)
        low, high = band
        method = kwargs.get("method", "welch")
        window_sec = kwargs.get("window_sec", 2 / (low + 0.000001))
        sf = self.get_sampling_rate()
        lfp_samples = self.get_samples()

        band_total = kwargs.get('band_total', False)
        _filter = kwargs.get('total_band', [1.5, 40])

        # if prefilt:
        #     lfp_samples = butter_filter(lfp_samples, sf, *_filter)
        # Compute the modified periodogram (Welch)
        if method == 'welch':
            nperseg = int(window_sec * sf)
            freqs, psd = welch(lfp_samples, sf, nperseg=nperseg)

        # The multaper method is more accurate but we will not use it
        # Welch's method is still very good
        # See MNE for the multitaper method
        # from mne.time_frequency import psd_array_multitaper
        # elif method == 'multitaper':
        #     psd, freqs = psd_array_multitaper(lfp_samples, sf, adaptive=True,
        #                                     normalization='full', verbose=0)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)

        if band_total:
            idx_band = np.logical_and(
                freqs >= _filter[0],
                freqs <= _filter[1])
            tp = simps(psd[idx_band], dx=freq_res)
        else:
            tp = simps(psd, dx=freq_res)

        output = {
            "bandpower": bp,
            "total_power": tp,
            "relative_power": bp/tp}
        return output

    def bandpower_ratio(self, first_band, second_band, win_sec, **kwargs):
        """
        Calculate the ratio in power between two bandpass filtered signals.

        Note that common ranges are:
        delta (1.5–4 Hz), theta (5-11 Hz)

        Parameters
        ----------
        first_band - 1d array
            lower and upper bands
        second_band - 1d array
            lower and upper bands
        win_sec - float
            length of the windows to bin lfp into in seconds.
            recommend 4 for eg.
        kwargs:
            first_name - str name of band 1, default "Band 1"
            second_name - str name of band 2, default "Band 2"

        Returns
        -------
        float - the ratio between the power signals.

        See also
        --------
        nc_lfp.NLfp().bandpower()
        """

        _results = oDict()
        name1 = kwargs.get("first_name", "Band 1")
        name2 = kwargs.get("second_name", "Band 2")
        if "window_sec" not in kwargs:
            kwargs["window_sec"] = win_sec

        b1 = self.bandpower(first_band, **kwargs)
        b2 = self.bandpower(second_band, **kwargs)
        if b1["total_power"] != b2["total_power"]:
            logging.error(
                "Differing total power in lfp bandpower ratio calculations")
        bp = b1["bandpower"] / b2["bandpower"]
        key1 = name1 + " Power"
        key2 = name2 + " Power"
        key3 = name1 + " " + name2 + " Power Ratio"
        _results[key1] = b1["bandpower"]
        _results[key1 + " (Relative)"] = b1["relative_power"]
        _results[key2] = b2["bandpower"]
        _results[key2 + " (Relative)"] = b2["relative_power"]
        _results[key3] = bp
        _results["Total Power"] = b1["total_power"]
        self.update_result(_results)
        return bp

    def save_to_hdf5(self, file_name=None, system=None):
        """
        Stores NLfp() object to HDF5 file

        Parameters
        ----------
        file_name : str
            Full file directory for the lfp data
        system : str
            Recoring system or data format

        Returns
        -------
        None

        Also see
        --------
        nc_hdf.Nhdf().save_lfp()

        """

        hdf = Nhdf()
        if file_name and system:
            if os.path.exists(file_name):
                self.set_filename(file_name)
                self.set_system(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')

        hdf.save_lfp(lfp=self)
        hdf.close()

    def load_lfp_NWB(self, file_name):
        """
        Decodes LFP data from NWB (HDF5) file format

        Parameters
        ----------
        file_name : str
            Full file directory for the lfp data

        Returns
        -------
        None

        """

        file_name, path = file_name.split('+')
        if os.path.exists(file_name):
            hdf = Nhdf()
            hdf.set_filename(file_name)

            _record_info = {}

            if path in hdf.f:
                g = hdf.f[path]
            elif '/processing/Neural Continuous/LFP/'+ path in hdf.f:
                path = '/processing/Neural Continuous/LFP/'+ path
                g = hdf.f[path]
            else:
                logging.error('Specified path does not exist!')

            for key, value in g.attrs.items():
                _record_info[key] = value

            self.set_record_info(_record_info)

            self._set_samples(hdf.get_dataset(group=g, name='data'))
            self._set_timestamp(hdf.get_dataset(group=g, name='timestamps'))
            self._set_total_samples(hdf.get_dataset(group=g, name='num_samples'))

            hdf.close()
        else:
            logging.error(file_name + ' does not exist!')

    def load_lfp_Axona(self, file_name):
        """
        Decodes LFP data from Axona file format

        Parameters
        ----------
        file_name : str
            Full file directory for the lfp data

        Returns
        -------
        None

        """

        words = file_name.split(sep=os.sep)
        file_directory = os.sep.join(words[0:-1])
        file_tag, file_extension = words[-1].split('.')
        set_file = file_directory + os.sep + file_tag + '.set'

        self._set_data_source(file_name)
        self._set_source_format('Axona')

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                while True:
                    line = f.readline()
                    try:
                        line = line.decode('latin-1')
                    except:
                        break

                    if line == '':
                        break
                    if line.startswith('trial_date'):
                        self._set_date(' '.join(line.replace(',', ' ').split()[1:]))
                    if line.startswith('trial_time'):
                        self._set_time(line.split()[1])
                    if line.startswith('experimenter'):
                        self._set_experiemnter(' '.join(line.split()[1:]))
                    if line.startswith('comments'):
                        self._set_comments(' '.join(line.split()[1:]))
                    if line.startswith('duration'):
                        self._set_duration(float(''.join(line.split()[1:])))
                    if line.startswith('sw_version'):
                        self._set_file_version(line.split()[1])
                    if line.startswith('num_chans'):
                        self._set_total_channel(int(''.join(line.split()[1:])))
                    if line.startswith('sample_rate'):
                        self._set_sampling_rate(float(''.join(re.findall(r'\d+.\d+|\d+', line))))
                    if line.startswith('bytes_per_sample'):
                        self._set_bytes_per_sample(int(''.join(line.split()[1:])))
                    if line.startswith('num_'+ file_extension[:3].upper() + '_samples'):
                        self._set_total_samples(int(''.join(line.split()[1:])))
                    if line.startswith("data_start"):
                        break

                num_samples = self.get_total_samples()
                bytes_per_sample = self.get_bytes_per_sample()

                f.seek(0, 0)
                header_offset = []
                while True:
                    try:
                        buff = f.read(10).decode('UTF-8')
                    except:
                        break
                    if buff == 'data_start':
                        header_offset = f.tell()
                        break
                    else:
                        f.seek(-9, 1)

                eeg_ID = re.findall(r'\d+', file_extension)
                self.set_file_tag(1 if not eeg_ID else int(eeg_ID[0]))
                max_ADC_count = 2**(8*bytes_per_sample-1)-1
                max_byte_value = 2**(8*bytes_per_sample)

                with open(set_file, 'r', encoding='latin-1') as f_set:
                    lines = f_set.readlines()
                    channel_lines = dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                                for line in lines if line.startswith('EEG_ch_')])
                    channel_id = channel_lines[self.get_file_tag()]
                    self.set_channel_id(channel_id)

                    gain_lines = dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                            for line in lines if 'gain_ch_' in line])
                    gain = gain_lines[channel_id-1]

                    for line in lines:
                        if line.startswith('ADC_fullscale_mv'):
                            self._set_fullscale_mv(int(re.findall(r'\d+.\d+|d+', line)[0]))
                            break
                    AD_bit_uvolt = 2*self.get_fullscale_mv()/ \
                                    (gain*np.power(2, 8*bytes_per_sample))

                record_size = bytes_per_sample
                sample_le = 256**(np.arange(0, bytes_per_sample, 1))

                if not header_offset:
                    print('Error: data_start marker not found!')
                else:
                    f.seek(header_offset, 0)
                    byte_buffer = np.fromfile(f, dtype='uint8')
                    len_bytebuffer = len(byte_buffer)
                    end_offset = len('\r\ndata_end\r')
                    lfp_wave = np.zeros([num_samples, ], dtype=np.float64)
                    for k in np.arange(0, bytes_per_sample, 1):
                        byte_offset = k
                        sample_value = (sample_le[k]* byte_buffer[byte_offset \
                                    :byte_offset+ len_bytebuffer- end_offset- record_size\
                                    :record_size])
                        if sample_value.size < num_samples:
                            sample_value = np.append(sample_value, np.zeros([num_samples-sample_value.size,]))
                        sample_value = sample_value.astype(np.float64, casting='unsafe', copy=False)
                        np.add(lfp_wave, sample_value, out=lfp_wave)
                    np.putmask(lfp_wave, lfp_wave > max_ADC_count, lfp_wave- max_byte_value)

                    self._set_samples(lfp_wave*AD_bit_uvolt)
                    self._set_timestamp(np.arange(0, num_samples, 1)/self.get_sampling_rate())

        else:
            logging.error(
                "No lfp file found for file {}".format(file_name))

    def load_lfp_Neuralynx(self, file_name):
        """
        Decodes LFP data from Neuralynx file format

        Parameters
        ----------
        file_name : str
            Full file directory for the lfp data

        Returns
        -------
        None

        """

        self._set_data_source(file_name)
        self._set_source_format('Neuralynx')

        # Format description for the NLX file:

        resamp_freq = 250 # NeuroChaT subsamples the original recording from 32000 to 250

        header_offset = 16*1024 # fixed for NLX files

        bytes_per_timestamp = 8
        bytes_chan_no = 4
        bytes_sample_freq = 4
        bytes_num_valid_samples = 4
        bytes_per_sample = 2
        samples_per_record = 512

        max_byte_value = np.power(2, bytes_per_sample*8)
        max_ADC_count = np.power(2, bytes_per_sample*8- 1)-1
        AD_bit_uvolt = 10**-6

        self._set_bytes_per_sample(bytes_per_sample)

        record_size = None
        with open(file_name, 'rb') as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('UTF-8')
                except:
                    break

                if line == '':
                    break
                if 'SamplingFrequency' in line:
                    self._set_sampling_rate(float(''.join(re.findall(r'\d+.\d+|\d+', line)))) # We are subsampling from the blocks of 512 samples per record
                if 'RecordSize' in line:
                    record_size = int(''.join(re.findall(r'\d+.\d+|\d+', line)))
                if 'Time Opened' in line:
                    self._set_date(re.search(r'\d+/\d+/\d+', line).group())
                    self._set_time(re.search(r'\d+:\d+:\d+', line).group())
                if 'FileVersion' in line:
                    self._set_file_version(line.split()[1])
                if 'ADMaxValue' in line:
                    max_ADC_count = float(''.join(re.findall(r'\d+.\d+|\d+', line)))
                if 'ADBitVolts' in line:
                    AD_bit_uvolt = float(''.join(re.findall(r'\d+.\d+|\d+', line)))*(10**6)

            self._set_fullscale_mv(max_byte_value*AD_bit_uvolt/2) # gain = 1 assumed to keep in similarity to Axona

            if not record_size:
                record_size = bytes_per_timestamp+ \
                             bytes_chan_no+ \
                             bytes_sample_freq+ \
                             bytes_num_valid_samples+ \
                             bytes_per_sample*samples_per_record

            time_offset = 0
            sample_freq_offset = bytes_per_timestamp+ bytes_chan_no
            num_valid_samples_offset = sample_freq_offset+ bytes_sample_freq
            sample_offset = num_valid_samples_offset+ bytes_num_valid_samples
            f.seek(0, 2)
            num_samples = int((f.tell()- header_offset)/record_size)

            f.seek(header_offset, 0)
            time = np.array([])
            lfp_wave = np.array([])
            sample_le = 256**(np.arange(0, bytes_per_sample, 1))
            for _ in np.arange(num_samples):
                sample_bytes = np.fromfile(f, dtype='uint8', count=record_size)
                block_start = int.from_bytes(sample_bytes[time_offset+ \
                                np.arange(bytes_per_timestamp)], byteorder='little', signed=False)/10**6
                valid_samples = int.from_bytes(sample_bytes[num_valid_samples_offset+ \
                                np.arange(bytes_num_valid_samples)], byteorder='little', signed=False)
                sampling_freq = int.from_bytes(sample_bytes[sample_freq_offset+ \
                                np.arange(bytes_sample_freq)], byteorder='little', signed=False)

                wave_bytes = sample_bytes[sample_offset+ np.arange(valid_samples* bytes_per_sample)]\
                                .reshape([valid_samples, bytes_per_sample])
                block_wave = np.dot(wave_bytes, sample_le)
                #    for k in np.arange(valid_samples):
                #        block_wave[k] = int.from_bytes(sample_bytes[sample_offset+ k*bytes_per_sample+ \
                #                    np.arange(bytes_per_sample)], byteorder='little', signed=False)
                np.putmask(block_wave, block_wave > max_ADC_count, block_wave - max_byte_value)
                block_wave = block_wave*AD_bit_uvolt
                block_time = block_start +  np.arange(valid_samples)/ sampling_freq
                interp_time = np.arange(block_start, block_time[-1], 1/resamp_freq)
                interp_wave = np.interp(interp_time, block_time, block_wave)
                time = np.append(time, interp_time)
                lfp_wave = np.append(lfp_wave, interp_wave)
            time -= time.min()
            self._set_samples(lfp_wave)
            self._set_total_samples(lfp_wave.size)
            self._set_timestamp(time)
            self._set_sampling_rate(resamp_freq)
