# -*- coding: utf-8 -*-
"""
This module implements NSpike Class for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import os

import re
#from imp import reload

import logging
from collections import OrderedDict as oDict
from copy import deepcopy

import numpy as np

#import nc_utils
#reload(nc_utils)
from neurochat.nc_utils import extrema, find, residual_stat

#from nc_lfp import NLfp
from neurochat.nc_hdf import Nhdf
from neurochat.nc_base import NBase

from scipy.optimize import curve_fit


class NSpike(NBase):
    """
    This data class is the placeholder for the dataset that contains information
    about the neural spikes. It decodes data from different formats and analyses
    single units in the recording.
     
    """
     
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._unit_no = kwargs.get('unit_no', 0)
        self._unit_stamp = []
        self._timestamp = []
        self._unit_list = []
        self._unit_Tags = []
        self._waveform = []
        self.set_record_info({'Timebase': 1,
                              'Samples per spike': 1,
                              'No of spikes': 0,
                              'Channel IDs': None})
        
        self.__type = 'spike'
        
    def get_type(self):
        """
        Returns the type of object. For NSpike, this is always `spike` type
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str

        """
        
        return self.__type 
    
    def get_unit_tags(self):
        
        """
        Returns the unit number or tags of the clustered units
        
        Parameters
        ----------
        None
        
        Returns
        -------
        list ot ndarray

        """
        
        return self._unit_Tags
    
    def set_unit_tags(self, new_tags):
        
        """
        Sets the number or tags of the clustered units
        
        Parameters
        ----------
        new_tags : list or ndarray
            Tags for each spiking wave 
        
        Returns
        -------
        None

        """
        
        if len(new_tags) == len(self._timestamp):
            self._unit_Tags = new_tags
            self._set_unit_list()
        else:
            logging.error('No of tags spikes does not match with no of spikes')

    def get_unit_list(self):
        """
        Gets the list of the units
        
        Parameters
        ----------
        None
        
        Returns
        -------
        list
            List of the unique tags of spiking-waveforms from clustering

        """
        
        return self._unit_list
    
    def _set_unit_list(self):
        """
        Sets the list of units from the list of unit tags
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None

        """
        
        self._unit_list = list(map(int, set(self._unit_Tags)))
        if 0 in self._unit_list:
            self._unit_list.remove(0)

    def set_unit_no(self, unit_no=None, spike_name=None):
        """
        Sets the unit number of the spike dataset to analyse
        
        Parameters
        ----------
        unit_no : int
            Unit or cell number to analyse
        
        Returns
        -------
        None

        """
        
        if isinstance(unit_no, int):
            if unit_no in self.get_unit_list():
                self._unit_no = unit_no
                self._set_unit_stamp()
        else:
            if spike_name is None:
                spike_name = self.get_spike_names()
            if len(unit_no) == len(spike_name):
                spikes = self.get_spike(spike_name)
                for i, num in enumerate(unit_no):
                    if num in spikes[i].get_unit_list():
                        spikes[i].set_unit_no(num)
            else:
                logging.error('Unit no. to set are not as many as child spikes!')

    def get_unit_no(self, spike_name=None):
        """
        Gets currently set unit number of the spike dataset to analyse
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Unit or cell number set to analyse

        """
        if spike_name is None:
            unit_no = self._unit_no
        else:
            unit_no = []
            spikes = self.get_spike(spike_name)
            for spike in spikes:
                unit_no.append(spike._unit_no)
        return unit_no

    def get_timestamp(self, unit_no=None):
        """
        Returns the timestamps of the spike-waveforms of spefied unit
        
        Parameters
        ----------
        None
        
        Returns
        -------
        ndarray
            Timestamps of the spiking waveforms
        """
        
        if unit_no is None:
            return self._timestamp
        else:
            if unit_no in self._unit_list:
                return self._timestamp[self._unit_Tags == unit_no]
            else:
                logging.warning('Unit ' + str(unit_no) + ' is not present in the spike data')

    def _set_timestamp(self, timestamp=None):
        """
        Sets the timestamps for all spiking waveforms in the recording
        
        Parameters
        ----------
        timestamp : list or ndarray
            Timestamps of all spiking waveforms
            
        Returns
        -------
        None

        """
        
        if timestamp is not None:
            self._timestamp = timestamp

    def get_unit_stamp(self):
        """
        Gets the timestamps for currently set unit to analyse
        
        Parameters
        ----------
        None
        
        Returns
        -------
        list or ndarray
            Timestamps for currently set unit

        """
        
        return self.get_timestamp(self._unit_no)

    def _set_unit_stamp(self):
        """
        Sets timestamps of the unit currently set to analyse
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Unit or cell number set to analyse

        """
        
        self._unit_stamp = self.get_unit_stamp()

    def get_unit_spikes_count(self, unit_no=None):
        """
        Returns the number of spikes in a unit
        
        Parameters
        ----------
        unit_no : int
            Units whose spike count is returned
        
        Returns
        -------
        int
            Number of units spikes of a unit in a recording session

        """
        
        if unit_no is None:
            unit_no = self._unit_no
        if unit_no in self._unit_list:
            return sum(self._unit_Tags == unit_no)

    def get_waveform(self):
        """
        Returns spike-waveforms
        
        Parameters
        ----------
        None
        
        Returns
        -------
        OrderedDict
            Dictionary of spiking waveforms where keys represent the channel number

        """
        
        return self._waveform

    def _set_waveform(self, spike_waves=[]):
        """
        Sets spike waveform to the NSpike() object
        
        Parameters
        ----------
        spike_waves : OrderedDict
            Spike waveforms where each key represents one channel
        
        Returns
        -------
        None

        """
        
        if spike_waves:
            self._waveform = spike_waves

    def get_unit_waves(self, unit_no=None):
        """
        Returns spike waveform of a specified unit
        
        Parameters
        ----------
        unit_no : int
            Unit whose waveforms are to be returned
        
        Returns
        -------
        OrderedDict
            Waveforms of the specified unit. If None, waveforms of currently set
            unit are returned

        """
        
        if unit_no is None:
            unit_no = self._unit_no
        _waves = oDict()
        for chan, wave in self._waveform.items():
            _waves[chan] = wave[self._unit_Tags == unit_no, :]
        
        return _waves

    def get_unit_stamps_in_ranges(self, ranges):
        """
        Return the unit timestamps in a list of ranges.

        Parameters
        ----------
        ranges : list
            A list of tuples indicating time ranges to get stamps in
        
        Returns
        -------
        list
            The timestamps
        """
        stamps = self.get_unit_stamp()
        new_stamps = [
            val for val in stamps
            if any(lower <= val <= upper for (lower, upper) in ranges)
        ]
        return new_stamps

    def subsample(self, sample_range=None):
        """
        Extract a time range from the spikes.
        
        Parameters
        ----------
        sample_range : tuple
            the time in seconds to extract from the spikes
        
        Returns
        -------
        NSpike
            subsampled version of initial spike object
        """
        if sample_range is None:
            return self
        new_spike = deepcopy(self)
        stamps = self.get_timestamp()
        lower, upper = sample_range
        sample_spike_idxs = (
            (stamps <= upper) & (stamps >= lower)).nonzero()
        new_spike_times = stamps[sample_spike_idxs]
        new_tags = self.get_unit_tags()[sample_spike_idxs]
        new_waveform = new_spike.get_waveform()
        for ch in new_waveform.keys():
            new_waveform[ch] = new_waveform[ch][sample_spike_idxs, :].squeeze()
        new_spike._set_timestamp(new_spike_times)
        new_spike.set_unit_tags(new_tags)
        new_spike._set_waveform(new_waveform)
        new_spike._set_duration(upper - lower)
        return new_spike

    def load(self, filename=None, system=None):
        """
        Loads spike datasets
        
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
        load_spike_axona(), load_spike_NLX(), load_spike_NWB()
            
        """
        
        if system is None:
            system = self._system
        else:
            self._system = system
        if filename is None:
            filename = self._filename
        else:
            self._filename = filename
        loader = getattr(self, 'load_spike_'+ system)
        loader(filename)

    def add_spike(self, spike=None, **kwargs):
        """
        Adds new spike node to current NSpike() object
        
        Parameters
        ----------
        spike : NSpike
            NSPike object. If None, new object is created
        
        Returns
        -------
        `:obj:NSpike`
            A new NSpike() object

        """
        new_spike = self._add_node(self.__class__, spike, 'spike', **kwargs)
        
        return new_spike

    def load_spike(self, names=None):
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
        
        if names is None:
            self.load()
        elif names == 'all':
            for spike in self._spikes:
                spike.load()
        else:
            logging.error("Spikes by name has yet to be implemented")
            # for name in names:
            #     spike = self.get_spikes_by_name(name)
            #     spike.load()

    def add_lfp(self, lfp=None, **kwargs):
        """
        Adds new LFP node to current NSpike() object
        
        Parameters
        ----------
        lfp : NLfp
            NLfp object. If None, new object is created
        
        Returns
        -------
        `:obj:Nlfp`
            A new NLfp() object

        """
        
        try:
            data_type = lfp.get_type()
        except:
            logging.error('The data type of the added object cannot be determined!')

        if data_type == 'lfp':
                cls= lfp.___class__ 
        else:
            cls = None

        new_lfp = self._add_node(cls, lfp, 'lfp', **kwargs)
        
        return new_lfp

    def load_lfp(self, names='all'):
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
        
        if names == 'all':
            for lfp in self._lfp:
                lfp.load()
        else:
            logging.error("Lfp by name has yet to be implemented")
            # for name in names:
            #     lfp = self.get_lfp_by_name(name)
            #     lfp.load()

    def wave_property(self):
        """
        Claulates different waveform properties for currently set unit
        
        Parameters
        ----------
        None
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """

        _result = oDict()
        graph_data = {}

        def argpeak(data):
            data = np.array(data)
            peak_loc = [j for j in range(7, len(data)) \
                        if data[j] <= 0 and data[j - 1] > 0]
            return peak_loc[0] if peak_loc else 0

        def argtrough1(data, peak_loc):
            data = data.tolist()
            trough_loc = [peak_loc - j for j in range(peak_loc - 2) \
                        if data[peak_loc - j] >= 0 and data[peak_loc - j - 1] <= 0]
            return trough_loc[0] if trough_loc else 0

        def wave_width(wave, peak, thresh=0.25):
            p_loc, p_val = peak
            Len = wave.size
            if p_loc:
                w_start = find(wave[:p_loc] <= thresh*p_val, 1, 'last')
                w_start = w_start[0] if w_start.size else 0
                w_end = find(wave[p_loc:] <= thresh*p_val, 1, 'first')
                w_end = p_loc + w_end[0] if w_end.size else Len
            else:
                w_start = 1
                w_end = Len

            return w_end- w_start

        num_spikes = self.get_unit_spikes_count()
        _result['Number of Spikes'] = num_spikes
        _result['Mean Spiking Freq'] = num_spikes/ self.get_duration()
        _waves = self.get_unit_waves()
        samples_per_spike = self.get_samples_per_spike()
        tot_chans = self.get_total_channels()
        meanWave = np.empty([samples_per_spike, tot_chans])
        stdWave = np.empty([samples_per_spike, tot_chans])

        width = np.empty([num_spikes, tot_chans])
        amp = np.empty([num_spikes, tot_chans])
        height = np.empty([num_spikes, tot_chans])
        for i, (chan, wave) in enumerate(_waves.items()):
            meanWave[:, i] = np.mean(wave, 0)
            stdWave[:, i] = np.std(wave, 0)
            slope = np.gradient(wave)[1][:, :-1]
            max_val = wave.max(1)

            peak_val, trough1_val = 0, 0
            if max_val.max() > 0:
                peak_loc = [argpeak(slope[I, :]) for I in range(num_spikes)]
                peak_val = [wave[I, peak_loc[I]] for I in range(num_spikes)]
                trough1_loc = [argtrough1(slope[I, :], peak_loc[I]) for I in range(num_spikes)]
                trough1_val = [wave[I, trough1_loc[I]] for I in range(num_spikes)]
                peak_loc = np.array(peak_loc)
                peak_val = np.array(peak_val)
                trough1_loc = np.array(trough1_loc)
                trough1_val = np.array(trough1_val)
                width[:, i] = np.array([wave_width(wave[I, :], (peak_loc[I], peak_val[I]), 0.25) \
                             for I in range(num_spikes)])

            amp[:, i] = peak_val - trough1_val
            height[:, i] = peak_val - wave.min(1)
        max_chan = amp.mean(0).argmax()
        width = width[:, max_chan]* 10**6/self.get_sampling_rate()
        amp = amp[:, max_chan]
        height = height[:, max_chan]

        graph_data = {'Mean wave': meanWave, 'Std wave': stdWave,
                      'Amplitude': amp, 'Width': width, 'Height': height,
                      'Max channel': max_chan}

        _result.update({'Mean amplitude': amp.mean(), 'Std amplitude': amp.std(),
                        'Mean height': height.mean(), 'Std height': height.std(),
                        'Mean width': width.mean(), 'Std width': width.std()})

        self.update_result(_result)
        
        return graph_data

    def isi(self, bins='auto', bound=None, density=False, 
            refractory_threshold=2):
        """
        Calulates the ISI histogram of the spike train
        
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
    
        """
        graph_data = oDict()
        _results = oDict()

        unitStamp = self.get_unit_stamp()
        isi = 1000*np.diff(unitStamp)

        below_refractory = isi[isi < refractory_threshold]

        graph_data['isiHist'], edges = np.histogram(isi, bins=bins, range=bound, density=density)
        graph_data['isiBins'] = edges[:-1]
        graph_data['isi'] = isi
        graph_data['maxCount'] = graph_data['isiHist'].max()
        graph_data['isiBefore'] = isi[:-1]
        graph_data['isiAfter'] = isi[1:]

        _results["Mean ISI"] = isi.mean()
        _results["Std ISI"] = isi.std()
        _results["Refractory violation"] = (
            below_refractory.size / unitStamp.size)

        self.update_result(_results)
        return graph_data

    def isi_corr(self, spike=None, **kwargs):
        """
        Calculates the correlation of ISI histogram.
        
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
    
        """
        
        graph_data = oDict()
        if spike is None:
            _unit_stamp = np.copy(self.get_unit_stamp())
        elif isinstance(spike, int):
            if spike in self.get_unit_list():
                _unit_stamp = self.get_timestamp(spike)
        else:
            if isinstance(spike, str):
                spike = self.get_spike(spike)
            if isinstance(spike, self.__class__):
                _unit_stamp = spike.get_unit_stamp()
            else:
                logging.error('No valid spike specified')

        _corr = self.psth(_unit_stamp, **kwargs)
        graph_data['isiCorrBins'] = _corr['bins']
        graph_data['isiAllCorrBins'] = _corr['all_bins']
        center = find(_corr['bins'] == 0, 1, 'first')[0]
        graph_data['isiCorr'] = _corr['psth']
        graph_data['isiCorr'][center] = graph_data['isiCorr'][center] \
                                    - np.min([self.get_unit_stamp().size, _unit_stamp.size])

        return graph_data

    def psth(self, event_stamp, **kwargs):
        """
        Calculates peri-stimulus time histogram (PSTH)
        
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
    
        """
        
        graph_data = oDict()
        bins = kwargs.get('bins', 1)
        if isinstance(bins, int):
            bound = np.array(kwargs.get('bound', [-500, 500]))
            bins = np.hstack((np.arange(bound[0], 0, bins), np.arange(0, bound[1] + bins, bins)))
        bins = bins/1000 # converted to sec
        n_bins = len(bins) - 1

        hist_count = np.zeros([n_bins, ])
        unitStamp = self.get_unit_stamp()
        for it in range(event_stamp.size):
            tmp_count, edges = np.histogram(unitStamp - event_stamp[it], bins=bins)
            hist_count = hist_count + tmp_count

        graph_data['psth'] = hist_count
        graph_data['bins'] = 1000*edges[:-1]

        # Included in case the last point is needed
        graph_data['all_bins'] = 1000*edges

        return graph_data

    def burst(self, burst_thresh=5, ibi_thresh=50):
        """
        Analysis of bursting properties of the spiking train
        
        Parameters
        ----------
        burst_thresh : int
            Minimum ISI between consecutive spikes in a burst
            
        ibi_thresh : int
            Minimum inter-burst interval between two bursting groups of spikes
 
        Returns
        -------
        None
    
        """

        _results = oDict()

        unitStamp = self.get_unit_stamp()
        isi = 1000*np.diff(unitStamp)

        burst_start = []
        burst_end = []
        burst_duration = []
        spikesInBurst = []
        bursting_isi = []
        num_burst = 0
        ibi = []
        duty_cycle = []
        k = 0
        while k < isi.size:
            if isi[k] <= burst_thresh:
                burst_start.append(k)
                spikesInBurst.append(2)
                bursting_isi.append(isi[k])
                burst_duration.append(isi[k])
                m = k+1
                while m < isi.size and isi[m] <= burst_thresh:
                    spikesInBurst[num_burst] += 1
                    bursting_isi.append(isi[m])
                    burst_duration[num_burst] += isi[m]
                    m += 1
                burst_duration[num_burst] += 1 # to compensate for the span of the last spike
                burst_end.append(m)
                k = m+1
                num_burst += 1
            else:
                k += 1
        if num_burst:
            for j in range(0, num_burst-1):
                ibi.append(unitStamp[burst_start[j+1]]- unitStamp[burst_end[j]])
            duty_cycle = np.divide(burst_duration[1:], ibi)/1000 # ibi in sec, burst_duration in ms
        else:
            logging.warning(
                'No burst detected in {}'.format(self.get_filename()))

        spikesInBurst = np.array(spikesInBurst) if spikesInBurst else np.array([])
        bursting_isi = np.array(bursting_isi) if bursting_isi else np.array([])
        ibi = 1000*np.array(ibi) if ibi else np.array([]) # in sec unit, so converted to ms
        burst_duration = np.array(burst_duration) if burst_duration else np.array([])
        duty_cycle = np.array(duty_cycle) if len(duty_cycle) else np.array([])

        _results['Total burst'] = num_burst
        _results['Total bursting spikes'] = spikesInBurst.sum()
        _results['Mean bursting ISI ms'] = bursting_isi.mean() if bursting_isi.any() else None
        _results['Std bursting ISI ms'] = bursting_isi.std() if bursting_isi.any() else None
        _results['Mean spikes per burst'] = spikesInBurst.mean() if spikesInBurst.any() else None
        _results['Std spikes per burst'] = spikesInBurst.std() if spikesInBurst.any() else None
        _results['Mean burst duration ms'] = burst_duration.mean() if burst_duration.any() else None
        _results['Std burst duration'] = burst_duration.std() if burst_duration.any() else None
        _results['Mean duty cycle'] = duty_cycle.mean() if duty_cycle.any() else None
        _results['Std duty cycle'] = duty_cycle.std() if duty_cycle.any() else None
        _results['Mean IBI'] = ibi.mean() if ibi.any() else None
        _results['Std IBI'] = ibi.std() if ibi.any() else None
        _results['Propensity to burst'] = spikesInBurst.sum()/ unitStamp.size
        
        self.update_result(_results)

    def theta_index(self, **kwargs):
        """
        Analysis of theta-modulation of a unit
        
        Parameters
        ----------
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
    
        """
        
        p_0 = kwargs.get('start', [6, 0.1, 0.05])
        lb = kwargs.get('lower', [4, 0, 0])
        ub = kwargs.get('upper', [14, 5, 0.1])

        _results = oDict()
        graph_data = self.isi_corr(**kwargs)
        corrBins = graph_data['isiCorrBins']
        corrCount = graph_data['isiCorr']
        m = corrCount.max()
        center = find(corrBins == 0, 1, 'first')[0]
        x = corrBins[center:]/1000
        y = corrCount[center:]
        y_fit = np.empty([corrBins.size,])

        ## This is for the double-exponent dip model
        # def fit_func(x, a, f, tau1, b, c1, tau2, c2, tau3):
        #     return  a*np.cos(2*np.pi*f*x)*np.exp(-np.abs(x)/tau1)+ b+ \
        #         c1*np.exp(-np.abs(x)/tau2)- c2*np.exp(-np.abs(x)/tau3)
        
        # popt, pcov = curve_fit(fit_func, x, y, \
        #                         p0=[m, p_0[0], p_0[1], m, m, p_0[2], m, 0.005], \
        #                         bounds=([0, lb[0], lb[1], 0, 0, lb[2], 0, 0], \
        #                         [m, ub[0], ub[1], m, m, ub[2], m, 0.01]),
        #                         max_nfev=100000)
        # a, f, tau1, b, c1, tau2, c2, tau3 = popt

        # This is for the single-exponent dip model
        def fit_func(x, a, f, tau1, b, c, tau2):
            return  a*np.cos(2*np.pi*f*x)*np.exp(-np.abs(x)/tau1)+ b+ \
                c*np.exp(-(x/tau2)**2)

        try:
            popt, pcov = curve_fit(
                fit_func, x, y,
                p0=[m, p_0[0], p_0[1], m, m, p_0[2]], 
                bounds=([0, lb[0], lb[1], 0, -m, lb[2]],
                [m, ub[0], ub[1], m, m, ub[2]]),
                max_nfev=100000)
        except Exception as e:
            logging.error("Failed curve_fit in theta_index: {} ".format(e))
            _results['Theta Index'] = None
            _results['TI fit freq Hz'] = None
            _results['TI fit tau1 sec'] = None
            _results['TI adj Rsq'] = None
            _results['TI Pearse R'] = None
            _results['TI Pearse P'] = None

            self.update_result(_results)
            return None

        a, f, tau1, b, c, tau2 = popt

        y_fit[center:] = fit_func(x, *popt)
        y_fit[:center] = np.flipud(y_fit[center:])

        gof = residual_stat(y, y_fit[center:], 6)

        graph_data['corrFit'] = y_fit
        _results['Theta Index'] = a/b
        _results['TI fit freq Hz'] = f
        _results['TI fit tau1 sec'] = tau1
        _results['TI adj Rsq'] = gof['adj Rsq']
        _results['TI Pearse R'] = gof['Pearson R']
        _results['TI Pearse P'] = gof['Pearson P']

        self.update_result(_results)

        return graph_data

    def theta_skip_index(self, **kwargs):
        """
        Analysis of theta-skipping of a unit
        
        Parameters
        ----------
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        
        p_0 = kwargs.get('start', [6, 0.1, 0.05])
        lb = kwargs.get('lower', [4, 0, 0])
        ub = kwargs.get('upper', [14, 5, 0.1])

        _results = oDict()
        graph_data = self.isi_corr(**kwargs)
        corrBins = graph_data['isiCorrBins']
        corrCount = graph_data['isiCorr']
        m = corrCount.max()
        center = find(corrBins == 0, 1, 'first')[0]
        x = corrBins[center:]/1000
        y = corrCount[center:]
        y_fit = np.empty([corrBins.size,])

        # This is for the double-exponent dip model
        def fit_func(x, a1, f1, a2, f2, tau1, b, c1, tau2, c2, tau3):
            return  (a1*np.cos(2*np.pi*f1*x)+ a2*np.cos(2*np.pi*f2*x))*np.exp(-np.abs(x)/tau1)+ b+ \
                c1*np.exp(-np.abs(x)/tau2)- c2*np.exp(-np.abs(x)/tau3)

        popt, pcov = curve_fit(fit_func, x, y, \
                                p0=[m, p_0[0], m, p_0[0]/2, p_0[1], m, m, p_0[2], m, 0.005], \
                                bounds=([0, lb[0], 0, lb[0]/2, lb[1], 0, 0, lb[2], 0, 0], \
                                [m, ub[0], m, ub[0]/2, ub[1], m, m, ub[2], m, 0.01]),\
                                max_nfev=100000)
        a1, f1, a2, f2, tau1, b, c1, tau2, c2, tau3 = popt

        ## This is for the single-exponent dip model
        # def fit_func(x, a1, f1, a2, f2, tau1, b, c, tau2):
        #     return  (a1*np.cos(2*np.pi*f1*x)+ a2*np.cos(2*np.pi*f2*x))*np.exp(-np.abs(x)/tau1)+ b+ \
        #         c*np.exp(-(x/tau2)**2)
        
        # popt, pcov = curve_fit(fit_func, x, y, \
        #                         p0=[m, p_0[0], m, p_0[0]/2, p_0[1], m, m, p_0[2]], \
        #                         bounds=([0, lb[0], 0, lb[0]/2, lb[1], 0, -m, lb[2]], \
        #                         [m, ub[0], m, ub[0]/2, ub[1], m, m, ub[2]]),
        #                         max_nfev=100000)
        # a1, f1, a2, f2, tau1, b, c, tau2 = popt

        temp_fit = fit_func(x, *popt)
        y_fit[center:] = temp_fit
        y_fit[:center] = np.flipud(temp_fit)

        peak_val, peak_loc = extrema(temp_fit[find(x >= 50/1000)])[0:2]

        if len(peak_val) >= 2:
            skipIndex = (peak_val[1]- peak_val[0])/np.max(np.array([peak_val[1], peak_val[0]]))
        else:
            skipIndex = None
        gof = residual_stat(y, temp_fit, 6)

        graph_data['corrFit'] = y_fit
        _results['Theta Skip Index'] = skipIndex
        _results['TS jump factor'] = a2/(a1+ a2) if skipIndex else None
        _results['TS f1 freq Hz'] = f1 if skipIndex else None
        _results['TS f2 freq Hz'] = f2 if skipIndex else None
        _results['TS freq ratio'] = f1/f2 if skipIndex else None
        _results['TS tau1 sec'] = tau1 if skipIndex else None
        _results['TS adj Rsq'] = gof['adj Rsq']
        _results['TS Pearse R'] = gof['Pearson R']
        _results['TS Pearse P'] = gof['Pearson P']

        self.update_result(_results)

        return graph_data

    def phase_dist(self, lfp = None, **kwargs):
        """
        Analysis of spike to LFP phase distribution
        
        Delegates to NLfp().phase_dist()
        
        Parameters
        ----------
        lfp : NLfp
            LFP object which contains the LFP data
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
        
        if lfp is None:
            logging.error('LFP data not specified!')
        else:
            try:
                lfp.phase_dist(self.get_unit_stamp(), **kwargs)
            except:
                logging.error('No phase_dist() method in lfp data specified!')

    def plv(self, lfp=None, **kwargs):
        """
        Calculates phase-locking value of spike train to underlying LFP signal.
        
        Delegates to NLfp().plv()
        
        Parameters
        ----------
        lfp : NLfp
            LFP object which contains the LFP data
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
        
        if lfp is None:
            logging.error('LFP data not specified!')
        else:
            try:
                lfp.plv(self.get_unit_stamp(), **kwargs)
            except:
                logging.error('No plv() method in lfp data specified!')
            
    def spike_lfp_causality(self, lfp=None, **kwargs):
        """
        Analyses spike to underlying LFP causality
        
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
        
        if lfp is None:
            logging.error('LFP data not specified!')
        else:
            try:
                lfp.spike_lfp_causality(self.get_unit_stamp(), **kwargs)
            except:
                logging.error('No sfc() method in lfp data specified!')

    def _set_total_spikes(self, spike_count=1):
        """
        Sets the total number of spikes as part of storing the recording information
                
        Parameters
        ----------
        spike_count : int
            Total number of spikes
 
        Returns
        -------
        None    

        """
        
        self._record_info['No of spikes'] = spike_count
        self.spike_count = spike_count
        
    def _set_total_channels(self, tot_channels=1):
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
        
    def _set_channel_ids(self, channel_ids):
        """
        Sets identity of the channels as part of storing the recording information
                
        Parameters
        ----------
        channel_ids : int
            Total number of channels
 
        Returns
        -------
        None    

        """
        
        self._record_info['Channel IDs'] = channel_ids
        
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
        
    def _set_timebase(self, timebase=1):
        """
        Sets timbase for spike event timestamps as part of storing the recording information
                
        Parameters
        ----------
        timebase : int
            Timebase for the spike event timestamps
 
        Returns
        -------
        None    

        """
        self._record_info['Timebase'] = timebase
        
    def _set_sampling_rate(self, sampling_rate=1):
        """
        Sets the sampling rate of the spike waveform as part of storing the recording information
                
        Parameters
        ----------
        sampling_rate : int
            Sampling rate of the spike waveforms
 
        Returns
        -------
        None    

        """
        self._record_info['Sampling rate'] = sampling_rate
        
    def _set_bytes_per_sample(self, bytes_per_sample=1):
        """
        Sets `bytes per sample` value as part of storing the recording information
                
        Parameters
        ----------
        bytes_per_sample : int
            Total number of bytes to represent each waveform sample in the binary file
 
        Returns
        -------
        None    

        """
        self._record_info['Bytes per sample'] = bytes_per_sample
        
    def _set_samples_per_spike(self, samples_per_spike=1):
        """
        Sets `samples per spike` value as part of storing the recording information
                
        Parameters
        ----------
        samples_per_spike : int
            Total number of samples to represent a spike waveform
 
        Returns
        -------
        None

        """
        
        self._record_info['Samples per spike'] = samples_per_spike
        
    def _set_fullscale_mv(self, adc_fullscale_mv=1):
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

    def get_total_spikes(self):
        """
        Returns total number of spikes in the recording
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Total number of spikes

        """

        return self._record_info['No of spikes']
    
    def get_total_channels(self):
        """
        Returns total number of electrode channels in the spike data file
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Total number of electrode channels

        """
        
        return self._record_info['No of channels']
    
    def get_channel_ids(self):
        """
        Returns the identities of individual channels
                
        Parameters
        ----------
        None
 
        Returns
        -------
        list
            Identities of individual channels 

        """
        
        return self._record_info['Channel IDs']
    
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
    
    def get_timebase(self):
        """
        Returns the timebase for spike event timestamps
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Timebase for spike event timestamps

        """
    
        return self._record_info['Timebase']
    
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
        Returns the number of bytes to represent each spike waveform sample
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Number of bytes to represent each sample of the spike waveforms

        """
    
        return self._record_info['Bytes per sample']
    
    def get_samples_per_spike(self):
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
        
        return self._record_info['Samples per spike']
    
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

    def save_to_hdf5(self, file_name=None, system=None):
        """
        Stores NSpike() object to HDF5 file
        
        Parameters
        ----------
        file_name : str
            Full file directory for the spike data
        system : str
            Recoring system or data format
        
        Returns
        -------
        None
        
        Also see
        --------
        nc_hdf.Nhdf().save_spike()
        
        """
        hdf = Nhdf()
        if file_name and system:
            if os.path.exists(file_name):
                self.set_filename(file_name)
                self.set_system(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')

        hdf.save_spike(spike=self)
        hdf.close()

    def load_spike_NWB(self, file_name):
        """
        Decodes spike data from NWB (HDF5) file format
        
        Parameters
        ----------
        file_name : str
            Full file directory for the spike data
        
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
            elif '/processing/Shank/'+ path in hdf.f:
                path = '/processing/Shank/'+ path
                g = hdf.f[path]
            else:
                logging.error('Specified shank datapath does not exist!')
                return
    
            for key, value in g.attrs.items():
                _record_info[key] = value
            self.set_record_info(_record_info)
    
            path_clust = 'Clustering'
            path_wave = 'EventWaveForm/WaveForm'
    
            if path_clust in g:
                g_clust = g[path_clust]
                self._set_timestamp(hdf.get_dataset(group=g_clust, name='times'))
                self.set_unit_tags(hdf.get_dataset(group=g_clust, name='num'))
                self._set_unit_list()
            else:
                logging.error('There is no /Clustering in the :' +path)
    
            if path_wave in g:
                g_wave = g[path_wave]
                self._set_total_spikes(hdf.get_dataset(group=g_wave, name='num_events'))
                chanIDs = hdf.get_dataset(group=g_wave, name='electrode_idx')
                self._set_channel_ids(chanIDs)
    
                spike_wave = oDict()
                data = hdf.get_dataset(group=g_wave, name='data')
                if len(data.shape) == 2:
                    num_events, num_samples = data.shape
                    tot_chans = 1
                elif len(data.shape) == 3:
                    num_events, num_samples, tot_chans = data.shape
                else:
                    logging.error(path_wave+ '/data contains for more than 3 dimensions!')
    
                if num_events != hdf.get_dataset(group=g_wave, name='num_events'):
                    logging.error('Mismatch between num_events and 1st dimension of '+ path_wave+ '/data')
                if num_samples != hdf.get_dataset(group=g_wave, name='num_samples'):
                    logging.error('Mismatch between num_samples and 2nd dimension of '+ path_wave+ '/data')
                for i in np.arange(tot_chans):
                    spike_wave['ch'+ str(i+1)] = data[:, :, i]
                self._set_waveform(spike_wave)
            else:
                logging.error('There is no /EventWaveForm/WaveForm in the :' +path)
            
            hdf.close()
        else:
            logging.error(file_name + ' does not exist!')
    
    def load_spike_Axona(self, file_name):
        """
        Decodes spike data from Axona file format
        
        Parameters
        ----------
        file_name : str
            Full file directory for the spike data
        
        Returns
        -------
        None
        
        """
        words = file_name.split(sep=os.sep)
        file_directory = os.sep.join(words[0:-1])
        file_tag = words[-1].split(sep='.')[0]
        tet_no = words[-1].split(sep='.')[1]
        set_file = file_directory + os.sep + file_tag + '.set'
        cut_file = file_directory + os.sep + file_tag + '_' + tet_no + '.cut'
        clu_file = file_directory + os.sep + file_tag + '.clu.' + tet_no

        self._set_data_source(file_name)
        self._set_source_format('Axona')

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
                    self._set_total_channels(int(''.join(line.split()[1:])))
                if line.startswith('timebase'):
                    self._set_timebase(int(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if line.startswith('bytes_per_timestamp'):
                    self._set_timestamp_bytes(int(''.join(line.split()[1:])))
                if line.startswith('samples_per_spike'):
                    self._set_samples_per_spike(int(''.join(line.split()[1:])))
                if line.startswith('sample_rate'):
                    self._set_sampling_rate(int(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if line.startswith('bytes_per_sample'):
                    self._set_bytes_per_sample(int(''.join(line.split()[1:])))
                if line.startswith('num_spikes'):
                    self._set_total_spikes(int(''.join(line.split()[1:])))
                if line.startswith('data_start'):
                    break

            num_spikes = self.get_total_spikes()
            bytes_per_timestamp = self.get_timestamp_bytes()
            bytes_per_sample = self.get_bytes_per_sample()
            samples_per_spike = self.get_samples_per_spike()

            f.seek(0, 0)
            header_offset = []
            while True:
                try:
                    buff = f.read(10).decode('UTF-8')
                    if buff == 'data_start':
                        header_offset = f.tell()
                        break
                    else:
                        f.seek(-9, 1)
                except:
                    break

            tot_channels = self.get_total_channels()
            self._set_channel_ids([(int(tet_no) - 1)*tot_channels + x for x in range(tot_channels)])
            max_ADC_count = 2**(8*bytes_per_sample - 1) - 1
            max_byte_value = 2**(8*bytes_per_sample)

            with open(set_file, 'r', encoding='latin-1') as f_set:
                lines = f_set.readlines()
                gain_lines = dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                            for line in lines if 'gain_ch_' in line])
                gains = np.array([gain_lines[ch_id] for ch_id in self.get_channel_ids()])
                for line in lines:
                    if line.startswith('ADC_fullscale_mv'):
                        self._set_fullscale_mv(int(re.findall(r'\d+.\d+|d+', line)[0]))
                        break
                AD_bit_uvolts = 2*self.get_fullscale_mv()*10**3/ \
                                 (gains*(2**(8*bytes_per_sample)))

            record_size = tot_channels*(bytes_per_timestamp + \
                            bytes_per_sample * samples_per_spike)
            time_be = 256**(np.arange(bytes_per_timestamp, 0, -1)-1)
            sample_le = 256**(np.arange(0, bytes_per_sample, 1))

            if not header_offset:
                print('Error: data_start marker not found!')
            else:
                f.seek(header_offset, 0)
                byte_buffer = np.fromfile(f, dtype='uint8')
                spike_time = np.zeros([num_spikes, ], dtype='uint32')
                for i in list(range(0, bytes_per_timestamp)):
                    byte = byte_buffer[i:len(byte_buffer):record_size]
                    byte = byte[:num_spikes]
                    spike_time = spike_time + time_be[i]*byte
                spike_time = spike_time/ self.get_timebase()
                spike_time = spike_time.reshape((num_spikes, ))

                spike_wave = oDict()


                for i in np.arange(tot_channels):
                    chan_offset = (i+1)*bytes_per_timestamp+ i*bytes_per_sample*samples_per_spike
                    chan_wave = np.zeros([num_spikes, samples_per_spike], dtype=np.float64)
                    for j in np.arange(0, samples_per_spike, 1):
                        sample_offset = j*bytes_per_sample + chan_offset
                        for k in np.arange(0, bytes_per_sample, 1):
                            byte_offset = k + sample_offset
                            sample_value = sample_le[k]* byte_buffer[byte_offset \
                                          : len(byte_buffer)+ byte_offset-record_size\
                                          :record_size]
                            sample_value = sample_value.astype(np.float64, casting='unsafe', copy=False)
                            np.add(chan_wave[:, j], sample_value, out=chan_wave[:, j])
                        np.putmask(chan_wave[:, j], chan_wave[:, j] > max_ADC_count, chan_wave[:, j]- max_byte_value)
                    spike_wave['ch'+ str(i+1)] = chan_wave*AD_bit_uvolts[i]

            unit_ID = None
            if os.path.isfile(cut_file):
                used = cut_file
                with open(cut_file, 'r') as f_cut:
                    while True:
                        line = f_cut.readline()
                        if line == '':
                            break
                        if line.startswith('Exact_cut'):
                            unit_ID = np.fromfile(f_cut, dtype='uint8', sep=' ')
            
            elif os.path.isfile(clu_file):
                used = clu_file
                data = np.loadtxt(clu_file)
                unit_ID = data[1:].flatten() - 1
                    
            else:
                logging.error(
                    "No cluster file found for spike file {} please make one at {} or {}".format(
                        file_name, cut_file, clu_file))
                return
            
            if unit_ID is None:
                string = "Unable to parse clusters from {}".format(used)
                print(string)
                logging.error(string)
                return

            self._set_timestamp(spike_time)
            self._set_waveform(spike_wave)
            self.set_unit_tags(unit_ID)

    def load_spike_Neuralynx(self, file_name):
        """
        Decodes spike data from Neuralynx file format
        
        Parameters
        ----------
        file_name : str
            Full file directory for the spike data
        
        Returns
        -------
        None
        
        """
        self._set_data_source(file_name)
        self._set_source_format('Neuralynx')

        # Format description for the NLX file:
        file_ext = file_name[-3:]
        if file_ext == 'ntt':
            tot_channels = 4
        elif file_ext == 'nst':
            tot_channels = 2
        elif file_ext == 'nse':
            tot_channels = 1
        header_offset = 16*1024 # fixed for NLX files

        bytes_per_timestamp = 8
        bytes_chan_no = 4
        bytes_cell_no = 4
        bytes_per_feature = 4
        num_features = 8
        bytes_features = bytes_per_feature*num_features
        bytes_per_sample = 2
        samples_per_record = 32
        channel_pack_size = bytes_per_sample*tot_channels# ch1|ch2|ch3|ch4 each with 2 bytes

        max_byte_value = np.power(2, bytes_per_sample*8)
        max_ADC_count = np.power(2, bytes_per_sample*8- 1)-1
        AD_bit_uvolts = np.ones([tot_channels, ])*10**-6 # Default value

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
                    self._set_sampling_rate(float(''.join(re.findall(r'\d+.\d+|\d+', line))))
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
                    AD_bit_uvolts = np.array([float(x)*(10**6) for x in re.findall(r'\d+.\d+|\d+', line)])
                if 'ADChannel' in line:
                    self._set_channel_ids(np.array([int(x) for x in re.findall(r'\d+', line)]))
                if 'NumADChannels' in line:
                    tot_channels = int(''.join(re.findall(r'\d+', line)))

            self._set_fullscale_mv((max_byte_value/2)*AD_bit_uvolts) # gain = 1 assumed to keep in similarity to Axona
            self._set_bytes_per_sample(bytes_per_sample)
            self._set_samples_per_spike(samples_per_record)
            self._set_timestamp_bytes(bytes_per_timestamp)
            self._set_total_channels(tot_channels)

            if not record_size:
                record_size = bytes_per_timestamp+ \
                             bytes_chan_no+ \
                             bytes_cell_no+ \
                             bytes_features+ \
                             bytes_per_sample*samples_per_record*tot_channels

            time_offset = 0
            unitID_offset = bytes_per_timestamp+ \
                           bytes_chan_no
            sample_offset = bytes_per_timestamp+ \
                           bytes_chan_no+ \
                           bytes_cell_no+ \
                           bytes_features
            f.seek(0, 2)
            num_spikes = int((f.tell()- header_offset)/record_size)
            self._set_total_spikes(num_spikes)

            f.seek(header_offset, 0)
            spike_time = np.zeros([num_spikes, ])
            unit_ID = np.zeros([num_spikes, ], dtype=int)
            spike_wave = oDict()
            sample_le = 256**(np.arange(bytes_per_sample))
            for i in np.arange(tot_channels):
                spike_wave['ch'+ str(i+1)] = np.zeros([num_spikes, samples_per_record])

            for i in np.arange(num_spikes):
                sample_bytes = np.fromfile(f, dtype='uint8', count=record_size)
                spike_time[i] = int.from_bytes(sample_bytes[time_offset+ np.arange(bytes_per_timestamp)], byteorder='little', signed=False)/10**6
                unit_ID[i] = int.from_bytes(sample_bytes[unitID_offset+ np.arange(bytes_cell_no)], byteorder='little', signed=False)

                for j in range(tot_channels):
                    sample_value = np.zeros([samples_per_record, bytes_per_sample])
                    ind = sample_offset+ j*bytes_per_sample+ np.arange(samples_per_record)*channel_pack_size
                    for k in np.arange(bytes_per_sample):
                        sample_value[:, k] = sample_bytes[ind+ k]
                    sample_value = sample_value.dot(sample_le)
                    np.putmask(sample_value, sample_value > max_ADC_count, sample_value- max_byte_value)
                    spike_wave['ch'+ str(j+1)][i, :] = sample_value*AD_bit_uvolts[j]
            spike_time -= spike_time.min()
            self._set_duration(spike_time.max())
            self._set_timestamp(spike_time)
            self._set_waveform(spike_wave)
            self.set_unit_tags(unit_ID)

    # def sfc(self, lfp=None, **kwargs):
    #     """
    #     Calculates spike-field coherence of spike train with underlying LFP signal.

    #     Delegates to NLfp().sfc()

    #     Parameters
    #     ----------
    #     lfp : NLfp
    #         LFP object which contains the LFP data
    #     **kwargs
    #         Keyword arguments

    #     Returns
    #     -------
    #     dict
    #         Graphical data of the analysis

    #     See also
    #     --------
    #     nc_lfp.NLfp().sfc()

    #     """

    #     if lfp is None:
    #         logging.error('LFP data not specified!')
    #     else:
    #         try:
    #             lfp.sfc(self.get_unit_stamp(), **kwargs)
    #         except:
    #             logging.error('No sfc() method in lfp data specified!')
