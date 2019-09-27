# -*- coding: utf-8 -*-
"""
This module implements NSpatial Class for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import os
import re

import logging

from collections import OrderedDict as oDict
from copy import deepcopy

from neurochat.nc_utils import chop_edges, corr_coeff, extrema,\
            find, find2d, find_chunk, histogram, histogram2d, \
            linfit, residual_stat, rot_2d, smooth_1d, smooth_2d, \
            centre_of_mass, find_true_ranges

from neurochat.nc_base import NAbstract
from neurochat.nc_circular import CircStat
from neurochat.nc_hdf import Nhdf

from neurochat.nc_spike import NSpike
from neurochat.nc_lfp import NLfp
from neurochat.nc_event import NEvent

import numpy as np
import numpy.random as nprand

import scipy as sc
from scipy.optimize import curve_fit

import scipy.signal as sg

from sklearn.linear_model import LinearRegression

class NSpatial(NAbstract):
    """
    This data class is the placeholder for the dataset that contains information
    about the spatial behaviour of the animal. It decodes data from different 
    formats and analyses the correlation of spatial information with the spiking
    activity of a unit.
    """    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._time = []
        self._timestamp = []
        self._total_samples = []
        self._fs = []
        self._pixel_size = 3
        self._pos_x = []
        self._pos_y = []
        self._direction = []
        self._speed = []
        self._ang_vel = []
        self._border_dist = []
        self._xbound = []
        self._ybound = []
        self._dist_map = []
        self.spike = []
        self.lfp = []
        
        self.__type = 'spatial'
        
    def get_type(self):
        """
        Returns the type of object. For NSpatial, this is always `spatial` type
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str

        """ 
        
        return self.__type        

    def subsample(self, sample_range=None):
        """
        Extract a time range from the positions.

        NOTE for now, the duration will be longer than sample time.
        Duration is actually from 0 to max recording length.
        This is to easier match ndata which assumes recordings start at 0.
        
        Parameters
        ----------
        sample_range : tuple
            the time in seconds to extract from the positions.
        
        Returns
        -------
        NSpike
            subsampled version of initial spatial object
        """
        if sample_range is None:
            return self
        new_spatial = deepcopy(self)
        lower, upper = sample_range
        times = self._time
        sample_spatial_idx = (
            (times <= upper) & (times >= lower)).nonzero()
        new_spatial._set_time(self._time[sample_spatial_idx])
        new_spatial._set_pos_x(self._pos_x[sample_spatial_idx])
        new_spatial._set_pos_y(self._pos_y[sample_spatial_idx])
        new_spatial._set_direction(self._direction[sample_spatial_idx])
        new_spatial._set_speed(self._speed[sample_spatial_idx])
        new_spatial.set_ang_vel(self._ang_vel[sample_spatial_idx])
        # NOTE can use to set proper duration
        #new_spatial._set_duration(upper-lower)
        return new_spatial

    def set_pixel_size(self, pixel_size):
        """
        Sets the size of pixel size by which the entire foraged arena is tessellated
        
        Parameters
        ----------
        pixel_size : int
            Pixel size of the foraged arena
        Returns
        -------
        None

        """
        
        self._pixel_size = pixel_size
        
    def _set_time(self, time):
        """
        Sets the time for all the spatial samples. It also resets the `timestamp`
        or the the temporal resolution of the spatial samples and recalculates
        the sampling rate.
        
        Parameters
        ----------
        time : list or ndarray
            Timestamps of all spatial samples
            
        Returns
        -------
        None

        """
  
        
        self._time = time
        self._set_timestamp()
        self._set_sampling_rate()
        
    def _set_timestamp(self, timestamp=None):
        """
        Sets the timestamps for the spatial information. Here, it is defined as
        the temporal resolution of the spatial samples, not the happening time of
        each sample. This way it is different from NLfp and NSpike timestamp 
        definition
        
        Parameters
        ----------
        timestamp : list or ndarray
            Timestamps of all spiking waveforms
            
        Returns
        -------
        None

        """

        
        if timestamp:
            self._timestamp = timestamp
        elif np.array(self._time).any():
            self._timestamp = np.diff(np.array(self._time)).mean()
            
    def _set_sampling_rate(self, sampling_rate=None):
        """
        Sets the sampling rate of the spatial information
                
        Parameters
        ----------
        sampling_rate : int
            Sampling rate of the spatial information
 
        Returns
        -------
        None    

        """
        
        if sampling_rate:
            self._fs = sampling_rate
        elif np.array(self._time).any():
            self._fs = 1/np.diff(np.array(self._time)).mean()

    def _set_pos_x(self, pos_x):
        """
        Sets the X-coordinate of the location of the animal
                
        Parameters
        ----------
        pos_x : ndarray
            X-ccordinate of the location of the animal
 
        Returns
        -------
        None

        """
        
        self._pos_x = pos_x
        
    def _set_pos_y(self, pos_y):
        """
        Sets the Y-coordinate of the location of the animal
                
        Parameters
        ----------
        pos_y : ndarray
            Y-ccordinate of the location of the animal
 
        Returns
        -------
        None

        """
        
        self._pos_y = pos_y
        
    def _set_direction(self, direction):
        """
        Sets the head-direction of the animal
                
        Parameters
        ----------
        direction : ndarray
            Head-direction of the animal
 
        Returns
        -------
        None

        """
        self._direction = direction
        
    def _set_speed(self, speed):
        """
        Sets the speed of the animal
                
        Parameters
        ----------
        speed : ndarray
            Speed of the animal
 
        Returns
        -------
        None

        """
        self._speed = speed
        
    def set_ang_vel(self, ang_vel):
        """
        Sets the angular head velocity (AHV) of the animal
                
        Parameters
        ----------
        ang_vel : ndarray
            Angular head velocity (AHV) of the animal
 
        Returns
        -------
        None

        """
        
        self._ang_vel = ang_vel
        
    def set_border(self, border):
        """
        Sets the distance of the animal from the arena border
                
        Parameters
        ----------
        border : ndarray
            Distance of the animal from the arena border
 
        Returns
        -------
        None

        """
        
        self._border_dist = border[0]
        self._xbound = border[1]
        self._ybound = border[2]
        self._dist_map = border[3]

    def get_total_samples(self):
        """
        Returns the number of spatial samples
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Total spatial samples

        """        
        return self._time.size
    
    def get_sampling_rate(self):
        """
        Returns the sampling rate of the spatial samples
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Spatial data sampling rate

        """
    
        return self._fs
    
    def get_duration(self):
        """
        Returns the duration of the experiment
                
        Parameters
        ----------
        None
 
        Returns
        -------
        float
            Duration of the experiment

        """
        if len(self._time) == 0:
            return 0
        return self._time[-1]
        
    
    def get_pixel_size(self):
        """
        Returns the pixel size of the recorded arena
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Pixel size

        """    
        return self._pixel_size
    
    def get_time(self):
        """
        Returns the time of individual spatial samples
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Total spatial samples

        """
    
        return self._time
    
    def get_timestamp(self):
        """
        Returns the temporal resolution of spatial samples
                
        Parameters
        ----------
        None
 
        Returns
        -------
        int
            Temporal resolution of spatial samples

        """
        
        return self._timestamp
    
    def get_pos_x(self):
        """
        Returns the X-ccordinates of animal's location
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            X-coordinates of animal's location

        """
    
        return self._pos_x
    
    def get_pos_y(self):
        """
        Returns the Y-ccordinates of animal's location
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            Y-coordinates of animal's location

        """
    
        return self._pos_y
    
    def get_direction(self):
        """
        Returns head direction of the animal
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            Head direction of the animal

        """
        
        return self._direction
    
    def get_speed(self):
        """
        Returns speed of the animal
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            Speed of the animal

        """       
        return self._speed
    
    def get_ang_vel(self):
        """
        Returns angular head velocity of the animal
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            Angular head velocity of the animal

        """
        
        return self._ang_vel
    
    def get_border(self):
        """
        Returns animal's distance from the border
                
        Parameters
        ----------
        None
 
        Returns
        -------
        ndarray
            Animal's distance from the border

        """
        
        return self._border_dist, self._xbound, self._ybound, self._dist_map

    def set_spike(self, spike, **kwargs):
        """
        Adds the NSpike object to NSpatial object 
                
        Parameters
        ----------
        spike : NSpike
            NSpike object to be added to the NSpatial object. If no spike object
            is provided, a new NSpike() object is created.
        **kwargs
            Keyword argumemts for creating the new NSpike instance
 
        Returns
        -------
        None

        """

        
        if spike is isinstance(spike, NSpike):
            self.spike = spike
        else:
            cls = NSpike if not spike else spike
            spike = cls(**kwargs)
        self.spike = spike

    def set_lfp(self, lfp, **kwargs):
        """
        Adds the NLfp object to NSpatial object 
                
        Parameters
        ----------
        lfp : NLfp
            NLfp object to be added to the NSpatial object. If no spike object
            is provided, a new NLfp() object is created.
        **kwargs
            Keyword argumemts for creating the new NLfp instance
 
        Returns
        -------
        None

        """
        
        if lfp is isinstance(lfp, NLfp):
            self.lfp = lfp
        else:
            cls = NLfp if not lfp else lfp
            lfp = cls(**kwargs)
        self.lfp = lfp

    def set_spike_name(self, name=None):
        """
        Sets the name of the spike dataset
        
        Parameters
        ----------
        name : str
            Name of the spike dataset
        
        Returns
        -------
        None

        """
        
        if name is not None:
            self.spike.set_name(name)
            
    def set_spike_filename(self, filename=None):
        """
        Sets file name of the spike dataset
        
        Parameters
        ----------
        name : str
            Full file directory of the spike dataset
        
        Returns
        -------
        None
        """
        
        if filename is not None:
            self.spike.set_filename()

    def set_lfp_name(self, name=None):
        """
        Sets the name of the lfp dataset
        
        Parameters
        ----------
        name : str
            Name of the lfp dataset
        
        Returns
        -------
        None

        """
        
        self.lfp.set_name(name)
        
    def set_lfp_filename(self, filename=None):
        """
        Sets file name of the lfp dataset
        
        Parameters
        ----------
        name : str
            Full file directory of the lfp dataset
        
        Returns
        -------
        None
        """
        self.lfp.set_filename(filename)

    def set_event(self, event, **kwargs):
        """
        Sets the NEvent() object to NSpatial().         
        
        Parameters
        ----------
        event
            NEvent or its childclass or NEvent() object
        
        Returns
        -------
        NEvent()
        
        """
        
        if event is isinstance(event, NEvent):
            self.event = event
        else:
            cls = NEvent if not event else event
            event = cls(**kwargs)
        self.event = event

    def set_event_name(self, name=None):
        """
        Sets the name of the event object.         
        
        Parameters
        ----------
        name : str
            Name of the vent dataset
        
        Returns
        -------
        None
        
        """
        
        self.event.set_name(name)
        
    def set_event_filename(self, filename=None):
        """
        Sets the filename for the event
        
        Parameters
        ----------
        filename : str
            Full file of the event dataset
        
        Returns
        -------
        None
        
        """
        
        self.event.set_filename(filename)

    def set_system(self, system=None):
        """
        Sets the data format or recording system.
        
        Parameters
        ----------
        system : str
            Data format or recording system
        
        Returns
        -------
        None
        
        """
        
        if system is not None:
            self._system = system

            if self.spike:
                self.spike.set_system(system)
            if self.lfp:
                self.lfp.set_system(system)

    def load_spike(self):
        """
        Loads the composing spike object         
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """
        
        self.spike.load()
        
    def load_lfp(self):
        """
        Loads the composite lfp object         
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """

        
        self.lfp.load()

    def save_to_hdf5(self, file_name=None, system=None):
        """
        Save spatial dataset to HDF5 file         
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """

        hdf = Nhdf()
        if file_name and system:
            if os.path.exists(file_name):
                self.set_filename(file_name)
                self.set_system(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')

        hdf.save_spatial(spatial=self)
        hdf.close()

    def load(self, filename=None, system=None):
        """
        Loads the spatial object         
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """

        if system is None:
            system = self._system
        else:
            self._system = system
        if filename is None:
            filename = self._filename
        else:
            filename = self._filename
        loader = getattr(self, 'load_spatial_'+ system)
        loader(filename)
        try:
            self.smooth_speed()
        except:
            logging.warning(self.get_system() + ' files may not have speed data!')
        if not np.array(self._ang_vel).any():
            self.set_ang_vel(self.calc_ang_vel())
        self.set_border(self.calc_border())

    def load_spatial_Axona(self, file_name):
        """
        Loads Axona format spatial data to the NSpatial() object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """

        try:
            f = open(file_name, 'rt')
            self._set_data_source(file_name)
            self._set_source_format('Axona')
            while True:
                line = f.readline()
                if line == '':
                    break
                elif line.startswith('time'):
                    spatial_data = np.loadtxt(f, dtype='float', usecols=range(5))
            self._set_time(spatial_data[:, 0])
            self._set_pos_x(spatial_data[:, 1]- np.min(spatial_data[:, 1]))
            self._set_pos_y(spatial_data[:, 2]- np.min(spatial_data[:, 2]))
            self._set_direction(spatial_data[:, 3])
            self._set_speed(spatial_data[:, 4])
            f.seek(0, 0)
            pixel_size = list(map(float, re.findall(r"\d+.\d+|\d+", f.readline())))
            self.set_pixel_size(pixel_size)
            self.smooth_direction()
        except:
            logging.error('File does not exist or is open in another process!')

    def load_spatial_NWB(self, file_name):
        """
        Loads HDF5 format spatial data to the NSpatial() object
        
        Parameters
        ----------
        None
        
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
            elif '/processing/Behavioural/Position' in hdf.f:
                path = '/processing/Behavioural/Position'
                g = hdf.f[path]
                logging.info('Path for spatial data set to: ' + path)
            else:
                logging.error('Path for spatial data does not exist!')
    
            for key, value in g.attrs.items():
                _record_info[key] = value
            
            self.set_record_info(_record_info)
    
            if path+ '/'+ 'location' in g:
                g_loc = g[path+ '/'+ 'location']
                data = hdf.get_dataset(group=g_loc, name='data')
                self._set_pos_x(data[:, 0])
                self._set_pos_y(data[:, 1])
                self._set_time(hdf.get_dataset(group=g_loc, name='timestamps'))
            else:
                logging.error('Spatial location information not found!')
    
            if path+ '/'+ 'direction' in g:
                g_dir = g[path+ '/'+ 'direction']
                data = hdf.get_dataset(group=g_dir, name='data')
                self._set_direction(data)
            else:
                logging.error('Spatial direction information not found!')
    
            if path+ '/'+ 'speed' in g:
                g_speed = g[path+ '/'+ 'speed']
                data = hdf.get_dataset(group=g_speed, name='data')
                self._set_speed(data)
            else:
                logging.error('Spatial speed information not found!')
    
            if path+ '/'+ 'angular velocity' in g:
                g_ang_vel = g[path+ '/'+ 'angular velocity']
                data = hdf.get_dataset(group=g_ang_vel, name='data')
                self.set_ang_vel(data)
            else:
                self.set_ang_vel(np.array([]))
                logging.warning('Spatial angular velocity information not found, will be calculated from direction!')
    
            hdf.close()
            
    def load_spatial_Neuralynx(self, file_name):
        """
        Loads Neuralynx format spatial data to the NSpatial() object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """

        self._set_data_source(file_name)
        self._set_source_format('Neuralynx')

        # Format description for the NLX file:
        header_offset = 16*1024 # fixed for NLX files

        bytes_start_record = 2
        bytes_origin_id = 2
        bytes_videoRec_size = 2
        bytes_per_timestamp = 8
        bytes_per_bitfield = 4*400
        bytes_sncrc = 2
        bytes_per_xloc = 4
        bytes_per_yloc = 4
        bytes_per_angle = 4
        bytes_per_target = 4*50

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

            if not record_size:
                record_size = bytes_start_record+ \
                             bytes_origin_id+ \
                             bytes_videoRec_size+  \
                             bytes_per_timestamp+ \
                             bytes_per_bitfield+ \
                             bytes_sncrc+ \
                             bytes_per_xloc+ \
                             bytes_per_yloc+ \
                             bytes_per_angle+ \
                             bytes_per_target

            time_offset = bytes_start_record+ \
                             bytes_origin_id+ \
                             bytes_videoRec_size
            xloc_offset = time_offset+ \
                         bytes_per_timestamp+ \
                         bytes_per_bitfield+ \
                         bytes_sncrc
            yloc_offset = xloc_offset+bytes_per_xloc
            angle_offset = yloc_offset+bytes_per_xloc

            f.seek(0, 2)
            self._total_samples = int((f.tell()- header_offset)/ record_size)
            spatial_data = np.zeros([self._total_samples, 4])

            f.seek(header_offset)
            for i in np.arange(self._total_samples):
                sample_bytes = np.fromfile(f, dtype='uint8', count=record_size)
                spatial_data[i, 0] = int.from_bytes(sample_bytes[time_offset+ np.arange(bytes_per_timestamp)], byteorder='little', signed=False)
                spatial_data[i, 1] = int.from_bytes(sample_bytes[xloc_offset+ np.arange(bytes_per_xloc)], byteorder='little', signed=False)
                spatial_data[i, 2] = int.from_bytes(sample_bytes[yloc_offset+ np.arange(bytes_per_yloc)], byteorder='little', signed=False)
                spatial_data[i, 3] = int.from_bytes(sample_bytes[angle_offset+ np.arange(bytes_per_angle)], byteorder='little', signed=False)

            spatial_data[:, 0] /= 10**6
            spatial_data[:, 0] -= np.min(spatial_data[:, 0])
            self._timestamp = np.mean(np.diff(spatial_data[:, 0]))
            self._set_sampling_rate(1/self._timestamp)
            self._set_time(spatial_data[:, 0])
            self._set_pos_x(spatial_data[:, 1]- np.min(spatial_data[:, 1]))
            self._set_pos_y(spatial_data[:, 2]- np.min(spatial_data[:, 2]))
            self._set_direction(spatial_data[:, 3])
            # Neuralynx data does not have any speed information
            self.smooth_direction()

    def smooth_speed(self):
        """
        Smoothes the speed data using a moving-average box filter
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """
        
        self._set_speed(smooth_1d(self.get_speed(), 'b', 5))
        
    def smooth_direction(self):
        """
        Smoothes the angular head direction data using a moving circular average
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See also
        --------
        nc_circular.CircStat().circ_smooth()
        
        """
        
        cs = CircStat()
        cs.set_theta(self.get_direction())
        self._set_direction(cs.circ_smooth(filttype='b', filtsize=5))

    def calc_ang_vel(self, npoint=5):
        """
        Calculates the angular head velocity of the animal from the direction data
        Each sample is the slope of a fitted line of five directional data centred
        around current sample.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None        
        
        """
        
        theta = self.get_direction()
        ang_vel = np.zeros(theta.shape)
        N = theta.size
        L = npoint
        l = int(np.floor(L/2))
        cs = CircStat()
        for i in np.arange(l):
            y = cs.circ_regroup(theta[:L-l+ i])
            ang_vel[i] = np.polyfit(np.arange(len(y)), y, 1)[0]

        for i in np.arange(l, N- l, 1):
            y = cs.circ_regroup(theta[i-l:i+l+ 1])
            ang_vel[i] = np.polyfit(np.arange(len(y)), y, 1)[0]

        for i in np.arange(N- l, N):
            y = cs.circ_regroup(theta[i- l:])
            ang_vel[i] = np.polyfit(np.arange(len(y)), y, 1)[0]

        return ang_vel*self.get_sampling_rate()

    def calc_border(self, **kwargs):
        """
        Identifies the border of the recording arena from the trace of the foraging of the
        animal in the arena
        
        Parameters
        ----------
        **kwargs
            Keyword arguments
        
        Returns
        -------
        border_dist : ndarray
            Distance of the animal from the border at each behavioural samples
        xedges : ndarray
            Pixelated edge of the x-axis
        yedges : ndarray
            Pixelated edge of the y-axis
        dist_mat : ndarray
            A matrix of distance of each pixel of the arena from the identified
            border
        
        """
        
        # define edges
        pixel = kwargs.get('pixel', 3)
        chop_bound = kwargs.get('chop_bound', 5)

        xedges = np.arange(0, np.ceil(np.max(self._pos_x)), pixel)
        yedges = np.arange(0, np.ceil(np.max(self._pos_y)), pixel)

        tmap, yedges, xedges = histogram2d(self._pos_y, self._pos_x, yedges, xedges)
        if abs(xedges.size- yedges.size) <= chop_bound:
            tmap = chop_edges(tmap, min(tmap.shape), min(tmap.shape))[2]
        else:
            tmap = chop_edges(tmap, tmap.shape[1], tmap.shape[0])[2]

        ybin, xbin = tmap.shape

        border = np.zeros(tmap.shape)
        border[tmap > 0] = False
        border[tmap == 0] = True

        for J in np.arange(ybin):
            for I in np.arange(xbin):
                if not border[J, I] and (J == ybin-1 or J == 0 or I == xbin-1 or I == 0):
                    border[J, I] = True


        # Optimize the border
        optBorder = np.zeros(border.shape)
        for i in np.arange(border.shape[0]):
            for j in np.arange(border.shape[1]):
                if border[i, j]:
                    if i == 0: # along the 1st row
                        if border[i, j] != border[i + 1, j]:
                            optBorder[i, j] = True
                    elif j == 0: # along the 1st column
                        if border[i, j] != border[i, j + 1]:
                            optBorder[i, j] = True
                    elif i == border.shape[0] - 1: # along the last row
                        if border[i, j] != border[i - 1, j]:
                            optBorder[i, j] = True
                    elif j == border.shape[1] - 1:# along the last column
                        if border[i, j] != border[i, j - 1]:
                            optBorder[i, j] = True
                    else: # other cases
                        if (border[i, j] != border[i, j + 1]) or (border[i, j] != border[i + 1, j])\
                                or (border[i, j] != border[i, j - 1]) or (border[i, j] != border[i - 1, j]):
                            optBorder[i, j] = True

        border = optBorder

        xborder = np.zeros(tmap.shape, dtype=bool)
        yborder = np.zeros(tmap.shape, dtype=bool)
        for J in np.arange(ybin):
            xborder[J, find(border[J, :], 1, 'first')] = True # 1 added/subed to the next pixel of the traversed arena as the border
            xborder[J, find(border[J, :], 1, 'last')] = True
        for I in np.arange(xbin):
            yborder[find(border[:, I], 1, 'first'), I] = True
            yborder[find(border[:, I], 1, 'last'), I] = True

        #        self.border = border
        border = xborder | yborder
        self.tmap = tmap*self._timestamp

        distMat = np.zeros(border.shape)
        xx, yy = np.meshgrid(np.arange(xbin), np.arange(ybin))
        borderDist = np.zeros(self._time.size)

        xedges = np.arange(xbin)*pixel
        yedges = np.arange(ybin)*pixel
        xind = histogram(self._pos_x, xedges)[1]
        yind = histogram(self._pos_y, yedges)[1]

        for J in np.arange(ybin):
            for I in np.arange(xbin):
                dist_arr = (
                    np.abs(xx[border] - xx[J, I]) +
                    np.abs(yy[border] - yy[J, I]))
                if dist_arr.size == 0:
                    logging.error("could not calculate border")
                    return None, None, None, None
                tmp_dist = np.min(dist_arr)
                if find(np.logical_and(xind == I, yind == J)).size:
                    borderDist[np.logical_and(xind == I, yind == J)] = tmp_dist
                distMat[J, I] = tmp_dist
        
        dist_mat= distMat*pixel
        border_dist= borderDist*pixel
        
        return border_dist, xedges, yedges, dist_mat

    @staticmethod
    def skaggs_info(firing_rate, visit_time):
        """
        Calculates the Skaggs information content of the spatial firing
        
        Parameters
        ----------
        firing_rate : ndarray
            Firing rate of the unit at each pixelated location or binned information,
            i.e., binned speed or head-direction            
        visit_time : ndarray
            Amount of time animal spent in each pixel or bin
        
        Returns
        -------
        float
            Skaggs information content
        
        """
        
        firing_rate[np.isnan(firing_rate)] = 0
        Li = firing_rate # Lambda
        L = np.sum(firing_rate*visit_time)/ visit_time.sum()
        P = visit_time/visit_time.sum()
        
        return np.sum(P[Li > 0]*(Li[Li > 0]/L)*np.log2(Li[Li > 0]/L))

    @staticmethod
    def spatial_sparsity(firing_rate, visit_time):
        """
        Calculates the spatial sparsity of the spatial firing
        
        Parameters
        ----------
        firing_rate : ndarray
            Firing rate of the unit at each pixelated location  
        visit_time : ndarray
            Amount of time animal spent in each pixel
        
        Returns
        -------
        float
            Spatial sparsity
        
        """
        
        firing_rate[np.isnan(firing_rate)] = 0
        Li = firing_rate # Lambda
        # L = np.sum(firing_rate*visit_time)/ visit_time.sum()
        P = visit_time/visit_time.sum()
        return np.sum(P*Li)**2/ np.sum(P*Li**2)

    def speed(self, ftimes, **kwargs):
        """
        Calculates the firing rate of the unit at different binned speeds.
        
        The spike rate vs speed is fitted with a linear equation and goodness of fit
        is measured
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """

        _results = oDict()
        graph_data = {}
        update = kwargs.get('update', True) # When update = True, it will use the
                                            #results for statistics, if False,
                                            #i.e. in Multiple Regression, it will ignore updating
        binsize = kwargs.get('binsize', 1)
        min_speed, max_speed = kwargs.get('range', [0, 40])

        speed = self.get_speed()
        max_speed = min(max_speed, np.ceil(speed.max()/binsize)*binsize)
        min_speed = max(min_speed, np.floor(speed.min()/binsize)*binsize)
        bins = np.arange(min_speed, max_speed, binsize)

        vid_count = histogram(ftimes, self.get_time())[0]
        visit_time, speedInd = histogram(speed, bins)[0:2]
        visit_time = visit_time/self.get_sampling_rate()

        rate = np.array([sum(vid_count[speedInd == i]) for i in range(len(bins))])/ visit_time
        rate[np.isnan(rate)] = 0

        _results['Speed Skaggs'] = self.skaggs_info(rate, visit_time)

        rate = rate[visit_time > 1]
        bins = bins[visit_time > 1]

        fit_result = linfit(bins, rate)

        _results['Speed Pears R'] = fit_result['Pearson R']
        _results['Speed Pears P'] = fit_result['Pearson P']
        graph_data['bins'] = bins
        graph_data['rate'] = rate
        graph_data['fitRate'] = fit_result['yfit']

        if update:
            self.update_result(_results)
        return graph_data

    def angular_velocity(self, ftimes, **kwargs):
        """
        Calculates the firing rate of the unit at different binned angular head velocity.
        
        The spike rate vs speed is fitted with a linear equation individually
        for the negative and positive angular velocities, and goodness of fit
        is measured
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        _results = oDict()
        graph_data = {}
        update = kwargs.get('update', True) # When update = True, it will use the
                                            #results for statistics, if False,
                                            #i.e. in Multiple Regression, it will ignore updating
        binsize = kwargs.get('binsize', 10)
        min_vel, max_vel = kwargs.get('range', [-100, 100])
        cutoff = kwargs.get('cutoff', 10)

        ang_vel = self.get_ang_vel()

        max_vel = min(max_vel, np.ceil(ang_vel.max()/binsize)*binsize)
        min_vel = max(min_vel, np.floor(ang_vel.min()/binsize)*binsize)
        bins = np.arange(min_vel, max_vel, binsize)

        vid_count = histogram(ftimes, self.get_time())[0]
        visit_time, velInd = histogram(ang_vel, bins)[0:2]
        visit_time = visit_time/self.get_sampling_rate()

        rate = np.array([sum(vid_count[velInd == i]) for i in range(len(bins))])/ visit_time
        rate[np.isnan(rate)] = 0

        _results['speedSkaggs'] = self.skaggs_info(rate, visit_time)

        rate = rate[visit_time > 1]
        bins = bins[visit_time > 1]


        fit_result = linfit(bins[bins <= -cutoff], rate[bins <= -cutoff])

        _results['Ang Vel Left Pears R'] = fit_result['Pearson R']
        _results['Ang Vel Left Pears P'] = fit_result['Pearson P']
        graph_data['leftBins'] = bins[bins <= -cutoff]
        graph_data['leftRate'] = rate[bins <= -cutoff]
        graph_data['leftFitRate'] = fit_result['yfit']

        fit_result = linfit(bins[bins >= cutoff], rate[bins >= cutoff])

        _results['Ang Vel Right Pears R'] = fit_result['Pearson R']
        _results['Ang Vel Right Pears P'] = fit_result['Pearson P']
        graph_data['rightBins'] = bins[bins >= cutoff]
        graph_data['rightRate'] = rate[bins >= cutoff]
        graph_data['rightFitRate'] = fit_result['yfit']

        if update:
            self.update_result(_results)
        return graph_data

    def place(self, ftimes, **kwargs):
        """
        Calculates the two-dimensional firing rate of the unit with respect to
        the location of the animal in the environment. This is called Firing map.
        
        Specificity indices are measured to assess the quality of location-specific firing of the unit.
        
        This method also plot the events of spike occurring superimposed on the
        trace of the animal in the arena, commonly known as Spike Plot.
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        _results = oDict()
        graph_data = {}
        update = kwargs.get('update', True)
        pixel = kwargs.get('pixel', 3)
        chop_bound = kwargs.get('chop_bound', 5)
        filttype, filtsize = kwargs.get('filter', ['b', 5])
        lim = kwargs.get('range', [0, self.get_duration()])
        brAdjust = kwargs.get('brAdjust', True)
        thresh = kwargs.get('fieldThresh', 0.2)
        required_neighbours = kwargs.get('minPlaceFieldNeighbours', 9)
        smooth_place = kwargs.get('smoothPlace', False)
        separate_border_data = kwargs.get(
            "separateBorderData", False)

        # xedges = np.arange(0, np.ceil(np.max(self._pos_x)), pixel)
        # yedges = np.arange(0, np.ceil(np.max(self._pos_y)), pixel)

        # Update the border to match the requested pixel size
        if separate_border_data:
            self.set_border(
                separate_border_data.calc_border(**kwargs))
            times = self._time
            lower, upper = (times.min(), times.max())
            new_times = separate_border_data._time
            sample_spatial_idx = (
                (new_times <= upper) & (new_times >= lower)).nonzero()
            self._border_dist = self._border_dist[sample_spatial_idx]
        else:  
            self.set_border(self.calc_border(**kwargs))

        xedges = self._xbound
        yedges = self._ybound

        spikeLoc = self.get_event_loc(ftimes, **kwargs)[1]
        posX = self._pos_x[np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])]
        posY = self._pos_y[np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])]

        tmap, yedges, xedges = histogram2d(posY, posX, yedges, xedges)

        if tmap.shape[0] != tmap.shape[1] & np.abs(tmap.shape[0]- tmap.shape[1]) <= chop_bound:
            tmap = chop_edges(tmap, min(tmap.shape), min(tmap.shape))[2]
        tmap /= self.get_sampling_rate()

        ybin, xbin = tmap.shape
        xedges = np.arange(xbin)*pixel
        yedges = np.arange(ybin)*pixel

        spike_count = histogram2d(spikeLoc[1], spikeLoc[0], yedges, xedges)[0]
        fmap = np.divide(spike_count, tmap, out=np.zeros_like(spike_count), where=tmap != 0)

        if brAdjust:
            nfmap = fmap/ fmap.max()
            if np.sum(np.logical_and(nfmap >= 0.2, tmap != 0)) >= 0.8*nfmap[tmap != 0].flatten().shape[0]:
                back_rate = np.mean(fmap[np.logical_and(nfmap >= 0.2, nfmap < 0.4)])
                fmap -= back_rate
                fmap[fmap < 0] = 0

        if filttype is not None:
            smoothMap = smooth_2d(fmap, filttype, filtsize)
        else :
            smoothMap = fmap
        
        if smooth_place:
            pmap = smoothMap
        else:
            pmap = fmap
        
        pmap[tmap == 0] = None
        pfield, largest_group = NSpatial.place_field(
            pmap, thresh, required_neighbours)
        # if largest_group == 0:
        #     if smooth_place:
        #         info = "where the place field was calculated from smoothed data"
        #     else:
        #         info = "where the place field was calculated from raw data"
        #     logging.info(
        #         "Lack of high firing neighbours to identify place field " +
        #         info)
        centroid = NSpatial.place_field_centroid(pfield, pmap, largest_group)
        #centroid is currently in co-ordinates, convert to pixels
        centroid = centroid * pixel + (pixel * 0.5)
        #flip x and y
        centroid = centroid[::-1]
        
        p_shape = pfield.shape
        maxes = [xedges.max(), yedges.max()]
        scales = (
             maxes[0] / p_shape[1],
             maxes[1] / p_shape[0])
        co_ords = np.array(np.where(pfield == largest_group))
        boundary = [None, None]
        for i in range(2):
            j = (i + 1) % 2
            boundary[i] = (
                co_ords[j].min() * scales[i],
                np.clip((co_ords[j].max()+1) * scales[i], 0, maxes[i]))
        inside_x = (
            (boundary[0][0] <= spikeLoc[0]) &
            (spikeLoc[0] <= boundary[0][1]))
        inside_y = (
            (boundary[1][0] <= spikeLoc[1]) &
            (spikeLoc[1] <= boundary[1][1]))
        co_ords = np.nonzero(np.logical_and(inside_x, inside_y))

        if update:
            _results['Spatial Skaggs'] = self.skaggs_info(fmap, tmap)
            _results['Spatial Sparsity'] = self.spatial_sparsity(fmap, tmap)
            _results['Spatial Coherence'] = np.corrcoef(fmap[tmap != 0].flatten(), smoothMap[tmap != 0].flatten())[0, 1]
            _results['Found strong place field'] = (largest_group != 0)
            _results['Place field Centroid x'] = centroid[0]
            _results['Place field Centroid y'] = centroid[1]
            _results['Place field Boundary x'] = boundary[0]
            _results['Place field Boundary y'] = boundary[1]
            _results['Number of Spikes in Place Field'] = co_ords[0].size
            _results['Percentage of Spikes in Place Field'] = co_ords[0].size*100 / ftimes.size
            self.update_result(_results)

        smoothMap[tmap == 0] = None

        graph_data['posX'] = posX
        graph_data['posY'] = posY
        graph_data['fmap'] = fmap
        graph_data['smoothMap'] = smoothMap
        graph_data['firingMap'] = fmap
        graph_data['tmap'] = tmap
        graph_data['xedges'] = xedges
        graph_data['yedges'] = yedges
        graph_data['spikeLoc'] = spikeLoc
        graph_data['placeField'] = pfield
        graph_data['largestPlaceGroup'] = largest_group
        graph_data['placeBoundary'] = boundary
        graph_data['indicesInPlaceField'] = co_ords
        graph_data['centroid'] = centroid

        return graph_data
    
    # Created by Sean Martin: 13/02/2019
    def place_field_centroid_zscore(self, ftimes, **kwargs):
        """
        A naive method to find the centroid of the place field using the 
        z-score of the normal distribution to remove outliers, 
        and then averaging remaining locations' co-ordinates.
                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        ndarray
            The centroid of the place field
        """

        _results = oDict()
        update = kwargs.get('update', True)
        lim = kwargs.get('range', [0, self.get_duration()])
        remove_outliers = kwargs.get('remove_outliers', True)
        threshold = kwargs.get('z_threshold', 2)

        spikeLoc = self.get_event_loc(ftimes, **kwargs)[1]
        
        if remove_outliers:
            z_scores = sc.stats.zscore(spikeLoc, axis=1)
            # Filter out locations with x or y outside of 3 std devs.
            filter_array = np.logical_and((abs(z_scores[0]) < threshold).astype(bool), (abs(z_scores[1]) < threshold).astype(bool))
            spikeLoc[0] = spikeLoc[0][filter_array]
            spikeLoc[1] = spikeLoc[1][filter_array]

        centroid = np.average(spikeLoc, axis=1)
        if update:
            _results['Place field Centroid x'] = centroid[0]
            _results['Place field Centroid y'] = centroid[1]
            self.update_result(_results)
        return centroid

    def non_moving_periods(self, **kwargs):
        """
        Returns a number of tuples indicating ranges where the subject is not moving

        kwargs
        ------
        should_smooth : bool
            flags if the speed data should be smoothed, default False
        min_range : float
            the minimum amount of time that the subject should not be moving for
        moving_thresh : float
            any speed above this thresh is considered to be movement
        """
        should_smooth = kwargs.get("should_smooth", False)
        min_range = kwargs.get("min_range", 150)
        moving_thresh = kwargs.get("moving_thresh", 2.5)

        if should_smooth:
            self.smooth_speed()
        not_moving = self.get_speed() < moving_thresh

        return find_true_ranges(
            self.get_time(), not_moving, min_range)

    def get_non_moving_times(self, **kwargs):
        """ Returns the times where the subject is not moving"""
        ranges = self.non_moving_periods(**kwargs)
        time_data = [
            val for val in self.get_time()
            if any(lower <= val <= upper for (lower, upper) in ranges)
        ]
        return time_data

    def loc_time_lapse(self, ftimes, **kwargs):
        """
        Calculates the firing rate map and idnetifies the location of the spiking events
        at certain intervals. This method is useful in observing the evolution of
        unit-activity as the animal traverses the environment.
        
        Following intervals ar used:
            0-1min, 0-2min, 0-4min, 0-8min, 0-16min or 0-end depending on the recording duration
            0-1min, 1-2min, 2-4min, 4-8min, 8-16min or 16-end depending on the recording duration
                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        graph_data = oDict()
        pixel = kwargs.get('pixel', 3)
        chop_bound = kwargs.get('chop_bound', 5)
        filter = kwargs.get('filter', ['b', 5])
        brAdjust = kwargs.get('brAdjust', True)

        lim = [0, 1*60]
        graph_data['0To1min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        lim = [0, 2*60]
        graph_data['0To2min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        lim = [0, 4*60]
        graph_data['0To4min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        lim = [0, 8*60]
        graph_data['0To8min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        # 0-1min, 1-2min,  2-4min, 4-8min
        lim = [1*60, 2*60]
        graph_data['1To2min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        lim = [2*60, 4*60]
        graph_data['2To4min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        lim = [4*60, 8*60]
        graph_data['4To8min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

        ## 0-16min, 8-16min 0-20min, 16-20min

        if self.get_duration() > 8*60:
            if self.get_duration() > 16*60:
                lim = [0, 16*60]
                graph_data['0To16min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

                lim = [8*60, 16*60]
                graph_data['8To16min'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

                lim = [0, self.get_duration()]
                graph_data['0ToEnd'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

                lim = [16*60, self.get_duration()]
                graph_data['16ToEnd'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

            else:
                lim = [0, self.get_duration()]
                graph_data['0ToEnd'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

                lim = [8*60, self.get_duration()]
                graph_data['8ToEnd'] = self.place(ftimes, range=lim, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)

            return graph_data

    def hd_rate(self, ftimes, **kwargs):
        """
        Calculates the firing rate of the unit with respect to the head-direciton
        of the animal in the environment. This is calle Tuning curve.
        
        Precited firing map from the locational firing is also calculated and
        distributive ratio is measured along with the Skaggs information.
        
        Spike-plot similar to locational firing is developed but in the circular bins
        which shows the direction of the animal's head at each spike's occurring time.
                                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        _results = oDict()
        graph_data = {}
        update = kwargs.get('update', True)
        binsize = kwargs.get('binsize', 5) # in degrees
        filttype, filtsize = kwargs.get('filter', ['b', 5])
        lim = kwargs.get('range', [0, self.get_duration()])

        bins = np.arange(0, 360, binsize)

        spike_hd = self.get_event_loc(ftimes, **kwargs)[2]
        direction = self.get_direction()[np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])]

        tcount, ind, bins = histogram(direction, bins)

        tcount = tcount/ self.get_sampling_rate()

        spike_count = histogram(spike_hd, bins)[0].astype(tcount.dtype)

        hd_rate = np.divide(spike_count, tcount, out=np.zeros_like(spike_count), where=tcount != 0, casting='unsafe')

        smoothRate = smooth_1d(hd_rate, filttype, filtsize)

        if update:
            _results['HD Skaggs'] = self.skaggs_info(hd_rate, tcount)
            cs = CircStat(rho=smoothRate, theta=bins)
            results = cs.calc_stat()
            _results['HD Rayl Z'] = results['RaylZ']
            _results['HD Rayl P'] = results['RaylP']
            _results['HD von Mises K'] = results['vonMisesK']
            
            _results['HD Mean'] = results['meanTheta']
            _results['HD Mean Rate'] = results['meanRho']
            _results['HD Res Vect'] = results['resultant']

            binInterp = np.arange(360)
            rateInterp = np.interp(binInterp, bins, hd_rate)

            _results['HD Peak Rate'] = np.amax(rateInterp)
            _results['HD Peak'] = binInterp[np.argmax(rateInterp)]

            half_max = np.amin(rateInterp)+ (np.amax(rateInterp)- np.amin(rateInterp))/2
            d = np.sign(half_max - rateInterp[0:-1]) - np.sign(half_max - rateInterp[1:])
            left_possible = find(d > 0)
            if len(left_possible) > 0:
                left_idx = left_possible[0]
            else:
                left_idx = None
            
            right_possible = find(d < 0)
            if len(right_possible) > 0:
                right_idx = right_possible[-1]
            else:
                right_idx = None
            _results['HD Half Width'] = None if (not left_idx or not right_idx or left_idx > right_idx) \
                                        else binInterp[right_idx]- binInterp[left_idx]

            pixel = kwargs.get('pixel', 3)
            placeData = self.place(ftimes, pixel=pixel)
            fmap = placeData['smoothMap']
            fmap[np.isnan(fmap)] = 0
            hdPred = np.zeros(bins.size)
            for i, b in enumerate(bins):
                hdInd = np.logical_and(direction >= b, direction < b+ binsize)
                tmap = histogram2d(self.get_pos_y()[hdInd], self.get_pos_x()[hdInd], placeData['yedges'], placeData['xedges'])[0]
                tmap /= self.get_sampling_rate()
                hdPred[i] = np.sum(fmap*tmap)/ tmap.sum()

            graph_data['hdPred'] = smooth_1d(hdPred, 'b', 5)
            self.update_result(_results)

        graph_data['hd'] = direction
        graph_data['hdRate'] = hd_rate
        graph_data['smoothRate'] = smoothRate
        graph_data['tcount'] = tcount
        graph_data['bins'] = bins
        graph_data['spike_hd'] = spike_hd

        cs = CircStat()
        cs.set_theta(spike_hd)
        graph_data['scatter_radius'], graph_data['scatter_bins'] = cs.circ_scatter(bins=2, step=0.05)


        return graph_data

    def hd_rate_ccw(self, ftimes, **kwargs):
        """
        Calculates the tuning curve but split into clock-wise vs counterclockwise
        head-directional movement.
                                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        _results = oDict()
        graph_data = {}
        update = kwargs.get('update', True)
        binsize = kwargs.get('binsize', 5) # in degrees
        filttype, filtsize = kwargs.get('filter', ['b', 5])
        lim = kwargs.get('range', [0, self.get_duration()])
        thresh = kwargs.get('thresh', 30)

        edges = np.arange(0, 360, binsize)

        spikeInd, spikeLoc, spike_hd = self.get_event_loc(ftimes, **kwargs)
        vidInd = np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
        direction = self.get_direction()[vidInd]

        ccwSpike_hd = spike_hd[self.get_ang_vel()[spikeInd] < -thresh]
        cwSpike_hd = spike_hd[self.get_ang_vel()[spikeInd] > thresh]

        ccw_dir = direction[self.get_ang_vel()[vidInd] < -thresh]
        cw_dir = direction[self.get_ang_vel()[vidInd] > thresh]


        binInterp = np.arange(360)

        tcount, ind, bins = histogram(cw_dir, edges)
        tcount = tcount/ self.get_sampling_rate()
        spike_count = histogram(cwSpike_hd, edges)[0].astype(tcount.dtype)
        cwRate = np.divide(spike_count, tcount, out=np.zeros_like(spike_count), where=tcount != 0, casting='unsafe')
        cwRate = np.interp(binInterp, bins, smooth_1d(cwRate, filttype, filtsize))

        tcount, ind, bins = histogram(ccw_dir, edges)
        tcount = tcount/ self.get_sampling_rate()
        spike_count = histogram(ccwSpike_hd, edges)[0].astype(tcount.dtype)
        ccwRate = np.divide(spike_count, tcount, out=np.zeros_like(spike_count), where=tcount != 0, casting='unsafe')
        ccwRate = np.interp(binInterp, bins, smooth_1d(ccwRate, filttype, filtsize))

        if update:
            _results['HD Delta'] = binInterp[np.argmax(ccwRate)]- binInterp[np.argmax(cwRate)]
            _results['HD Peak CW'] = np.argmax(cwRate)
            _results['HD Peak CCW'] = np.argmax(ccwRate)
            _results['HD Peak Rate CW'] = np.amax(cwRate)
            _results['HD Peak Rate CCW'] = np.amax(ccwRate)
            self.update_result(_results)

        graph_data['bins'] = binInterp
        graph_data['hdRateCW'] = cwRate
        graph_data['hdRateCCW'] = ccwRate

        return graph_data

    def hd_time_lapse(self, ftimes):
        """
        Calculates the tuning curve and idnetifies the location of the spiking events
        at certain intervals. This method is useful in observing the evolution of
        unit-activity as the animal traverses the environment.
        
        Following intervals ar used:
            0-1min, 0-2min, 0-4min, 0-8min, 0-16min or 0-end depending on the recording duration
            0-1min, 1-2min, 2-4min, 4-8min, 8-16min or 16-end depending on the recording duration
                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        #### Breaking down the spike plot for firing evolution
        # 0-1min,  0-2min, 0-4min, 0-8min
        graph_data = oDict()
        lim = [0, 1*60]
        graph_data['0To1min'] = self.hd_rate(ftimes, range=lim, update=False)

        lim = [0, 2*60]
        graph_data['0To2min'] = self.hd_rate(ftimes, range=lim, update=False)

        lim = [0, 4*60]
        graph_data['0To4min'] = self.hd_rate(ftimes, range=lim, update=False)

        lim = [0, 8*60]
        graph_data['0To8min'] = self.hd_rate(ftimes, range=lim, update=False)

        # 0-1min, 1-2min,  2-4min, 4-8min
        lim = [1*60, 2*60]
        graph_data['1To2min'] = self.hd_rate(ftimes, range=lim, update=False)

        lim = [2*60, 4*60]
        graph_data['2To4min'] = self.hd_rate(ftimes, range=lim, update=False)

        lim = [4*60, 8*60]
        graph_data['4To8min'] = self.hd_rate(ftimes, range=lim, update=False)

        ## 0-16min, 8-16min 0-20min, 16-20min

        if self.get_duration() > 8*60:
            if self.get_duration() > 16*60:
                lim = [0, 16*60]
                graph_data['0To16min'] = self.hd_rate(ftimes, range=lim, update=False)

                lim = [8*60, 16*60]
                graph_data['8To16min'] = self.hd_rate(ftimes, range=lim, update=False)

                lim = [0, self.get_duration()]
                graph_data['0ToEnd'] = self.hd_rate(ftimes, range=lim, update=False)

                lim = [16*60, self.get_duration()]
                graph_data['16ToEnd'] = self.hd_rate(ftimes, range=lim, update=False)

            else:
                lim = [0, self.get_duration()]
                graph_data['0ToEnd'] = self.hd_rate(ftimes, range=lim, update=False)

                lim = [8*60, self.get_duration()]
                graph_data['8ToEnd'] = self.hd_rate(ftimes, range=lim, update=False)

            return graph_data

    def hd_shuffle(self, ftimes, **kwargs):
        """
        Shuffling analysis of the unit to see if the head-directional firing specifity
        is by chance or actually correlated to the head-direction of the animal
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        
        _results = oDict()
        graph_data = {}
        nshuff = kwargs.get('nshuff', 500)
        limit = kwargs.get('limit', 0)
        bins = kwargs.get('bins', 100)
        # limit = 0 implies enirely random shuffle, limit = 'x' implies nshuff number of shuffles in the range [-x x]
        dur = self.get_time()[-1]
        shift = nprand.uniform(low=limit- dur, high=dur- limit, size=nshuff)
        raylZ = np.zeros((nshuff,))
        vonMisesK = np.zeros((nshuff,))
        for i in np.arange(nshuff):
            shift_ftimes = ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes > dur] -= dur
            shift_ftimes[shift_ftimes < 0] += dur

            hdData = self.hd_rate(shift_ftimes, update=False)
            cs = CircStat(rho=hdData['smoothRate'], theta=hdData['bins'])
            results = cs.calc_stat()
            raylZ[i] = results['RaylZ']
            vonMisesK[i] = results['vonMisesK']

        graph_data['raylZ'] = raylZ
        graph_data['vonMisesK'] = vonMisesK
        hdData = self.hd_rate(ftimes, update=False)
        cs.set_rho(hdData['smoothRate'])
        results = cs.calc_stat()
        graph_data['refRaylZ'] = results['RaylZ']
        graph_data['refVonMisesK'] = results['vonMisesK']

        graph_data['raylZCount'], ind, graph_data['raylZEdges'] = histogram(raylZ, bins=bins)
        graph_data['raylZPer95'] = np.percentile(raylZ, 95)

        graph_data['vonMisesKCount'], ind, graph_data['vonMisesKEdges'] = histogram(vonMisesK, bins=bins)
        graph_data['vonMisesKPer95'] = np.percentile(vonMisesK, 95)

        _results['HD Shuff Rayl Z Per 95'] = np.percentile(raylZ, 95)
        _results['HD Shuff von Mises K Per 95'] = np.percentile(vonMisesK, 95)
        self.update_result(_results)

        return graph_data

    def hd_shift(self, ftimes, shift_ind=np.arange(-10, 11)):
        """
        Analysis of firing specificity of the unit with respect to animal's head
        direction to oberve whether it represents past direction or anicipates a
        future direction.
                
        Parameters
        ----------
        shift_ind : ndarray
            Index of spatial resolution shift for the spike event time. Shift -1
            implies shift to the past by 1 spatial time resolution, and +2 implies
            shift to the future by 2 spatial time resoultion.
 
        Returns
        -------
        dict
            Graphical data of the analysis
    
        """
        
        _results = oDict()
        graph_data = {}
        shift = shift_ind/self.get_sampling_rate()
        shiftlen = shift.size
        dur = self.get_time()[-1]
        delta = np.zeros((shiftlen,))
        skaggs = np.zeros((shiftlen,))
        peakRate = np.zeros((shiftlen,))

        for i in np.arange(shiftlen):
            shift_ftimes = ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes > dur] -= dur
            shift_ftimes[shift_ftimes < 0] += dur

            hdData = self.hd_rate_ccw(shift_ftimes, update=False)
            delta[i] = hdData['bins'][np.argmax(hdData['hdRateCCW'])]- hdData['bins'][np.argmax(hdData['hdRateCW'])]
            hdData = self.hd_rate(shift_ftimes, update=False)
            peakRate[i] = np.amax(hdData['smoothRate'])
            skaggs[i] = self.skaggs_info(hdData['hdRate'], hdData['tcount'])

        graph_data['delta'] = delta
        graph_data['skaggs'] = skaggs
        graph_data['peakRate'] = peakRate
        graph_data['shiftTime'] = shift*1000 # changing to milisecond

        # Find out the optimum skaggs location
        shiftUpsamp = np.arange(shift[0], shift[-1], np.mean(np.diff(shift))/10)
        skaggsUpsamp = np.interp(shiftUpsamp, shift, skaggs)
        peakRateUpsamp = np.interp(shiftUpsamp, shift, peakRate)

        dfit_result = linfit(shift, delta)
        deltaFit = dfit_result['yfit']
        sortInd = np.argsort(deltaFit)
        _results['HD ATI'] = np.interp(0, deltaFit[sortInd], shift[sortInd])*1000 if dfit_result['Pearson R'] >= 0.85 else np.nan

        graph_data['deltaFit'] = deltaFit
        imax = sg.argrelmax(skaggsUpsamp)[0]
        maxloc = find(skaggsUpsamp[imax] == skaggsUpsamp.max())
        _results['HD Opt Shift Skaggs'] = np.nan if maxloc.size != 1 else \
                    (np.nan if imax[maxloc] == 0 or imax[maxloc] == skaggsUpsamp.size else shiftUpsamp[imax[maxloc]][0]*1000) # in milisecond

        imax = sg.argrelmax(peakRateUpsamp)[0]
        maxloc = find(peakRateUpsamp[imax] == peakRateUpsamp.max())
        _results['HD Opt Shift Peak Rate'] = np.nan if maxloc.size != 1 else \
                    (np.nan if imax[maxloc] == 0 or imax[maxloc] == peakRateUpsamp.size else shiftUpsamp[imax[maxloc]][0]*1000) # in milisecond
        self.update_result(_results)

        return graph_data

    @staticmethod
    def place_field(pmap, thresh=0.2, required_neighbours=9):
        """
        Calculates a mapping over the captured arena.
        For each bin in the place map, it is assigned to an integer group.
        These groups denote which neighbouring area the bin belongs to.

        Parameters
        ----------
        pmap : ndarray
            The firing map to calculate place fields from
        thresh : float
            The fraction of the peak firing that a bin must exceed
        required_neighbours : int
            The number of adjacent bins that must be together to form a field
        """

        def alongColumn(pfield, ptag, J, I):
            """
            Iterates along the columns of the ptags to find vertical neighbours

            Parameters
            ----------
            pfield : ndarray
                The place field map, consisting of 
                1 for groups satisying rules and 0 otherwise
            ptag : ndarray
                The place field map, grouped into tags of neighbouring areas
            J : int
                The vertical index to start searching at
            I : int
                The horizontal index to start searching at
            """

            Ji = J
            Ii = I
            rows=[]
            while J+1 < ptag.shape[0]:
                if not pfield[J+1, I] or ptag[J+1, I]:
                    break
                else:
                    ptag[J+1, I] = ptag[J, I]
                    rows.append(J+1)
                    J += 1
            J = Ji
            while J-1 >0:
                if not pfield[J-1, I] or ptag[J-1, I]:
                    break
                else:
                    ptag[J-1, I] = ptag[J, I]
                    rows.append(J-1)
                    J-=1
            for J in rows:
                if J != Ji:
                    ptag = alongRows(pfield, ptag, J, Ii)
            return ptag

        def alongRows(pfield, ptag, J, I):
            """
            Iterates along the columns of the ptags, finds horizontal neighbours

            Parameters
            ----------
            pfield : ndarray
                The place field map, consisting of 
                1 for groups satisying rules and 0 otherwise
            ptag : ndarray
                The place field map, grouped into tags of neighbouring areas
            J : int
                The vertical index to start searching at
            I : int
                The horizontal index to start searching at
            """

            Ii = I
            columns=[]
            while I+1<= ptag.shape[1]:
                if not pfield[J, I+1] or ptag[J, I+1]:
                    break
                else:
                    ptag[J, I+1] = ptag[J, I]
                    columns.append(I+1)
                    I += 1
            I = Ii
            while I-1 >=0:
                if not pfield[J, I-1] or ptag[J, I-1]:
                    break
                else:
                    ptag[J, I-1] = ptag[J, I]
                    columns.append(I-1)
                I-=1
            for I in columns:
                if I!= Ii:
                    ptag = alongColumn(pfield, ptag, J, I)
            return ptag
        
        # Finding the place map firing field:
        # Rules to form a field: 
            # 1. There are sufficient spikes in bin 
            # 2. The bin shares a side with other bins which contain spikes
        
        # Apply Rule 1
        where_are_NaNs = np.isnan(pmap)
        pmap[where_are_NaNs] = 0
        pmap = pmap/pmap.max()
        weights = pmap
        pmap = pmap > thresh

        # Pad the place field with a single layer of zeros to compare neighbours
        pfield = np.zeros(np.add(pmap.shape, 2))
        pfield[1:-1, 1:-1] = pmap

        # Apply rule 2
        has_neighbour_horizontal = np.logical_or(
            pfield[0:-2, 1:-1], pfield[2:, 1:-1])
        has_neighbour_vertical = np.logical_or(
            pfield[1:-1, 0:-2], pfield[1:-1, 2:])

        # Combine rules 1 and 2
        pfield[1:-1, 1:-1] = np.logical_and(
            pmap, 
            np.logical_or(has_neighbour_horizontal, has_neighbour_vertical))
        
        # Initialise all tags to 0
        ptag = np.zeros(pfield.shape, dtype = int)

        # Find the first non zero entry of the pfield
        J, I = find2d(pfield)
        
        #Group all the neighbouring pixels
        group = 1
        for (j, i) in zip(J, I): 
            if ptag[j, i] == 0:
                ptag[j, i] = group
                group = group + 1
                # Tag all neighbours as being of the same group
                ptag = alongColumn(pfield, ptag, j, i)
        
        # Remove the padding
        ptag = ptag[1:-1, 1:-1]

        # Find the largest field, and also remove fields that are too small
        # If there are no large enough fields, label all bins as 0
        uniques, counts = np.unique(ptag[ptag > 0], return_counts=True)
        max_count, largest_group_num = 0, 0
        reduction = 0
        for unique, count in zip(uniques, counts):
            # Don't consider groups that are small
            unique = unique - reduction
            if count < required_neighbours:
                ptag[ptag == unique] = 0
                ptag[ptag > unique] = ptag[ptag > unique] - 1
                reduction = reduction + 1
            # Define the largest group to be the one with largest weight
            # Could also be the one with the largest area
            else:
                interest_weights = weights[ptag == unique]
                weight = np.sum(interest_weights)
                if weight > max_count:
                    max_count = weight
                    largest_group_num = unique

        return ptag, largest_group_num

    @staticmethod
    def place_field_centroid(pfield, fmap, group_num, **kwargs):
        """
        Calculate the centroid of a place field
    
        Parameters
        ----------
        pfield : ndarray
            Input place field consisting of a map of groups
        fmap : ndarray
            Input firing map
        group_num : int
            The group to get the centroid for
        **kwargs :
            Keyword arguments
    
        Returns
        -------
        ndarray
            A list of co-ordinates for each place field group
        """
        # For each group, get the list of co-ordinates from the pfield
        co_ords = np.array(np.where(pfield == group_num))
        weights = fmap[co_ords[0], co_ords[1]]
        return centre_of_mass(co_ords, weights, axis=1)
           
      
    def get_event_loc(self, ftimes, **kwargs):
        """
        Calculates location of the event from its timestamps.
                                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking or any other events
        **kwargs
            Keyword arguments
 
        Returns
        -------
        ndarray        
            Index of the events in spatial-timestamps
        [ Two item list containing
            ndarray
                x-coordinates of the event location
            ndarray
                y-ccordinates of the event location
        ]
        ndarray
            direction of the animal at the time of the event            
        """
        
        
        time = self.get_time()
        lim = kwargs.get('range', [0, time.max()])

        # Sean - Why is zero idx is always thrown away?
        keep_zero_idx = kwargs.get('keep_zero_idx', False)
        
        hist = histogram(
            ftimes[np.logical_and(
                    ftimes >= lim[0], ftimes < lim[1])], 
            time)
        vidInd = hist[1]

        if keep_zero_idx:
            retInd = vidInd
        else:
            retInd = vidInd[vidInd != 0]
        
        return retInd, [self._pos_x[retInd], self._pos_y[retInd]], self._direction[retInd]

    def loc_shuffle(self, ftimes, **kwargs):
        """
        Shuffling analysis of the unit to see if the locational firing specifity
        is by chance or actually correlated to the location of the animal
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        
        _results = oDict()
        graph_data = {}

        nshuff = kwargs.get('nshuff', 500)
        limit = kwargs.get('limit', 0)
        bins = kwargs.get('bins', 100)
        brAdjust = kwargs.get('brAdjust', False)
        pixel = kwargs.get('pixel', 3)
        chop_bound = kwargs.get('chop_bound', 5)
        filter = kwargs.get('filter', ['b', 5])
        # limit = 0 implies enirely random shuffle, limit = 'x' implies nshuff number of shuffles in the range [-x x]
        dur = self.get_time()[-1]
        shift = nprand.uniform(low=limit- dur, high=dur- limit, size=nshuff)
        skaggs = np.zeros((nshuff,))
        sparsity = np.zeros((nshuff,))
        coherence = np.zeros((nshuff,))
        for i in np.arange(nshuff):
            shift_ftimes = ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes > dur] -= dur
            shift_ftimes[shift_ftimes < 0] += dur

            placeData = self.place(shift_ftimes, filter=filter, \
                          chop_bound=chop_bound, pixel=pixel, brAdjust=brAdjust, update=False)
            skaggs[i] = self.skaggs_info(placeData['fmap'], placeData['tmap'])
            sparsity[i] = self.spatial_sparsity(placeData['fmap'], placeData['tmap'])
            coherence[i] = np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]

        graph_data['skaggs'] = skaggs
        graph_data['coherence'] = coherence
        graph_data['sparsity'] = sparsity

        placeData = self.place(ftimes, pixel=pixel, filter=filter, brAdjust=brAdjust,\
                              chop_bound=chop_bound, update=False)
        graph_data['refSkaggs'] = self.skaggs_info(placeData['fmap'], placeData['tmap'])
        graph_data['refSparsity'] = self.spatial_sparsity(placeData['fmap'], placeData['tmap'])
        graph_data['refCoherence'] = np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]

        graph_data['skaggsCount'], graph_data['skaggsEdges'] = np.histogram(skaggs, bins=bins)
        graph_data['skaggs95'] = np.percentile(skaggs, 95)

        graph_data['sparsityCount'], graph_data['sparsityEdges'] = np.histogram(sparsity, bins=bins)
        graph_data['sparsity05'] = np.percentile(sparsity, 5)

        graph_data['coherenceCount'], graph_data['coherenceEdges'] = np.histogram(coherence, bins=bins)
        graph_data['coherence95'] = np.percentile(coherence, 95)

        _results['Loc Skaggs 95'] = np.percentile(skaggs, 95)
        _results['Loc Sparsity 05'] = np.percentile(sparsity, 95)
        _results['Loc Coherence 95'] = np.percentile(coherence, 95)

        self.update_result(_results)

        return graph_data

    def loc_shift(self, ftimes, shift_ind=np.arange(-10, 11), **kwargs):
        """
        Analysis of firing specificity of the unit with respect to animal's location
        to oberve whether it represents past location of the animal or anicipates a
        future location.
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        shift_ind : ndarray
            Index of spatial resolution shift for the spike event time. Shift -1
            implies shift to the past by 1 spatial time resolution, and +2 implies
            shift to the future by 2 spatial time resoultion.
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        
        _results = oDict()
        graph_data = {}

        brAdjust = kwargs.get('brAdjust', False)
        pixel = kwargs.get('pixel', 3)
        chop_bound = kwargs.get('chop_bound', 5)
        _filter = kwargs.get('filter', ['b', 5])

        # limit = 0 implies enirely random shuffle, limit = 'x' implies nshuff number of shuffles in the range [-x x]
        shift = shift_ind/self.get_sampling_rate()
        shiftlen = shift.size
        dur = self.get_time()[-1]
        skaggs = np.zeros((shiftlen,))
        sparsity = np.zeros((shiftlen,))
        coherence = np.zeros((shiftlen,))

        for i in np.arange(shiftlen):
            shift_ftimes = ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes > dur] -= dur
            shift_ftimes[shift_ftimes < 0] += dur

            placeData = self.place(shift_ftimes, pixel=pixel, filter=_filter, \
                                  brAdjust=brAdjust, chop_bound=chop_bound, update=False)
            skaggs[i] = self.skaggs_info(placeData['fmap'], placeData['tmap'])
            sparsity[i] = self.spatial_sparsity(placeData['fmap'], placeData['tmap'])
            coherence[i] = np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]

        graph_data['skaggs'] = skaggs
        graph_data['sparsity'] = sparsity
        graph_data['coherence'] = coherence

        graph_data['shiftTime'] = shift

        # Find out the optimum skaggs location
        shiftUpsamp = np.arange(shift[0], shift[-1], np.mean(np.diff(shift))/4)
        skaggsUpsamp = np.interp(shiftUpsamp, shift, skaggs)
        sparsityUpsamp = np.interp(shiftUpsamp, shift, sparsity)
        coherenceUpsamp = np.interp(shiftUpsamp, shift, coherence)

        imax = sg.argrelmax(skaggsUpsamp)[0]
        maxloc = find(skaggsUpsamp[imax] == skaggsUpsamp.max())
        _results['Loc Opt Shift Skaggs'] = np.nan if maxloc.size != 1 else (np.nan if imax[maxloc] == 0 or imax[maxloc] == skaggsUpsamp.size else shiftUpsamp[imax[maxloc]])

        imin = sg.argrelmin(sparsityUpsamp)[0]
        minloc = find(sparsityUpsamp[imin] == sparsityUpsamp.min())
        _results['Loc Opt Shift Sparsity'] = np.nan if minloc.size != 1 else (np.nan if imin[minloc] == 0 or imin[minloc] == sparsityUpsamp.size else shiftUpsamp[imin[minloc]])

        imax = sg.argrelmax(coherenceUpsamp)[0]
        maxloc = find(coherenceUpsamp[imax] == coherenceUpsamp.max())
        _results['Loc Opt Shift Coherence'] = np.nan if maxloc.size != 1 else (np.nan if imax[maxloc] == 0 or imax[maxloc] == coherenceUpsamp.size else shiftUpsamp[imax[maxloc]])

        self.update_result(_results)

        return graph_data

    def loc_auto_corr(self, ftimes, **kwargs):
        """
        Calculates the two-dimensional correlation of firing map which is the
        map of the firing rate of the animal with respect to its location
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit        
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        graph_data = {}

        minPixel = kwargs.get('minPixel', 20)
        pixel = kwargs.get('pixel', 3)

        if 'update' in kwargs.keys():
            del kwargs['update']
        placeData = self.place(ftimes, update=False, **kwargs)

        fmap = placeData['smoothMap']
        fmap[np.isnan(fmap)] = 0
        leny, lenx = fmap.shape

        xshift = np.arange(-(lenx-1), lenx)
        yshift = np.arange(-(leny-1), leny)

        corrMap = np.zeros((yshift.size, xshift.size))

        for J, ysh  in enumerate(yshift):
            for I, xsh in enumerate(xshift):
                if ysh >= 0:
                    map1YInd = np.arange(ysh, leny)
                    map2YInd = np.arange(leny - ysh)
                elif ysh < 0:
                    map1YInd = np.arange(leny + ysh)
                    map2YInd = np.arange(-ysh, leny)

                if xsh >= 0:
                    map1XInd = np.arange(xsh, lenx)
                    map2XInd = np.arange(lenx - xsh)
                elif xsh < 0:
                    map1XInd = np.arange(lenx + xsh)
                    map2XInd = np.arange(-xsh, lenx)
                map1 = fmap[tuple(np.meshgrid(map1YInd, map1XInd))]
                map2 = fmap[tuple(np.meshgrid(map2YInd, map2XInd))]
                if map1.size < minPixel:
                    corrMap[J, I] = -1
                else:
                    corrMap[J, I] = corr_coeff(map1, map2)

        graph_data['corrMap'] = corrMap
        graph_data['xshift'] = xshift*pixel
        graph_data['yshift'] = yshift*pixel

        return graph_data

    def loc_rot_corr(self, ftimes, **kwargs):
        """
        Calculates the rotational correlation of the locational firing rate of the animal with
        respect to location, also called firing map    
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit        
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
    
        """    
        
        graph_data = {}

        binsize = kwargs.get('binsize', 3) #degrees
#        filttype, filtsize = kwargs.get('filter', ['b', 3])

        bins = np.arange(0, 360, binsize)
        placeData = self.place(ftimes, update=False, **kwargs)

        fmap = placeData['smoothMap']
        fmap[np.isnan(fmap)] = 0

        rotCorr = [corr_coeff(rot_2d(fmap, theta), fmap) for k, theta in enumerate(bins)]

        graph_data['rotAngle'] = bins
        graph_data['rotCorr'] = rotCorr

        return graph_data

    def border(self, ftimes, **kwargs):        
        """
        Analysis of the firing characteristic of a unit with respect to the
        environmental border
                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit        
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
    
        """
                
        _results = oDict()
        graph_data = {}

        dist, xedges, yedges, distMat = self.get_border()
        pixel = np.diff(xedges).mean()

        update = kwargs.get('update', True)
        thresh = kwargs.get('thresh', 0.2)
        cbinsize = kwargs.get('cbinsize', 5) # Circular binsize in degrees
        lim = kwargs.get('range', [0, self.get_duration()])

        steps = kwargs.get('nstep', 5)

        distBins = np.arange(dist.min(), dist.max() + pixel, pixel)

        if 'update' in kwargs.keys():
            del kwargs['update']

        placeData = self.place(ftimes, range=lim, update=False, **kwargs)
        fmap = placeData['smoothMap']

        xind = np.array([])
        yind = np.array([])
        if placeData['xedges'].max() < xedges.max():
            xind = xedges <= placeData['xedges'].max()
            xedges = xedges[xind]
        if placeData['yedges'].max() < yedges.max():
            yind = yedges <= placeData['yedges'].max()
            yedges = yedges[xind]

        if xind.any():
            distMat = distMat[:, xind]
        if yind.any():
            distMat = distMat[yind, :]

        nanInd = np.isnan(fmap)
        fmap[nanInd] = 0

        smoothRate = np.zeros(distBins.shape) # Calculated from smooth FR map not by smoothing from raw rate
        for i, edge in enumerate(distBins):
            edge_ind = distMat == edge
            if edge_ind.any() and np.logical_and(np.logical_not(nanInd), distMat == edge).any():
                smoothRate[i] = fmap[np.logical_and(np.logical_not(nanInd), distMat == edge)].mean()
#        smoothRate = smooth_1d(smoothRate, filttype, filtsize)

        fmap /= fmap.max()

        tcount = histogram(dist, distBins)[0]

        tcount = tcount/ self.get_sampling_rate()

        spikeDist = dist[self.get_event_loc(ftimes)[0]]
        spike_count = histogram(spikeDist, distBins)[0].astype(tcount.dtype)

        distRate = np.divide(spike_count, tcount, out=np.zeros_like(spike_count),\
                            where=tcount != 0, casting='unsafe') # for skaggs only
        pixelCount = histogram(distMat[np.logical_not(nanInd)], distBins)[0]
        distCount = np.divide(histogram(distMat[fmap >= thresh], distBins)[0], pixelCount, \
                             out=np.zeros_like(distBins), where=pixelCount != 0, casting='unsafe')

        circBins = np.arange(0, 360, cbinsize)

        X, Y = np.meshgrid(xedges, np.flipud(yedges))
        X = X- xedges[-1]/2
        Y = Y- yedges[-1]/2
        angDist = np.arctan2(Y, X)* 180/np.pi
        angDist[angDist < 0] += 360

        meanDist = distMat[fmap >= thresh].mean()

        cs = CircStat()
        cs.set_theta(angDist[np.logical_and(distMat <= meanDist, fmap >= thresh)])
        angDistCount = cs.circ_histogram(circBins)[0]

        # Circular linear map
        circLinMap = np.zeros((distBins.size, circBins.size))

        for i, edge in enumerate(distBins):
            cs.set_theta(angDist[np.logical_and(distMat == edge, fmap >= thresh)])
            circLinMap[i, :] = cs.circ_histogram(circBins)[0]

        perSteps = np.arange(0, 1, 1/steps)
        perDist = np.zeros(steps)

        for i in np.arange(steps):
            perDist[i] = distMat[np.logical_and(np.logical_not(nanInd), \
                        np.logical_and(fmap >= perSteps[i], fmap < perSteps[i]+ 1/steps))].mean()
        if update:
            _results['Border Skaggs'] = self.skaggs_info(distRate, tcount)

            angDistExt = np.append(angDistCount, angDistCount)

            segsize = find_chunk(angDistExt > 0)[0]
            _results['Border Ang Ext'] = max(segsize)*cbinsize

            cBinsInterp = np.arange(0, 360, 0.1)
            dBinsInterp = np.arange(0, distBins[-1] + pixel, 0.1)
            graph_data['cBinsInterp'] = cBinsInterp
            graph_data['dBinsInterp'] = dBinsInterp
            graph_data['circLinMap'] = sc.interpolate.interp2d(circBins, distBins, circLinMap, kind='cubic')(cBinsInterp, dBinsInterp)

            self.update_result(_results)

        graph_data['distBins'] = distBins
        graph_data['distCount'] = distCount
        graph_data['circBins'] = circBins
        graph_data['angDistCount'] = angDistCount
        graph_data['distRate'] = distRate
        graph_data['smoothRate'] = smoothRate
        graph_data['perSteps'] = perSteps*100
        graph_data['perDist'] = perDist

        return graph_data

    def gradient(self, ftimes, **kwargs):
        """
        Analysis of gradient cell, a unit whose firing rate gradually increases 
        as the animal traverses from the border to the cneter of the environment
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit        
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
        """
        
        _results = oDict()
        graph_data = {}

        alim = kwargs.get('alim', 0.25)
        blim = kwargs.get('blim', 0.25)
        clim = kwargs.get('clim', 0.5)

        graph_data = self.border(ftimes, **kwargs)

        x = graph_data['distBins']
        y = graph_data['smoothRate']
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]
        y = np.log(y, out=np.zeros_like(y), where=y != 0, casting='unsafe')
        ai = y.max()
        y0 = y[x == 0]
        bi = ai- y0

        d_half = x.mean()
        for i, dist in enumerate(x):
            if i < x.size-1 and (y0+ bi/2) > y[i] and (y0+ bi/2) <= y[i+1]:
                d_half = x[i:i+2].mean()

        ci = np.log(2)/d_half

        def fit_func(x, a, b, c):
            return a- b*np.exp(-c*x)

        popt, pcov = curve_fit(fit_func, x, y, \
                                p0=[ai, bi, ci], \
                                bounds=([(1- alim)*ai, (1- blim)*bi, (1- clim)*ci], \
                                [(1+ alim)*ai, (1+ blim)*bi, (1+ clim)*ci]), \
                                max_nfev=100000)
        a, b, c = popt

        y_fit = fit_func(x, *popt)


        gof = residual_stat(y, y_fit, 3)
        rateFit = np.exp(y_fit)

        graph_data['distBins'] = x
#        graph_data['smoothRate'] = y
        graph_data['rateFit'] = rateFit
        graph_data['diffRate'] = b*c*np.multiply(rateFit, np.exp(-c*x))

        _results['Grad Pearse R'] = gof['Pearson R']
        _results['Grad Pearse P'] = gof['Pearson P']
        _results['Grad adj Rsq'] = gof['adj Rsq']
        _results['Grad Max Growth Rate'] = c*np.exp(a-1)
        _results['Grad Inflect Dist'] = np.log(b)/c

        self.update_result(_results)
        return graph_data

    def grid(self, ftimes, **kwargs):        
        """
        Analysis of Grid cells characterised by formation of grid-like pattern
        of high activity in the firing-rate map        
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit   
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis
    
        """
        
        _results = oDict()
        tol = kwargs.get('angtol', 2)
        binsize = kwargs.get('binsize', 3)
        bins = np.arange(0, 360, binsize)

        graph_data = self.loc_auto_corr(ftimes, update=False, **kwargs)
        corrMap = graph_data['corrMap']
        corrMap[np.isnan(corrMap)] = 0
        xshift = graph_data['xshift']
        yshift = graph_data['yshift']

        pixel = np.int(np.diff(xshift).mean())

        ny, nx = corrMap.shape
        rpeaks = np.zeros(corrMap.shape, dtype=bool)
        cpeaks = np.zeros(corrMap.shape, dtype=bool)
        for j in np.arange(ny):
            rpeaks[j, extrema(corrMap[j, :])[1]] = True
        for i in np.arange(nx):
            cpeaks[extrema(corrMap[:, i])[1], i] = True
        ymax, xmax = find2d(np.logical_and(rpeaks, cpeaks))

        peakDist = np.sqrt((ymax- find(yshift == 0))**2+ (xmax- find(xshift == 0))**2)
        sortInd = np.argsort(peakDist)
        ymax, xmax, peakDist = ymax[sortInd], xmax[sortInd], peakDist[sortInd]

        ymax, xmax, peakDist = (ymax[1:7], xmax[1:7], peakDist[1:7]) if ymax.size >= 7 else ([], [], [])
        theta = np.arctan2(yshift[ymax], xshift[xmax])*180/np.pi
        theta[theta < 0] += 360
        sortInd = np.argsort(theta)
        ymax, xmax, peakDist, theta = (ymax[sortInd], xmax[sortInd], peakDist[sortInd], theta[sortInd])

        graph_data['ymax'] = yshift[ymax]
        graph_data['xmax'] = xshift[xmax]

        meanDist = peakDist.mean()
        X, Y = np.meshgrid(xshift, yshift)
        distMat = np.sqrt(X**2 + Y**2)/pixel

        if len(ymax) == np.logical_and(peakDist > 0.75*meanDist, peakDist < 1.25*meanDist).sum(): # if all of them are within tolerance(25%)
            maskInd = np.logical_and(distMat > 0.5*meanDist, distMat < 1.5*meanDist)
            rotCorr = np.array([corr_coeff(rot_2d(corrMap, theta)[maskInd], corrMap[maskInd]) for k, theta in enumerate(bins)])
            ramax, rimax, ramin, rimin = extrema(rotCorr)
            mThetaPk, mThetaTr = (np.diff(bins[rimax]).mean(), np.diff(bins[rimin]).mean()) if rimax.size and rimin.size else (None, None)
            graph_data['rimax'] = rimax
            graph_data['rimin'] = rimin
            graph_data['anglemax'] = bins[rimax]
            graph_data['anglemin'] = bins[rimin]
            graph_data['rotAngle'] = bins
            graph_data['rotCorr'] = rotCorr

            if mThetaPk is not None and mThetaTr is not None:
                isGrid = True if 60 - tol < mThetaPk < 60 + tol and 60 - tol < mThetaTr < 60 + tol else False
            else:
                isGrid = False

            meanAlpha = np.diff(theta).mean()
            psi = theta[np.array([2, 3, 4, 5, 0, 1])]- theta
            psi[psi < 0] += 360
            meanPsi = psi.mean()

            _results['Is Grid'] = isGrid and 120 - tol < meanPsi < 120 + tol and 60 - tol < meanAlpha < 60 + tol
            _results['Grid Mean Alpha'] = meanAlpha
            _results['Grid Mean Psi'] = meanPsi
            _results['Grid Spacing'] = meanDist*pixel
            _results['Grid Score'] = rotCorr[rimax].max()- rotCorr[rimin].min() # Difference between highest Pearson R at peaks and lowest at troughs
            _results['Grid Orientation'] = theta[0]

        else:
            _results['Is Grid'] = False

        self.update_result(_results)
        return graph_data

    def multiple_regression(self, ftimes, **kwargs):        
        """
        Multiple-rgression analysis where firing rate for each variable, namely
        location, head-direction, speed, AHV, and distance from border, are used
        to regress the instantaneous firing rate of the unit.
                
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit   
        **kwargs
            Keyword arguments
 
        Returns
        -------
        dict
            Graphical data of the analysis

        """
        
        _results = oDict()
        graph_data = oDict()
        subsampInterv = kwargs.get('subsampInterv', 0.1)
        episode = kwargs.get('episode', 120)
        nrep = kwargs.get('nrep', 1000)
        sampRate = 1/subsampInterv
        stamp = subsampInterv
 
        a_size = np.round(self.get_duration(), 4)
        time = np.linspace(
            0, a_size, endpoint=True,
            num=np.round(a_size/stamp)+1)
        time = np.round(time, 4)
        Y = histogram(ftimes, time)[0]* sampRate # Instant firing rate

        nt = time.size
        xloc, yloc, loc, hd, speed, ang_vel, distBorder = list(np.zeros((7, nt)))
        tmp = self.place(ftimes)
        placeRate, xedges, yedges = (tmp['smoothMap'], tmp['xedges'], tmp['yedges'])
        placeRate[np.isnan(placeRate)] = 0
        for i in np.arange(nt):
            ind = find(np.logical_and(self.get_time() >= time[i], self.get_time() < time[i]+ stamp))
            if len(ind) == 0:
                continue
            xloc[i] = np.median(self.get_pos_x()[ind])
            yloc[i] = np.median(self.get_pos_y()[ind])
            if histogram(yloc[i], yedges)[1] < yedges.size and histogram(xloc[i], xedges)[1] < xedges.size:
                loc[i] = placeRate[histogram(yloc[i], yedges)[1], histogram(xloc[i], xedges)[1]]
            hd[i] = np.median(self.get_direction()[ind])
            speed[i] = np.median(self.get_speed()[ind])
            ang_vel[i] = np.median(self.get_ang_vel()[ind])
            distBorder[i] = np.median(self.get_border()[0][ind])

        tmp = self.hd_rate(ftimes, update=False)
        hd_rate, hdBins = (tmp['hdRate'], tmp['bins'])
        cs = CircStat()
        cs.set_theta(hd)
        hd = hd_rate[cs.circ_histogram(hdBins)[1]] # replaced by corresponding rate
        # Speed+ ang_vel will be linearly modelled, so no transformation required; ang_vel will be replaced by the non-linear rate
        tmp = self.border(ftimes, update=False)
        borderRate, borderBins = (tmp['distRate'], tmp['distBins'])
        distBorder = borderRate[histogram(distBorder, borderBins)[1]] # replaced by corresponding rate

        ns = int(episode/stamp) # row to select in random

        X = np.vstack((loc, hd, speed, ang_vel, distBorder)).transpose()
        lm = LinearRegression(fit_intercept=True, normalize=True)

        Rsq = np.zeros((nrep, 6))
        for i in np.arange(nrep):
            ind = np.random.permutation(time.size)[:ns]
            lm.fit(X[ind, :], Y[ind])
            Rsq[i, 0] = lm.score(X[ind, :], Y[ind])
            for j in np.arange(5):
                varind = np.array([k for k in range(5) if k != j])
                lm.fit(X[np.ix_(ind, varind)], Y[ind]) #np.ix_ is used for braodcasting the index arrays
                Rsq[i, j+1] = Rsq[i, 0]- lm.score(X[np.ix_(ind, varind)], Y[ind])

        meanRsq = Rsq.mean(axis=0)
        # Regresssion parameters are alays stored in following order
        varOrder = ['Total', 'Loc', 'HD', 'Speed', 'Ang Vel', 'Dist Border']

#        graph_data['order'] = varOrder
        graph_data['Rsq'] = Rsq
        graph_data['meanRsq'] = meanRsq
        graph_data['maxRsq'] = Rsq.max(axis=0)
        graph_data['minRsq'] = Rsq.min(axis=0)
        graph_data['stdRsq'] = Rsq.std(axis=0)
        
        _results['Mult Rsq'] = meanRsq[0]
        for i, key in enumerate(varOrder):
            if i > 0:
                _results['Semi Rsq '+key] = meanRsq[i]
        self.update_result(_results)

        return graph_data

    def interdependence(self, ftimes, **kwargs):
        """
        Interdependence analysis where firing rate of each variable is predicted
        from another variable and the distributive ratio is measured between the
        predicted firing rate and the caclulated firing rate.
        
        Parameters
        ----------
        ftimes : ndarray
            Timestamps of the spiking activity of a unit
        **kwargs
            Keyword arguments
 
        Returns
        -------
        None
        
        """        

        _results = oDict()
        pixel = kwargs.get('pixel', 3)
        hdbinsize = kwargs.get('hdbinsize', 5)
        spbinsize = kwargs.get('spbinsize', 1)
        sprange = kwargs.get('sprange', [0, 40])
        abinsize = kwargs.get('abinsize', 10)
        ang_velrange = kwargs.get('ang_velrange', [-500, 500])

        placeData = self.place(ftimes, pixel=pixel, update=False)
        fmap = placeData['smoothMap']
        fmap[np.isnan(fmap)] = 0
        xloc = self.get_pos_x()
        yloc = self.get_pos_y()
        xedges = placeData['xedges']
        yedges = placeData['yedges']

        hdData = self.hd_rate(ftimes, binsize=hdbinsize, update=False)
        bins = hdData['bins']
        predRate = np.zeros(bins.size)
        for i, b in enumerate(bins):
            ind = np.logical_and(hdData['hd'] >= b, hdData['hd'] < b + hdbinsize)
            tmap = histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap /= self.get_sampling_rate()
            predRate[i] = np.sum(fmap*tmap)/ tmap.sum()
        _results['DR HP'] = np.abs(np.log((1 + hdData['smoothRate'])/ (1 + predRate))).sum()/bins.size

        spData = self.speed(ftimes, binsize=spbinsize, range=sprange, update=False)
        bins = spData['bins']
        predRate = np.zeros(bins.size)
        speed = self.get_speed()
        for i, b in enumerate(bins):
            ind = np.logical_and(speed >= b, speed < b + spbinsize)
            tmap = histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap /= self.get_sampling_rate()
            predRate[i] = np.sum(fmap*tmap)/ tmap.sum()
        _results['DR SP'] = np.abs(np.log((1 + spData['rate'])/ (1 + predRate))).sum()/bins.size

        ang_velData = self.angular_velocity(ftimes, binsize=abinsize, range=ang_velrange, update=False)
        bins = np.hstack((ang_velData['leftBins'], ang_velData['rightBins']))
        predRate = np.zeros(bins.size)
        ang_vel = self.get_ang_vel()
        for i, b in enumerate(bins):
            ind = np.logical_and(ang_vel >= b, ang_vel < b + abinsize)
            tmap = histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap /= self.get_sampling_rate()
            predRate[i] = np.sum(fmap*tmap)/ tmap.sum()
        ang_velObs = np.hstack((ang_velData['leftRate'], ang_velData['rightRate']))
        _results['DR AP'] = np.abs(np.log((1 + ang_velObs)/ (1 + predRate))).sum()/bins.size

        borderData = self.border(ftimes, update=False)
        bins = borderData['distBins']
        dbinsize = np.diff(bins).mean()
        predRate = np.zeros(bins.size)
        border = self.get_border()[0]
        for i, b in enumerate(bins):
            ind = np.logical_and(border >= b, border < b + dbinsize)
            tmap = histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap /= self.get_sampling_rate()
            predRate[i] = np.sum(fmap*tmap)/ tmap.sum()
        _results['DR BP'] = np.abs(np.log((1 + borderData['distRate']) / (1 + predRate))).sum()/bins.size

        self.update_result(_results)

#    def __getattr__(self, arg):
#        if hasattr(self.spike, arg):
#            return getattr(self.spike, arg)
#        elif hasattr(self.lfp, arg):
#            return getattr(self.lfp, arg)
