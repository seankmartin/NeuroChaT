# -*- coding: utf-8 -*-
"""
This module implements Nhdf Class for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie

"""
import os
import io

import logging

import numpy as np

import h5py


class Nhdf(object):
    """
    The Nhdf class manages the import and export of various NeuroChaT dataset to
    a HDF5 file dataset. It also creates and manages the nomenclature for storage
    paths within the file.

    """

    def __init__(self, **kwargs):
        self._filename = kwargs.get('filename', '')
        self.f = None

        self.__type = 'hdf'

        if os.path.exists(self._filename):
            self.file()

    def get_type(self):
        """
        Returns the type of object. For Nhdf, this is always `hdf` type

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self.__type

    def get_filename(self):
        """
        Returns the full file of the HDF5 dataset

        Parameters
        ----------
        None

        Returns
        -------
        str

        """

        return self._filename

    def set_filename(self, filename=None):
        """
        Sets the full file of the HDF5 dataset

        Parameters
        ----------
        filename : str
            Filename of the HDF5 dataset

        Returns
        -------
        None

        """

        if filename:
            self._filename = filename
        try:
            self.file()
        except:
            logging.error('Invalid file!')

    def get_file_object(self):
        """
        Returns the file object that is opened using h5py

        Parameters
        ----------
        None

        Returns
        -------
        object
            h5py file object

        """
        if isinstance(self.f, io.IOBase):
            return self.f
        else:
            logging.warning(
                'The file Nhdf instance is not open yet, use Nhdf.File() method to open it!')

    def file(self):
        """
        Opens the file, and returns the file object

        Parameters
        ----------
        None

        Returns
        -------
        object
            h5py file object

        """
        self.close()
        try:
            self.f = h5py.File(self._filename, 'a')
            self.initialize()
        except:
            logging.error('Cannot open ' + self._filename)

        return self.f

    def close(self):
        """
        Closes the h5py file object

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if isinstance(self.f, h5py.File):
            self.f.close()
            self.f = None

    def initialize(self):
        """
        Initializes the basic groups for the HDF5 file

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        groups = ['acquisition', 'processing',
                  'analysis', 'epochs', 'general', 'stimulus']
        for g in groups:
            self.f.require_group(g)

    def get_groups_in_path(self, path=''):
        """
        Returns the names of groups or datasets in a path

        Parameters
        ----------
        path : str
            path to HDF5 file group

        Returns
        -------
        list
            Names of the groups or datasets in the path

        """
        items = []
        if path in self.f:
            items = list(self.f[path].keys())
        else:
            logging.warning('No groups in the path:' + path)

        return items

    @staticmethod
    def resolve_hdfname(data=None):
        """
        Resolves and returns the name of the HDF5 file from the filenames of the NeuroChaT data

        Parameters
        ----------
        data
            One of the NeuroChaT data objects

        Returns
        -------
        hdf_name : str
            Hdf5 file name

        """
        try:
            data_type = data.get_type()
        except:
            logging.error('The type of the data cannot be extracted!')

        hdf_name = None
        file_name = data.get_filename()
        system = data.get_system()
        if system == 'NWB':
            hdf_name = file_name.split('+')[0]
        if os.path.exists(file_name):
            f_path, f_name = os.path.split(file_name)
            if system == 'Axona':
                if data_type == 'spike' or data_type == 'lfp':
                    hdf_name = os.sep.join(
                        [f_path, os.path.splitext(f_name)[0] + '.hdf5'])
                elif data_type == 'spatial':
                    hdf_name = os.sep.join(
                        [f_path,
                         '_'.join(os.path.splitext(f_name)[0].split('_')[:-1]) + '.hdf5'])
            elif system == 'Neuralynx':
                hdf_name = os.sep.join(
                    [f_path, f_path.split(os.sep)[-1] + '.hdf5'])

        return hdf_name

    def resolve_datapath(self, data=None):
        """
        Resolves and returns path of the dataset from NeuroChaT data objects

        Parameters
        ----------
        data
            NeuroChaT data objects

        Returns
        -------
        str
            Path of the NeuroChaT data

        """

        # No resolution for NWB file, this function will not be called if the system == 'NWB'
        try:
            data_type = data.get_type()
        except:
            logging.error('The type of the data cannot be extracted!')
        path = None
        tag = self.get_file_tag(data)

        if data_type == 'spatial':
            path = '/processing/Behavioural/Position'
        elif tag and data_type == 'spike':
            path = '/processing/Shank/' + tag
        elif tag and data_type == 'lfp':
            path = '/processing/Neural Continuous/LFP/' + tag

        return path

    @staticmethod
    def get_file_tag(data=None):
        """
        Resolves and returns the file tag or extension to name the group of the
        neural data in the HDF5 file

        Parameters
        ----------
        data : NSpike or NLfp
            Neural data objects of NeuroChaT

        Returns
        -------
        str
            File extention (Axona) or name (Neuralynx) of the neural datasets

        """
        try:
            data_type = data.get_type()
        except:
            logging.error('The type of the data cannot be extracted!')
        # data is one of NSpike or Nlfp instance
        tag = None
        if data_type == 'spike' or data_type == 'lfp':
            f_name = data.get_filename()
            system = data.get_system()
            if system == 'NWB':
                tag = f_name.split('+')[-1].split('/')[-1]
            else:
                name, ext = os.path.splitext(os.path.basename(f_name))
                ext = ext[1:]
                if system == 'Axona':
                    tag = ext
                elif system == 'Neuralynx':
                    tag = name
        return tag

    def resolve_analysis_path(self, spike=None, lfp=None):
        """
        Resolves and returns path of the dataset where analysis results will be
        stroed. This path is also the unique unit ID.

        Parameters
        ----------
        spike : NSpike
            Spike data object
        lfp : NLfp
            Lfp data object

        Returns
        -------
        str
            Unique unit ID resolved from spike and lfp filenames which is also the
            name of the path to store the data of NeuroChaT analysis

        """
        # Each input is an object
        try:
            data_type = spike.get_type()
        except:
            logging.error('The type of the data cannot be extracted!')

        path = ''
        if data_type == 'spike':
            tag = self.get_file_tag(spike)
            if spike.get_system() == 'Axona' or not tag.startswith('TT'):
                tag = 'TT' + tag
            path += tag + '_SS_' + str(spike.get_unit_no())
        else:
            logging.error('Please specify a valid spike data!')

        try:
            data_type = lfp.get_type()
        except:
            logging.error('The type of the data cannot be extracted!')

        if data_type == 'lfp':
            path += '_' + self.get_file_tag(lfp)

        return path

    def save_dataset(self, path=None, name=None, data=None, create_group=True):
        """
        Stores a dataset to a specific path

        Parameters
        ----------
        path : str
            Path of a group in HDF5 file
        name : str
            Name of the new dataset
        data : ndarrray or list of numbers
            Data to be stored
        create_group : bool
            If True, creates a new group if the 'path' is not in the file

        Returns
        -------
        None

        """
        # Path is an abosulte path of the group where the dataset will be stored
        # Abosulte path for the dataset will be created using the group path and the name of the dataset
        # If create_group = False (default), it will check if the path exists. If not error will be generated
        # If creat_group = True, it will create the group]\
        if not path:
            logging.error('Invalid group path specified!')
        if not name:
            logging.error('Please provide a name for the dataset!')
        if (path in self.f) or create_group:
            g = self.f.require_group(path)
            if name in g:
                del g[name]
            # This conditional restricts the None data to store, need to change
            if isinstance(data, list):
                data = [np.nan if item is None else item for item in data]
                try:
                    data = np.array(data)
                except:
                    pass
            try:
                g.create_dataset(name=name, data=data)
            except:
                logging.error('Error in creating ' +
                              name + ' dataset to hdf5 file')
        else:
            logging.error('hdf5 file path can be created or restored!')

    def get_dataset(self, group=None, path='', name=''):
        """
        Stores a dataset to a specific path

        Parameters
        ----------
        group : str
            Path of a group in HDF5 file
        path : str
            Name of the member group. This path is relative to the 'group'
        name : str
            Name of the dataset

        Returns
        -------
        ndarray or numeric objects
            Value of the dataset

        """

        if isinstance(group, h5py.Group):
            g = group
        else:
            g = self.f
        if path in g:
            if isinstance(g[path], h5py.Dataset):
                return np.array(g[path])
            elif isinstance(g[path], h5py.Group):
                g = g[path]
                if name in g:
                    return np.array(g[name])
                else:
                    logging.error(
                        'Specify a valid name for the required dataset')
        elif name in g:
            return np.array(g[name])
        else:
            logging.error(path + ' not found!' +
                          'Specify a valid path or name or check if a proper group is specified!')

#    def delete_group_data(self, path = None):
#        # Deletes everything within a group, not the group itself
#        if path in self.f:
#            g = self.f[path]
#        if g.keys():
#            for node in g.keys():
#                del self.f[path+ '/'+ node]

    def save_dict_recursive(self, path=None, name=None, data=None, create_group=True):
        """
        Stores a dictionary dataset to a specific path. If the dictionary is nested,
        it creates a group for each of the outermost keys.

        Parameters
        ----------
        path : str
            Path of a group in HDF5 file
        name : str
            Name of the new dataset
        data : ndarrray or list of numbers
            Data to be stored
        create_group : bool
            If True, creates a new group if the 'path' is not in the file

        Returns
        -------
        None

        """

        if not isinstance(data, dict):
            logging.error(
                'Nhdf class method save_dict_recursive() takes only dictionary data input!')
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    self.save_dict_recursive(
                        path=path + name + '/', name=key, data=data[key], create_group=create_group)
                else:
                    self.save_dataset(path=path + name, name=key,
                                      data=value, create_group=create_group)

    def save_attributes(self, path=None, attr=None):
        """
        Stores an attribute to a group or dataset

        Parameters
        ----------
        path : str
            Path of a group or dataset in HDF5 file
        attr : dict
            Attribute names and values in a dictionary

        Returns
        -------
        None

        """
        # path has to be the absolute path of a group
        if path in self.f:
            g = self.f[path]
            if isinstance(attr, dict):
                for key, val in attr.items():
                    g.attrs[key] = val
            else:
                logging.error('Please specify the attributes in a dictionary!')
        else:
            logging.error('Please provide a valid hdf5 path!')

    def save_object(self, obj=None):
        """
        Stores a NeuroChaT dataset to the HDF5 file. It resolves the name first and
        then stores the data in the storage path

        Parameters
        ----------
        obj
            One of the NeuroChaT data types

        Returns
        -------
        None

        """
        try:
            obj_type = obj.get_type()
        except:
            logging.error(
                'Cannot get object! It is not of one of the neurochat data types!')

        try:
            fun = getattr(self, 'save_' + obj_type)
            fun(obj)
        except:
            logging.error('Failed to save the dataset!')

    def save_spatial(self, spatial=None):
        """
        Stores NSpatial() dataset to the HDF5 file

        Parameters
        ----------
        spatial : NSpatial()
            Spatial data object in NeuroChaT

        Returns
        -------
        None

        """

        # derive the path from the filename to ensure uniqueness
        self.set_filename(self.resolve_hdfname(data=spatial))
        # Get the lfp data path/group
        path = self.resolve_datapath(data=spatial)
        # delete old data
        if path in self.f:
            del self.f[path]

        # Create group afresh
        g = self.f.require_group(path)

        self.save_attributes(path=path, attr=spatial.get_record_info())

        g_loc = g.require_group(path + '/' + 'location')
        g_dir = g.require_group(path + '/' + 'direction')
        g_speed = g.require_group(path + '/' + 'speed')
        g_ang_vel = g.require_group(path + '/' + 'angular velocity')

        loc = np.empty((spatial.get_total_samples(), 2))
        loc[:, 0] = spatial.get_pos_x()
        loc[:, 1] = spatial.get_pos_y()

        g_loc.create_dataset(name='data', data=loc)
        g_loc.create_dataset(name='num_samples',
                             data=spatial.get_total_samples())
        g_loc.create_dataset(name='timestamps', data=spatial.get_time())
        #            g_loc.create_dataset(name='unit', data=spatial.getUnit(var='speed')) # Unit information needs to be included
        # need to implement the spatial.getUnit() method

        g_dir.create_dataset(name='data', data=spatial.get_direction())
        g_dir.create_dataset(name='num_samples',
                             data=spatial.get_total_samples())
        g_dir.create_dataset(name='timestamps', data=spatial.get_time())
        #            g_dir.create_dataset(name='timestamps', data=h5py.SoftLink(g_loc.name+ '/timestamps'))

        g_speed.create_dataset(name='data', data=spatial.get_speed())
        g_speed.create_dataset(
            name='num_samples', data=spatial.get_total_samples())
        g_speed.create_dataset(name='timestamps', data=spatial.get_time())

        g_ang_vel.create_dataset(name='data', data=spatial.get_ang_vel())
        g_ang_vel.create_dataset(
            name='num_samples', data=spatial.get_total_samples())
        g_ang_vel.create_dataset(name='timestamps', data=spatial.get_time())

        self.close()

    def save_lfp(self, lfp=None):
        """
        Stores NLfp() dataset to the HDF5 file

        Parameters
        ----------
        lfp : NLfp()
            LFP data object in NeuroChaT

        Returns
        -------
        None

        """

        # derive the path from the filename to ensure uniqueness
        self.set_filename(self.resolve_hdfname(data=lfp))
        # Get the lfp data path/group
        path = self.resolve_datapath(data=lfp)
        # delete old data
        if path in self.f:
            del self.f[path]

        # Create group afresh
        g = self.f.require_group(path)

        self.save_attributes(path=path, attr=lfp.get_record_info())

        g.create_dataset(name='data', data=lfp.get_samples())
        g.create_dataset(name='num_samples', data=lfp.get_total_samples())
        g.create_dataset(name='timestamps', data=lfp.get_timestamp())

        self.close()

    def save_spike(self, spike=None):
        """
        Stores NSpike() dataset to the HDF5 file

        Parameters
        ----------
        spike : NSpike()
            Spike data object in NeuroChaT

        Returns
        -------
        None

        """

        # derive the path from the filename to ensure uniqueness
        self.set_filename(self.resolve_hdfname(data=spike))
        # Get the spike data path/group
        path = self.resolve_datapath(data=spike)

        # delete old data
        if path in self.f:
            del self.f[path]

        # Create group afresh
        g = self.f.require_group(path)

        self.save_attributes(path=path, attr=spike.get_record_info())

        g_clust = g.require_group(path + '/' + 'Clustering')
        g_wave = g.require_group(path + '/' + 'EventWaveForm/WaveForm')

        # From chX dictionary, create a higher order np array

        # NC waves are stroed in waves['ch1'], waves['ch2'] etc. ways
        waves = spike.get_waveform()
        stacked_channels = np.empty((spike.get_total_spikes(
        ), spike.get_samples_per_spike(), spike.get_total_channels()))
        i = 0
        for key, val in waves.items():
            stacked_channels[:, :, i] = val
            i += 1
        g_wave.create_dataset(name='data', data=stacked_channels)
        g_wave.create_dataset(name='electrode_idx',
                              data=spike.get_channel_ids())
        g_wave.create_dataset(name='num_events', data=spike.get_total_spikes())
        g_wave.create_dataset(name='num_samples',
                              data=spike.get_samples_per_spike())
        g_wave.create_dataset(name='timestamps', data=spike.get_timestamp())

        # save Clutser number
        g_clust.create_dataset(name='cluster_nums', data=spike.get_unit_list())
        g_clust.create_dataset(name='num', data=spike.get_unit_tags())
        g_clust.create_dataset(name='times', data=spike.get_timestamp())

        self.close()

    def save_cluster(self, clust=None):
        """
        Stores NClust() dataset to the HDF5 file

        Parameters
        ----------
        clust : NClust()
            Cluster data object in NeuroChaT

        Returns
        -------
        None

        """

        # Nclust is a NSpike derivative (inherited from NSpike) to add clustering facilities to the NSpike data
        # But we will consider putting it within NSpike itself
        # This will store data to Shank's Clustering and Feature Extraction group

        logging.warning('save_cluster() method is not implemented yet!')
