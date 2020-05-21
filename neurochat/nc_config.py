# -*- coding: utf-8 -*-
"""
This module implements Configuration Class for NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import os.path
import logging
from collections import OrderedDict as oDict

import yaml

# default parameters by analaysis type
from neurochat.nc_defaults import ANALYSES, PARAMETERS

_MAPPING_TAG = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def __dict_representer(dumper, data):
    return dumper.represent_mapping(_MAPPING_TAG, data.items())


def __dict_constructor(loader, node):
    return oDict(loader.construct_pairs(node))


yaml.add_representer(oDict, __dict_representer)
yaml.add_constructor(_MAPPING_TAG, __dict_constructor)


class Configuration(object):
    """
    The Configuration object is the placeholder for all NeuroChaT settings.

    The object consists of specification of data and analysis along with
    the parameters for each analysis type.

    It also facilitates saving these setting to a .ncfg file
    and retriving them from the .ncfg file.
    The .ncfg file is a YAML-formatted file.

    Attributes
    ----------
    filename: str
        Full file name of the configuration storage (.ncfg)
    format: str
        Recording system or format of the data file
    analysis_mode: str
        Mode of analysis in NeuroChaT.
        Options are 'Single Unit', 'Single Session' and 'Listed Units'
    mode_id: int
        Numeric ID of modes in NeuroChaT
        Respectively 0, 1, and 2 for the three modes
    graphic_format: str
        File format for output graphics. Options are 'PDF' or 'Postscript'
    unit_no: int
        Unit number to be analyzed. Used in 'Single Unit' mode
    spatial_file: str
        Full file of the spatial dataset
    spike_file: str
        Full file of the spike dataset
    lfp_file: str
        Full file of the lfp dataset
    nwb_file: str
        Full file of the NWB format dataset
    excel_dir: str
        Full file of the Excel list of unit.
        Used only in 'Listed Units' mode
    analyses: dict
        Dictionary of analysis methods as key and
        their selection as boolean values
    parameters: dict
        (key: value) pairs of parameter names and their values
    special_analysis: dict
        A dict indicating a menu analysis method.

    """

    def __init__(self, filename=[]):
        """
        See also Configuration.

        Parameters
        ----------
        filename : str
            Full file name of the configuration storage (.ncfg)

        """
        self.filename = filename
        self.format = 'Axona'
        self.analysis_mode = 'Single Unit'
        self.mode_id = 0
        self.graphic_format = 'pdf'
        self.valid_graphics = {'PDF': 'pdf', 'Postscript': 'ps'}
        self.unit_no = 0
        self.cell_type = ''
        self.spike_file = ''
        self.spatial_file = ''
        self.lfp_file = ''
        self.nwb_file = ''
        self.excel_file = ''
        self.data_directory = ''
        self.config_directory = ''
        self.analyses = {}
        self.parameters = {}
        self.special_analysis = {}

        for k, v in PARAMETERS.items():  # setting the default parameters
            self.parameters.update(v)

        for k, v in ANALYSES.items():  # setting the default analyses
            self.analyses.update({k: v})

        self.options = oDict()
        self.mode_dict = oDict([('Single Unit', 0),
                                ('Single Session', 1),
                                ('Listed Units', 2)])

    def set_special_analysis(self, in_dict={}):
        """
        Set special analysis to the input dictionary.

        Parameters
        ----------
        in_dict: dict
            Dictionary to set special analysis to.

        Returns
        -------
        None

        """
        self.special_analysis = in_dict

    def get_special_analysis(self):
        """
        Get the special analysis dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary of special analyses.

        """
        return self.special_analysis

    def set_param(self, name=None, value=None):
        """
        Set the value of a parameter.

        Parameters
        ----------
        name : str
            Name of the parameter
        value : any
            Value of the parameter

        Returns
        -------
        None

        """
        if isinstance(name, str):
            self.parameters[name] = value
        else:
            logging.error('Parameter name and/or value is inappropriate')

    def get_params(self, name=None):
        """
        Get the value of parameter.

        If a list of parameter names are provided,
        then a list of values are returned.

        Parameters
        ----------
        name : str or list of str
            Name of the parameter(s)

        Returns
        -------
        params
            Parameter value(s)

        """
        if isinstance(name, list):
            params = {}
            not_found = []
            for pname in name:
                if pname in self.parameters.keys():
                    params[pname] = self.parameters[pname]
                else:
                    not_found.append(pname)
            if not_found:
                logging.warning(
                    'Following parameters not found- ' + ','.join(not_found))
            return params
        elif isinstance(name, str):
            if name in self.parameters.keys():
                return self.parameters[name]
            else:
                logging.error(name + ' is not found in parameter list')

    def get_params_by_analysis(self, analysis=None):
        """
        Return the parameters and their values for a given analysis.

        Parameters
        ----------
        analysis : str
            Name of the analysis

        Returns
        -------
        params : dict
            Dictionary of parameters and their values

        """
        if analysis in PARAMETERS.keys():
            # get default params first
            params = PARAMETERS[analysis]
            # Update from set parameter
            params.update(self.get_params(list(params.keys())))

            return params
        else:
            logging.error(
                'Specific analysis is not found in defaults PARAMETERS!')

    def set_analysis(self, name=None, value=None):
        """
        Set the selection of an analysis.

        Parameters
        ----------
        name : str
            Name of the analysis
        value : bool
            Boolean value to indicate analysis selection

        Returns
        -------
        None

        """
        if name == 'all' and isinstance(value, bool):
            for name in self.analyses.keys():
                self.analyses[name] = value
        elif isinstance(name, str) and isinstance(value, bool):
            self.analyses[name] = value
        elif isinstance(name, list) and isinstance(value, bool):
            for n in name:
                if isinstance(n, str):
                    self.analyses[n] = value
        else:
            logging.error('Analysis name and/or value is inappropriate')

    def get_analysis(self, name=None):
        """
        Return the selection of an analysis.

        If name is 'all', selection values for all the analyses are returned.

        Parameters
        ----------
        name : str
            Name of the analysis.
            Enter 'all' for returning values for all the analyses

        Returns
        -------
        bool
            True if selected, False if not.

        """
        if name == 'all':
            return self.analyses.values()
        elif name in self.analyses.keys():
            return self.analyses[name]
        else:
            logging.error(name + ' is not found in parameter list')

    def get_param_list(self):
        """
        Return the list of all parameters.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of parameter names

        """
        return list(self.parameters.keys())

    def get_analysis_list(self):
        """
        Return a list of analyses.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of analysis names.

        """
        return list(self.analyses.keys())

    def set_data_format(self, file_format=None):
        """
        Set the format of the data or recording system.

        Parameters
        ----------
        file_format  : str
            Format of the data or recording system

        Returns
        -------
        None

        """
        if file_format:
            self.format = file_format

    def get_data_format(self):
        """
        Return the data format or recording system.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Data format or recording system

        """
        return self.format

    def set_analysis_mode(self, analysis_mode=None):
        """
        Set the mode of analysis.

        Parameters
        ----------
        analysis_mode : str
            Mode of the analysis

        Returns
        -------
        None

        """
        if analysis_mode in self.mode_dict.keys():
            self.analysis_mode = analysis_mode
            self.mode_id = self.mode_dict[analysis_mode]
        elif analysis_mode in self.mode_dict.values():
            self.mode_id = analysis_mode
            for key, val in self.mode_dict.items():
                if val == analysis_mode:
                    self.analysis_mode = key
        else:
            logging.error('No/Invalid analysis mode!')

    def get_analysis_mode(self):
        """
        Return the mode of analysis and mode ID.

        Parameters
        ----------
        None

        Returns
        -------
        (str, int)
            Analysis mode set, ID of analysis mode

        """
        return self.analysis_mode, self.mode_id

    def get_all_modes(self):
        """
        Return the analysis modes in NeuroChaT.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Modes and their IDs

        """
        return self.mode_dict

    def set_graphic_format(self, graphic_format=None):
        """
        Set output graphics file format.

        Parameters
        ----------
        graphic_format : str
            Format of output graphic export.
            Options are 'PDF' or 'Postscript'.

        Returns
        -------
        None

        """
        if graphic_format in self.valid_graphics.keys():
            self.graphic_format = self.valid_graphics[graphic_format]
        elif graphic_format in self.valid_graphics.values():
            self.graphic_format = graphic_format
        else:
            logging.error('No/Invalid graphic format!')

    def get_graphic_format(self):
        """
        Return output graphics file format.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Export format of output graphics

        """
        return self.graphic_format

    def set_unit_no(self, unit_no=None):
        """
        Set the unit no to analyse in 'Single Unit' analysis.

        Parameters
        ----------
        unit_no : int
            Unit number the user is intending to analyse

        Returns
        -------
        None

        """
        if isinstance(unit_no, int):
            self.unit_no = unit_no

    def get_unit_no(self):
        """
        Return the unit number that is already set.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Unit number

        """
        return self.unit_no

    def set_cell_type(self, cell_type=None):
        """
        Set the type of cell to analyse.

        Parameters
        ----------
        cell_type : str
            Cell type of interest

        Returns
        -------
        None

        """
        self.cell_type = cell_type

    def get_cell_type(self):
        """
        Return the type of cell set to analyse.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Cell type set for analyses

        """
        return self.cell_type

    def set_spike_file(self, spike_file=None):
        """
        Set filename of the spike data.

        Parameters
        ----------
        spike_file : str
            Filename of the spike data

        Returns
        -------
        None

        """
        if isinstance(spike_file, str):
            self.spike_file = spike_file

    def get_spike_file(self):
        """
        Return the filename of the spike data.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of spike data

        """
        return self.spike_file

    def set_spatial_file(self, spatial_file=None):
        """
        Set filename of the spatial data.

        Parameters
        ----------
        spatial_file : str
            Filename of the spatial data

        Returns
        -------
        None

        """
        if isinstance(spatial_file, str):
            self.spatial_file = spatial_file

    def get_spatial_file(self):
        """
        Return the filename of the spatial data.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the spatial data

        """
        return self.spatial_file

    def set_lfp_file(self, lfp_file=None):
        """
        Set filename of the lfp data.

        Parameters
        ----------
        lfp_file : str
            Filename of the lfp data

        Returns
        -------
        None

        """
        if isinstance(lfp_file, str):
            self.lfp_file = lfp_file

    def get_lfp_file(self):
        """
        Return the filename of the lfp data.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the lfp data

        """
        return self.lfp_file

    def set_nwb_file(self, nwb_file=None):
        """
        Set filename of the HDF5 data.

        Parameters
        ----------
        nwb_file : str
            Filename of the HDF5 data

        Returns
        -------
        None

        """
        if isinstance(nwb_file, str):
            self.nwb_file = nwb_file

    def get_nwb_file(self):
        """
        Return the filename of the HDF5 data.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the HDF5 data

        """
        return self.nwb_file

    def get_excel_file(self):
        """
        Return the filename of the Excel list.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Filename of the Excel list

        """
        return self.excel_file

    def set_excel_file(self, excel_file=None):
        """
        Set filename of the Excel list.

        Parameters
        ----------
        excel_file : str
            Filename of the Excel list

        Returns
        -------
        None

        """
        if excel_file:
            self.excel_file = excel_file
        else:
            logging.error('Invalid/No excel filename specified')

    def set_data_dir(self, directory=None):
        """
        Set the data directory.

        Parameters
        ----------
        directory : str
            Data directory

        Returns
        -------
        None

        """
        if os.path.exists(directory):
            self.data_directory = directory
        else:
            logging.error('Invalid/No directory specified')

    def get_data_dir(self):
        """
        Return the data directory.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Data directory

        """
        return self.data_directory

    def set_config_dir(self, directory=None):
        """
        Set the directory of configuration file.

        Parameters
        ----------
        directory : str
            Directory of configuration file

        Returns
        -------
        None

        """
        if os.path.exists(directory):
            self.config_directory = directory
        else:
            logging.error('Invalid/No directory specified')

    def get_config_dir(self):
        """
        Return the directory of configuration file.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Name of the configuration file

        """
        return self.config_directory

    def set_config_file(self, filename):
        """
        Set the name of the configuration file.

        Parameters
        ----------
        directory : str
            Name of configuration file

        Returns
        -------
        None

        """
        self.filename = filename

    def get_config_file(self):
        """
        Return the name of the configuration file.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Name of configuration file

        """
        return self.filename

    def save_config(self, filename=None):
        """
        Export the configuration data to .ncfg file.

        Parameters
        ----------
        filename : str
            Name of the configuration file

        Returns
        -------
        None

        """
        if filename:
            self.set_config_file(filename)
        if not self.filename:
            logging.warning('No/invalid filename')
        else:
            self._save()

    def load_config(self, filename=None):
        """
        Import the configuration data from a .ncfg file.

        Parameters
        ----------
        filename: str
            Name of the configuration file

        Returns
        -------
        None

        """
        if filename:
            self.set_config_file(filename)
        if not self.filename:
            logging.warning('No/Invalid filename')
        else:
            self._load()

    def _save(self):
        """Save configuration data to the specified .ncfg(YAML) file."""
        try:
            with open(self.filename, 'w') as f:
                settings = oDict([('format', self.format),
                                  ('analysis_mode', self.analysis_mode),
                                  ('mode_id', self.mode_id),
                                  ('graphic_format', self.graphic_format),
                                  ('unit_no', self.unit_no),
                                  ('lfp_file', self.lfp_file),
                                  ('cell_type', self.cell_type),
                                  ('spike_file', self.spike_file),
                                  ('spatial_file', self.spatial_file),
                                  ('nwb_file', self.nwb_file),
                                  ('excel_file', self.excel_file),
                                  ('data_directory', self.data_directory)])

                cfgData = oDict([('settings', settings),
                                 ('analyses', self.analyses),
                                 ('parameters', self.parameters)])

                yaml.dump(cfgData, f, default_flow_style=False)
        except Exception:
            logging.error(
                'Configuration cannot be saved in the specified file!')

    def _load(self):
        """Load configuration data from the .ncfg(YAML) file."""
        with open(self.filename, 'r') as f:
            cfgData = yaml.load(f, Loader=yaml.FullLoader)
            settings = cfgData.get('settings')
            for key, val in settings.items():
                self.__setattr__(key, val)
            self.analyses = cfgData.get('analyses')
            self.parameters = cfgData.get('parameters')
