# -*- coding: utf-8 -*-
"""
This module implements NeuroChaT Class for the NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import os.path
import logging
import inspect
from collections import OrderedDict as oDict
from copy import deepcopy

import numpy as np
import pandas as pd

from PyQt5 import QtCore

from neurochat.nc_utils import NLog, angle_between_points, log_exception
from neurochat.nc_utils import remove_extension
from neurochat.nc_data import NData
from neurochat.nc_spike import NSpike
from neurochat.nc_lfp import NLfp
from neurochat.nc_spatial import NSpatial
from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_hdf import Nhdf
from neurochat.nc_clust import NClust
from neurochat.nc_config import Configuration
import neurochat.nc_plot as nc_plot
import neurochat.nc_containeranalysis as nca

import matplotlib.pyplot as plt
import matplotlib.figure

from matplotlib.backends.backend_pdf import PdfPages


class NeuroChaT(QtCore.QThread):
    """
    The NeuroChaT class is the controller class in NeuroChaT software.

    The NeuroChaT class is the backend to the NeuroChaT GUI.
    It reads data, parameter and analysis specifications from the
    Configuration class and executes accordingly.
    It also interfaces the GUI to the rest of the NeuroChaT elements.

    Attributes
    ----------
    ndata : NData
        NData object to store neural data.
    config : Configuration
        Configuration object
    log : NLog
        Central logger object
    hdf : Nhdf
        Nhdf object
    """

    finished = QtCore.pyqtSignal()

    def __init__(self, config=Configuration(), data=NData(), parent=None):
        """See NeuroChaT class description."""
        super().__init__(parent)
        self.ndata = data
        self.config = config
        self.log = NLog()
        self.hdf = Nhdf()
        self.reset()

    def reset(self):
        """
        Reset NeuroChaT's internal attributes.

        This prepares it for another set of analysis or a new session.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.__count = 0
        self.nwb_files = []
        self.graphic_files = []
        self.cellid = []
        self.results = []
        self.save_to_file = False
        self._pdf_file = None

        if not self.get_graphic_format():
            self.set_graphic_format('PDF')
        nc_plot.set_backend(self.get_graphic_format())

    def get_output_files(self):
        """
        Return a DataFrame of output graphic files and HDF5 files.

        This should be called after thecompletion of the analysis.
        Index are the unit IDs of the analysed units.

        Parameters
        ----------
        None

        Returns
        -------
        op_files : pandas.DataFrame
            Column 1 contains the name of the output graphic files.
            Column 2 gives the name of the NWB files

        """
        op_files = {'Graphics Files': self.graphic_files,
                    'NWB Files': self.nwb_files}
        op_files = pd.DataFrame.from_dict(op_files)
        op_files.index = self.cellid
        op_files = op_files[['Graphics Files', 'NWB Files']]

        return op_files

    def update_results(self, _results):
        """
        Update the results with new analysis results.

        Parameters
        ----------
        _results : OrderedDict
            Dictionary of the new results

        Returns
        -------
        None

        """
        # without copy, list contains a reference to the original dictionary
        # and old results are replaced by the new one
        self.results.append(_results.copy())

    def get_results(self):
        """
        Return the parametric results of the analyses.

        Parameters
        ----------
        None

        Returns
        -------
        results : OrderedDict
            Parametric results of the analysis
        """
        try:
            keys = []
            for d in self.results:
                [keys.append(k) for k in list(d.keys()) if k not in keys]
            results = pd.DataFrame(self.results, columns=keys)
            results.index = self.cellid
        except Exception as ex:
            log_exception(
                ex, "Error in getting results")

        return results

    def open_pdf(self, filename=None):
        """
        Open the PDF file object using PdfPages.

        PdfPages is from matplotlib.backends.backend_pdf.

        Parameters
        ----------
        filename : str
            Filename of the PDF output

        Returns
        -------
        None

        """
        if filename is not None:
            words = filename.split(os.sep)
            directory = os.sep.join(words[:-1])
            if os.path.exists(directory):
                self._pdf_file = filename  # Current PDF file being handled
                try:
                    self.pdf = PdfPages(self._pdf_file)
                    self.save_to_file = True
                except PermissionError:
                    logging.error(
                        "Please close PDF with name {} before writing to it".format(
                            self._pdf_file))
                    self.save_to_file = False
                    self._pdf_file = None
            else:
                self.save_to_file = False
                self._pdf_file = None
                logging.error('Cannot create PDF, file path is invalid')
        else:
            logging.error('No valid PDf file is specified')

    def close_pdf(self):
        """
        Close the PDF file object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self._pdf_file is not None:
            self.pdf.close()
            logging.info('Output graphics saved to ' + self._pdf_file)
        else:
            logging.warning('No PDF file for graphic output!')

    def close_fig(self, fig):
        """
        Close a matplotlib.figure.Figure() object after saving it.

        These figures are saved to the output PDF stored on this object.
        If a tuple or list of such figures are provided
        each of them are saved and closed accordingly.

        Parameters
        ----------
        fig
           matplotlib.figure.Figure() or a list or tuple of them.

        Returns
        -------
        None

        """
        if isinstance(fig, (tuple, list)):
            for f in fig:
                if isinstance(f, matplotlib.figure.Figure):
                    if self.save_to_file:
                        try:
                            self.pdf.savefig(f, dpi=400)
                        except PermissionError:
                            logging.error(
                                "Please close pdf before saving output to it")
                    plt.close(f)
                else:
                    logging.error('Invalid matplotlib.figure instance')
        elif isinstance(fig, matplotlib.figure.Figure):
            if self.save_to_file:
                try:
                    self.pdf.savefig(fig)
                except PermissionError:
                    logging.error(
                        "Please close pdf before saving output to it")
            plt.close(fig)
        else:
            logging.error('Invalid matplotlib.figure instance')

    def run(self):
        """
        After calling start(), the NeuroChaT thread calls this function.

        It verifies the input specifications and calls the mode() method.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.reset()
        verified = True
        no_spike = False
        no_spatial = False
        no_lfp = False

        # Handles menu functions
        if not any(self.get_analysis('all')):
            verified = False
            special_analysis = self.get_special_analysis()
            if special_analysis:
                key = special_analysis["key"]
                logging.info("Starting special analysis {}".format(
                    key))
                if key == "place_cell_plots":
                    self.place_cell_plots(
                        special_analysis["directory"],
                        special_analysis["dpi"])
                elif key == "angle_calculation":
                    self.open_pdf(special_analysis["pdf_name"])
                    self.angle_calculation(special_analysis["excel_file"])
                    self.close_pdf()
            else:
                logging.error('No analysis method has been selected')
        else:
            mode_id = self.get_analysis_mode()[1]
            if (
                (mode_id == 0 or mode_id == 1) and
                (self.get_data_format() == 'Axona' or
                    self.get_data_format() == 'Neuralynx')):
                if not os.path.isfile(self.get_spike_file()):
                    no_spike = True

                if not os.path.isfile(self.get_spatial_file()):
                    no_spatial = True

                if not os.path.isfile(self.get_lfp_file()):
                    no_lfp = True

                if no_spike and no_lfp:
                    verified = False
                    name_spike = (
                        "None" if self.get_spike_file() == ""
                        else self.get_spike_file())
                    name_lfp = (
                        "None" if self.get_lfp_file() == ""
                        else self.get_lfp_file())
                    logging.error(
                        "No spike or LFP files found, respectively: {} and {}".format(
                            name_spike, name_lfp))

            elif mode_id == 2:
                if not os.path.isfile(self.get_excel_file()):
                    verified = False
                    logging.error('Excel file does not exist')

        if verified:
            self.__count = 0
            self.ndata.set_data_format(self.get_data_format())
            self.mode()
        self.finished.emit()

    def mode(self):
        """
        Read data and perform analysis based on the mode set in config.

        This is the principle method in NeuroChaT.
        The method reads the specifications and analyses data according to the
        mode that is set in the Configuration file.

        This sets the input and output data files and sets the NData() object.
        After this, it calls the execute() method for running the analyses.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        info = {'spat': [], 'spike': [], 'unit': [],
                'lfp': [], 'nwb': [], 'graphics': [], 'cellid': []}
        mode_id = self.get_analysis_mode()[1]

        # All the cells in the same tetrode will use the same lfp channel
        if mode_id == 0 or mode_id == 1:
            spatial_file = self.get_spatial_file()
            spike_file = self.get_spike_file()
            lfp_file = self.get_lfp_file()

            using_nwb = (self.get_data_format() == "NWB")

            if using_nwb:
                _spike_exists = self.hdf.path_exists(spike_file)
            else:
                _spike_exists = os.path.isfile(spike_file)

            if _spike_exists:
                self.ndata.set_spike_file(spike_file)
                self.ndata.load_spike()

                units = (
                    [self.get_unit_no()]
                    if mode_id == 0 else self.ndata.get_unit_list())
                if not units:
                    logging.error('No unit found in analysis')
                else:
                    for unit_no in units:
                        info['spat'].append(spatial_file)
                        info['spike'].append(spike_file)
                        info['unit'].append(unit_no)
                        info['lfp'].append(lfp_file)
            else:
                info['spat'].append(spatial_file)
                info['spike'].append(spike_file)
                info['lfp'].append(lfp_file)
                info['unit'].append(0)

        # Read files from an excel list
        elif mode_id == 2:
            try:
                excel_file = self.get_excel_file()
                if os.path.exists(excel_file):
                    excel_info = pd.read_excel(excel_file)
                    excel_info = excel_info.replace(
                        to_replace=np.nan, value="__EMPTY__")
                    for pd_row in excel_info.itertuples():
                        row = []
                        for val in pd_row:
                            row.append(val)
                        if row[1] == "__EMPTY__":
                            logging.error("Directory is not set in excel file")
                            raise ValueError(
                                "Directory is not set in excel file")
                        if row[2] == "__EMPTY__":
                            if self.get_data_format() == 'NWB':
                                logging.error(
                                    "HDF5 filename is not set in excel file")
                                raise ValueError(
                                    "HDF5 filename is not set in excel file")
                            else:
                                row[2] = ".no_spatial.None"
                        if row[3] == "__EMPTY__":
                            row[3] = ".no_spike.NONE"
                        if row[4] == "__EMPTY__":
                            row[4] = 0
                        if row[5] == "__EMPTY__":
                            row[5] = ".no_lfp.NONE"

                        spike_file = row[1] + os.sep + row[3]
                        unit_no = int(row[4])
                        lfp_id = str(row[5])

                        if self.get_data_format() == 'Axona':
                            end = "" if row[2][-4:] == ".txt" else ".txt"
                            spatial_file = row[1] + os.sep + row[2] + end
                            if os.path.isfile(spike_file):
                                lfp_file = remove_extension(
                                    spike_file) + lfp_id
                            else:
                                lfp_file = row[1] + os.sep + row[5]

                        elif self.get_data_format() == 'Neuralynx':
                            spatial_file = row[1] + os.sep + row[2] + '.nvt'
                            lfp_file = row[1] + os.sep + lfp_id + '.ncs'

                        elif self.get_data_format() == 'NWB':
                            # excel list: directory| hdf5 file name w/o extension|
                            # spike group| unit_no| lfp group
                            hdf_name = row[1] + os.sep + row[2] + '.hdf5'
                            spike_file = (
                                hdf_name + '+/processing/Shank/' + row[3])
                            spatial_file = (
                                hdf_name + '+/processing/Behavioural/Position')
                            lfp_file = (
                                hdf_name + '+/processing/Neural Continuous/LFP/'
                                + lfp_id)

                        info['spat'].append(spatial_file)
                        info['spike'].append(spike_file)
                        info['unit'].append(unit_no)
                        info['lfp'].append(lfp_file)
            except BaseException as e:
                log_exception(e, "Parsing excel file")
                logging.warning(
                    "Please check if the data format is set correctly.")
                return

        if info['unit']:
            last_used_info = {
                'spat': None,
                'spike': None,
                'lfp': None,
            }
            for i, unit_no in enumerate(info['unit']):
                do_border = False
                data_for_hdf = None
                logging.info('Starting a new unit ({})...'.format(unit_no))
                using_nwb = (self.get_data_format() == "NWB")
                if using_nwb:
                    _spat_exists = self.hdf.path_exists(info['spat'][i])
                    _lfp_exists = self.hdf.path_exists(info['lfp'][i])
                    _spike_exists = self.hdf.path_exists(info['spike'][i])
                else:
                    _spat_exists = os.path.isfile(info['spat'][i])
                    _lfp_exists = os.path.isfile(info['lfp'][i])
                    _spike_exists = os.path.isfile(info['spike'][i])
                if _spat_exists:
                    if last_used_info['spat'] == info['spat'][i]:
                        logging.info(
                            "Using loaded spatial file {}".format(info['spat'][i]))
                    else:
                        logging.info(
                            "Loading spatial file {}".format(info['spat'][i]))
                        self.ndata.set_spatial_file(info['spat'][i])
                        self.ndata.spatial.load()
                        last_used_info['spat'] = info['spat'][i]
                        do_border = True
                else:
                    logging.warning(
                        'Spatial data does not exist or was not selected')
                    self.ndata.spatial = NSpatial(name='S0')
                    self.ndata.spatial.set_filename(".no_spatial.NONE")

                if _lfp_exists:
                    if last_used_info['lfp'] == info['lfp'][i]:
                        logging.info(
                            "Using loaded lfp file {}".format(info['lfp'][i]))
                    else:
                        logging.info(
                            "Loading LFP file {}".format(info['lfp'][i]))
                        self.ndata.set_lfp_file(info['lfp'][i])
                        self.ndata.lfp.load()
                        last_used_info['lfp'] = info['lfp'][i]
                    data_for_hdf = self.ndata.lfp
                else:
                    logging.warning(
                        'lfp data does not exist or was not selected')
                    self.ndata.lfp = NLfp(name='L0')
                    self.ndata.lfp.set_filename(".no_lfp.NONE")

                if _spike_exists:
                    if last_used_info['spike'] == info['spike'][i]:
                        logging.info(
                            "Using loaded spike file {}".format(info['spike'][i]))
                    else:
                        logging.info(
                            "Loading spike file {}".format(info['spike'][i]))
                        self.ndata.set_spike_file(info['spike'][i])
                        self.ndata.spike.load()
                        last_used_info['spike'] = info['spike'][i]
                    data_for_hdf = self.ndata.spike
                else:
                    logging.warning(
                        'Spike data does not exist or was not selected')
                    self.ndata.spike = NSpike(name='C0')
                    self.ndata.spike.set_filename(".no_spike.NONE")

                self.ndata.set_unit_no(info['unit'][i])

                self.ndata.reset_results()

                if data_for_hdf is None:
                    logging.error("Could not analyse this dataset")
                    continue

                cell_id = self.hdf.resolve_analysis_path(
                    spike=self.ndata.spike, lfp=self.ndata.lfp)
                nwb_name = self.hdf.resolve_hdfname(
                    data=data_for_hdf)
                pdf_name = (
                    remove_extension(nwb_name, keep_dot=False) +
                    '_' + cell_id + '.' + self.get_graphic_format())

                info['nwb'].append(nwb_name)
                info['cellid'].append(cell_id)
                info['graphics'].append(pdf_name)

                self.open_pdf(pdf_name)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.text(0.1, 0.6, 'Cell ID = ' + cell_id + '\n' +
                        'HDF5 file = ' + nwb_name.split(os.sep)[-1] + '\n' +
                        'Graphics file = ' + pdf_name.split(os.sep)[-1],
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        clip_on=True)
                ax.set_axis_off()
                self.close_fig(fig)

                # Set and open hdf5 file for saving graph data
                # within self.execute()
                self.hdf.set_filename(nwb_name)
                if '/analysis/' + cell_id in self.hdf.f:
                    del self.hdf.f['/analysis/' + cell_id]
                self.execute(name=cell_id, do_border=do_border)

                self.close_pdf()

                _results = self.ndata.get_results()

                self.update_results(_results)
                self.hdf.save_dict_recursive(
                    path='/analysis/' + cell_id + '/',
                    name='results', data=_results)

                self.hdf.close()
                self.ndata.save_to_hdf5()  # Saving data to hdf file

                self.__count += 1
                logging.info('Units already analysed = ' + str(self.__count))

        logging.info('Total cells analysed: ' + str(self.__count))
        self.cellid = info['cellid']
        self.nwb_files = info['nwb']
        self.graphic_files = info['graphics']

    def execute(self, name=None, do_border=True):
        """
        Execute each analysis that is selected in the configuration.

        This also exports the plot data from individual analyses to the
        hdf file and figures to the graphics file that are set in the mode()
        method.

        Parameters
        ----------
        name : str, optional
            Name of the unit or the unique unit ID. Defaults to None.
        do_border : bool, optional
            If true, calculate the border. Defaults to True.

        Returns
        -------
        None

        """
        if do_border:
            try:
                logging.info('Calculating environmental border...')
                self.set_border(self.calc_border())

            except BaseException as ex:
                logging.warning(
                    'Border calculation was not properly completed!')

        if self.get_analysis('wave_property'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for wave property analysis")
            else:
                logging.info('Assessing waveform properties...')
                try:
                    graph_data = self.wave_property()  # gd = graph_data
                    fig = nc_plot.wave_property(
                        graph_data, [int(self.get_total_channels() / 2), 2])
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/waveProperty/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Assessing waveform property')

        if self.get_analysis('isi'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for isi analysis")
            else:
                logging.info(
                    'Calculating inter-spike interval distribution...')
                try:
                    params = self.get_params_by_analysis('isi')
                    graph_data = self.isi(
                        bins=int(params['isi_length'] / params['isi_bin']),
                        bound=[0, params['isi_length']],
                        refractory_threshold=params['isi_refractory'])
                    fig = nc_plot.isi(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/isi/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(
                        ex, 'Assessing interspike interval distribution')

        if self.get_analysis('isi_corr'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for isi correlation analysis")
            else:
                logging.info(
                    'Calculating inter-spike interval autocorrelation histogram...')
                try:
                    params = self.get_params_by_analysis('isi_corr')

                    graph_data = self.isi_corr(
                        bins=params['isi_corr_bin_long'],
                        bound=[
                            -params['isi_corr_len_long'],
                            params['isi_corr_len_long']])
                    fig = nc_plot.isi_corr(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/isiCorrLong/', graph_data=graph_data)
                    # Autocorr 10ms
                    graph_data = self.isi_corr(
                        bins=params['isi_corr_bin_short'],
                        bound=[
                            -params['isi_corr_len_short'],
                            params['isi_corr_len_short']])
                    fig = nc_plot.isi_corr(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/isiCorrShort/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Assessing ISI autocorrelation')

        if self.get_analysis('theta_cell'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for theta cell analysis")
            else:
                logging.info('Estimating theta-modulation index...')
                try:
                    params = self.get_params_by_analysis('theta_cell')

                    graph_data = self.theta_index(
                        start=[
                            params['theta_cell_freq_start'],
                            params['theta_cell_tau1_start'],
                            params['theta_cell_tau2_start']],
                        lower=[
                            params['theta_cell_freq_min'], 0, 0],
                        upper=[
                            params['theta_cell_freq_max'],
                            params['theta_cell_tau1_max'],
                            params['theta_cell_tau2_max']],
                        bins=params['isi_corr_bin_long'],
                        bound=[
                            -params['isi_corr_len_long'],
                            params['isi_corr_len_long']])
                    fig = nc_plot.theta_cell(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/theta_cell/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Theta-index analysis')

        if self.get_analysis('theta_skip_cell'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for theta skip analysis")
            else:
                logging.info('Estimating theta-skipping index...')
                try:
                    params = self.get_params_by_analysis('theta_cell')

                    graph_data = self.theta_skip_index(
                        start=[
                            params['theta_cell_freq_start'],
                            params['theta_cell_tau1_start'],
                            params['theta_cell_tau2_start']],
                        lower=[
                            params['theta_cell_freq_min'], 0, 0],
                        upper=[
                            params['theta_cell_freq_max'],
                            params['theta_cell_tau1_max'],
                            params['theta_cell_tau2_max']],
                        bins=params['isi_corr_bin_long'],
                        bound=[
                            -params['isi_corr_len_long'],
                            params['isi_corr_len_long']])
                    fig = nc_plot.theta_cell(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/theta_skip_cell/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Theta-skipping cell index analysis')

        if self.get_analysis('burst'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for burst analysis")
            else:
                logging.info('Analysing bursting properties...')
                try:
                    params = self.get_params_by_analysis('burst')

                    self.burst(
                        burst_thresh=params['burst_thresh'],
                        ibi_thresh=params['ibi_thresh'])

                except BaseException as ex:
                    log_exception(ex, 'Analysing bursting properties')

        if self.get_analysis('speed'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for speed analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for speed analysis")
            else:
                logging.info('Calculating spike-rate vs running speed...')
                try:
                    params = self.get_params_by_analysis('speed')

                    graph_data = self.speed(
                        range=[params['speed_min'], params['speed_max']],
                        binsize=params['speed_bin'],
                        update=True)
                    fig = nc_plot.speed(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/speed/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Analysis of spike rate vs speed')

        if self.get_analysis('ang_vel'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for angular velocity analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for angular velocity analysis")
            else:
                logging.info(
                    'Calculating spike-rate vs angular head velocity...')
                try:
                    params = self.get_params_by_analysis('ang_vel')

                    graph_data = self.angular_velocity(
                        range=[params['ang_vel_min'], params['ang_vel_max']],
                        binsize=params['ang_vel_bin'],
                        cutoff=params['ang_vel_cutoff'], update=True)
                    fig = nc_plot.angular_velocity(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/ang_vel/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(
                        ex, 'Analysis of spike rate vs angular velocity')

        if self.get_analysis('hd_rate'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for head-directional analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for head-directional analysis")
            else:
                logging.info('Assessing head-directional tuning...')
                try:
                    params = self.get_params_by_analysis('hd_rate')

                    hdData = self.hd_rate(
                        binsize=params['hd_bin'],
                        filter=['b', params['hd_rate_kern_len']],
                        pixel=params['loc_pixel_size'],
                        update=True)
                    fig = nc_plot.hd_firing(hdData)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/hd_rate/', graph_data=hdData)

                    hdData = self.hd_rate_ccw(
                        binsize=params['hd_bin'],
                        filter=['b', params['hd_rate_kern_len']],
                        thresh=params['hd_ang_vel_cutoff'],
                        pixel=params['loc_pixel_size'],
                        update=True)
                    fig = nc_plot.hd_rate_ccw(hdData)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/hd_rate_CCW/', graph_data=hdData)

                except BaseException as ex:
                    log_exception(
                        ex, 'Analysis of spike rate vs head direction')

        if self.get_analysis('hd_shuffle'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for hd shuffle analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for hd shuffle analysis")
            else:
                logging.info(
                    'Shuffling analysis of head-directional tuning...')
                try:
                    params = self.get_params_by_analysis('hd_shuffle')

                    graph_data = self.hd_shuffle(
                        bins=params['hd_shuffle_bins'],
                        nshuff=params['hd_shuffle_total'],
                        limit=params['hd_shuffle_limit'])
                    fig = nc_plot.hd_shuffle(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/hd_shuffle/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Head directional shuffling analysis')

        if self.get_analysis('hd_time_lapse'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for hd time lapse analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for hd time lapse analysis")
            else:
                logging.info('Time-lapsed head-directional tuning...')
                try:
                    graph_data = self.hd_time_lapse()

                    fig = nc_plot.hd_spike_time_lapse(graph_data)
                    self.close_fig(fig)

                    fig = nc_plot.hd_rate_time_lapse(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/hd_time_lapse/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Head directional time-lapse analysis')

        if self.get_analysis('hd_time_shift'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for hd time shift analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for hd time shift analysis")
            else:
                logging.info(
                    'Time-shift analysis of head-directional tuning...')
                try:
                    params = self.get_params_by_analysis('hd_time_shift')

                    hdData = self.hd_shift(
                        shift_ind=np.arange(params['hd_shift_min'],
                                            params['hd_shift_max'] +
                                            params['hd_shift_step'],
                                            params['hd_shift_step']))
                    fig = nc_plot.hd_time_shift(hdData)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/hd_time_shift/', graph_data=hdData)

                except BaseException as ex:
                    log_exception(ex, 'Head directional time-shift analysis')

        if self.get_analysis('loc_rate'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for locational rate analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for locational rate analysis")
            else:
                logging.info('Assessing of locational tuning...')
                try:
                    params = self.get_params_by_analysis('loc_rate')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    place_data = self.ndata.place(
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['loc_rate_kern_len']],
                        fieldThresh=params['loc_field_thresh'],
                        smoothPlace=params['loc_field_smooth'],
                        minPlaceFieldNeighbours=params['loc_field_bins'],
                        brAdjust=True, update=True)
                    fig1 = nc_plot.loc_firing(
                        place_data, colormap=params['loc_colormap'],
                        style=params['loc_style'],
                        levels=params['loc_contour_levels'])
                    self.close_fig(fig1)
                    fig2 = nc_plot.loc_firing_and_place(
                        place_data, colormap=params['loc_colormap'],
                        style=params['loc_style'],
                        levels=params['loc_contour_levels'])
                    self.close_fig(fig2)
                    self.plot_data_to_hdf(
                        name=name + '/loc_rate/', graph_data=place_data)

                except BaseException as ex:
                    log_exception(ex, 'Analysis of locational firing rate')

        if self.get_analysis('loc_shuffle'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for locational shuffle analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for locational shuffle analysis")
            else:
                logging.info('Shuffling analysis of locational tuning...')
                try:
                    params = self.get_params_by_analysis('loc_shuffle')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    place_data = self.loc_shuffle(
                        bins=params['loc_shuffle_nbins'],
                        nshuff=params['loc_shuffle_total'],
                        limit=params['loc_shuffle_limit'],
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['loc_rate_kern_len']],
                        brAdjust=True, update=False)
                    fig = nc_plot.loc_shuffle(place_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/loc_shuffle/', graph_data=place_data)

                except BaseException as ex:
                    log_exception(ex, 'Locational shuffling analysis')

        if self.get_analysis('loc_time_lapse'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for locational time lapse analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for locational time lapse analysis")
            else:
                logging.info('Time-lapse analysis of locational tuning...')
                try:
                    params = self.get_params_by_analysis('loc_time_lapse')
                    params2 = self.get_params_by_analysis('loc_rate')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    graph_data = self.loc_time_lapse(
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['loc_rate_kern_len']],
                        brAdjust=True)

                    fig = nc_plot.loc_spike_time_lapse(graph_data)
                    self.close_fig(fig)

                    fig = nc_plot.loc_rate_time_lapse(
                        graph_data, colormap=params2['loc_colormap'],
                        style=params2['loc_style'],
                        levels=params2['loc_contour_levels'])
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/loc_time_lapse/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Locational time-lapse analysis')

        if self.get_analysis('loc_time_shift'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for locational time shift analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for locational time shift analysis")
            else:
                logging.info('Time-shift analysis of locational tuning...')
                try:
                    params = self.get_params_by_analysis('loc_time_shift')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    plot_data = self.loc_shift(
                        shift_ind=np.arange(params['loc_shift_min'],
                                            params['loc_shift_max'] +
                                            params['loc_shift_step'],
                                            params['loc_shift_step']),
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[
                            filttype, params['loc_rate_kern_len']],
                        brAdjust=True, update=False)
                    fig = nc_plot.loc_time_shift(plot_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/loc_time_shift/', graph_data=plot_data)

                except BaseException as ex:
                    log_exception(ex, 'Locational time-shift analysis')

        if self.get_analysis('spatial_corr'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for spatial correlation analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for spatial correlation analysis")
            else:
                logging.info(
                    'Spatial and rotational correlation of locational tuning...')
                try:
                    params = self.get_params_by_analysis('spatial_corr')

                    if params['spatial_corr_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    plot_data = self.loc_auto_corr(
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['spatial_corr_kern_len']],
                        minPixel=params['spatial_corr_min_obs'], brAdjust=True)
                    fig = nc_plot.loc_auto_corr(
                        plot_data, colormap=params['spatial_corr_colormap'],
                        style=params['spatial_corr_style'],
                        levels=params['spatial_corr_contour_levels'])
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/spatial_corr/', graph_data=plot_data)

                    plot_data = self.loc_rot_corr(
                        binsize=params['rot_corr_bin'],
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['spatial_corr_kern_len']],
                        minPixel=params['spatial_corr_min_obs'], brAdjust=True)
                    fig = nc_plot.rot_corr(plot_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/spatial_corr/', graph_data=plot_data)

                except BaseException as ex:
                    log_exception(ex, 'Assessing spatial autocorrelation')

        if self.get_analysis('grid'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for grid cell analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for grid cell analysis")
            else:
                logging.info('Assessing gridness...')
                try:
                    params = self.get_params_by_analysis('grid')
                    params2 = self.get_params_by_analysis('spatial_corr')

                    if params['spatial_corr_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    graph_data = self.grid(
                        angtol=params['grid_ang_tol'],
                        binsize=params['grid_ang_bin'],
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['spatial_corr_kern_len']],
                        minPixel=params['spatial_corr_min_obs'],
                        brAdjust=True)  # Add other paramaters
                    fig = nc_plot.grid(
                        graph_data, colormap=params2['spatial_corr_colormap'],
                        style=params2['spatial_corr_style'],
                        levels=params2['spatial_corr_contour_levels'])
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/grid/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Grid cell analysis')

        if self.get_analysis('border'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for border analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for border analysis")
            else:
                logging.info('Estimating tuning to border...')
                try:
                    params = self.get_params_by_analysis('border')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    graph_data = self.border(
                        update=True, thresh=params['border_firing_thresh'],
                        cbinsize=params['border_ang_bin'],
                        nstep=params['border_stair_steps'],
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['loc_rate_kern_len']],
                        brAdjust=True)

                    fig = nc_plot.border(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/border/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Border cell analysis')

        if self.get_analysis('gradient'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for gradient analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for gradient analysis")
            else:
                logging.info('Calculating gradient-cell properties...')
                try:
                    params = self.get_params_by_analysis('gradient')

                    if params['loc_rate_filter'] == 'Gaussian':
                        filttype = 'g'
                    else:
                        filttype = 'b'

                    graph_data = self.gradient(
                        alim=params['grad_asymp_lim'],
                        blim=params['grad_displace_lim'],
                        clim=params['grad_growth_rate_lim'],
                        pixel=params['loc_pixel_size'],
                        chop_bound=params['loc_chop_bound'],
                        filter=[filttype, params['loc_rate_kern_len']],
                        brAdjust=True)
                    fig = nc_plot.gradient(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/gradient/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Gradient cell analysis')

        if self.get_analysis('multiple_regression'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for multi-regression analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for multi-regression analysis")
            else:
                logging.info('Multiple-regression analysis...')
                try:
                    params = self.get_params_by_analysis('multiple_regression')

                    graph_data = self.multiple_regression(
                        nrep=params['mra_nrep'],
                        episode=params['mra_episode'],
                        subsampInterv=params['mra_interval'])

                    fig = nc_plot.multiple_regression(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/multiple_regression/', graph_data=graph_data)

                except Exception as ex:
                    log_exception(ex, "Multiple-regression analysis")

        if self.get_analysis('inter_depend'):
            if self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for interdependence analysis")
            elif self.ndata.spatial.get_filename() == ".no_spatial.NONE":
                logging.error(
                    "Spatial data is required for interdependence analysis")
            else:
                logging.info('Assessing dependence of spatial variables...')
                try:
                    self.interdependence(
                        pixel=3, hdbinsize=5, spbinsize=1, sprange=[0, 40],
                        abinsize=10, angvelrange=[-500, 500])

                except BaseException as ex:
                    log_exception(ex, 'Error in interdependence analysis')

        if self.get_analysis('lfp_spectrum'):
            if self.ndata.lfp.get_filename() == ".no_lfp.NONE":
                logging.error(
                    "LFP data is required for spectrum analysis")
            else:
                logging.info(
                    "Analysing LFP power spectrum...")
                try:
                    params = self.get_params_by_analysis('lfp_spectrum')

                    graph_data = self.spectrum(
                        window=params['lfp_pwelch_seg_size'],
                        noverlap=params['lfp_pwelch_overlap'],
                        nfft=params['lfp_pwelch_nfft'],
                        ptype='psd', prefilt=True,
                        filtset=[params['lfp_prefilt_order'],
                                 params['lfp_prefilt_lowcut'],
                                 params['lfp_prefilt_highcut'], 'bandpass'],
                        fmax=params['lfp_pwelch_freq_max'],
                        db=False, tr=False)
                    fig = nc_plot.lfp_spectrum(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/lfp_spectrum/', graph_data=graph_data)

                    graph_data = self.spectrum(
                        window=params['lfp_stft_seg_size'],
                        noverlap=params['lfp_stft_overlap'],
                        nfft=params['lfp_stft_nfft'],
                        ptype='psd', prefilt=True,
                        filtset=[params['lfp_prefilt_order'],
                                 params['lfp_prefilt_lowcut'],
                                 params['lfp_prefilt_highcut'], 'bandpass'],
                        fmax=params['lfp_stft_freq_max'],
                        db=True, tr=True)
                    fig = nc_plot.lfp_spectrum_tr(
                        graph_data, colormap=params['lfp_spectrum_colormap'])
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/lfp_spectrum_TR/', graph_data=graph_data)

                    # Default band ranges are from Muessig et al. 2019
                    # They are theta delta ranges. From
                    # Coordinated Emergence of Hippocampal Replay and
                    # Theta Sequences during Post - natal Development
                    first_name = (
                        str(params["lfp_highband_lowcut"]) + "Hz to " +
                        str(params["lfp_highband_highcut"]) + "Hz")
                    second_name = (
                        str(params["lfp_lowband_lowcut"]) + "Hz to " +
                        str(params["lfp_lowband_highcut"]) + "Hz")
                    total_band = [
                        params["lfp_prefilt_lowcut"], params['lfp_prefilt_highcut']]
                    self.bandpower_ratio(
                        [params["lfp_highband_lowcut"],
                            params["lfp_highband_highcut"]],
                        [params["lfp_lowband_lowcut"],
                            params["lfp_lowband_highcut"]],
                        params["lfp_band_win_len"], band_total=True,
                        total_band=total_band,
                        first_name=first_name, second_name=second_name)

                except BaseException as ex:
                    log_exception(ex, 'Analysing lfp spectrum')

        if self.get_analysis('spike_phase'):
            if self.ndata.lfp.get_filename() == ".no_lfp.NONE":
                logging.error(
                    "LFP data is required for spike phase analysis")
            elif self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for spike phase analysis")
            else:
                logging.info('Analysing distribution of spike-phase in lfp...')
                try:
                    params = self.get_params_by_analysis('spike_phase')

                    graph_data = self.phase_dist(
                        binsize=params['phase_bin'],
                        rbinsize=params['phase_raster_bin'],
                        fwin=[params['phase_freq_min'],
                              params['phase_freq_max']],
                        pratio=params['phase_power_thresh'],
                        aratio=params['phase_amp_thresh'],
                        filtset=[params['lfp_prefilt_order'],
                                 params['lfp_prefilt_lowcut'],
                                 params['lfp_prefilt_highcut'], 'bandpass'])
                    fig = nc_plot.spike_phase(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/spike_phase/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Assessing spike-phase distribution')

        if self.get_analysis('phase_lock'):
            if self.ndata.lfp.get_filename() == ".no_lfp.NONE":
                logging.error(
                    "LFP data is required for phase lock analysis")
            elif self.ndata.spike.get_filename() == ".no_spike.NONE":
                logging.error(
                    "Spike data is required for phase lock analysis")
            else:
                logging.info(
                    'Analysis of Phase-locking value and spike-field coherence...')
                try:
                    params = self.get_params_by_analysis('phase_lock')

                    reparam = {
                        'window': [
                            params['phase_loc_win_low'],
                            params['phase_loc_win_up']],
                        'nfft': params['phase_loc_nfft'],
                        'fwin': [2, params['phase_loc_freq_max']],
                        'nsample': 2000,
                        'slide': 25,
                        'nrep': 500,
                        'mode': 'tr'}

                    graph_data = self.plv(**reparam)
                    fig = nc_plot.plv_tr(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/phase_lock_TR/', graph_data=graph_data)

                    reparam.update({'mode': 'bs', 'nsample': 100})
                    graph_data = self.plv(**reparam)
                    fig = nc_plot.plv_bs(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/phase_lock_BS/', graph_data=graph_data)

                    reparam.update({'mode': None})
                    graph_data = self.plv(**reparam)
                    fig = nc_plot.plv(graph_data)
                    self.close_fig(fig)
                    self.plot_data_to_hdf(
                        name=name + '/phase_lock/', graph_data=graph_data)

                except BaseException as ex:
                    log_exception(ex, 'Spike-phase locking analysis')

        if self.get_analysis('lfp_spike_causality'):
            logging.warning(
                'Unit-LFP analysis has not been implemented yet!')

    def open_hdf_file(self, filename=None):
        """
        Set the filename and open the file object for the HDF5 file.

        Parameters
        ----------
        filename : str
            Filename of the HDF5 object

        Returns
        -------
        None

        """
        if not filename:
            filename = self.config.get_nwb_file()

        self.hdf.set_filename(filename=filename)

    def close_hdf_file(self):
        """
        Close the HDF5 file object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.hdf.close()

    def get_hdf_groups(self, path=''):
        """
        Return the names of groups or datasets in a path.

        Parameters
        ----------
        path : str
            path to HDF5 file group

        Returns
        -------
        list
            Names of the groups or datasets in the path

        """
        return self.hdf.get_groups_in_path(path=path)

    def exist_hdf_path(self, path=''):
        """
        Check and return if an HDF5 file path exists.

        Parameters
        ----------
        path : str
            path to HDF5 file group

        Returns
        -------
        exists : bool
            True if the path exists

        """
        exists = False
        if path in self.hdf.f:
            exists = True
        return exists

    def plot_data_to_hdf(self, name=None, graph_data=None):
        """
        Store plot data to the HDF5 file in the '/analysis/' path.

        Parameters
        ----------
        name : str
            Unit ID which is the name of the group in the '/analysis/' path
        graph_data : dict
            Dictionary of data that are plotted

        Returns
        -------
        None

        """
        self.hdf.save_dict_recursive(path='/analysis/',
                                     name=name, data=graph_data)

    def set_neuro_data(self, ndata):
        """
        Set a new NData() object or its subclass object.

        Parameters
        ----------
        ndata : NData
            Object of NData class or its subclass.

        Returns
        -------
        None

        """
        if inspect.isclass(ndata):
            ndata = ndata()
        if isinstance(ndata, NData):
            self.ndata = ndata
        else:
            logging.warning('Inappropriate NeuroData object or class')

    def get_neuro_data(self):
        """
        Return the NData() object from this class.

        Parameters
        ----------
        None

        Returns
        -------
        NData
            NeuroChaT's ndata attribute

        """
        return self.ndata

    def set_configuration(self, config):
        """
        Set a new Configuration() object or its subclass object.

        Parameters
        ----------
        config : Configuration
            Object of Configuration class or its subclass.

        Returns
        -------
        None

        """
        if inspect.isclass(config):
            config = config()
        if isinstance(config, Configuration):
            self.config = config
        else:
            logging.warning('Inappropriate Configuration object or class')

    def get_configuration(self):
        """
        Return the Configuration() object from this class.

        Parameters
        ----------
        None

        Returns
        -------
        Configuration
            NeuroChaT's config attribute

        """
        return self.config

    def convert_to_nwb(self, excel_file=None):
        """
        Take a list of datasets in Excel file and converts them into NWB.

        This method currently supports Axona and Neuralynx data formats.

        Parameters
        ----------
        excel_file : str
            Name of the excel file that contains data specifications

        Returns
        -------
        None

        """
        if self.get_data_format() == 'NWB':
            logging.error(
                'NWB files do not need to be converted!' +
                'Check file format option again!')
        info = {'spat': [], 'spike': [], 'lfp': []}
        export_info = oDict({'dir': [], 'nwb': [], 'spike': [], 'lfp': []})
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spike_file = row[1] + os.sep + row[3]
                lfp_id = row[4]

                if self.get_data_format() == 'Axona':
                    spatial_file = row[1] + os.sep + row[2] + '.txt'
                    lfp_file = remove_extension(spike_file) + lfp_id

                elif self.get_data_format() == 'Neuralynx':
                    spatial_file = row[1] + os.sep + row[2] + '.nvt'
                    lfp_file = row[1] + os.sep + lfp_id + '.ncs'

                info['spat'].append(spatial_file)
                info['spike'].append(spike_file)
                info['lfp'].append(lfp_file)

        if info['spike']:
            for i, spike_file in enumerate(info['spike']):

                logging.info('Converting file groups: ' + str(i + 1))
                self.ndata.set_spatial_file(info['spat'][i])
                self.ndata.set_spike_file(info['spike'][i])
                self.ndata.set_lfp_file(info['lfp'][i])
                self.ndata.load()
                self.ndata.save_to_hdf5()

                f_name = self.hdf.resolve_hdfname(data=self.ndata.spike)
                export_info['dir'].append(
                    os.sep.join(f_name.split(os.sep)[:-1]))
                export_info['nwb'].append(
                    f_name.split(os.sep)[-1].split('.')[0])

                export_info['spike'].append(
                    self.hdf.get_file_tag(self.ndata.spike))
                export_info['lfp'].append(
                    self.hdf.get_file_tag(self.ndata.lfp))

        export_info = pd.DataFrame(export_info, columns=[
                                   'dir', 'nwb', 'spike', 'lfp'])
        words = excel_file.split(os.sep)
        name = 'NWB_list_' + words[-1]
        export_info.to_excel(
            os.path.join(os.sep.join(words[:-1]), name), index=False)
        logging.info('Conversion process completed!')

    @staticmethod
    def sortingextractor_to_nwb(sorting, plot_waveforms=False):
        """
        Convert a SpikeInterface sortingextractor to NWB format.

        This allows for Neurochat to later analyse it.
        See examples/spike_inteface_convert.py for an example.

        Parameters
        ----------
        sorting : spikeinterface.extractors.SortingExtractor
            The sortingextractor to convert.
        plot_waveforms : bool, optional.
            Defaults to False. Whether to plot unit waveforms.

        Returns
        -------
        None

        """
        def plot_all_waveforms(sorting, out_folder):
            """Local function to plot all spike interface sorting waveforms."""
            unit_ids = sorting.get_unit_ids()

            waveform_eg = sorting.get_unit_spike_features(
                unit_ids[0], "waveforms")
            total_channels = waveform_eg.shape[1]

            wf_by_group = [
                sorting.get_unit_spike_features(u, "waveforms") for u in unit_ids]
            tetrode = 0
            for i, wf in enumerate(wf_by_group):
                try:
                    tetrode = sorting.get_unit_property(unit_ids[i], "group")
                except Exception:
                    try:
                        tetrode = sorting.get_unit_property(
                            unit_ids[i], "ch_group")
                    except Exception:
                        logging.warning(
                            "Unable to find property cluster group or group in units")
                        tetrode += 1
                        print("Will use tetrode {}".format(tetrode))

                fig, axes = plt.subplots(total_channels)
                for j in range(total_channels):
                    try:
                        wave = wf[:, j, :]
                    except Exception:
                        wave = wf[j, :]

                    axes[j].plot(wave.T, color="k", lw=0.3)

                if 0 in unit_ids:
                    to_use_id = unit_ids[i] + 1
                else:
                    to_use_id = unit_ids[i]

                o_loc = os.path.join(
                    out_folder, "tet{}_unit{}_waveforms.png".format(
                        tetrode, to_use_id))
                fig.savefig(o_loc, dpi=200)
                plt.close("all")

        sorting_path = sorting.params.get("dat_path", "")
        if os.path.isdir(os.path.dirname(sorting_path)):
            out_folder = os.path.dirname(sorting_path)
            out_name = (
                os.path.splitext(os.path.basename(sorting_path))[0] + "_NC_NWB.h5")
        else:
            out_folder = os.getcwd()
            out_name = "NC_NWB.h5"
            logging.warning(
                "Sorting Extractor has no data path, saving waveforms to cwd")

        unit_ids = sorting.get_unit_ids()
        if 0 in unit_ids:
            logging.warning(
                "Unit numbers loaded from spike interface contain 0" +
                ", as such, all unit numbers will be incremented by 1.")

        if plot_waveforms:
            plot_out_folder = os.path.join(out_folder, "nc_waveforms")
            os.makedirs(plot_out_folder, exist_ok=True)
            logging.info("Plotting waveforms to {}".format(plot_out_folder))
            plot_all_waveforms(sorting, plot_out_folder)

        groups = []
        for unit in unit_ids:
            try:
                tetrode = sorting.get_unit_property(unit, "group")
            except BaseException:
                try:
                    tetrode = sorting.get_unit_property(unit, "ch_group")
                except BaseException:
                    tetrode = None
            if tetrode is not None:
                if tetrode not in groups:
                    groups.append(tetrode)
        logging.info("All groups found in sorting: {}".format(groups))

        spike = NSpike()
        hdf_path = os.path.join(out_folder, out_name)
        nhdf = Nhdf(filename=hdf_path)
        logging.info(
            "Converting spike extractor to {}".format(hdf_path))

        # This occurs if no probe/tetrode information is found
        if len(groups) == 0:
            spike.load_spike_spikeinterface(sorting)
            spike.set_unit_no(spike.get_unit_list()[0])
            nhdf.save_spike(spike=spike)

        else:
            for g in groups:
                spike.load_spike_spikeinterface(sorting, group=g)
                spike.set_unit_no(spike.get_unit_list()[0])
                nhdf.save_spike(spike=spike)

    def verify_units(self, excel_file=None):
        """
        Take a list of datasets and verify the specifications of the units.

        The verification tool is useful for prescreening of units before the
        batch-mode analysis using 'Listed Units' mode of NeuroChaT.

        Parameters
        ----------
        excel_file : str
            Name of the excel file that contains data specifications

        Returns
        -------
        None

        """
        info = {'spike': [], 'unit': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spike_file = row[1] + os.sep + row[3]
                unit_no = int(row[4])
                if self.get_data_format() == 'NWB':
                    # excel list: directory| spike group| unit_no
                    hdf_name = row[1] + os.sep + row[3] + '.hdf5'
                    spike_file = hdf_name + '+/processing/Shank' + '/' + row[4]
                info['spike'].append(spike_file)
                info['unit'].append(unit_no)
            n_units = excel_info.shape[0]

            excel_info = excel_info.assign(
                fileExists=pd.Series(np.zeros(n_units, dtype=bool)))
            excel_info = excel_info.assign(
                unitExists=pd.Series(np.zeros(n_units, dtype=bool)))

            if info['spike']:
                for i, spike_file in enumerate(info['spike']):

                    logging.info('Verifying unit: ' + str(i + 1))
                    if os.path.exists(spike_file):
                        excel_info.loc[i, 'fileExists'] = True
                        self.ndata.set_spike_file(spike_file)
                        self.ndata.load_spike()
                        units = self.ndata.get_unit_list()

                        if info['unit'][i] in units:
                            excel_info.loc[i, 'unitExists'] = True

            excel_info.to_excel(excel_file, index=False)
            logging.info('Verification process completed!')
        else:
            logging.error('Excel  file does not exist!')

    def angle_calculation(self, excel_file=None, should_plot=True):
        """
        Find the angle between place field centroids from an excel file.

        This takes an input excel file which lists unit specifications
        and finds the angle between the place field centroids.
        The results of the analysis are written back to the input Excel file.

        Parameters
        ----------
        excel_file : str
            Name of the excel file that contains data specifications

        Returns
        -------
        None

        """
        params = self.get_params_by_analysis('loc_rate')

        if params['loc_rate_filter'] == 'Gaussian':
            filttype = 'g'
        else:
            filttype = 'b'

        collection = NDataContainer()
        excel_info = collection.add_files_from_excel(excel_file)
        if excel_info is None:
            return

        n_units = len(collection)
        if (n_units % 3 != 0):
            logging.error(
                "angle_calculation: Can't compute the angle for a number of"
                + " units not divisible by 3, given " + str(n_units))
            return

        excel_info = excel_info.assign(CentroidX=pd.Series(np.zeros(n_units)))
        excel_info = excel_info.assign(CentroidY=pd.Series(np.zeros(n_units)))
        excel_info = excel_info.assign(
            AngleInDegrees=pd.Series(np.zeros(n_units)))
        excel_info = excel_info.assign(
            StrongPlaceField=pd.Series(np.zeros(n_units)))
        excel_info = excel_info.assign(
            Skaggs=pd.Series(np.zeros(n_units)))

        centroids = []
        figs = []
        for i, data in enumerate(collection):
            place_data = data.place(
                pixel=params['loc_pixel_size'],
                chop_bound=params['loc_chop_bound'],
                filter=[filttype, params['loc_rate_kern_len']],
                fieldThresh=params['loc_field_thresh'],
                smoothPlace=params['loc_field_smooth'],
                brAdjust=True, update=True)
            centroid = place_data['centroid']
            centroids.append(centroid)
            excel_info.loc[i, "CentroidX"] = centroid[0]
            excel_info.loc[i, "CentroidY"] = centroid[1]
            _res = data.get_results()
            excel_info.loc[i, "Skaggs"] = _res["Spatial Skaggs"]
            excel_info.loc[i, "StrongPlaceField"] = (
                _res["Found strong place field"])
            if should_plot:
                fig = nc_plot.loc_firing_and_place(place_data)
                figs.append(fig)

            if (i + 1) % 3 == 0:  # then spit out the angle
                first_centroid = centroids[0]
                second_centroid = centroids[1]
                angle = angle_between_points(
                    first_centroid, second_centroid, centroid)
                excel_info.loc[i, "AngleInDegrees"] = angle
                if should_plot:
                    fig = nc_plot.plot_angle_between_points(
                        centroids,
                        place_data['xedges'].max(),
                        place_data['yedges'].max())
                    figs.append(fig)
                centroids = []

        if should_plot:
            self.close_fig(figs)

        try:
            split_up = remove_extension(
                excel_file, keep_dot=False, return_ext=True)
            output_file = split_up[0] + "_result." + split_up[1]
            excel_info.to_excel(output_file, index=False)
        except PermissionError:
            logging.warning(
                "Please close the excel file to"
                + " write the result back to it at"
                + " {}".format(output_file))

        logging.info('Angle calculation completed! Value was {}'.format(angle))

    def cluster_evaluate(self, excel_file=None):
        """
        Take a list of unit specifications and evaluates clustering quality.

        The results of the analysis are written back to the input Excel file.

        Parameters
        ----------
        excel_file : str
            Name of the excel file that contains data specifications

        Returns
        -------
        None

        """
        info = {'spike': [], 'unit': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spike_file = row[1] + os.sep + row[2]
                unit_no = int(row[3])
                if self.get_data_format() == 'NWB':
                    # excel list: directory| spike group| unit_no
                    hdf_name = row[1] + os.sep + row[2] + '.hdf5'
                    spike_file = hdf_name + '+/processing/Shank' + '/' + row[3]
                info['spike'].append(spike_file)
                info['unit'].append(unit_no)
            n_units = excel_info.shape[0]

            excel_info = excel_info.assign(BC=pd.Series(np.zeros(n_units)))
            excel_info = excel_info.assign(Dh=pd.Series(np.zeros(n_units)))

            if info['spike']:
                for i, spike_file in enumerate(info['spike']):

                    logging.info('Evaluating unit: ' + str(i + 1))
                    if os.path.exists(spike_file):
                        self.ndata.set_spike_file(spike_file)
                        self.ndata.load_spike()
                        units = self.ndata.get_unit_list()

                        if info['unit'][i] in units:
                            nclust = NClust(spike=self.ndata.spike)
                            bc, dh = nclust.cluster_separation(
                                unit_no=info['unit'][i])
                            excel_info.loc[i, 'BC'] = np.max(bc)
                            excel_info.loc[i, 'Dh'] = np.min(dh)
            excel_info.to_excel(excel_file, index=False)
            logging.info('Cluster evaluation completed!')
        else:
            logging.error('Excel  file does not exist!')

    def cluster_similarity(self, excel_file=None):
        """
        Take a list of specifications for pairwise comparison of units.

        The results are written back to the input Excel file.

        Parameters
        ----------
        excel_file : str
            Name of the excel file that contains unit specifications

        Returns
        -------
        None

        """
        nclust_1 = NClust()
        nclust_2 = NClust()
        info = {'spike_1': [], 'unit_1': [], 'spike_2': [], 'unit_2': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spike_file = row[1] + os.sep + row[2]
                unit_1 = int(row[3])
                if self.get_data_format() == 'NWB':
                    # excel list: directory| spike group| unit_no
                    hdf_name = row[1] + os.sep + row[2] + '.hdf5'
                    spike_file = hdf_name + '+/processing/Shank' + '/' + row[3]
                info['spike_1'].append(spike_file)
                info['unit_1'].append(unit_1)

                spike_file = row[4] + os.sep + row[5]
                unit_2 = int(row[6])
                if self.get_data_format() == 'NWB':
                    # excel list: directory| spike group| unit_no
                    hdf_name = row[4] + os.sep + row[5] + '.hdf5'
                    spike_file = hdf_name + '+/processing/Shank' + '/' + row[6]
                info['spike_2'].append(spike_file)
                info['unit_2'].append(unit_2)

            n_comparison = excel_info.shape[0]

            excel_info = excel_info.assign(
                BC=pd.Series(np.zeros(n_comparison)))
            excel_info = excel_info.assign(
                Dh=pd.Series(np.zeros(n_comparison)))

            if info['spike_1']:
                for i in np.arange(n_comparison):
                    logging.info(
                        'Evaluating unit similarity row: ' + str(i + 1))
                    if os.path.exists(info['spike_1']) and os.path.exists(
                            info['spike_2']):
                        nclust_1.load(
                            filename=info['spike_1'],
                            system=self.get_data_format())
                        nclust_2.load(
                            filename=info['spike_2'],
                            system=self.get_data_format())
                        bc, dh = nclust_1.cluster_similarity(
                            nclust=nclust_2,
                            unit_1=info['unit_1'][i], unit_2=info['unit_2'][i])
                        excel_info.loc[i, 'BC'] = bc
                        excel_info.loc[i, 'Dh'] = dh
            excel_info.to_excel(excel_file, index=False)
            logging.info('Cluster similarity analysis completed!')
        else:
            logging.error('Excel  file does not exist!')

    def place_cell_plots(self, directory, dpi=400):
        """
        Plot png images of place cell figures, looping over a directory.

        Currently only works for axona files, but can be extended.
        Extension performed by supporting more formats in NDataContainer.

        Parameters
        ----------
        dir : str
            The directory to get files from.
        dpi : int
            The desired dpi of the pngs.

        Returns
        -------
        None

        """
        try:
            container = NDataContainer(load_on_fly=True)
            container.add_axona_files_from_dir(
                directory, tetrode_list=[i for i in range(1, 17)])
            nca.place_cell_summary(
                container, dpi=dpi,
                filter_place_cells=False, filter_low_freq=False,
                point_size=10)
        except Exception as ex:
            log_exception(
                ex, "In walking a directory for place cell summaries")
        return

    def append_selection_to_excel(self, excel_file):
        """
        Append the current selection of files to the Excel file.

        Parameters
        ----------
        excel_file : str
            The path to the Excel file to append the selection to.

        Returns
        -------
        None

        """
        if os.path.exists(excel_file):
            try:
                excel_info = pd.read_excel(excel_file)
            except PermissionError:
                logging.error(
                    "Please close {} before writing to it".format(excel_file))
                return
        else:
            if self.get_data_format() == 'NWB':
                excel_info = pd.DataFrame(
                    columns=[
                        "Directory", "NWB Name", "Electrode Group",
                        "Cell ID", "LFP Group"])
            else:
                excel_info = pd.DataFrame(
                    columns=[
                        "Directory", "Position File", "Spike File",
                        "Cell ID", "LFP Chan"])
        logging.info(
            "Saving selected files to {}".format(excel_file))
        logging.info(
            "ALL files in this excel sheet should be in {} format".format(
                self.get_data_format()))
        row_info = [None] * 5
        spike_file = self.get_spike_file()
        lfp_file = self.get_lfp_file()
        spatial_file = self.get_spatial_file()
        if self.get_data_format() != 'NWB':
            try:
                dname_spike = os.path.dirname(spike_file)
                if dname_spike == "":
                    dname_spike = None
                bname_spike = os.path.basename(spike_file)
            except BaseException:
                dname_spike = None
                bname_spike = ""
            try:
                dname_spatial = os.path.dirname(spatial_file)
                bname_spatial = os.path.basename(spatial_file)
                if dname_spatial == "":
                    dname_spatial = None
            except BaseException:
                dname_spatial = None
                bname_spatial = ""
            try:
                dname_lfp = os.path.dirname(lfp_file)
                bname_lfp = os.path.basename(lfp_file)
                if dname_lfp == "":
                    dname_lfp = None
            except BaseException:
                dname_lfp = None
                bname_lfp = ""

            count = 0
            dnames = [dname_spike, dname_lfp, dname_spatial]
            for i, val in enumerate(dnames):
                dname_cpy = deepcopy(dnames)
                if val is not None:
                    dname_cpy.pop(i)
                    for val2 in dname_cpy:
                        if val2 is not None:
                            if val != val2:
                                count += 1

            if count > 0:
                raise ValueError(
                    "All input files must be in the same directory" +
                    " or be left blank, provided {}, {}, {}".format(
                        spike_file, lfp_file, spatial_file))
            dname = next(
                (item for item in dnames if item is not None), "")

        if self.get_data_format() == 'Axona':
            row_info[0] = dname
            row_info[1] = bname_spatial
            row_info[2] = bname_spike
            row_info[3] = self.get_unit_no()
            if os.path.isfile(spike_file):
                row_info[4] = os.path.splitext(bname_lfp)[1]
            else:
                row_info[4] = bname_lfp

        elif self.get_data_format() == 'Neuralynx':
            row_info[0] = dname
            row_info[1] = bname_spatial
            row_info[2] = bname_spike
            row_info[3] = self.get_unit_no()
            row_info[4] = os.path.splitext(bname_lfp)[0]

        elif self.get_data_format() == 'NWB':
            fname = spike_file.split("+")[0]
            row_info[0] = os.path.dirname(fname)
            row_info[1] = os.path.splitext(os.path.basename(fname))[0]
            if "+" in spike_file:
                path = spike_file.split("+")[1]
                row_info[2] = path.split("/")[-1]
            else:
                row_info[2] = None
            row_info[3] = self.get_unit_no()
            if "+" in lfp_file:
                path = lfp_file.split("+")[1]
                row_info[4] = path.split("/")[-1]
            else:
                row_info[4] = None

        df_to_append = pd.DataFrame(
            data=[row_info, ], columns=list(excel_info.columns))
        excel_info = excel_info.append(df_to_append, ignore_index=True)

        try:
            excel_info.to_excel(excel_file, index=False)
        except PermissionError:
            logging.error(
                "Please close {} before saving".format(excel_file))

    def __getattr__(self, arg):
        """Forward __getattr__ to configuration class."""
        if hasattr(self.config, arg):
            return getattr(self.config, arg)
        elif hasattr(self.ndata, arg):
            return getattr(self.ndata, arg)
        else:
            logging.warning(
                'No ' + arg + ' method or attribute in NeuroChaT class')
