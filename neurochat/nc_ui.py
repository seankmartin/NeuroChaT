# -*- coding: utf-8 -*-
"""
This module implements NeuroChaT_Ui, the main class for NeuroChaT graphical user interface. It contains
other graphical and data objects and connects to the NeuroChaT class for setting configuration and analysis
in NeuroChaT.

@author: Md Nurul Islam; islammn at tcd dot ie

"""
# Form implementation generated from reading ui file 'NeuroChaT.ui'
#
# Created: Thu Jan 12 13:04:12 2017
#      by: PyQt5 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!
import os
import sys
import logging
import webbrowser


from PyQt5 import QtCore, QtWidgets, QtGui

#from imp import reload

#import nc_data
# reload(nc_data)
#from nc_data import Nhdf

#import nc_uiutils
# reload(nc_uiutils)
from neurochat.nc_uiutils import NOut, PandasModel, ScrollableWidget, add_radio_button, \
    add_push_button, add_check_box, add_combo_box, add_log_box, add_label, \
    add_line_edit, add_group_box, add_spin_box, add_double_spin_box, xlt_from_utf8

from neurochat.nc_uimerge import UiMerge

#import nc_control
# reload(nc_control)
from neurochat.nc_control import NeuroChaT
from neurochat.nc_utils import make_dir_if_not_exists, log_exception
from neurochat.nc_utils import remove_extension

import pandas as pd

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)


class NeuroChaT_Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.nout = NOut()
        sys.stdout.write = self.nout.write
        self._control = NeuroChaT(parent=self)
        self._results_ui = UiResults(self)
        self._mode_dict = self._control.get_all_modes()
        self._default_loc = os.path.join(
            os.path.expanduser("~"), ".nc_saved", "last_dir_location.txt")
        make_dir_if_not_exists(self._default_loc)
        if os.path.isfile(self._default_loc):
            with open(self._default_loc, "r") as f:
                default_dir = f.readline()
            if os.path.isdir(default_dir):
                os.chdir(default_dir)
            else:
                self._curr_dir = "/home/"
        else:
            self._curr_dir = "/home/"
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the elements of NeuroChaT_ui class

        """

        self.setObjectName(xlt_from_utf8("MainWindow"))
        self.setEnabled(True)
        self.setFixedSize(725, 420)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(xlt_from_utf8("centralwidget"))

        layer_6_1 = QtWidgets.QVBoxLayout()
        self.mode_label = add_label(text="Analysis Mode", obj_name="modeLabel")
        self.mode_box = add_combo_box(obj_name="modeBox")
        layer_6_1.addWidget(self.mode_label)
        layer_6_1.addWidget(self.mode_box)

        layer_6_2 = QtWidgets.QVBoxLayout()
        self.unit_label = add_label(text="Unit No", obj_name="unitLabel")
        self.unit_no_box = add_combo_box(obj_name="unitNoBox")
        self.unit_no_box.addItems([str(i) for i in list(range(256))])
#        self.unit_no_box.setEditable(True)
        layer_6_2.addWidget(self.unit_label)
        layer_6_2.addWidget(self.unit_no_box)

        layer_6_3 = QtWidgets.QVBoxLayout()
        self.chan_label = add_label(text="LFP Ch No", obj_name="chanLabel")
        self.lfp_chan_box = add_combo_box(obj_name="lfpChanBox")
        self.lfp_chan_box.setSizeAdjustPolicy(
            self.lfp_chan_box.AdjustToContents)
#        self.lfp_chan_box.setEditable(True)
        layer_6_3.addWidget(self.chan_label)
        layer_6_3.addWidget(self.lfp_chan_box)

        layer_5_1 = QtWidgets.QHBoxLayout()
        layer_5_1.addLayout(layer_6_1)
#        layer_5_1.addStretch(1)
        layer_5_1.addLayout(layer_6_2)
        layer_5_1.addLayout(layer_6_3)

        layer_5_2 = QtWidgets.QHBoxLayout()
        self.browse_button = add_push_button(
            text="Browse", obj_name="browseButton")
        self.filename_line = add_line_edit(obj_name="filenameLine",
                                           text="Select spike(.n) &/or position file(.txt)")
        layer_5_2.addWidget(self.browse_button)
        layer_5_2.addWidget(self.filename_line)

        layer_4_1 = QtWidgets.QVBoxLayout()
        layer_4_1.addLayout(layer_5_1)
        layer_4_1.addLayout(layer_5_2)

        self.cell_type_box = self.select_cell_type_ui()
        layer_3_2 = QtWidgets.QVBoxLayout()
        layer_3_2.addLayout(layer_4_1)
        layer_3_2.addWidget(self.cell_type_box)

        self.inp_format_label = add_label(
            text="Input Data Format", obj_name="inpFormatLabel")
        self.file_format_box = add_combo_box(obj_name="fileFormatBox")
        self.graphic_format_box = self.selectGraphicFormatUi()
        self.start_button = add_push_button(
            text="Start", obj_name="startButton")
        self.save_log_button = add_push_button(
            text="Save log", obj_name="saveLogButton")
        self.clear_log_button = add_push_button(
            text="Clear log", obj_name="clearLogButton")
        layer_3_1 = QtWidgets.QVBoxLayout()
        layer_3_1.addWidget(self.inp_format_label)
        layer_3_1.addWidget(self.file_format_box)
        layer_3_1.addWidget(self.graphic_format_box)
        layer_3_1.addStretch(1)
        self.start_button.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        layer_3_1.addWidget(self.start_button)
        layer_3_1.addStretch(1)

        layer_2_1 = QtWidgets.QHBoxLayout()
        layer_2_1.addLayout(layer_3_1)
        layer_2_1.addLayout(layer_3_2, 2)

        layer_2_2 = QtWidgets.QHBoxLayout()
        self.save_log_button.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.clear_log_button.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        layer_2_2.addWidget(self.save_log_button, 0, QtCore.Qt.AlignLeft)
        layer_2_2.addWidget(self.clear_log_button, 0, QtCore.Qt.AlignRight)

        self.log_text = add_log_box(obj_name="logText")
        layer_1_1 = QtWidgets.QVBoxLayout()
        layer_1_1.addLayout(layer_2_1)
        layer_1_1.addLayout(layer_2_2)
        layer_1_1.addWidget(self.log_text)

        layer_1_2 = self.select_analysis_ui()

        final_layer = QtWidgets.QHBoxLayout()
        final_layer.addLayout(layer_1_1, 4)
        final_layer.addLayout(layer_1_2, 2)

        # self.mode_box.addItems(["Single Unit", "Single Session", "Listed Units", "Multiple Sessions"])
        self.mode_box.addItems(
            ["Single Unit", "Single Session", "Listed Units"])
        self.file_format_box.addItems(["Axona", "Neuralynx", "NWB"])
        self.lfp_chan_getitems()

        self.centralwidget.setLayout(final_layer)

        self.menu_ui()
        self.retranslate_ui()

        # Set up child windows
        self._results_ui.setup_ui()
        self._merge_ui = UiMerge(self)
        self._param_ui = UiParameters(self)

        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)

        # Set the callbacks
        self.behaviour_ui()

    def behaviour_ui(self):
        """
        Sets up the behaviour of NeuroChaT_ui widgets

        """
        # self.connect(self.nout, QtCore.SIGNAL('update_log(QString)'), self.update_log)
        self.nout.emitted[str].connect(self.update_log)
        self.file_format_box.currentIndexChanged[int].connect(
            self.data_format_select)
        self.mode_box.currentIndexChanged[int].connect(self.mode_select)
        self.pdf_button.setChecked(True)
        self.graphic_format_group.buttonClicked.connect(
            self.graphic_format_select)
        self.unit_no_box.currentIndexChanged[int].connect(self.set_unit_no)
        self.lfp_chan_box.currentIndexChanged[int].connect(self.set_lfp_chan)
        self.select_all_box.stateChanged.connect(self.select_all)
        self.browse_button.clicked.connect(self.browse)
        self.clear_log_button.clicked.connect(self.clear_log)
        self.save_log_button.clicked.connect(self.save_log)
        self.open_file_act.triggered.connect(self.browse)
        self.save_session_act.triggered.connect(self.save_session)
        self.load_session_act.triggered.connect(self.load_session)

        self.start_button.clicked.connect(self.start)

        self.exit_act.triggered.connect(self.exit_nc)

        self.export_results_act.triggered.connect(self.export_results)
        self.export_graphic_info_act.triggered.connect(
            self.export_graphic_info)

        self.merge_act.triggered.connect(self.merge_output)
        self.accumulate_act.triggered.connect(self.accumulate_output)

        self.angle_act.triggered.connect(self.angle_calculation)
        self.multi_place_cell_act.triggered.connect(self.place_cell_plots)

        self.view_help_act.triggered.connect(self.view_help)
        # self.tutorial_act.triggered.connect(self.tutorial)
        self.about_nc_act.triggered.connect(self.about_nc)

        self.verify_units_act.triggered.connect(self.verify_units)
        self.evaluate_act.triggered.connect(self.cluster_evaluate)
        self.compare_units_act.triggered.connect(self.compare_units)
        self.convert_files_act.triggered.connect(self.convert_to_nwb)
        self.param_set_act.triggered.connect(self.set_parameters)

        self._results_ui.export_button.clicked.connect(self.export_results)

        self.cell_type_group.buttonClicked.connect(self.cell_type_select)

    def menu_ui(self):
        """
        Sets up the menu items in NeuroChaT GUI

        """

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 722, 21))
        self.menubar.setObjectName(xlt_from_utf8("menubar"))
        self.file_menu = self.menubar.addMenu('&File')
        self.settings_menu = self.menubar.addMenu('&Settings')
        self.utilities_menu = self.menubar.addMenu('&Utilities')
        self.multifile_menu = self.menubar.addMenu('&Multiple Files')
        self.help_menu = self.menubar.addMenu('&Help')

        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(xlt_from_utf8("statusbar"))
        self.setStatusBar(self.statusbar)

        self.open_file_act = self.file_menu.addAction("Open...")
        self.open_file_act.setShortcut(QtGui.QKeySequence("Ctrl+O"))

        self.file_menu.addSeparator()

        self.save_session_act = self.file_menu.addAction("Save session...")
        self.save_session_act.setShortcut(QtGui.QKeySequence("Ctrl+S"))

        self.load_session_act = self.file_menu.addAction("Load session...")
        self.load_session_act.setShortcut(QtGui.QKeySequence("Ctrl+L"))

        self.file_menu.addSeparator()

        self.exit_act = self.file_menu.addAction("Exit")
        self.exit_act.setShortcut(QtGui.QKeySequence("Ctrl+Q"))

        self.param_set_act = self.settings_menu.addAction("Parameters")
        self.param_set_act.setShortcut(QtGui.QKeySequence("Ctrl+P"))

        self.export_results_act = self.utilities_menu.addAction(
            "Export results")
        self.export_results_act.setShortcut(QtGui.QKeySequence("Ctrl+T"))

        self.export_graphic_info_act = self.utilities_menu.addAction(
            "Export graphic file info")
        self.export_graphic_info_act.setShortcut(QtGui.QKeySequence("Ctrl+G"))

        self.utilities_menu.addSeparator()

        self.merge_act = self.utilities_menu.addAction("Merge Output PS/PDF")
        self.accumulate_act = self.utilities_menu.addAction(
            "Accumulate output PS/PDF")

        self.utilities_menu.addSeparator()

        self.verify_units_act = self.utilities_menu.addAction("Verify units")
        self.evaluate_act = self.utilities_menu.addAction(
            "Evaluate clustering")
        self.compare_units_act = self.utilities_menu.addAction(
            "Compare single units")
        self.convert_files_act = self.utilities_menu.addAction(
            "Convert to NWB format")

        self.angle_act = self.multifile_menu.addAction(
            "Centroid Angle Calculation")
        self.angle_act.setStatusTip(
            "Select an excel file which specifies files " +
            "in the order of: " +
            "directory | position_file | spike_file | unit_no | eeg extension")

        self.multi_place_cell_act = self.multifile_menu.addAction(
            "Directory place cell summary")
        self.multi_place_cell_act.setStatusTip(
            "Select a folder to analyse units for place cells")

        self.view_help_act = self.help_menu.addAction(
            "NeuroChaT documentation")
        self.view_help_act.setShortcut(QtGui.QKeySequence("F1"))
        # self.tutorial_act = self.help_menu.addAction("NeuroChaT tutorial")
        # self.help_menu.addSeparator()
        self.about_nc_act = self.help_menu.addAction("About NeuroChaT")

    def selectGraphicFormatUi(self):
        """
        Sets up the graphic format selection panel in NeuroChaT GUI

        """

        self.pdf_button = add_radio_button(text="PDF", obj_name="pdfButton")
        self.ps_button = add_radio_button(
            text="Postscript", obj_name="psButton")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pdf_button)
        layout.addWidget(self.ps_button)

        graphic_format_box = add_group_box(
            title="Graphic Format", obj_name="graphicFormatBox")
        graphic_format_box.setLayout(layout)

        self.graphic_format_group = QtWidgets.QButtonGroup(graphic_format_box)
        self.graphic_format_group.setObjectName(
            xlt_from_utf8("graphicFormatGroup"))
        self.graphic_format_group.addButton(self.pdf_button)
        self.graphic_format_group.addButton(self.ps_button)

        return graphic_format_box

    def select_cell_type_ui(self):
        """
        Sets up the cell type selection panel in NeuroChaT GUI

        """

        positions = [(i, j) for j in range(3) for i in range(4)]

        self.place_cell_button = add_radio_button(
            text="Place", obj_name="place_cell_button")

        self.hd_cell_button = add_radio_button(
            text="Head-directional", obj_name="hd_cell_button")

        self.grid_cell_button = add_radio_button(
            text="Grid", obj_name="grid_cell_button")

        self.boundary_cell_button = add_radio_button(
            text="Boundary", obj_name="boundary_cell_button")

        self.gradient_cell_button = add_radio_button(
            text="Gradient", obj_name="gradientCell_button")

        self.hd_by_place_cell_button = add_radio_button(
            text="HDxPlace", obj_name="hdXPlaceCell_button")

        self.theta_cell_button = add_radio_button(
            text="Theta-rhythmic", obj_name="theta_cell_button")

        self.theta_skip_cell_button = add_radio_button(
            text="Theta-skipping", obj_name="theta_skip_cell_button")

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.place_cell_button, *positions[0])
        layout.addWidget(self.hd_cell_button, *positions[1])
        layout.addWidget(self.grid_cell_button, *positions[2])
        layout.addWidget(self.boundary_cell_button, *positions[3])
        layout.addWidget(self.gradient_cell_button, *positions[4])
        layout.addWidget(self.hd_by_place_cell_button, *positions[5])
        layout.addWidget(self.theta_cell_button, *positions[6])
        layout.addWidget(self.theta_skip_cell_button, *positions[7])

        cell_type_box = add_group_box(
            title="Select Cell Type", obj_name="cellTypeBox")
        cell_type_box.setLayout(layout)

        self.cell_type_group = QtWidgets.QButtonGroup(cell_type_box)
        self.cell_type_group.setObjectName(xlt_from_utf8("graphicFormatGoup"))
        self.cell_type_group.addButton(self.place_cell_button)
        self.cell_type_group.addButton(self.hd_cell_button)
        self.cell_type_group.addButton(self.grid_cell_button)
        self.cell_type_group.addButton(self.boundary_cell_button)
        self.cell_type_group.addButton(self.gradient_cell_button)
        self.cell_type_group.addButton(self.hd_by_place_cell_button)
        self.cell_type_group.addButton(self.theta_cell_button)
        self.cell_type_group.addButton(self.theta_skip_cell_button)

        return cell_type_box

    def select_analysis_ui(self):
        """
        Sets up the analysis type selection panel in NeuroChaT GUI

        """

        self.wave_property = add_check_box(
            text="Waveform Properties", obj_name="wave_property")

        self.isi = add_check_box(text="Interspike Interval", obj_name="isi")

        self.isi_corr = add_check_box(
            text="ISI Autocorrelation", obj_name="isi_corr")

        self.theta_cell = add_check_box(
            text="Theta-modulated Cell Index", obj_name="theta_cell")

        self.theta_skip_cell = add_check_box(
            text="Theta-skipping Cell Index", obj_name="theta_skip_cell")

        self.burst = add_check_box(text="Burst Property", obj_name="burst")

        self.speed = add_check_box(
            text="Spike Rate vs Running Speed", obj_name="speed")

        self.ang_vel = add_check_box(
            text="Spike Rate vs Angular Velocity", obj_name="ang_vel")

        self.hd_rate = add_check_box(
            text="Spike Rate vs Head Direction", obj_name="hd_rate")

        self.hd_shuffle = add_check_box(
            text="Head Directional Shuffling Analysis", obj_name="hd_shuffle")

        self.hd_time_lapse = add_check_box(
            text="Head Directional Time Lapse Analysis", obj_name="hd_time_lapse")

        self.hd_time_shift = add_check_box(
            text="Head Directional Time Shift Analysis", obj_name="hd_time_shift")

        self.loc_rate = add_check_box(
            text="Spike Rate vs Location", obj_name="loc_rate")

        self.loc_shuffle = add_check_box(
            text="Locational Shuffling Analysiss", obj_name="loc_shuffle")

#        self.place_field = add_check_box("Place Field Map", "placeField")

        self.loc_time_lapse = add_check_box(
            text="Locational Time Lapse Analysis", obj_name="loc_time_lapse")

        self.loc_time_shift = add_check_box(
            text="Locational Time Shift Analysis", obj_name="loc_time_shift")

        self.spatial_corr = add_check_box(
            text="Spatial Autocorrelation", obj_name="spatial_corr")

        self.grid = add_check_box(text="Grid Cell Analysis", obj_name="grid")

        self.border = add_check_box(
            text="Border Cell Analysis", obj_name="border")

        self.gradient = add_check_box(
            text="Gradient Cell Analysis", obj_name="gradient")

        self.multiple_regression = add_check_box(
            text="Multiple Regression", obj_name="multiple_regression")

        self.inter_depend = add_check_box(
            text="Interdependence Analysis", obj_name="inter_depend")

        self.lfp_spectrum = add_check_box(
            text="LFP Frequency Spectrum", obj_name="lfp_spectrum")

        self.spike_phase = add_check_box(
            text="Unit LFP-phase Distribution", obj_name="spike_phase")

        self.phase_lock = add_check_box(
            text="Unit LFP-phase Locking", obj_name="phase_lock")

        self.lfp_spike_causality = add_check_box(
            text="Unit-LFP Causality", obj_name="lfp_spike_causality")

        self.scroll_layout = QtWidgets.QVBoxLayout()
        self.scroll_layout.addWidget(self.wave_property)
        self.scroll_layout.addWidget(self.isi)
        self.scroll_layout.addWidget(self.isi_corr)
        self.scroll_layout.addWidget(self.theta_cell)
        self.scroll_layout.addWidget(self.theta_skip_cell)
        self.scroll_layout.addWidget(self.burst)
        self.scroll_layout.addWidget(self.speed)
        self.scroll_layout.addWidget(self.ang_vel)
        self.scroll_layout.addWidget(self.hd_rate)
        self.scroll_layout.addWidget(self.hd_shuffle)
        self.scroll_layout.addWidget(self.hd_time_lapse)
        self.scroll_layout.addWidget(self.hd_time_shift)
        self.scroll_layout.addWidget(self.loc_rate)
        self.scroll_layout.addWidget(self.loc_shuffle)
#        self.scroll_layout.addWidget(self.place_field)
        self.scroll_layout.addWidget(self.loc_time_lapse)
        self.scroll_layout.addWidget(self.loc_time_shift)
        self.scroll_layout.addWidget(self.spatial_corr)
        self.scroll_layout.addWidget(self.grid)
        self.scroll_layout.addWidget(self.border)
        self.scroll_layout.addWidget(self.gradient)
        self.scroll_layout.addWidget(self.multiple_regression)
        self.scroll_layout.addWidget(self.inter_depend)
        self.scroll_layout.addWidget(self.lfp_spectrum)
        self.scroll_layout.addWidget(self.spike_phase)
        self.scroll_layout.addWidget(self.phase_lock)
        self.scroll_layout.addWidget(self.lfp_spike_causality)

        self.function_widget = ScrollableWidget()
        self.function_widget.setContents(self.scroll_layout)

        self.function_select_label = add_label(
            text="Analysis Selection", obj_name="funcSelectLabel")
        self.select_all_box = add_check_box(
            text="Select All", obj_name="selectAll")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.function_select_label, 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.select_all_box, 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.function_widget, 5)

        return layout

    def retranslate_ui(self):
        """
        Sets up the title and icon in NeuroChaT GUI

        """

        self.setWindowTitle(_translate("MainWindow", "NeuroChaT", None))
        # <-- absolute dir the script is in
        script_dir = os.path.dirname(__file__)
        rel_path = "../NeuroChat.png"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.setWindowIcon(QtGui.QIcon(abs_file_path))

    def start(self):
        """
        Called when start button is clicked. Starts the entire backend operation
        in NeuroChaT

        """
        self._get_config()
        self._control.finished.connect(self.restore_start_button)
        self._control.start()
        # self.nthread = QtCore.QThread()
        # self.worker = Worker(self.startAnalysis)
        # self.worker.moveToThread(self.nthread)
        # self.nthread.started.connect(self.worker.run)
        # [self.worker.finished.connect(x) for x in [self.restoreStartButton, self.nthread.quit]]

        # self.nthread.start()

    def restore_start_button(self):
        """
        After reactivating the start button, it displays the results in the UI table.

        """
        pd_model = PandasModel(self._control.get_results())
        self._results_ui.set_data(pd_model)
        self._results_ui.show()

    def export_results(self):
        """
        Called when 'Export Results' button is clicked. Opens a file dialogue for the
        selection of an Excel file, and exports the results in the table to the file

        """

        excel_file = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export analysis results to...', 
                os.getcwd() + os.sep + 'nc_results.xlsx', 
                "Excel Files (*.xlsx .*xls)")[0]
        )
        if not excel_file:
            logging.warning(
                "No excel file selected! Results cannot be exported!")
        else:
            try:
                results = self._control.get_results()
                results.to_excel(excel_file)
                logging.info("Analysis results exported to: " +
                             excel_file.rstrip("\n\r").split(os.sep)[-1])
            except:
                logging.error('Failed to export results!')

    def export_graphic_info(self):
        """
        Called when 'Export graphic file info' menu is clicked. Opens a file dialogue for the
        selection of an Excel file, and exports the graphic file infor in the table to the file

        """

        excel_file = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export information to...', 
                os.getcwd() + os.sep + 'nc_graphicInfo.xlsx', 
                "Excel Files (*.xlsx .*xls)")[0]
        )
        if not excel_file:
            logging.warning(
                "No excel file selected! Information cannot be exported!")
        else:
            try:
                info = self._control.get_output_files()
                info.to_excel(excel_file)
                logging.info("Graphics information exported to: " +
                             excel_file.rstrip("\n\r").split(os.sep)[-1])
            except:
                logging.error('Failed to export graphics information!')

    def closeEvent(self, event):
        """
        Called when NeuroChaT window is about to close. Opens a dialogue for saving the
        session information in NeuroChaT configuration file (.ncfg).

        """
        def save_last(event):
            try:
                with open(self._default_loc, "w") as f:
                    f.write(os.getcwd())
                event.accept()
            except Exception as e:
                log_exception(e, "Failed to save last location {} in {}".format(
                    os.getcwd(), self._default_loc))
                event.ignore()

        reply = QtWidgets.QMessageBox.question(self, "Message",
                                               "Save current session before you quit?",
                                               QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close | QtWidgets.QMessageBox.Cancel,
                                               QtWidgets.QMessageBox.Save)
        if reply == QtWidgets.QMessageBox.Save:
            config_file = QtCore.QDir.toNativeSeparators(
                QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save session as...', os.getcwd(), "*.ncfg")[0])
            if config_file:
                try:
                    self._get_config()
                    self._control.save_config(config_file)
                    logging.info("Session saved in: " + config_file)
                    os.chdir(os.path.dirname(config_file))
                    save_last(event)
                except:
                    logging.error('Failed to save configuration!')
                    event.ignore()
        elif reply == QtWidgets.QMessageBox.Close:
            save_last(event)
        else:
            event.ignore()

    def exit_nc(self):
        """
        Called when 'Exit' menu item is clicked. Closes the NeuroChaT window.

        """
        self.close()
#        print('ask for saving session...')
#        QtCore.QCoreApplication.instance().quit()

    def data_format_select(self, ind):
        """
        Called when there is a change in the data format selection combo box.
        Sets the data format to the selected item.

        """

        data_format = self.file_format_box.itemText(ind)
        self._control.set_data_format(data_format)
        logging.info("Input data format set to: " + data_format)
        self._set_dictation()
        if data_format == 'Axona' or data_format == 'Neuralynx':
            self._control.set_nwb_file('')

    def mode_select(self, ind):
        """
        Called when there is a change in the analysis mode selection combo box.
        Sets the data analysis mode to the selected item.

        """

        self._control.set_analysis_mode(ind)
        logging.info("Analysis mode set to: " + self.mode_box.itemText(ind))
        self._set_dictation()

    def graphic_format_select(self):
        """
        Called when there is a change in the graphic format selection button groups.
        Sets the output graphic format to the selected item.

        """

        button = self.graphic_format_group.checkedButton()
        text = button.text()
        self._control.set_graphic_format(text)
        logging.info("Graphic file format set to: " + text)

    def cell_type_select(self):
        """
        Called when there is a change in the cell type selection button groups.
        Sets the cell type to the selected item.

        """

        button = self.cell_type_group.checkedButton()
        text = button.text()
        self._control.set_cell_type(text)
        logging.info("Cell type set to: " + text)
        self.cell_type_analysis(text)

    def select_all(self):
        """
        Called when 'Select All' box is checked or unchecked. It checks or unchecks all other analyses.

        """

        if self.select_all_box.isChecked():
            logging.info("Selected ALL analyses")
            for checkbox in self.function_widget.findChildren(QtWidgets.QCheckBox):
                checkbox.setChecked(True)
        else:
            logging.info("Deselected ALL analyses")
            for checkbox in self.function_widget.findChildren(QtWidgets.QCheckBox):
                checkbox.setChecked(False)

    def lfp_chan_getitems(self):
        """
        Returns the list of LFP files (Neuralynx) or their file extension (Axona)
        once the spike data is set using the 'Browse' button

        """

        file_format = self.file_format_box.itemText(
            self.file_format_box.currentIndex())
        items = [""]
        if file_format == "Neuralynx":
            files = os.listdir(os.getcwd())
            items = [f for f in files if f.endswith('ncs')]

        elif file_format == "Axona":
            files = os.listdir(os.getcwd())
            items = [f.split('.')[-1]
                     for f in files if '.eeg' in f or '.egf' in f]

        elif file_format == "NWB":
            try:
                path = '/processing/Neural Continuous/LFP'
                self._control.open_hdf_file()
                items = self._control.get_hdf_groups(path=path)
                self._control.close_hdf_file()

#                hdf = Nhdf()
#                hdf.set_filename(self._control.get_nwb_file())
#                if path in hdf.f:
#                    items = list(hdf.f[path].keys())
#                else:
#                    logging.warning('No Lfp channel stored in the path:'+ path)
            except:
                logging.error('Cannot read the hdf file')
        else:
            items = [str(i) for i in list(range(256))]

        self.lfp_chan_box.clear()
        self.lfp_chan_box.addItems(items)

    def unit_getitems(self):
        """Returns the list of units once spike data is set."""
        try:
            file_format = self.file_format_box.itemText(
                self.file_format_box.currentIndex())
            if file_format == "NWB":
                self._control.open_hdf_file()
                _, path = self._control.get_spike_file().split("+")
                path = path + '/Clustering/cluster_nums'
                items = self._control.hdf.f[path]
            else:
                data = self._control.ndata
                data.load_spike()
                items = data.get_unit_list()
            if len(items) == 0:
                logging.error("No units in this file.")
                items = [i for i in range(256)]
            str_items = [str(x) for x in items]
            self.unit_no_box.clear()
            self.unit_no_box.addItems(str_items)
            if file_format == "NWB":
                self._control.close_hdf_file()
        except Exception as e:
            log_exception(e, "Populating unit list failed")
            items = [str(i) for i in range(256)]
            self.unit_no_box.clear()
            self.unit_no_box.addItems(items)

    def browse(self):
        """
        Opens a file dialog asking the user to select spike and spatial data files.
        Once selected, it also set the LFP channels in the 'LFP Ch No' combo box.

        """

        mode_id = self.mode_box.currentIndex()
        file_format = self._control.get_data_format()
        if mode_id == 0 or mode_id == 1:
            if file_format == "Axona" or file_format == "Neuralynx":
                if file_format == "Axona":
                    spike_filter = "".join(
                        ["*." + str(x) + ";;" for x in list(range(1, 129))])
                    spatial_filter = "*.txt"
                elif file_format == "Neuralynx":
                    spike_filter = "*.ntt;; *.nst;; *.nse"
                    spatial_filter = "*.nvt"
                spike_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                                  'Select spike file...', os.getcwd(), spike_filter)[0])
                if not spike_file:
                    logging.warning("No spike file selected")
                else:
                    words = spike_file.rstrip("\n\r").split(os.sep)
                    directory = os.sep.join(words[0:-1])
                    os.chdir(directory)
#                    spike_file = words[-1]
#                    self._curr_dir = directory
                    self._control.set_spike_file(spike_file)
                    self._control.ndata.set_spike_file(spike_file)
                    logging.info("New spike file added: " +
                                 words[-1])
                    spatial_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                                        'Select spatial file...', os.getcwd(), spatial_filter)[0])
                    if not spatial_file:
                        logging.warning("No spatial file selected")
                    else:
                        words = spatial_file.rstrip("\n\r").split(os.sep)
#                        spatial_file = words[-1]
                        self._control.set_spatial_file(spatial_file)
                        logging.info("New spatial file added: " +
                                     words[-1])
                    self.lfp_chan_getitems()
                    self.unit_getitems()

            elif file_format == "NWB":
                nwb_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                                'Select NWB file...', os.getcwd(), "*.hdf5")[0])
                if not nwb_file:
                    logging.warning("No NWB file selected")
                else:
                    words = nwb_file.rstrip("\n\r").split(os.sep)
                    directory = os.sep.join(words[0:-1])
                    os.chdir(directory)
                    self._control.set_nwb_file(nwb_file)

                    logging.info("New NWB file added: " +
                                 words[-1])
                    try:
                        path = '/processing/Shank'
                        self._control.open_hdf_file()
                        items = self._control.get_hdf_groups(path=path)
                        # hdf = Nhdf()
                        # hdf.set_filename(nwb_file)
                        # path = '/processing/Shank'
                        # if path in hdf.f:
                        #     items = list(hdf.f[path].keys())
                        # else:
                        #     logging.warning('No Shank data stored in the path:'+ path)
                        if items:
                            item, ok = QtWidgets.QInputDialog.getItem(
                                self, "Select electrode group",
                                "Electrode groups: ", items, 0, False)
                            if ok:
                                self._control.set_spike_file(
                                    nwb_file + '+' + path + '/' + item)
                                logging.info(
                                    'Spike data set to electrode group: '
                                    + path + '/' + item)

                                path = '/processing/Behavioural/Position'
                                if self._control.exist_hdf_path(path=path):
                                    self._control.set_spatial_file(
                                        nwb_file + '+' + path)
                                    logging.info(
                                        'Position data set to group: ' + path)
                                else:
                                    logging.warning(
                                        path +
                                        ' not found! Spatial data not set.')
                        self._control.close_hdf_file()
                    except Exception as e:
                        log_exception(
                            e, "Cannot read hdf file in nc_ui browse")

                    self.lfp_chan_getitems()
                    self.unit_getitems()

        elif mode_id == 2:
            excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                              'Select Excel file...', os.getcwd(), "*.xlsx;; .*xls")[0])
            if not excel_file:
                logging.warning("No excel file selected")
            else:
                words = excel_file.rstrip("\n\r").split(os.sep)
                directory = os.sep.join(words[0:-1])
                os.chdir(directory)
                # self._curr_dir = directory
                # excel_file = words[-1]
                self._control.set_excel_file(excel_file)
                logging.info("New excel file added: " +
                             words[-1])
#        elif mode_id == 3:
#            data_directory = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(self, \
#                           'Select data directory...', os.getcwd()))
#            if not data_directory:
#                logging.warning("No data directory selected")
#            else:
##                self._curr_dir = data_directory
#                self._control.set_dataDir(data_directory)
#                logging.info("New directory added: "+ data_directory)

    def update_log(self, msg):
        """
        Updates the log-box with new message

        Parameters
        ----------
        msg
            New log message or record

        Returns
        -------
        None

        """

        self.log_text.insert_log(msg)

    def clear_log(self):
        """
        Clears the texts in the log box.

        """

        self.log_text.clear()
        logging.info("Log cleared!")

    def save_log(self):
        """
        Opens a file dialog for the user to select a text file where the current
        texts of the log-box are exported

        """

        text = self.log_text.get_text()
        name = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save log as...', os.getcwd(), "Text files(*.txt)")[0])
        if not name:
            logging.warning("File not specified. Log is not saved")
        else:
            try:
                file = open(name, 'w')
                file.write(text)
                logging.info("Log saved in: " + name)
            except:
                logging.error(
                    'Log is not saved! See if the file is open in another application!')

    def set_unit_no(self, value):
        """
        Called when the selection in the 'Unit No' is changed. Sets the unit number accordingly.

        """
        if value < 0:
            return
        unit_no = int(self.unit_no_box.itemText(value))
        self._control.set_unit_no(unit_no)
        logging.info("Selected Unit: " + str(unit_no))

    def set_lfp_chan(self, value):
        """
        Called when the selection in the 'LFP Ch No' is changed. Sets the lfp channel accordingly.

        """

        lfpID = self.lfp_chan_box.itemText(value)

        if lfpID:
            logging.info("Selected LFP channel: " + lfpID)
            data_format = self._control.get_data_format()
            if data_format == 'Axona':
                spike_file = self._control.get_spike_file()
                lfp_file = remove_extension(spike_file) + lfpID
            elif data_format == 'Neuralynx':
                spike_file = self._control.get_spike_file()
                print(os.sep.join(spike_file.split(
                    os.sep)[:-1]) + os.sep + lfpID)
                lfp_file = os.sep.join(spike_file.split(os.sep)[
                                       :-1]) + os.sep + lfpID
            elif data_format == 'NWB':
                nwb_file = self._control.get_nwb_file()
                lfp_file = nwb_file + '+' + '/processing/Neural Continuous/LFP' + '/' + lfpID
                # Will implement later
            else:
                logging.error('The input data format not supported!')
            self._control.set_lfp_file(lfp_file)

    def view_help(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "../docs/index.html"
        url = os.path.join(script_dir, rel_path)
        if not os.path.isfile(url):
            url = "https://seankmartin.github.io/NeuroChaT/"
        webbrowser.open_new(url)

    def tutorial(self):
        pass

    def about_nc(self):
        url = "https://github.com/shanemomara/omaraneurolab/wiki/NeuroChaT"
        webbrowser.open_new(url)

    def _set_dictation(self):
        """
        Sets the dictation in the text-box for the browse button as the input data format changes.

        """

        _dictation = ["Select spike(.n) &/or position file(.txt)",
                      "Select spike(.n) &/or position file(.txt)",
                      "Select excel(.xls/.xlsx) file with unit list",
                      "Select folder"]
        file_format = self._control.get_data_format()
        analysis_mode, mode_id = self._control.get_analysis_mode()
        if file_format == "Neuralynx":
            _dictation[:2] = [
                "Select spike(.ntt/.nst/.nse) &/or position file(.nvt)"] * 2
        elif file_format == "NWB":
            _dictation[:2] = ["Select .hdf5 file"] * 2
        self.filename_line.setText(_dictation[mode_id])

    def _get_config(self):
        """
        Retrives all the configurations from the GUI elements and sets them to the
        Configuration() object through the NeuroChaT() object.

        """
        # Get selected function from functioWidget
        for checkbox in self.function_widget.findChildren(QtWidgets.QCheckBox):
            self._control.set_analysis(
                checkbox.objectName(), checkbox.isChecked())

        for checkbox in self._param_ui.findChildren(QtWidgets.QCheckBox):
            self._control.set_param(
                checkbox.objectName(), checkbox.isChecked())

        for spinbox in self._param_ui.findChildren(QtWidgets.QSpinBox):
            self._control.set_param(spinbox.objectName(), spinbox.value())

        for spinbox in self._param_ui.findChildren(QtWidgets.QDoubleSpinBox):
            self._control.set_param(spinbox.objectName(), spinbox.value())

        for combobox in self._param_ui.findChildren(QtWidgets.QComboBox):
            text = combobox.currentText()
            try:
                value = int(text)
                self._control.set_param(combobox.objectName(), value)
            except:
                self._control.set_param(combobox.objectName(), text)

    def save_session(self):
        """
        Prompts the user to select a .ncfg file and saves the current settings and parameters from the GUI elements to the file.

        """

        ncfg_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save session as...', os.getcwd(), "*.ncfg")[0])
        if not ncfg_file:
            logging.warning("File not specified. Session is not saved")
        else:
            self._get_config()
            self._control.save_config(ncfg_file)
            logging.info("Session saved in: " + ncfg_file)
            os.chdir(os.path.dirname(ncfg_file))

    def load_session(self):
        """
        Prompts the user to select a .ncfg file and loads the settings and parameters from the file to the GUI elements.

        """

        ncfg_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select NCFG file...', os.getcwd(), "(*.ncfg)")[0])
        if not ncfg_file:
            logging.error("No saved session selected! Loading failed!")
        else:
            self._control.load_config(ncfg_file)
            os.chdir(os.path.dirname(ncfg_file))

        index = self.file_format_box.findText(self._control.get_data_format())
        if index >= 0:
            self.file_format_box.setCurrentIndex(index)
        mode, mode_id = self._control.get_analysis_mode()
        self.mode_box.setCurrentIndex(mode_id)

        getattr(self, self._control.get_graphic_format() +
                '_button').setChecked(True)
        self.graphic_format_select()

        self.unit_getitems()
        index = self._control.get_unit_no()
        if index >= 0 & index < 256:
            self.unit_no_box.setCurrentIndex(index)

        file = self._control.get_lfp_file()
        if self._control.get_data_format() == 'Axona':
            file_tag = file.split('.')[-1]
        elif self._control.get_data_format() == 'Neuralynx':
            file_tag = file.split(os.sep)[-1].split('.')[0]
        elif self._control.get_data_format() == 'NWB':
            file_tag = file.split('/')[-1]

        self.lfp_chan_getitems()
        index = self.lfp_chan_box.findText(file_tag)
        if index >= 0:
            self.lfp_chan_box.setCurrentIndex(index)

        cell_type = self._control.get_cell_type()
        select_by_type = False
        for button in self.cell_type_group.buttons():
            if button.text == cell_type:
                select_by_type = True
                button.click()
        if not select_by_type:
            for key in self._control.get_analysis_list():
                getattr(self, key).setChecked(self._control.get_analysis(key))

        param_list = self._control.get_param_list()

        for name in param_list:
            param_widget = self._param_ui.findChild(
                QtWidgets.QWidget, name)
            if isinstance(param_widget, QtWidgets.QComboBox):
                index = param_widget.findText(
                    str(self._control.get_params(name)))
                if index >= 0:
                    param_widget.setCurrentIndex(index)
            elif isinstance(param_widget, QtWidgets.QCheckBox):
                param_widget.setChecked(self._control.get_params(name))
            else:
                param_widget.setValue(self._control.get_params(name))

    def merge_output(self):
        """
        Opens the UiMerge() object for the user to merge the selected PDF or Postscript files.

        """
        self._merge_ui.merge_enable = True
        self._merge_ui.setWindowTitle(QtWidgets.QApplication.translate(
            "mergeWindow", "Merge PDF/PS", None))
        self._merge_ui.set_default()
        self._merge_ui.show()
        logging.info(
            "Tool to MERGE graphic files activated! Only PDF files can be merged!")

    def accumulate_output(self):
        """
        Opens the UiMerge() object for the user to accumulate the selected PDF or Postscript files.

        """

        self._merge_ui.merge_enable = False
        self._merge_ui.setWindowTitle(QtWidgets.QApplication.translate(
            "mergeWindow", "Accumulate PDF/PS", None))
        self._merge_ui.set_default()
        self._merge_ui.show()
        logging.info("Tool to ACCUMULATE graphic files activated")

    def compare_units(self):
        """
        Opens a file dialog for selecting the Excel list that contains specifications for comparing units
        and compares the units through NeuroChaT().cluster_similarity() method.

        See also
        --------
        NeuroChaT().cluster_similarity()

        """

        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select unit-pair list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning(
                "No excel file selected! Comparing units is unsuccessful!")
        else:
            self._merge_ui.filename_line.setText(excel_file)
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])
        excel_data = pd.read_excel(excel_file)
        print('Create an excel read warpper from PANDAS')
        print(excel_data)

    def angle_calculation(self):
        """
        Open an excel file and calculate angles between centroids.

        See also
        --------
        NeuroChaT.angle_calculation
        """
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning("No excel file selected!")
        else:
            self._get_config()
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])

            pdf_name = excel_file[:excel_file.find(".")] + "_output.pdf"
            dict_info = {
                "key": self.angle_calculation.__name__,
                "excel_file": excel_file,
                "pdf_name": pdf_name}
            self._control.set_special_analysis(dict_info)
            self.start()

    def place_cell_plots(self):
        """
        Plot place cell figures for each set file in a directory.
        """
        directory = (
            QtCore.QDir.toNativeSeparators(
                QtWidgets.QFileDialog.getExistingDirectory(
                    self, 'Select data folder', os.getcwd(),
                    QtWidgets.QFileDialog.ShowDirsOnly
                    | QtWidgets.QFileDialog.DontResolveSymlinks)))
        if directory:
            dict_info = {
                "key": self.place_cell_plots.__name__,
                "directory": directory,
                "dpi": 400}
            self._control.set_special_analysis(dict_info)
            self.start()

    def verify_units(self):
        """
        Opens a file dialog for selecting the Excel list that contains specifications for verifying the units
        and verifies the unit using the NeuroChaT().verify_units() method.

        See also
        --------
        NeuroChaT().compare_units()

        """

        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning(
                "No excel file selected! Verification of units is unsuccessful!")
        else:
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])
            self._control.verify_units(excel_file)

    def cluster_evaluate(self):
        """
        Opens a file dialog for selecting the Excel list that contains specifications for cluster evaluation
        and evaluates the clusters using the NeuroChaT().cluster_evaluate() method.

        See also
        --------
        NeuroChaT().cluster_evaluate()

        """

        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning(
                "No excel file selected! Verification of units is unsuccessful!")
        else:
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])
            self._control.cluster_evaluate(excel_file)

    def convert_to_nwb(self):
        """
        Opens a file dialog for selecting the Excel list that contains specifications for NWB file for conversion.
        It then converts the files using the NeuroChaT().convert_to_nwb() method.

        See also
        --------
        NeuroChaT().convert_to_nwb()

        """

        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning(
                "No excel file selected! Conversion to NWB is unsuccessful!")
        else:
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])
            self._control.convert_to_nwb(excel_file)

    def set_parameters(self):
        """
        Shows the UiParameters() widget once the user clicks the 'Parameters' menu item for setting the paramters.

        """
        self._param_ui.show()

    def cell_type_analysis(self, cell_type):
        """
        Sets the analysis checkboxes based on the type of cell selected.

        """

        if cell_type == "Place":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(True)
            self.loc_shuffle.setChecked(True)
#            self.place_field.setChecked(True)
            self.loc_time_lapse.setChecked(True)
            self.loc_time_shift.setChecked(True)

            self.spatial_corr.setChecked(True)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "Head-directional":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(True)
            self.hd_shuffle.setChecked(True)
            self.hd_time_lapse.setChecked(True)
            self.hd_time_shift.setChecked(True)

            self.loc_rate.setChecked(False)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(False)
            self.loc_time_lapse.setChecked(False)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "Grid":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(True)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(True)
            self.loc_time_lapse.setChecked(True)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(True)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "Boundary":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(True)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(True)
            self.loc_time_lapse.setChecked(True)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(False)
            self.border.setChecked(True)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "Gradient":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(True)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(True)
            self.loc_time_lapse.setChecked(True)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(True)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "HDxPlace":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(True)
            self.hd_shuffle.setChecked(True)
            self.hd_time_lapse.setChecked(True)
            self.hd_time_shift.setChecked(True)

            self.loc_rate.setChecked(True)
            self.loc_shuffle.setChecked(True)
#            self.place_field.setChecked(True)
            self.loc_time_lapse.setChecked(True)
            self.loc_time_shift.setChecked(True)

            self.spatial_corr.setChecked(True)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(True)
            self.inter_depend.setChecked(True)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(False)
            self.spike_phase.setChecked(False)
            self.phase_lock.setChecked(False)
            self.lfp_spike_causality.setChecked(False)

        elif cell_type == "Theta-rhythmic":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(True)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(False)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(False)
            self.loc_time_lapse.setChecked(False)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(False)
            self.inter_depend.setChecked(False)

            self.theta_cell.setChecked(True)
            self.theta_skip_cell.setChecked(False)

            self.lfp_spectrum.setChecked(True)
            self.spike_phase.setChecked(True)
            self.phase_lock.setChecked(True)
            self.lfp_spike_causality.setChecked(True)

        elif cell_type == "Theta-skipping":
            self.wave_property.setChecked(True)
            self.isi.setChecked(True)
            self.isi_corr.setChecked(True)

            self.burst.setChecked(True)

            self.speed.setChecked(True)
            self.ang_vel.setChecked(True)

            self.hd_rate.setChecked(False)
            self.hd_shuffle.setChecked(False)
            self.hd_time_lapse.setChecked(False)
            self.hd_time_shift.setChecked(False)

            self.loc_rate.setChecked(False)
            self.loc_shuffle.setChecked(False)
#            self.place_field.setChecked(False)
            self.loc_time_lapse.setChecked(False)
            self.loc_time_shift.setChecked(False)

            self.spatial_corr.setChecked(False)

            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)

            self.multiple_regression.setChecked(False)
            self.inter_depend.setChecked(False)

            self.theta_cell.setChecked(False)
            self.theta_skip_cell.setChecked(True)

            self.lfp_spectrum.setChecked(True)
            self.spike_phase.setChecked(True)
            self.phase_lock.setChecked(True)
            self.lfp_spike_causality.setChecked(True)


class UiResults(QtWidgets.QDialog):
    """
    NeuroChaT user interface for displaying the analysis results and and facilitating their export.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def setup_ui(self):
        """
        Sets up the GUI elements of the widget and their behaviour. Clicking on the
        'Export Results' button calls the NeuroChaT_Ui.export_results() method.

        See also
        --------
        PandasModel

        """

        self.setObjectName(xlt_from_utf8("resultsWindow"))
        self.setEnabled(True)
        self.setFixedSize(725, 220)
        self.setWindowTitle(QtWidgets.QApplication.translate(
            "resultsWindow", "Analysis results", None))

        # layout
        self.layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableView()
        self.table.resizeColumnsToContents()
        self.table.showGrid()
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self.export_button = add_push_button(
            text="Export results", obj_name="Export")

        self.layout.addWidget(self.export_button)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)
        self.set_default()

    def set_data(self, pd_model):
        """
        Sets the PandasModel as the data model for the table-view.

        Parameters
        ----------
        pd_model : PandasModel
            PandasModel as the table-data

        """

        self.table.setModel(pd_model)

#    def show_results(self):
#        self.table.show()
#        self.show()

    def set_default(self):
        """
        Not implemented. Can be used for clearing the table and the data model
        underneath.

        """

        pass

# class UiConvert(QtWidgets.QDialog):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#    def setup_ui(self):
#
#        self.setObjectName(xlt_from_utf8("convertWindow"))
#        self.setEnabled(True)
#        self.setFixedSize(324, 220)
#        self.setWindowTitle("Convert Files")
#
#        self.inp_format_label = add_label(self, (20, 10, 111, 17), "inpFormatLabel", "Convert From")
#
#        self.file_format_box = add_combo_box(self, (20, 30, 111, 22), "fileFormatBox")
#        self.file_format_box.addItems(["Axona", "Neuralynx"])
#
#        self.browse_excel_button = add_push_button(self, (30, 60, 60, 23), "browseExcel", "Browse")
#
#        self.filename_line = add_line_edit(self, (95, 60, 215, 23), "filename", "")
#        self.filename_line.setText("Select Excel (.xls/.xlsx) file")


class UiParameters(QtWidgets.QDialog):
    """
    NeuroChaT user interface for setting analysis specific parameters.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        self.behaviour_ui()

    def setup_ui(self):
        """
        Sets the GUI elements for the widget.

        """
        self.setObjectName(xlt_from_utf8("paramSetWindow"))
        self.setEnabled(True)
        self.setFixedSize(640, 320)
        self.setWindowTitle(QtWidgets.QApplication.translate(
            "paramSetWindow", "Parameter settings", None))

        self.selectLabel = add_label(
            self, (10, 10, 180, 17), "inpFormatLabel", "Select analysis to set parameters")

        self.param_list = QtWidgets.QListWidget(self)
        self.param_list.setGeometry(QtCore.QRect(10, 30, 170, 280))
        self.param_list.setObjectName(xlt_from_utf8("paramList"))

        items = []
        widget_names = []
        for checkbox in self.parent.function_widget.findChildren(QtWidgets.QCheckBox):
            items.append(checkbox.text())
            widget_names.append(checkbox.objectName())

        self.param_list.addItems(items)

        self.param_stack = QtWidgets.QStackedWidget(self)
        self.param_stack.setGeometry(QtCore.QRect(190, 30, 440, 280))
        self.param_stack.setObjectName(xlt_from_utf8("paramStack"))

        self.param_stack.addWidget(self.waveform_page())

        self.param_stack.addWidget(self.isi_page())

        self.param_stack.addWidget(self.isi_corr_page())

        self.param_stack.addWidget(self.theta_cell_page())

        self.param_stack.addWidget(self.theta_skip_cell_page())

        self.param_stack.addWidget(self.burst_page())

        self.param_stack.addWidget(self.speed_page())

        self.param_stack.addWidget(self.ang_vel_page())

        self.param_stack.addWidget(self.hd_rate_page())

        self.param_stack.addWidget(self.hd_shuffle_page())

        self.param_stack.addWidget(self.hd_time_lapse_page())

        self.param_stack.addWidget(self.hd_time_shift_page())

        self.param_stack.addWidget(self.loc_rate_page())

        self.param_stack.addWidget(self.loc_shuffle_page())

#        self.param_stack.addWidget(self.plce_field_page())

        self.param_stack.addWidget(self.loc_time_lapse_page())

        self.param_stack.addWidget(self.loc_time_shift_page())

        self.param_stack.addWidget(self.spatial_corr_page())

        self.param_stack.addWidget(self.grid_page())

        self.param_stack.addWidget(self.border_page())

        self.param_stack.addWidget(self.gradient_page())

        self.param_stack.addWidget(self.multiple_regresison_page())

        self.param_stack.addWidget(self.inter_depend_page())

        self.param_stack.addWidget(self.lfp_spectrum_page())

        self.param_stack.addWidget(self.spike_phase_page())

        self.param_stack.addWidget(self.phase_lock_page())

        self.param_stack.addWidget(self.lfp_spike_causality_page())

    def behaviour_ui(self):
        """
        Sets the behaviour of the GUI elements.

        """

        self.param_list.itemActivated.connect(self.change_stack_page)
        self.loc_rate_filter.activated[str].connect(self.set_loc_rate_filter)
        self.spatial_corr_filter.activated[str].connect(
            self.set_spat_corr_filter)
        self.param_list.itemActivated.connect(self.change_stack_page)

    def change_stack_page(self):
        """
        Changes the stacked widgets of parameter setting according to the analysis selected from the lis on left of the window.

        """

        self.param_stack.setCurrentWidget(
            self.param_stack.widget(self.param_list.currentRow()))

    def set_loc_rate_filter(self, filt_type):
        """
        Sets the ui elements for the filters for locational firing rate map.

        """

        if filt_type == "Gaussian":
            self.loc_rate_kern_len.setSingleStep(1)
            self.loc_rate_kern_len.setValue(3)
        elif filt_type == "Box":
            self.loc_rate_kern_len.setSingleStep(2)
            self.loc_rate_kern_len.setValue(5)

    def set_spat_corr_filter(self, filt_type):
        """
        Sets the ui elements for the filters for spatial autocorrelation of locational firing rate map.

        """

        if filt_type == "Gaussian":
            self.spatial_corr_kern_len.setSingleStep(1)
            self.spatial_corr_kern_len.setValue(3)
        elif filt_type == "Box":
            self.spatial_corr_kern_len.setSingleStep(2)
            self.spatial_corr_kern_len.setValue(5)

    def waveform_page(self):
        """
        Sets the ui elements for the 'Waveform Analysis' parameters.

        """
        widget = ScrollableWidget()
        self.waveform_gb1 = add_group_box(title="", obj_name="waveform_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.waveform_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.waveform_gb1)

        widget.setContents(layout)
        return widget

    def isi_page(self):
        """
        Sets the ui elements for the 'isi' analysis parameters.

        """

        widget = ScrollableWidget()

        # Box- 1
        self.isi_gb1 = add_group_box(title="Histogram", obj_name="isi_gb1")
        self.isi_bin = add_spin_box(min_val=1, max_val=50, obj_name="isi_bin")
        self.isi_bin.setValue(1)

        self.isi_length = add_spin_box(
            min_val=10, max_val=1000, obj_name="isi_length")
        self.isi_length.setValue(200)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Histogram Binsize",
                          self.isi_bin, "ms [range: 1-50]")
        box_layout.addRow("Histogram Length", self.isi_length,
                          "ms [range: 10-1000]")

        self.isi_gb1.setLayout(box_layout)

        # Box- 2
        self.isi_gb2 = add_group_box(title="log-log plot", obj_name="isi_gb2")

        self.isi_log_no_bins = add_spin_box(
            min_val=10, max_val=100, obj_name="isiLogNoBins")
        self.isi_log_no_bins.setValue(70)

        self.isi_log_length = add_spin_box(
            min_val=10, max_val=1000, obj_name="isiLogLength")
        self.isi_log_length.setValue(350)

        box_layout = ParamBoxLayout()
        box_layout.addRow("No. of Histogram Bins",
                          self.isi_log_no_bins, "[range: 10-100]")
        box_layout.addRow("Histogram Length",
                          self.isi_log_length, "ms [range: 10-1000]")

        self.isi_gb2.setLayout(box_layout)

        # Box- 3
        self.isi_gb3 = add_group_box(title="Refractory", obj_name="isi_gb3")

        self.isi_refractory = add_double_spin_box(
            min_val=0, max_val=10, obj_name="isi_refractory")
        self.isi_refractory.setSingleStep(0.1)
        self.isi_refractory.setValue(2)

        box_layout = ParamBoxLayout()
        box_layout.addRow(
            "Refractory Threshold", self.isi_refractory, "ms [range 0 - 10]")
        self.isi_gb3.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.isi_gb1)
        layout.addWidget(self.isi_gb2)
        layout.addWidget(self.isi_gb3)

        widget.setContents(layout)

        return widget

    def isi_corr_page(self):
        """
        Sets the ui elements for the 'isi_corr' analysis parameters.

        """

        widget = ScrollableWidget()

        # Box- 1
        self.isi_corr_gb1 = add_group_box(
            title="Zoomed In", obj_name="isi_corr_gb1")

        self.isi_corr_bin_short = add_spin_box(
            min_val=1, max_val=10, obj_name="isi_corr_bin_short")
        self.isi_corr_bin_short.setValue(1)

        self.isi_corr_len_short = add_spin_box(
            min_val=5, max_val=50, obj_name="isi_corr_len_short")
        self.isi_corr_len_short.setValue(10)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Autcorrelation Histogram Binsize",
                          self.isi_corr_bin_short, "ms [range: 1-10]")
        box_layout.addRow("Autocorrelation Length",
                          self.isi_corr_len_short, "ms [range: 5-50]")

        self.isi_corr_gb1.setLayout(box_layout)

        # Box- 2
        self.isi_corr_gb2 = add_group_box(
            title="Zoomed Out", obj_name="isi_corr_gb2")

        self.isi_corr_bin_long = add_spin_box(
            min_val=1, max_val=50, obj_name="isi_corr_bin_long")
        self.isi_corr_bin_long.setValue(1)

        self.isi_corr_len_long = add_spin_box(
            min_val=10, max_val=1000, obj_name="isi_corr_len_long")
        self.isi_corr_len_long.setValue(350)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Autcorrelation Histogram Binsize",
                          self.isi_corr_bin_long, "ms [range: 1-50]")
        box_layout.addRow("Autocorrelation Length",
                          self.isi_corr_len_long, "ms [range: 10-1000]")

        self.isi_corr_gb2.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.isi_corr_gb1)
        layout.addWidget(self.isi_corr_gb2)

        widget.setContents(layout)
        return widget

    def theta_cell_page(self):
        """
        Sets the ui elements for the 'theta_cell' analysis parameters.

        """

        widget = ScrollableWidget()
        self.theta_cell_gb1 = add_group_box(
            title="Curve Fitting Parameters", obj_name="theta_cell_gb1")

        self.theta_cell_freq_min = add_double_spin_box(
            min_val=1, max_val=10, obj_name="theta_cell_freq_min")
        self.theta_cell_freq_min.setValue(6)
        self.theta_cell_freq_min.setSingleStep(0.5)

        self.theta_cell_freq_max = add_double_spin_box(
            min_val=8, max_val=16, obj_name="theta_cell_freq_max")
        self.theta_cell_freq_max.setValue(12)
        self.theta_cell_freq_max.setSingleStep(0.5)

        self.theta_cell_freq_start = add_double_spin_box(
            min_val=5, max_val=10, obj_name="theta_cell_freq_start")
        self.theta_cell_freq_start.setValue(6)
        self.theta_cell_freq_start.setSingleStep(0.5)

        self.theta_cell_tau1_max = add_double_spin_box(
            min_val=0.5, max_val=15, obj_name="theta_cell_tau1_max")
        self.theta_cell_tau1_max.setValue(5)
        self.theta_cell_tau1_max.setSingleStep(0.5)

        self.theta_cell_tau1_start = add_double_spin_box(
            min_val=0, max_val=15, obj_name="theta_cell_tau1_start")
        self.theta_cell_tau1_start.setValue(0.1)
        self.theta_cell_tau1_start.setSingleStep(0.05)

        self.theta_cell_tau2_max = add_double_spin_box(
            min_val=0, max_val=0.1, obj_name="theta_cell_tau2_max")
        self.theta_cell_tau2_max.setValue(0.05)
        self.theta_cell_tau2_max.setSingleStep(0.005)

        self.theta_cell_tau2_start = add_double_spin_box(
            min_val=0, max_val=0.1, obj_name="theta_cell_tau2_start")
        self.theta_cell_tau2_start.setValue(0.05)
        self.theta_cell_tau2_start.setSingleStep(0.005)

        box_layout = ParamBoxLayout()
        box_layout.addRow(
            "Minimum Frequency", self.theta_cell_freq_min, "Hz [range: 1-10, step: 0.5]")
        box_layout.addRow(
            "Maximum Frequency", self.theta_cell_freq_max, "Hz [range: 8-16, step: 0.5]")
        box_layout.addRow(
            "Starting Frequency", self.theta_cell_freq_start, "Hz [range: 5-10, step: 0.5]")
        box_layout.addRow("Max Time Constant (Tau-1)",
                          self.theta_cell_tau1_max, "sec [range: 0.5-10, step: 0.5]")
        box_layout.addRow(
            "Starting Tau-1", self.theta_cell_tau1_start, "sec [range: 0-15, step: 0.05]")
        box_layout.addRow("Gaussian Time Constant (Tau-2)",
                          self.theta_cell_tau2_max, "sec [range: 0-0.1, step: 0.005]")
        box_layout.addRow(
            "Start of Tau-2", self.theta_cell_tau2_start, "sec [range: 0-0.1, step: 0.005]")

        self.theta_cell_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.theta_cell_gb1)

        widget.setContents(layout)

        return widget

    def theta_skip_cell_page(self):
        """
        Sets the ui elements for the 'theta_skip_cell' analysis parameters.

        """

        widget = ScrollableWidget()
        self.theta_skip_cell_gb1 = add_group_box(
            title="Curve Fitting Parameters", obj_name="theta_skip_cell_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel("Uses the parameters from 'Theta-modulated Cell Index' analysis\n" +
                                              "The fitting parameters for 2nd frequency component is derived \n\rfrom the 1st component"))
        self.theta_skip_cell_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.theta_skip_cell_gb1)

        widget.setContents(layout)

        return widget

    def burst_page(self):
        """
        Sets the ui elements for the 'burst' analysis parameters.

        """

        widget = ScrollableWidget()

        # Box- 1
        self.burst_gb1 = add_group_box(
            title="Bursting conditions", obj_name="burst_gb1")

        self.burst_thresh = add_spin_box(
            min_val=1, max_val=15, obj_name="burst_thresh")
        self.burst_thresh.setValue(5)

        self.spikes_to_burst = add_spin_box(
            min_val=2, max_val=10, obj_name="spikesToBurst")
        self.spikes_to_burst.setValue(2)

        self.ibi_thresh = add_spin_box(
            min_val=5, max_val=1000, obj_name="ibi_thresh")
        self.ibi_thresh.setValue(50)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Burst Threshold",
                          self.burst_thresh, "ms [range: 1-15]")
        box_layout.addRow("Spikes to Burst",
                          self.spikes_to_burst, "[range: 2-10]")
        box_layout.addRow("Interburst Interval Lower Cutoff",
                          self.ibi_thresh, "ms [range: 5-1000]")

        self.burst_gb1.setLayout(box_layout)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.burst_gb1)

        widget.setContents(layout)
        return widget

    def speed_page(self):
        """
        Sets the ui elements for the 'burst' analysis parameters.

        """

        widget = ScrollableWidget()
        self.speed_gb1 = add_group_box(
            title="Analyses Parameters", obj_name="speed_gb1")

        self.speed_bin = add_spin_box(
            min_val=1, max_val=10, obj_name="speed_bin")
        self.speed_bin.setValue(1)

        self.speed_min = add_spin_box(
            min_val=0, max_val=10, obj_name="speed_min")
        self.speed_min.setValue(0)

        self.speed_max = add_spin_box(
            min_val=10, max_val=200, obj_name="speed_max")
        self.speed_max.setValue(40)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Speed Binsize", self.speed_bin,
                          "cm/sec [range: 1-10]")
        box_layout.addRow("Minimum Speed", self.speed_min,
                          "cm/sec [range: 0-10]")
        box_layout.addRow("Maximum Speed", self.speed_max,
                          "cm/sec [range: 10-200]")

        self.speed_gb1.setLayout(box_layout)

        self.speed_gb2 = add_group_box(
            title="Smoothing Box Kernal Length", obj_name="speed_gb2")

        self.speed_kern_len = add_spin_box(
            min_val=1, max_val=25, obj_name="speedKernLen")
        self.speed_kern_len.setValue(3)
        self.speed_kern_len.setSingleStep(2)
        # set validator in controller to accept only the odd numbers

        self.speed_rate_kern_len = add_spin_box(
            min_val=1, max_val=7, obj_name="speed_rate_kern_len")
        self.speed_rate_kern_len.setValue(3)
        self.speed_rate_kern_len.setSingleStep(2)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Speed", self.speed_kern_len,
                          "samples [range: 1-25, odds]")
        box_layout.addRow("Spike Rate", self.speed_rate_kern_len,
                          "bins [range: 1-7, odds]")

        self.speed_gb2.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.speed_gb1)
        layout.addWidget(self.speed_gb2)

        widget.setContents(layout)
        return widget

    def ang_vel_page(self):
        """
        Sets the ui elements for the 'ang_vel' analysis parameters.

        """

        widget = ScrollableWidget()

        self.ang_vel_gb1 = add_group_box(
            title="Analyses Parameters", obj_name="ang_vel_gb1")

        self.ang_vel_bin = add_spin_box(
            min_val=5, max_val=50, obj_name="ang_vel_bin")
        self.ang_vel_bin.setValue(10)
        self.ang_vel_bin.setSingleStep(5)

        self.ang_vel_min = add_spin_box(
            min_val=-500, max_val=0, obj_name="ang_vel_min")
        self.ang_vel_min.setValue(-200)

        self.ang_vel_max = add_spin_box(
            min_val=0, max_val=500, obj_name="ang_vel_max")
        self.ang_vel_max.setValue(200)

        self.ang_vel_cutoff = add_spin_box(
            min_val=0, max_val=100, obj_name="ang_vel_cutoff")
        self.ang_vel_cutoff.setValue(10)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Angular Velocity Binsize",
                          self.ang_vel_bin, "deg/sec [range: 1-50]")
        box_layout.addRow("Minimum Velocity", self.ang_vel_min,
                          "deg/sec [range: -500 to 0]")
        box_layout.addRow("Maximum Velocity", self.ang_vel_max,
                          "deg/sec [range: 0 to 500]")
        box_layout.addRow("Cutoff Velocity", self.ang_vel_cutoff,
                          "deg/sec [range: 0 to 100]")

        self.ang_vel_gb1.setLayout(box_layout)

        self.ang_vel_gb2 = add_group_box(
            title="Smoothing Box Kernal Length", obj_name="ang_vel_gb2")

        self.ang_vel_kern_len = add_spin_box(
            min_val=1, max_val=25, obj_name="ang_vel_kern_len")
        self.ang_vel_kern_len.setValue(3)
        self.ang_vel_kern_len.setSingleStep(2)
        # Set controller to recieve odd values

        self.ang_vel_rate_kern_len = add_spin_box(
            min_val=1, max_val=5, obj_name="ang_vel_rate_kern_len")
        self.ang_vel_rate_kern_len.setValue(3)
        self.ang_vel_rate_kern_len.setSingleStep(2)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Head Direction", self.ang_vel_kern_len,
                          "samples [range: 1-25, odds]")
        box_layout.addRow(
            "Spike Rate", self.ang_vel_rate_kern_len, "bins [range: 1-5, odds]")

        self.ang_vel_gb2.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ang_vel_gb1)
        layout.addWidget(self.ang_vel_gb2)

        widget.setContents(layout)
        return widget

    def hd_rate_page(self):
        """
        Sets the ui elements for the 'hd_rate' analysis parameters.

        """

        widget = ScrollableWidget()
        self.hd_rate_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="hd_rate_gb1")

        self.hd_bin = add_combo_box(obj_name="hd_bin")
        hd_bin_items = [str(d) for d in range(1, 360) if 360 %
                        d == 0 and d >= 5 and d <= 45]
        self.hd_bin.addItems(hd_bin_items)

        self.hd_ang_vel_cutoff = add_spin_box(
            min_val=0, max_val=100, obj_name="hd_ang_vel_cutoff")
        self.hd_ang_vel_cutoff.setValue(10)
        self.hd_ang_vel_cutoff.setSingleStep(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Head Directional Binsize", self.hd_bin, "degree")
        box_layout.addRow("Angular Velocity Cutoff",
                          self.hd_ang_vel_cutoff, "deg/sec [range: 0-100, step: 5]")

        self.hd_rate_gb1.setLayout(box_layout)

        self.hd_rate_gb2 = add_group_box(
            title="Smoothing Box Kernal Length", obj_name="hd_rate_gb2")

        self.hd_rate_kern_len = add_spin_box(
            min_val=1, max_val=11, obj_name="hd_rate_kern_len")
        self.hd_rate_kern_len.setValue(5)
        self.hd_rate_kern_len.setSingleStep(2)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Spike Rate", self.hd_rate_kern_len,
                          "bins [range: 1-11, odds]")

        self.hd_rate_gb2.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hd_rate_gb1)
        layout.addWidget(self.hd_rate_gb2)

        widget.setContents(layout)

        return widget

    def hd_shuffle_page(self):
        """
        Sets the ui elements for the 'hd_shuffle' analysis parameters.

        """

        widget = ScrollableWidget()
        self.hd_shuffle_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="hd_shuffle_gb1")

        self.hd_shuffle_total = add_spin_box(
            min_val=100, max_val=10000, obj_name="hd_shuffle_total")
        self.hd_shuffle_total.setValue(500)
        self.hd_shuffle_total.setSingleStep(50)

        self.hd_shuffle_limit = add_spin_box(
            min_val=0, max_val=500, obj_name="hd_shuffle_limit")
        self.hd_shuffle_limit.setValue(0)
        self.hd_shuffle_limit.setSingleStep(2)

        self.hd_shuffle_bins = add_spin_box(
            min_val=10, max_val=200, obj_name="hd_shuffle_bins")
        self.hd_shuffle_bins.setValue(100)
        self.hd_shuffle_bins.setSingleStep(10)

        box_layout = ParamBoxLayout()
        box_layout.addRow("No of Shuffles", self.hd_shuffle_total,
                          "[range: 100-10000, step: 50]")
        box_layout.addRow("Shuffling Limit", self.hd_shuffle_limit,
                          "sec [range: 0-500, 0 for Random, step: 2]")
        box_layout.addRow("No of Histogram Bins",
                          self.hd_shuffle_bins, "[range: 10-200, step: 10]")

        self.hd_shuffle_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hd_shuffle_gb1)

        widget.setContents(layout)

        return widget

    def hd_time_lapse_page(self):
        """
        Sets the ui elements for the 'hd_time_lapse' analysis parameters.

        """

        widget = ScrollableWidget()

        self.hd_time_lapse_gb1 = add_group_box(
            title="", obj_name="hd_time_lapse_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.hd_time_lapse_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hd_time_lapse_gb1)

        widget.setContents(layout)
        return widget

    def hd_time_shift_page(self):
        """
        Sets the ui elements for the 'hd_time_shift' analysis parameters.

        """
        widget = ScrollableWidget()

        self.hd_time_shift_gb1 = add_group_box(
            title="Shift Specifications", obj_name="hdTimeShift_gb1")

        self.hd_shift_max = add_spin_box(
            min_val=1, max_val=100, obj_name="hd_shift_max")
        self.hd_shift_max.setValue(10)

        self.hd_shift_min = add_spin_box(
            min_val=-100, max_val=-1, obj_name="hd_shift_min")
        self.hd_shift_min.setValue(-10)

        self.hd_shift_step = add_spin_box(
            min_val=1, max_val=3, obj_name="hd_shift_step")
        self.hd_shift_step.setValue(1)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Maximum Shift", self.hd_shift_max,
                          "indices [range: 1 to 100]")
        box_layout.addRow("Minimum Shift", self.hd_shift_min,
                          "indices [range: -1 to -100]")
        box_layout.addRow("Index Steps", self.hd_shift_step, "[1, 2, 3]")

        self.hd_time_shift_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.hd_time_shift_gb1)

        widget.setContents(layout)
        return widget

    def loc_rate_page(self):
        """
        Sets the ui elements for the 'loc_rate' analysis parameters.

        """

        widget = ScrollableWidget()
        # Box- 1
        self.loc_rate_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="loc_rate_gb1")

        self.loc_pixel_size = add_spin_box(
            min_val=1, max_val=100, obj_name="loc_pixel_size")
        self.loc_pixel_size.setValue(3)

        self.loc_chop_bound = add_spin_box(
            min_val=3, max_val=20, obj_name="loc_chop_bound")
        self.loc_chop_bound.setValue(5)

        self.loc_field_thresh = add_double_spin_box(
            min_val=0.0, max_val=1.0, obj_name="loc_field_thresh")
        self.loc_field_thresh.setValue(0.20)
        self.loc_field_thresh.setSingleStep(0.01)

        self.loc_field_smooth = add_check_box(obj_name='loc_field_smooth')
        self.loc_field_smooth.setChecked(False)

        self.loc_style = add_combo_box(obj_name="loc_style")
        self.loc_style.addItems(
            ["contour", "digitized", "interpolated"])

        self.loc_colormap = add_combo_box(obj_name="loc_colormap")
        self.loc_colormap.addItems(
            ["viridis", "default", "gray", "plasma",
             "inferno", "magma", "cividis"])
#        self.locAngVelCutoff = add_spin_box(min_val=0, 100, "locAngVelCutoff")
#        self.locAngVelCutoff.setValue(30)
#        self.locAngVelCutoff.setSingleStep(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow(
            "Pixel Size", self.loc_pixel_size, "cm [range: 1-100]")
        box_layout.addRow(
            "Bound for Chopping Edges",
            self.loc_chop_bound, "pixels [range: 3-20]")
#        box_layout.addRow("Angular Velocity Cutoff", self.locAngVelCutoff, "deg/sec [range: 0-100, step: 5]")
        box_layout.addRow(
            "Place Field Threshold",
            self.loc_field_thresh, "ratio [range: 0-1, step: 0.01]")
        box_layout.addRow(
            "Whether to Smooth Firing Map Before Centroid Calculation",
            self.loc_field_smooth, "True or False")

        self.loc_rate_gb1.setLayout(box_layout)

        # Box- 2
        self.loc_rate_gb2 = add_group_box(
            title="Smoothing Box Kernal", obj_name="loc_rate_gb2")

        self.loc_rate_filter = add_combo_box(obj_name="loc_rate_filter")
        self.loc_rate_filter.addItems(["Box", "Gaussian"])

        self.loc_rate_kern_len = add_spin_box(
            min_val=1, max_val=11, obj_name="loc_rate_kern_len")
        self.loc_rate_kern_len.setValue(5)
        self.loc_rate_kern_len.setSingleStep(2)
        # Change step size to 0.5 if Gaussian is selected

        box_layout = ParamBoxLayout()
        box_layout.addRow(
            "Smoothing Filter", self.loc_rate_filter, "")
        box_layout.addRow(
            "Spike Rate Pixels/Sigma",
            self.loc_rate_kern_len, "[range: 1-11]\n\r Box: odds")

        self.loc_rate_gb2.setLayout(box_layout)

        # Box 3
        self.loc_rate_gb3 = add_group_box(
            title="Plotting Style", obj_name="loc_rate_gb3")
        box_layout = ParamBoxLayout()
        box_layout.addRow(
            "Firing Rate Map Style", self.loc_style, "")
        box_layout.addRow(
            "Firing Rate Map Colormap", self.loc_colormap, "")
        self.loc_rate_gb3.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.loc_rate_gb1)
        layout.addWidget(self.loc_rate_gb2)
        layout.addWidget(self.loc_rate_gb3)

        widget.setContents(layout)

        return widget

    def loc_shuffle_page(self):
        """
        Sets the ui elements for the 'loc_shuffle' analysis parameters.

        """

        widget = ScrollableWidget()
        self.loc_shuffle_gb1 = add_group_box(
            title="Analyses Parameters", obj_name="loc_shuffle_gb1")

        self.loc_shuffle_total = add_spin_box(
            min_val=100, max_val=10000, obj_name="loc_shuffle_total")
        self.loc_shuffle_total.setValue(500)
        self.loc_shuffle_total.setSingleStep(50)

        self.loc_shuffle_limit = add_spin_box(
            min_val=0, max_val=500, obj_name="loc_shuffle_limit")
        self.loc_shuffle_limit.setValue(0)
        self.loc_shuffle_limit.setSingleStep(2)

        self.loc_shuffle_nbins = add_spin_box(
            min_val=10, max_val=200, obj_name="loc_shuffle_nbins")
        self.loc_shuffle_nbins.setValue(100)
        self.loc_shuffle_nbins.setSingleStep(10)

        box_layout = ParamBoxLayout()
        box_layout.addRow("No of Shuffles", self.loc_shuffle_total,
                          "[range: 100-10000, step: 50]")
        box_layout.addRow("Shuffling Limit", self.loc_shuffle_limit,
                          "sec [range: 0-500, 0 for Random, step: 2]")
        box_layout.addRow("No of Histogram Bins",
                          self.loc_shuffle_nbins, "[range: 10-200, step: 10]")

        self.loc_shuffle_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.loc_shuffle_gb1)

        widget.setContents(layout)

        return widget

#    def plce_field_page(self):
#        widget = ScrollableWidget()
#
#        self.placeField_gb1 = add_group_box(title="", obj_name="placeField_gb1")
#
#        box_layout = QtWidgets.QVBoxLayout()
#        box_layout.addWidget(QtWidgets.QLabel("No parameter to set"))
#        self.placeField_gb1.setLayout(box_layout)
#
#        layout = QtWidgets.QVBoxLayout()
#        layout.addWidget(self.placeField_gb1)
#
#        widget.setContents(layout)
#        return widget

    def loc_time_lapse_page(self):
        """
        Sets the ui elements for the 'loc_time_lapse' analysis parameters.

        """

        widget = ScrollableWidget()

        self.loc_time_lapse_gb1 = add_group_box(
            title="", obj_name="loc_time_lapse_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.loc_time_lapse_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.loc_time_lapse_gb1)

        widget.setContents(layout)
        return widget

    def loc_time_shift_page(self):
        """
        Sets the ui elements for the 'loc_time_shift' analysis parameters.

        """

        widget = ScrollableWidget()

        self.loc_time_shift_gb1 = add_group_box(
            title="Shift Specifications", obj_name="loc_time_shift_gb1")

        self.loc_shift_max = add_spin_box(
            min_val=1, max_val=100, obj_name="loc_shift_max")
        self.loc_shift_max.setValue(10)

        self.loc_shift_min = add_spin_box(
            min_val=-100, max_val=-1, obj_name="loc_shift_min")
        self.loc_shift_min.setValue(-10)

        self.loc_shift_step = add_spin_box(
            min_val=1, max_val=3, obj_name="loc_shift_step")
        self.loc_shift_step.setValue(1)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Maximum Shift", self.loc_shift_max,
                          "indices [range: 1 to 100]")
        box_layout.addRow("Minimum Shift", self.loc_shift_min,
                          "indices [range: -1 to -100]")
        box_layout.addRow("Index Steps", self.loc_shift_step, "[1, 2, 3]")

        self.loc_time_shift_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.loc_time_shift_gb1)

        widget.setContents(layout)
        return widget

    def spatial_corr_page(self):
        """
        Sets the ui elements for the 'spatial_corr' analysis parameters.

        """

        widget = ScrollableWidget()
        # Box- 1
        self.spatial_corr_gb1 = add_group_box(
            title="2D Correlation", obj_name="spatial_corr_gb1")

        self.spatial_corr_min_obs = add_spin_box(
            min_val=0, max_val=100, obj_name="spatial_corr_min_obs")
        self.spatial_corr_min_obs.setValue(20)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Minimum No. of Valid Pixels",
                          self.spatial_corr_min_obs, "[range: 1-100]")

        self.spatial_corr_gb1.setLayout(box_layout)

        # Box- 2
        self.spatial_corr_gb2 = add_group_box(
            title="Rotational Correlation", obj_name="spatial_corr_gb2")

        self.rot_corr_bin = add_combo_box(obj_name="rot_corr_bin")
        rot_corr_bin_items = [str(d) for d in range(
            1, 360) if 360 % d == 0 and d >= 3 and d <= 45]
        self.rot_corr_bin.addItems(rot_corr_bin_items)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Rotational Correlation Binsize",
                          self.rot_corr_bin, "degree")

        self.spatial_corr_gb2.setLayout(box_layout)

#        # Box- 3
        self.spatial_corr_gb3 = add_group_box(
            title="Smoothing Box Kernal", obj_name="spatial_corr_gb3")

        self.spatial_corr_filter = add_combo_box(
            obj_name="spatial_corr_filter")
        self.spatial_corr_filter.addItems(["Box", "Gaussian"])

        self.spatial_corr_kern_len = add_spin_box(
            min_val=1, max_val=11, obj_name="spatial_corr_kern_len")
        self.spatial_corr_kern_len.setValue(5)
        self.spatial_corr_kern_len.setSingleStep(2)
        # Change step size to 0.5 if Gaussian is selected

        box_layout = ParamBoxLayout()
        box_layout.addRow("Smoothing Filter", self.spatial_corr_filter, "")
        box_layout.addRow("Correlation Pixels Num/Sigma", self.spatial_corr_kern_len,
                          "[range: 1-11]\n\r Box: odds")

        self.spatial_corr_gb3.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.spatial_corr_gb1)
        layout.addWidget(self.spatial_corr_gb2)
        layout.addWidget(self.spatial_corr_gb3)

        widget.setContents(layout)

        return widget

    def grid_page(self):
        """
        Sets the ui elements for the 'grid' analysis parameters.

        """

        widget = ScrollableWidget()

        self.grid_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="grid_gb1")

        self.grid_ang_tol = add_spin_box(
            min_val=1, max_val=5, obj_name="grid_ang_tol")
        self.grid_ang_tol.setValue(2)

        self.grid_ang_bin = add_combo_box(obj_name="grid_ang_bin")
        grid_ang_bin_items = [str(d) for d in range(
            1, 360) if 360 % d == 0 and d >= 3 and d <= 45]
        self.grid_ang_bin.addItems(grid_ang_bin_items)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Angular Tolerance",
                          self.grid_ang_tol, "degree [range: 1 to 5]")
        box_layout.addRow("Angular Binsize", self.grid_ang_bin, "degree")

        self.grid_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.grid_gb1)

        widget.setContents(layout)
        return widget

    def border_page(self):
        """
        Sets the ui elements for the 'border' analysis parameters.

        """

        widget = ScrollableWidget()

        self.border_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="border_gb1")

        self.border_firing_thresh = add_double_spin_box(
            min_val=0, max_val=1, obj_name="border_firing_thresh")
        self.border_firing_thresh.setValue(0.1)
        self.border_firing_thresh.setSingleStep(0.05)

        self.border_ang_bin = add_combo_box(obj_name="border_ang_bin")
        border_ang_bin_items = [str(d) for d in range(
            1, 360) if 360 % d == 0 and d >= 3 and d <= 45]
        self.border_ang_bin.addItems(border_ang_bin_items)

        self.border_stair_steps = add_spin_box(
            min_val=4, max_val=10, obj_name="border_stair_steps")
        self.border_stair_steps.setValue(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Firing Threshold", self.border_firing_thresh,
                          "degree [range: 0 to 1, step: 0.05]")
        box_layout.addRow("Angular Binsize", self.border_ang_bin, "degree")
        box_layout.addRow("Stair Plot Steps",
                          self.border_stair_steps, "4 to 10")

        self.border_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.border_gb1)

        widget.setContents(layout)
        return widget

    def gradient_page(self):
        """
        Sets the ui elements for the 'gradient' analysis parameters.

        """
        widget = ScrollableWidget()

        self.gradient_gb1 = add_group_box(
            title="Gompertz Function Parameters", obj_name="gradient_gb1")

        self.grad_asymp_lim = add_double_spin_box(
            min_val=0.1, max_val=1, obj_name="grad_asymp_lim")
        self.grad_asymp_lim.setValue(0.25)
        self.grad_asymp_lim.setSingleStep(0.05)

        self.grad_displace_lim = add_double_spin_box(
            min_val=0.1, max_val=1, obj_name="grad_displace_lim")
        self.grad_displace_lim.setValue(0.25)
        self.grad_displace_lim.setSingleStep(0.05)

        self.grad_growth_rate_lim = add_double_spin_box(
            min_val=0.1, max_val=1, obj_name="grad_growth_rate_lim")
        self.grad_growth_rate_lim.setValue(0.5)
        self.grad_growth_rate_lim.setSingleStep(0.05)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Asymptote a \xb1 ", self.grad_asymp_lim,
                          "*a [range 0.1 to 1, step: 0.05]")
        box_layout.addRow("Displacement b \xb1 ",
                          self.grad_displace_lim, "*b [range 0.1 to 1, step: 0.05]")
        box_layout.addRow(
            "Growth Rate c \xb1 ", self.grad_growth_rate_lim, "*c [range 0.1 to 1, step: 0.05]")

        self.gradient_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gradient_gb1)

        widget.setContents(layout)
        return widget

    def multiple_regresison_page(self):
        """
        Sets the ui elements for the 'multiple_regression' analysis parameters.

        """

        widget = ScrollableWidget()
        self.mra_gb1 = add_group_box(
            title="Analyses Paramters", obj_name="mra_gb1")

        self.mra_interval = add_double_spin_box(
            min_val=0.1, max_val=2, obj_name="mra_interval")
        self.mra_interval.setValue(0.1)
        self.mra_interval.setSingleStep(0.1)

        self.mra_episode = add_spin_box(
            min_val=60, max_val=300, obj_name="mra_episode")
        self.mra_episode.setValue(120)
        self.mra_episode.setSingleStep(30)

        self.mra_nrep = add_spin_box(
            min_val=100, max_val=2000, obj_name="mra_nrep")
        self.mra_nrep.setValue(1000)
        self.mra_nrep.setSingleStep(100)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Subsampling Interval",
                          self.mra_interval, "sec [range: 0.1-1, step: 0.1]")
        box_layout.addRow("Chunk Length for Regression",
                          self.mra_episode, "sec [range: 60-300, step: 30]")
        box_layout.addRow("No of Replication", self.mra_nrep,
                          "[range: 100-2000, step: 100]")

        self.mra_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.mra_gb1)

        widget.setContents(layout)

        return widget

    def inter_depend_page(self):
        """
        Sets the ui elements for the 'inter_depend' analysis parameters.

        """
        widget = ScrollableWidget()

        self.inter_depend_gb1 = add_group_box(
            title="", obj_name="interDepend_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel(
            "Uses the parameters from other anlyses"))
        self.inter_depend_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.inter_depend_gb1)

        widget.setContents(layout)
        return widget

    def lfp_spectrum_page(self):
        """
        Sets the ui elements for the 'lfp_spectrum' analysis parameters.

        """

        widget = ScrollableWidget()
        # Box- 1
        self.lfp_spectrum_gb1 = add_group_box(
            title="Pre-filter (Butterworth) Properties", obj_name="lfp_spectrum_gb1")

        self.lfp_prefilt_lowcut = add_double_spin_box(
            min_val=0.1, max_val=4, obj_name="lfp_prefilt_lowcut")
        self.lfp_prefilt_lowcut.setValue(1.5)
        self.lfp_prefilt_lowcut.setSingleStep(0.1)

        self.lfp_prefilt_highcut = add_spin_box(
            min_val=10, max_val=500, obj_name="lfp_prefilt_highcut")
        self.lfp_prefilt_highcut.setValue(40)
        self.lfp_prefilt_highcut.setSingleStep(5)

        self.lfp_prefilt_order = add_spin_box(
            min_val=1, max_val=20, obj_name="lfp_prefilt_order")
        self.lfp_prefilt_order.setValue(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Lower Cutoff Frequency",
                          self.lfp_prefilt_lowcut, "Hz [range: 0.1-4, step: 0.1]")
        box_layout.addRow("Higher Cutoff Frequency",
                          self.lfp_prefilt_highcut, "Hz [range: 10-500, step: 5]")
        box_layout.addRow(
            "Filter Order", self.lfp_prefilt_order, "[range: 1-20]")

        self.lfp_spectrum_gb1.setLayout(box_layout)

        # Box- 2
        self.lfp_spectrum_gb2 = add_group_box(
            title="Spectrum Analysis (PWELCH) Settings", obj_name="lfp_spectrum_gb2")

        self.lfp_pwelch_seg_size = add_double_spin_box(
            min_val=0.5, max_val=100, obj_name="lfp_pwelch_seg_size")
        self.lfp_pwelch_seg_size.setValue(2)
        self.lfp_pwelch_seg_size.setSingleStep(0.5)

        self.lfp_pwelch_overlap = add_double_spin_box(
            min_val=0.5, max_val=50, obj_name="lfp_pwelch_overlap")
        self.lfp_pwelch_overlap.setValue(1)
        self.lfp_pwelch_overlap.setSingleStep(0.5)

        self.lfp_pwelch_nfft = add_combo_box(obj_name="lfp_pwelch_nfft")
        nfft_items = [str(2**d) for d in range(7, 14)]
        self.lfp_pwelch_nfft.addItems(nfft_items)
        self.lfp_pwelch_nfft.setCurrentIndex(3)

        self.lfp_pwelch_freq_max = add_spin_box(
            min_val=10, max_val=500, obj_name="lfp_pwelch_freq_max")
        self.lfp_pwelch_freq_max.setValue(40)
        self.lfp_pwelch_freq_max.setSingleStep(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Segment Size", self.lfp_pwelch_seg_size,
                          "sec [range: 0.5-100, step: 0.5]")
        box_layout.addRow("Overlap", self.lfp_pwelch_overlap,
                          "sec [range: 0.5-50, step: 0.5]")
        box_layout.addRow("NFFT", self.lfp_pwelch_nfft,
                          "[range: 128-8192, step: 128]")
        box_layout.addRow(
            "Maximum Frequency", self.lfp_pwelch_freq_max, "Hz [range: 10-500, step: 5]")

        self.lfp_spectrum_gb2.setLayout(box_layout)

        #        # Box- 3
        self.lfp_spectrum_gb3 = add_group_box(
            title="STFT Settings", obj_name="lfp_spectrum_gb3")

        self.lfp_stft_seg_size = add_double_spin_box(
            min_val=0.5, max_val=100, obj_name="lfp_stft_seg_size")
        self.lfp_stft_seg_size.setValue(2)
        self.lfp_stft_seg_size.setSingleStep(0.5)

        self.lfp_stft_overlap = add_double_spin_box(
            min_val=0.5, max_val=50, obj_name="lfp_stft_overlap")
        self.lfp_stft_overlap.setValue(1)
        self.lfp_stft_overlap.setSingleStep(0.5)

        self.lfp_stft_nfft = add_combo_box(obj_name="lfp_stft_nfft")
        nfft_items = [str(2**d) for d in range(7, 14)]
        self.lfp_stft_nfft.addItems(nfft_items)
        self.lfp_stft_nfft.setCurrentIndex(3)

        self.lfp_stft_freq_max = add_spin_box(
            min_val=10, max_val=500, obj_name="lfp_stft_freq_max")
        self.lfp_stft_freq_max.setValue(40)
        self.lfp_stft_freq_max.setSingleStep(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Segment Size", self.lfp_stft_seg_size,
                          "sec [range: 0.5-100, step: 0.5]")
        box_layout.addRow("Overlap", self.lfp_stft_overlap,
                          "sec [range: 0.5-50, step: 0.5]")
        box_layout.addRow("NFFT", self.lfp_stft_nfft,
                          "[range: 128-8192, step: 128]")
        box_layout.addRow("Maximum Frequency",
                          self.lfp_stft_freq_max, "Hz [range: 10-500, step: 5]")

        self.lfp_spectrum_gb3.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.lfp_spectrum_gb1)
        layout.addWidget(self.lfp_spectrum_gb2)
        layout.addWidget(self.lfp_spectrum_gb3)

        widget.setContents(layout)

        return widget

    def spike_phase_page(self):
        """
        Sets the ui elements for the 'spike_phase' analysis parameters.

        """

        widget = ScrollableWidget()

        self.spike_phase_gb1 = add_group_box(
            title="Analysis Parameters", obj_name="spike_phase_gb1")

        self.phase_freq_min = add_double_spin_box(
            min_val=1, max_val=10, obj_name="phase_freq_min")
        self.phase_freq_min.setValue(6)
        self.phase_freq_min.setSingleStep(0.5)

        self.phase_freq_max = add_double_spin_box(
            min_val=8, max_val=16, obj_name="phase_freq_max")
        self.phase_freq_max.setValue(12)
        self.phase_freq_max.setSingleStep(0.5)

        self.phase_power_thresh = add_double_spin_box(
            min_val=0, max_val=1, obj_name="phase_power_thresh")
        self.phase_power_thresh.setValue(0.1)
        self.phase_power_thresh.setSingleStep(0.05)

        self.phase_amp_thresh = add_double_spin_box(
            min_val=0, max_val=1, obj_name="phase_amp_thresh")
        self.phase_amp_thresh.setValue(0.15)
        self.phase_amp_thresh.setSingleStep(0.05)

        self.phase_bin = add_combo_box(obj_name="phase_bin")
        phase_bin_items = [str(d) for d in range(
            1, 360) if 360 % d == 0 and d >= 5 and d <= 45]
        self.phase_bin.addItems(phase_bin_items)

        self.phase_raster_bin = add_spin_box(
            min_val=1, max_val=15, obj_name="phase_raster_bin")
        self.phase_raster_bin.setValue(2)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Frequency of Interest (Min)",
                          self.phase_freq_min, "Hz [range: 1-10, step: 0.5]")
        box_layout.addRow("Frequency of Interest (Max)",
                          self.phase_freq_max, "Hz [range: 1-10, step: 0.5]")
        box_layout.addRow("Band to Total Power Ratio (Min)",
                          self.phase_power_thresh, "[range: 0-1, step: 0.05]")
        box_layout.addRow("Segment to Overall Amplitude Ratio (Min)",
                          self.phase_amp_thresh, "[range: 0-1, step: 0.05]")
        box_layout.addRow("Phase Plot Binsize", self.phase_bin, "degree")
        box_layout.addRow("Phase Raster Plot Binsize",
                          self.phase_raster_bin, "degree[range: 1-15]")

        self.spike_phase_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.spike_phase_gb1)

        widget.setContents(layout)

        return widget

    def phase_lock_page(self):
        """
        Sets the ui elements for the 'phase_lock' analysis parameters.

        """

        widget = ScrollableWidget()

        self.phase_lock_gb1 = add_group_box(
            title="Analysis Parameters", obj_name="phase_lock_gb1")

        self.phase_lock_win_low = add_double_spin_box(
            min_val=-1, max_val=-0.1, obj_name="phase_loc_win_low")
        self.phase_lock_win_low.setValue(-0.4)
        self.phase_lock_win_low.setSingleStep(0.05)

        self.phase_lock_win_up = add_double_spin_box(
            min_val=0.1, max_val=1, obj_name="phase_loc_win_up")
        self.phase_lock_win_up.setValue(0.4)
        self.phase_lock_win_up.setSingleStep(0.05)

        self.phase_lock_nfft = add_combo_box(obj_name="phase_loc_nfft")
        nfft_items = [str(2**d) for d in range(7, 14)]
        self.phase_lock_nfft.addItems(nfft_items)
        self.phase_lock_nfft.setCurrentIndex(3)

        self.phase_lock_freq_max = add_spin_box(
            min_val=10, max_val=500, obj_name="phase_loc_freq_max")
        self.phase_lock_freq_max.setValue(40)
        self.phase_lock_freq_max.setSingleStep(5)

        box_layout = ParamBoxLayout()
        box_layout.addRow("Analysis Window (Lower)",
                          self.phase_lock_win_low, "sec [range: -1 to -0.1, step: 0.05]")
        box_layout.addRow("Analysis Window (Upper)",
                          self.phase_lock_win_up, "sec [range: 0.1 to 1, step: 0.05]")
        box_layout.addRow("NFFT", self.phase_lock_nfft,
                          "[range: 128-8192, step: 128]")
        box_layout.addRow("Frequency of Interest (Max)",
                          self.phase_lock_freq_max, "Hz [range: 1-10, step: 0.5]")

        self.phase_lock_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.phase_lock_gb1)

        widget.setContents(layout)

        return widget

    def lfp_spike_causality_page(self):
        """
        Sets the ui elements for the 'lfp_spike_causality' analysis parameters.

        """
        widget = ScrollableWidget()

        self.causality_gb1 = add_group_box(title="", obj_name="causality_gb1")

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(QtWidgets.QLabel("Analysis not implemented yet!"))
        self.causality_gb1.setLayout(box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.causality_gb1)

        widget.setContents(layout)
        return widget


class ParamBoxLayout(QtWidgets.QVBoxLayout):
    """
    Subclass of QtWidgets.QVBoxLayout to faciliate adding new widget item to the analysis
    parameter selection window.

    """

    def __int__(self):
        super().__init__()

    def addRow(self, label_1, widg, label_2):
        """
        Adds a new row of widget using the QtWidgets.QHBoxLayout().

        Parameters
        ----------
        label_1 : str
            Name of the parameter
        widg
            PyQt5 widget to add
        label_2 : str
            Additional description of the parameter, i.e., unit, range etc.

        """

        widg.resize(widg.sizeHint())
        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addWidget(QtWidgets.QLabel(label_1), 0, QtCore.Qt.AlignLeft)
        hLayout.addWidget(widg, 0, QtCore.Qt.AlignLeft)
        hLayout.addWidget(QtWidgets.QLabel(label_2), 0, QtCore.Qt.AlignLeft)
        self.addLayout(hLayout)
