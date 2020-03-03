# -*- coding: utf-8 -*-
"""
This module implements UiMerge Class for NeuroChaT that provides the graphical
interface and functionalities for merging and accumulating the output graphics of NeuroChaT 

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import os
import shutil
import logging

from PyQt5 import QtCore, QtWidgets

from neurochat.nc_uigetfiles import UiGetFiles

from neurochat.nc_uiutils import add_radio_button, add_push_button, add_line_edit,\
    xlt_from_utf8

from PyPDF2 import PdfFileReader, PdfFileMerger

import pandas as pd


class UiMerge(QtWidgets.QDialog):
    """
    This class invokes a graphical user interface where the user can upload
    a list of PDF or Postscript files in Excel format or can use a filepicker 
    to manually pick the files to merge in a file or accumulate in a folder.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.merge_enable = True
        self._get_files_ui = UiGetFiles(self)

        self.setup_ui()
        self._behaviour_ui()

        self.files = []
        self.dst_directory = []  # destination directory for accumulation
        self.dst_file = []  # destination files for mergeing

    def setup_ui(self):
        """
        Sets up the GUI elements

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.setObjectName(xlt_from_utf8("mergeWindow"))
        self.setEnabled(True)
        self.setFixedSize(324, 220)

        self.use_list_button = add_radio_button(
            self, (10, 20, 102, 23), "useList", "Use Excel List")

        self.choose_files_button = add_radio_button(
            self, (10, 80, 102, 23), "chooseFiles", "Choose files")

        self.input_format_group = QtWidgets.QButtonGroup(self)
        self.input_format_group.setObjectName(
            xlt_from_utf8("mergeButtonGroup"))
        self.input_format_group.addButton(self.use_list_button)
        self.input_format_group.addButton(self.choose_files_button)

        self.browse_excel_button = add_push_button(
            self, (30, 50, 50, 23), "browseExcel", "Browse")

        self.filename_line = add_line_edit(
            self, (95, 50, 215, 23), "filename", "")

        self.select_button = add_push_button(
            self, (30, 110, 70, 23), "select", "Select now")

        self.save_in_button = add_push_button(
            self, (30, 140, 50, 23), "saveIn", "Save In")

        self.save_filename_line = add_line_edit(
            self, (95, 140, 215, 23), "saveFilename", "")

        self.start_button = add_push_button(
            self, (95, 180, 40, 23), "start", "Start")

        self.cancel_button = add_push_button(
            self, (165, 180, 50, 23), "cancel", "Cancel")

        self.set_default()

        self._get_files_ui.setup_ui()

    def set_default(self):
        """
        Sets up the defaults of the GUI

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.use_list_button.setChecked(True)
        self.select_button.setEnabled(False)
        self.filename_line.setText("Select Excel (.xls/.xlsx) file")
        self.save_filename_line.setText("")

    def _behaviour_ui(self):
        """
        Sets up the behaviour of the GUI elements

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self._get_files_ui.behaviour_ui()

        self.input_format_group.buttonClicked.connect(self.merge_files)
        self.browse_excel_button.clicked.connect(self.browse_excel_merge)
        self.select_button.clicked.connect(self.select_files_merge)
        self.save_in_button.clicked.connect(self.save_in_merge)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.close)

    def merge_files(self):
        """
        Calling this method toggles the UI selection for using an Excel list or
        picking files manually.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        button = self.input_format_group.checkedButton()
        if button.objectName() == "useList":
            self.browse_excel_button.setEnabled(True)
            self.filename_line.setEnabled(True)
            self.select_button.setEnabled(False)
            logging.info("Browse an excel list of output graphic files")
        elif button.objectName() == "chooseFiles":
            self.browse_excel_button.setEnabled(False)
            self.filename_line.setEnabled(False)
            self.select_button.setEnabled(True)
            logging.info("Select output graphic files")

    def browse_excel_merge(self):
        """
        Opens a dialogue for selecting the Excel list of PDF/Postscript files and reads the
        file information.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.files = []
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                                          'Select output graphic file list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning(
                "No excel file selected! Merging/accumulating is unsuccessful!")
        else:
            self.filename_line.setText(excel_file)
            logging.info("New excel file added: " +
                         excel_file.rstrip("\n\r").split(os.sep)[-1])
        data = pd.read_excel(excel_file)
        self.files = data.values.T.tolist()[0]
        for i, f in enumerate(self.files):
            if not os.path.exists(f):
                self.files.pop(i)
                logging.warning(f + ' file does not exist!')

    def select_files_merge(self):
        """
        Invokes the UiGetFiles class for manual selection of the PDF or Postscript
        files for merging or accumulating.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self._get_files_ui.show()
        self.files = self._get_files_ui.get_files()

    def save_in_merge(self):
        """
        Opens a dialogue for selecting the file or folder where the PDF/Postscript files
        qill be merged or accumulated.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if self.merge_enable:
            self.dst_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save as...', os.getcwd(), "*.pdf;; *.ps")[0])
            self.save_filename_line.setText(self.dst_file)
            if os.path.exists(self.dst_file):
                logging.info('PDF files will be merged to ' + self.dst_file)
        else:
            self.dst_directory = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                                                           'Select data directory...', os.getcwd()))
            self.save_filename_line.setText(self.dst_directory)
            if os.path.exists(self.dst_directory):
                logging.info('Files will be accumulated to ' +
                             self.dst_directory)

    def start(self):
        """
        Executes the merging or accumulating operation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if self.files:
            if self.merge_enable:
                merger = PdfFileMerger()
                for f in self.files:
                    if os.path.exists(f):
                        merger.append(PdfFileReader(f, 'rb'))
                    else:
                        logging.warning('Cannot merge file ' + f)
                try:
                    merger.write(self.dst_file)
                    logging.info('Files merged to ' + self.dst_file)
                except:
                    logging.error('Cannot merge files to ' +
                                  self.dst_directory)
            else:
                if os.path.exists(self.dst_directory) and os.access(self.dst_directory, os.W_OK):
                    for f in self.files:
                        try:
                            shutil.move(f, os.path.join(
                                self.dst_directory, f.split(os.sep)[-1]))
                        except:
                            logging.warning(
                                'Cannot move file ' + f + ' to ' + self.dst_directory)
                    logging.info('Files moved to ' + self.dst_directory)
                else:
                    logging.error('Destination folder ' + self.dst_directory +
                                  ' does not exist or not write accessible!')
