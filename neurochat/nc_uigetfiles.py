# -*- coding: utf-8 -*-
"""
This module implements UiGetFiles Class for NeuroChaT that provides the graphical
interface and functionalities for manually selecting files.

@author: Md Nurul Islam; islammn at tcd dot ie

"""

import os
from functools import partial

from PyQt5 import QtCore, QtWidgets, QtGui

from neurochat.nc_uiutils import add_push_button, add_combo_box, add_label, add_line_edit,\
    xlt_from_utf8


class UiGetFiles(QtWidgets.QDialog):
    DOWN = 1
    UP = -1

    def __init__(self, parent=None, filters=['.pdf', '.ps']):
        """
        Instantiate the UiGetFiles class. 

        Parameters
        ----------
        parent
            Parent widget if any
        filters : list of str
            File filters for manual selection

        Attributes
        ----------
        parent
            Parent widget
        filters : list of str
            Approved filters
        current_filter : str
            Currently set filter
        files : list
            List of selected files

        """

        super().__init__(parent)
        self.parent = parent
        self.filters = filters
        self.current_filter = None
        self.files = []
        self.dir_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
        self.file_icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)

    def setup_ui(self):
        """
        Sets up the elements of UiGetFiles class

        """

        self.setObjectName(xlt_from_utf8("getFilesWindow"))
        self.setEnabled(True)
        self.setFixedSize(680, 400)
        self.setWindowTitle(QtWidgets.QApplication.translate(
            "getFilesWindow", "Select files", None))

        self.dir_list = QtWidgets.QListView(self)
        self.dir_list.setObjectName(xlt_from_utf8("dir_list"))
        self.dir_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self.dir_model = QtGui.QStandardItemModel(self.dir_list)
        self.dir_list.setModel(self.dir_model)

        self.file_list = QtWidgets.QListView(self)
        self.file_list.setObjectName(xlt_from_utf8("file_list"))
        self.file_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self.file_model = QtGui.QStandardItemModel(self.file_list)
        self.file_list.setModel(self.file_model)

        button_layout = QtWidgets.QVBoxLayout()
        self.add_button = add_push_button(text="Add", obj_name="addButton")
        self.remove_button = add_push_button(
            text="Remove", obj_name="removeButton")
        self.up_button = add_push_button(text="Move Up", obj_name="upButton")
        self.down_button = add_push_button(
            text="Move Down", obj_name="downButton")
        self.done_button = add_push_button(text="Done", obj_name="doneButton")
        self.cancel_button = add_push_button(
            text="Cancel", obj_name="cancelButton")

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.up_button)
        button_layout.addWidget(self.down_button)
        button_layout.addWidget(self.done_button)
        button_layout.addWidget(self.cancel_button)

        box_layout = QtWidgets.QVBoxLayout()

        self.filter_label = add_label(text="File Type", obj_name="modeLabel")
        self.filter_box = add_combo_box(obj_name="filterBox")
        self.filter_box.addItems(self.filters)

        self.folder_label = add_label(
            text="Current Folder", obj_name="folderLabel")
        self.folder_line = add_line_edit(
            text=os.getcwd(), obj_name="folderLine")

        self.dir_box = add_combo_box(obj_name="dirBox")
        self.dir_box.addItems(os.getcwd().split(os.sep))
        self.dir_box.setCurrentIndex(
            self.dir_box.findText(os.getcwd().split(os.sep)[-1]))
        self.back_button = add_push_button(text="Back", obj_name="backButton")

        dir_layout = QtWidgets.QHBoxLayout()
        dir_layout.addWidget(self.dir_box)
        dir_layout.addWidget(self.back_button)
        dir_layout.addStretch()

        box_layout.addWidget(self.filter_label)
        box_layout.addWidget(self.filter_box)
        box_layout.addWidget(self.folder_label)
        box_layout.addWidget(self.folder_line)
        box_layout.addLayout(dir_layout)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.dir_list)
        bottom_layout.addLayout(button_layout)
        bottom_layout.addWidget(self.file_list)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(box_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def behaviour_ui(self):
        """
        Sets up the behaviour of UiGetFiles class

        """

        self.filter_box.currentIndexChanged[str].connect(self.filter_changed)
        self.folder_line.textEdited[str].connect(self.line_edited)
        self.dir_box.currentIndexChanged[int].connect(self.dir_changed)
        self.back_button.clicked.connect(self.hierarchy_changed)
        self.dir_list.activated[QtCore.QModelIndex].connect(
            self.item_activated)
        self.add_button.clicked.connect(self.add_items)
        self.remove_button.clicked.connect(self.remove_items)
        self.up_button.clicked.connect(partial(self.move_items, 'up'))
        self.down_button.clicked.connect(partial(self.move_items, 'down'))
        self.done_button.clicked.connect(self.done_merge)
        self.cancel_button.clicked.connect(self.close_dialog)

        self.folder_line.setText(os.getcwd())
        self.line_edited(os.getcwd())

    def filter_changed(self, value):
        """
        Called if the filter changed to update for the new selection

        Parameters
        ----------
        value
            Currently set filter

        Returns
        -------
        None

        """

        self.current_filter = value
        self.update_list(self.folder_line.text())

    def line_edited(self, value):
        """
        Called if the directory text box in the widget is changed to update the list of new subdirectories

        Parameters
        ----------
        value
            Newly set text in the directory box.

        Returns
        -------
        None

        """

        if os.path.exists(value):
            self.dir_box.clear()
            self.dir_box.addItems(value.split(os.sep))
            self.dir_box.setCurrentIndex(
                self.dir_box.findText(value.split(os.sep)[-1]))

    def dir_changed(self, value):
        """
        Called if the subdirectoy combo-box in the widget is changed to update the list of new subdirectories

        Parameters
        ----------
        value
            Newly set item number in the combo-box of subdirectories.

        Returns
        -------
        None

        """

        directory = os.sep.join([self.dir_box.itemText(i)
                                 for i in range(value + 1)])
        if os.sep not in directory:
            directory += os.sep
        self.folder_line.setText(directory)
        self.update_list(directory)

    def update_list(self, directory):
        """
        Updates the list of folders and sets the item model for the scrollable list 

        Parameters
        ----------
        directory
            New directory whose folders and files are listed

        Returns
        -------
        None

        """
        self.dir_model.clear()
        if os.path.isdir(directory):
            dir_content = os.listdir(directory)
            for f in dir_content:
                if os.path.isdir(os.path.join(directory, f)):
                    item = QtGui.QStandardItem(self.dir_icon, f)
                    item.setEditable(False)
                    self.dir_model.appendRow(item)

            for f in dir_content:
                if os.path.isfile(os.path.join(directory, f)) and f.endswith(self.filter_box.currentText()):
                    item = QtGui.QStandardItem(self.file_icon, f)
                    item.setEditable(False)
                    self.dir_model.appendRow(item)

    def hierarchy_changed(self):
        """
        Called if the directory hierarchy is changed

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        curr_ind = self.dir_box.currentIndex() - 1
        if curr_ind >= 0:
            self.dir_box.setCurrentIndex(curr_ind)

    def item_activated(self, qind):
        """
        Called if any of the model item in the list of folders and files is double-clicked

        Parameters
        ----------
        quind
            Indix of new model item

        Returns
        -------
        None

        """

        data = self.dir_model.itemFromIndex(qind).text()
        directory = os.path.join(self.folder_line.text(), data)
        if os.path.isdir(directory):
            # setText() does not invoke textEdited(), manually calling line_eidted()
            self.folder_line.setText(directory)
            self.line_edited(directory)

    def add_items(self):
        """
        Called if the add button is clicked. Adds selected model item to the right side
        selected file box if that passes the filter

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        qind = self.dir_list.selectedIndexes()
        items = [self.dir_model.itemFromIndex(i) for i in qind]
        curr_files = [self.file_model.item(i).text()
                      for i in range(self.file_model.rowCount())]

        for it in items:
            if os.path.isfile(os.path.join(self.folder_line.text(), it.text())) \
                    and it.text().endswith(self.filter_box.currentText()):
                if not curr_files or it.text() not in curr_files:
                    self.file_model.appendRow(it.clone())

        self.file_list.setModel(self.file_model)

    def remove_items(self):
        """
        Removes the item which is added to the selection list. Updates the item
        model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        qind = self.file_list.selectedIndexes()
        rows = [i.row() for i in qind][::-1]
        for r in rows:
            # Rows change after each removal, so the one in the highest index are not deleted
            self.file_model.removeRow(r)
        self.file_list.setModel(self.file_model)

    def move_items(self, direction='down'):
        """
        Moves item in the item model by changing their indices.

        Parameters
        ----------
        direction : str
            Direction of moving 'down' or 'up'

        Returns
        -------
        None

        """
        # -1= move up, +1= move down
        qind = self.file_list.selectedIndexes()
        rows = [i.row() for i in qind]
        if direction == 'up':
            rows.sort(reverse=False)
            newRows = [r - 1 for r in rows]
        elif direction == 'down':
            rows.sort(reverse=True)
            newRows = [r + 1 for r in rows]

        for i, row in enumerate(rows):
            if not 0 <= newRows[i] < self.file_model.rowCount():
                continue
            rowItem = self.file_model.takeRow(row)
            self.file_model.insertRow(newRows[i], rowItem)

        self.file_list.setModel(self.file_model)

    def done_merge(self):
        """
        Sets the list of files and closes the widget.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.files = [os.path.join(self.folder_line.text(), self.file_model.item(i).text())
                      for i in range(self.file_model.rowCount())]
        self.close_dialog()

    def close_dialog(self):
        """
        Closes the widget for file selection.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.file_model.clear()
        self.close()

    def get_files(self):
        """
        Returns the list of files.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of selected files

        """

        return self.files
