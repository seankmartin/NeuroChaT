# -*- coding: utf-8 -*-
"""
This module implements utility functions and classes for NeuroChaT software

@author: Md Nurul Islam; islammn at tcd dot ie

"""

from PyQt5 import QtCore, QtWidgets, QtGui

try:
    xlt_from_utf8 = QtCore.QString.fromUtf8
except AttributeError:
    def xlt_from_utf8(s):
        return s


class ScrollableWidget(QtWidgets.QWidget):
    """
    Subclassed from PyQt5.QtWidgets.QWidget, this class creates a scrollable widget.

    """

    def __init__(self):
        super().__init__()
        self.parent_layout = QtWidgets.QVBoxLayout(self)
        self.cont_widget = QtWidgets.QWidget()
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(False)

    def setContents(self, cont_layout):
        """
        Sets the contents of the scrollable widget

        Parameters
        ----------
        cont_layout
            PyQt5 layout that is the container for scrollable elements.

        """
        self.cont_layout = cont_layout
        self.cont_widget.setLayout(self.cont_layout)
        self.scroll_area.setWidget(self.cont_widget)
        self.parent_layout.addWidget(self.scroll_area)
        self.setLayout(self.parent_layout)


class NLogBox(QtWidgets.QTextEdit):
    """
    Subclassed from PyQt5.QtWidgets.QTextEdit, this class creates a formatted 
    text-editable log-box for NeuroChaT

    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def insert_log(self, msg):
        """
        Formats further the HTML 'msg' to categorally add color to the log-texts
        and displays it in the log-box or in any log-handler.

        Parameters
        ----------
        msg
            Log record that is to be displayed

        """

        level = msg.split(':')[0].upper()
#        level= "WARNING"
        if level == "WARNING":
            color = "darkorange"
        elif level == "ERROR":
            color = "red"
        elif level == "INFO":
            color = "black"
        else:
            color = "blue"
        msg = '<font color=' + color + '>' + \
            msg[msg.find(":") + 1:] + '</font><br>'
        self.insertHtml(msg)
        self.moveCursor(QtGui.QTextCursor.End)

    def get_text(self):
        """
        Returned the texts of log-box in plain text format     

        Parameters
        ----------
        None

        Returns
        -------
        str
            Plain text of log-box

        """

        return self.toPlainText()


class NOut(QtCore.QObject):
    """
    Subclassed from PyQt5.QtCore.QObject, it implements the Qt signalling mechanism
    so that when a text is written using NOut() object, it emits a text to the 
    output to an output console or file or where the emitted signal is connected to. 

    In NeuroChaT, the sys.stdout.write is replaced with NOut().write, which means
    print('some text') will print to GUI log box in GUI and to the standard output
    console when used in API.

    """

    emitted = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def write(self, text):
        """
        Emits the texts as Qt signal

        Parameters
        ----------
        text : str
            Text to be emitted

        Returns
        -------
        None

        """
#        self.emit(QtCore.SIGNAL('update_log(QString)'), text)
        self.emitted.emit(text)


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a QT table view with a pandas dataframe and implements methods
    that are to be overriden

    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        """
        Overrides the rowCount() methods

        Parameters
        ----------
        parent : QtCore.QModelIndex
            Specific model for item index. Usually not required.

        Returns
        -------
        int
            Count of row in the item model

        """

        return self._data.shape[0]

    def columnCount(self, parent=None):
        """
        Overrides the columnCount() methods

        Parameters
        ----------
        parent : QtCore.QModelIndex
            Specific model for item index. Usually not required.

        Returns
        -------
        int
            Count of column in the item model

        """

        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """
        Data model for the QtCore.QAbstractTableModel.

        Parameters
        ----------
        index
            Pandas DataFrame index
        role : Qt.ItemDataRole
            Usually set for QtCore.Qt.DisplayRole which means the data to rendered
            in the form of text

        Returns
        -------
        str
            Returns the data in the DataFrame's 'index' location in str format

        """

        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, index, orientation, role):
        """
        Data model for the QtCore.QAbstractTableModel.

        Parameters
        ----------
        index
            Pandas DataFrame index
        orientation : Qt.Orientation
            Orientation of the data
        role : Qt.ItemDataRole
            Usually set for QtCore.Qt.DisplayRole which means the data to be rendered
            in the form of text

        Returns
        -------

            Data in the DataFrame().columns[index] if orientation is 'Horizontal'
            or DataFrame().index[index] if orientation is 'Vertical'

        """

        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[index]
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[index]
        return None


def add_log_box(obj_name):
    """
    Returns a NLogBox() object

    Parameters
    ----------
    obj_name : str
        Name of the newly created object

    Returns
    -------
    NLogBox
        Instance of NLogBox Class

    """

    logTextBox = NLogBox()
#    logTextBox.seObjectName(xlt_from_utf8(obj_name))
    return logTextBox
#    return logTextBox.widget


def add_radio_button(parent=None, position=None, obj_name='', text=None):
    """
    Returns a QtWidgets.QRadioButton() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    text : str
        Button text

    Returns
    -------
    QtWidgets.QRadioButton
        Instance of QtWidgets.QRadioButton Class

    """

    button = QtWidgets.QRadioButton(parent)
    button.setObjectName(xlt_from_utf8(obj_name))

    if position:
        button.setGeometry(QtCore.QRect(*position))
    if text:
        button.setText(text)

    return button


def add_push_button(parent=None, position=None, obj_name='', text=None):
    """
    Returns a QtWidgets.QPushButton() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    text : str
        Button text

    Returns
    -------
    QtWidgets.QPushButton
        Instance of QtWidgets.QPushButton Class

    """

    button = QtWidgets.QPushButton(parent)
    button.setObjectName(xlt_from_utf8(obj_name))

    if position:
        button.setGeometry(QtCore.QRect(*position))
    if text:
        button.setText(text)

    return button


def add_check_box(parent=None, position=None, obj_name='', text=None):
    """
    Returns a QtWidgets.QCheckBox() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    text : str
        Check box text

    Returns
    -------
    QtWidgets.QCheckBox
        Instance of QtWidgets.QCheckBox Class

    """

    box = QtWidgets.QCheckBox(parent)
    box.setObjectName(xlt_from_utf8(obj_name))

    if position:
        box.setGeometry(QtCore.QRect(*position))
    if text:
        box.setText(text)

    return box


def add_combo_box(parent=None, position=None, obj_name=''):
    """
    Returns a QtWidgets.QComboBox() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object

    Returns
    -------
    QtWidgets.QComboBox
        Instance of QtWidgets.QComboBox Class

    """

    box = QtWidgets.QComboBox(parent)
    box.setObjectName(xlt_from_utf8(obj_name))

    if position:
        box.setGeometry(QtCore.QRect(*position))

    return box


def add_label(parent=None, position=None, obj_name='', text=None):
    """
    Returns a QtWidgets.QLabel() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    text : str
        Label text

    Returns
    -------
    QtWidgets.QLabel
        Instance of QtWidgets.QLabel Class

    """

    label = QtWidgets.QLabel(parent)
    label.setObjectName(xlt_from_utf8(obj_name))

    if position:
        label.setGeometry(QtCore.QRect(*position))
    if text:
        label.setText(text)

    return label


def add_line_edit(parent=None, position=None, obj_name='', text=None):
    """
    Returns a QtWidgets.QLineEdit() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    text : str
        Line-edit text

    Returns
    -------
    QtWidgets.QLineEdit
        Instance of QtWidgets.QLineEdit Class

    """

    line = QtWidgets.QLineEdit(parent)
    line.setObjectName(xlt_from_utf8(obj_name))

    if position:
        line.setGeometry(QtCore.QRect(*position))
    if text:
        line.setText(text)

    return line


def add_group_box(parent=None, position=None, obj_name='', title=None):
    """
    Returns a QtWidgets.QGroupBox() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object
    title : str
        Title of the group-box

    Returns
    -------
    QtWidgets.QGroupBox
        Instance of QtWidgets.QGroupBox Class

    """

    groupBox = QtWidgets.QGroupBox(parent)
    groupBox.setObjectName(xlt_from_utf8(obj_name))

    if position:
        groupBox.setGeometry(QtCore.QRect(*position))
    if title:
        groupBox.setTitle(title)

    return groupBox


def add_widget(parent=None, position=None, obj_name=''):
    """
    Returns a QtWidgets.QWidget() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    obj_name : str
        Name of the newly created object

    Returns
    -------
    QtWidgets.QWidget
        Instance of QtWidgets.QWidget Class

    """

    widget = QtWidgets.QWidget(parent)
    widget.setObjectName(xlt_from_utf8(obj_name))

    if position:
        widget.setGeometry(QtCore.QRect(*position))

    return widget


def add_spin_box(parent=None, position=None, obj_name='', min_val=0, max_val=128):
    """
    Returns a QtWidgets.QSpinBox() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    min_val : int
        Minimum value of the spin-box
    max_val : int
        Maximum value of the spin-box
    obj_name : str
        Name of the newly created object

    Returns
    -------
    QtWidgets.QSpinBox
        Instance of QtWidgets.QSpinBox Class

    """

    box = QtWidgets.QSpinBox(parent)
    box.setObjectName(xlt_from_utf8(obj_name))

    if position:
        box.setGeometry(QtCore.QRect(*position))

    box.setMinimum(min_val)
    box.setMaximum(max_val)

    return box


def add_double_spin_box(parent=None, position=None, min_val=0, max_val=1, obj_name=""):
    """
    Returns a QtWidgets.QDoubleSpinBox() object

    Parameters
    ----------
    parent
        Parent widget
    position : tuple
        Position in the parent object
    min_val : float
        Minimum value of the spin-box
    max_val : float
        Maximum value of the spin-box
    obj_name : str
        Name of the newly created object

    Returns
    -------
    QtWidgets.QDoubleSpinBox
        Instance of QtWidgets.QDoubleSpinBox Class

    """
    box = QtWidgets.QDoubleSpinBox(parent)
    box.setObjectName(xlt_from_utf8(obj_name))
    if position:
        box.setGeometry(QtCore.QRect(*position))
    box.setMinimum(min_val)
    box.setMaximum(max_val)

    return box
