#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Use this module to invoke the graphical user interface of NeuroChaT.

Run this code in command line or from IDE and it will start the graphical window
of the NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import datetime
import logging
import time
import os
import traceback
import sys

from PyQt5 import QtWidgets

from neurochat.nc_ui import NeuroChaT_Ui
from neurochat.nc_utils import make_dir_if_not_exists

default_write = sys.stdout.write
default_loc = os.path.join(
    os.path.expanduser("~"), ".nc_saved", "nc_errorlog.txt")
make_dir_if_not_exists(default_loc)
this_logger = logging.getLogger(__name__)
handler = logging.FileHandler(default_loc)
this_logger.addHandler(handler)


def excepthook(exc_type, exc_value, exc_traceback):
    """
    Any uncaught exceptions will be logged from here.

    """
    # Don't catch CTRL+C exceptions
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    now = datetime.datetime.now()
    this_logger.critical(
        "Uncaught Exception at {}".format(now), exc_info=(
            exc_type, exc_value, exc_traceback))

    QtWidgets.QApplication.quit()
    sys.stdout.write = default_write
    print("A fatal error occurred in NeuroChaT")
    print("The error info was: {}".format(
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)
                ).strip()))
    print(
        "Please report this to {} and provide the file {}".format(
            "us", default_loc))


def main():
    sys.excepthook = excepthook
    logging.basicConfig(level=logging.INFO)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)
    app = QtWidgets.QApplication(sys.argv)
    app.quitOnLastWindowClosed()
    ui = NeuroChaT_Ui()
    ui.show()
    ret = app.exec_()
    sys.stdout.write = default_write
    sys.exit(ret)


if __name__ == '__main__':
    main()
