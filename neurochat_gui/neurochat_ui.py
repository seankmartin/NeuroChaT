#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Use this module to invoke the graphical user interface of NeuroChaT.

Run this code in command line or from IDE and it will start the graphical window
of the NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import sys
import logging
from PyQt5 import QtWidgets
from neurochat.nc_ui import NeuroChaT_Ui


def main():
    logging.basicConfig(level=logging.INFO)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)
    app = QtWidgets.QApplication(sys.argv)
    app.quitOnLastWindowClosed()
    ui = NeuroChaT_Ui()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
