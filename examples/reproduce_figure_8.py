"""
Use this file to reproduce Figure 8 in the NeuroChaT paper.
Please replace all instances of 
C:\\Users\\smartin5\\Recordings\\NC_eg\\ with the folder
that you downloaded 112512_1.hdf5 to
"""

import os
import logging
from neurochat.nc_config import Configuration
from neurochat.nc_control import NeuroChaT


def main():
    nc = NeuroChaT()
    config = Configuration()

    root = os.path.dirname(os.path.realpath(__file__))
    name = "Figure8Reproduce.ncfg"
    config_loc = os.path.join(root, "Configs", name)
    config.set_config_file(config_loc)
    config.load_config()

    nc.set_configuration(config)
    nc.run()

    excel_file = nc._pdf_file[:-3] + "xlsx"
    try:
        results = nc.get_results()
        results.to_excel(excel_file)
        logging.info("Analysis results exported to: " +
                     excel_file.rstrip("\n\r").split(os.sep)[-1])
    except:
        logging.error('Failed to export results!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)
    main()
