import os

import matplotlib.pyplot as plt

from neurochat.nc_spike import NSpike
from neurochat.nc_hdf import Nhdf
import neurochat.nc_plot as nc_plot

import logging


def spikeinterface_test(folder_name):
    """An example sorting extractor, which loads from Phy."""
    import spikeinterface.extractors as se
    to_exclude = ["mua", "noise"]
    sorting = se.PhySortingExtractor(
        folder_name, exclude_cluster_groups=to_exclude, load_waveforms=True,
        verbose=False)
    return sorting


def read_hdf(hdf_path, verbose=False, group=3):
    """
    Read the NWB at hdf_path into NeuroChaT.

    Parameters
    ----------
    hdf_path : str
        Path to the hdf5 file
    verbose : bool, optional.
        Defaults to False, indicates whether to print information.
    group : int, optional.
        Defaults to 3, indicates the group in the hdf5 file to use.

    Returns
    -------
    NSpike
        The loaded NSpike object.

    """
    if verbose:
        from skm_pyutils.py_print import print_h5
        print_h5(hdf_path)

    spike_file = hdf_path + "+/processing/Shank/" + str(group)
    spike = NSpike()
    spike.set_system("NWB")
    spike.set_filename(spike_file)
    spike.load()
    unit_no = spike.get_unit_list()[0]
    spike.set_unit_no(unit_no)

    if verbose:
        print(spike)

    return spike


if __name__ == '__main__':
    """Can set whether to write or read hdf5 here, and also the paths."""
    to_write = True
    to_read = True

    logging.basicConfig(level=logging.INFO)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)

    if to_write:
        write_name = r"/media/sean/Elements/Ham_Data/Batch_2/A9_CAR-SA1/CAR-SA1_20191130_1_PreBox/phy_klusta"
        from neurochat.nc_control import NeuroChaT
        plot_waveforms = True
        sorting = spikeinterface_test(write_name)
        NeuroChaT.sortingextractor_to_nwb(
            sorting, plot_waveforms=plot_waveforms)

    if to_read:
        read_name = r"/media/sean/Elements/Ham_Data/Batch_2/A9_CAR-SA1/CAR-SA1_20191130_1_PreBox/CAR-SA1_20191130_1_PreBox_NC_NWB.hdf5"
        spike = read_hdf(read_name)
        print(spike)
