"""Can set up this file to provide example data for tests."""
import os
from neurochat.nc_data import NData


def load_data():
    dir = r'C:\Users\smartin5\recording_example'
    spike_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.2")
    pos_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms_2.txt")
    lfp_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.eeg")
    unit_no = 7
    ndata = NData()
    ndata.set_spike_file(spike_file)
    ndata.set_spatial_file(pos_file)
    ndata.set_lfp_file(lfp_file)
    ndata.load()
    ndata.set_unit_no(unit_no)

    return ndata
