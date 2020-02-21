"""
Use this file to reproduce Figure 6 in the NeuroChaT paper
Please change line 13 to reflect the directory you saved 040512_1.hdf5 in
"""
import os
import numpy as np
from neurochat.nc_data import NData
import neurochat.nc_plot as nc_plot


def load_h5_data():
    # Change this to the directory where the file 040513_1.hdf5 is saved
    data_dir = r'C:\Users\smartin5\Recordings\NC_eg'
    main_file = "040513_1.hdf5"
    spike_file = "/processing/Shank/6"
    pos_file = "/processing/Behavioural/Position"
    lfp_file = "/processing/Neural Continuous/LFP/eeg"
    unit_no = 3

    def m_file(x): return os.path.join(data_dir, main_file + "+" + x)
    ndata = NData()
    ndata.set_data_format(data_format='NWB')
    ndata.set_spatial_file(m_file(pos_file))
    ndata.set_spike_file(m_file(spike_file))
    ndata.set_lfp_file(m_file(lfp_file))
    ndata.load()
    ndata.set_unit_no(unit_no)

    return ndata


def main():
    ndata = load_h5_data()
    graph_data = ndata.place()
    fig = nc_plot.loc_firing(
        graph_data, style="digitized", colormap="default")
    fig.savefig("Figure6a_place.png")

    graph_data = ndata.hd_rate()
    figs = nc_plot.hd_firing(graph_data)
    figs[1].savefig("Figure6a_hd.png")

    graph_data = ndata.multiple_regression()
    fig = nc_plot.multiple_regression(graph_data)
    fig.savefig("Figure6b.png")

    graph_data = ndata.loc_shuffle(bins=50)
    fig = nc_plot.loc_shuffle(graph_data)
    fig.savefig("Figure6c.png")

    graph_data = ndata.loc_shift(shift_ind=np.arange(-10, 21))
    figs = nc_plot.loc_time_shift(
        graph_data)
    figs[0].savefig("Figure6d.png")


if __name__ == "__main__":
    main()
