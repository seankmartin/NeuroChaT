import os

import matplotlib.pyplot as plt

from neurochat.nc_spike import NSpike
from neurochat.nc_hdf import Nhdf
import neurochat.nc_plot as nc_plot

import logging


def main(fname, out_folder, plot=False):
    sorting = spikeinterface_test(fname)

    if plot:
        print("Plotting waveforms to {}".format(out_folder))
        plot_all_waveforms(sorting, out_folder)

    spike = NSpike()
    spike.load_spike_spikeinterface(sorting)
    unit_no = 12
    spike.set_unit_no(unit_no)
    print(spike)
    results = spike.wave_property()
    fig = nc_plot.wave_property(results)
    out_loc = os.path.join(out_folder, "wave_test{}.png".format(unit_no))
    print("Plotting neurochat waveform to {}".format(out_loc))
    fig.savefig(out_loc)

    hdf_path = os.path.join(out_folder, "test.h5")
    nhdf = Nhdf(filename=hdf_path)
    nhdf.save_spike(spike=spike)


def spikeinterface_test(folder_name):
    """TEST"""
    import spikeinterface.extractors as se
    to_exclude = ["mua", "noise"]
    sorting = se.PhySortingExtractor(
        folder_name, exclude_cluster_groups=to_exclude, load_waveforms=True,
        verbose=False)
    return sorting


def plot_all_waveforms(sorting, out_folder):
    unit_ids = sorting.get_unit_ids()

    waveform_eg = sorting.get_unit_spike_features(unit_ids[0], "waveforms")
    total_channels = waveform_eg.shape[1]

    wf_by_group = [
        sorting.get_unit_spike_features(u, "waveforms") for u in unit_ids]
    tetrode = 0
    for i, wf in enumerate(wf_by_group):
        try:
            tetrode = sorting.get_unit_property(unit_ids[i], "group")
        except Exception:
            try:
                tetrode = sorting.get_unit_property(
                    unit_ids[i], "ch_group")
            except Exception:
                print("Unable to find cluster group or group in units")
                print(sorting.get_shared_unit_property_names())
                tetrode += 1
                print("Will use tetrode {}".format(tetrode))

        fig, axes = plt.subplots(total_channels)
        for j in range(total_channels):
            try:
                wave = wf[:, j, :]
            except Exception:
                wave = wf[j, :]

            axes[j].plot(wave.T, color="k", lw=0.3)
        o_loc = os.path.join(
            out_folder, "tet{}_unit{}_forms.png".format(
                tetrode, unit_ids[i]))
        print("Saving waveform {} on tetrode {} to {}".format(
            i, tetrode, o_loc))
        fig.savefig(o_loc, dpi=200)
        plt.close("all")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)

    fname = r"/media/sean/Elements/Ham_Data/Batch_2/A9_CAR-SA1/CAR-SA1_20191130_1_PreBox/phy_klusta"
    home = os.path.abspath(os.path.expanduser("~"))
    file_name = os.path.join(home, "neurochat_temp",
                             "waveform_output", "dummy.png")
    if not os.path.isdir(os.path.dirname(file_name)):
        inp = input("Will create directory {}, is this ok? (y/n) ".format(
            os.path.dirname(file_name))).lower().strip()
        if inp != "y":
            "Ok, quitting"
            exit(0)
        else:
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

    out_folder = os.path.dirname(file_name)
    main(fname, out_folder)
