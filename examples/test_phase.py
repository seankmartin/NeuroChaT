import os
import numpy as np
import matplotlib.pyplot as plt
from neurochat.nc_data import NData
from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_utils import make_dir_if_not_exists


def main(dir):
    save_dir = os.path.join(dir, "plots", "phase")
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir)
    container.setup()
    spike_names = container.get_file_dict()["Spike"]

    for i, ndata in enumerate(container):
        name = spike_names[container._index_to_data_pos(i)[0]]
        results = ndata.phase_at_spikes(
            should_filter=True)
        positions = results["positions"]
        phases = results["phases"]
        good_place = results["good_place"]
        directions = results["directions"]
        co_ords = {}
        co_ords["north"] = np.nonzero(
            (45 <= directions) &
            (directions < 135))
        co_ords["south"] = np.nonzero(
            (225 <= directions) &
            (directions <= 315))
        if (phases.size != 0) and good_place:
            for direction in "north", "south":
                dim_pos = positions[1][co_ords[direction]]
                directional_phases = phases[co_ords[direction]]
                fig, ax = plt.subplots()
                ax.scatter(dim_pos, directional_phases)
                # ax.hist2d(dim_pos, directional_phases, bins=[10, 90])
                parts = os.path.basename(name[0]).split(".")
                end_name = (
                    parts[0] + "_unit" + str(ndata.get_unit_no()) + "_" +
                    direction + ".png")
                out_name = os.path.join(save_dir, end_name)
                make_dir_if_not_exists(out_name)
                fig.savefig(out_name)
                plt.close(fig)


def single_data_phase(
        dir, spike_name, pos_name, lfp_name):
    ndata = load_data(dir, spike_name, pos_name, lfp_name)
    spike_file = os.path.join(dir, spike_name)
    save_dir = os.path.join(dir, "plots", "phase")
    results = ndata.phase_at_spikes(
        should_filter=True)
    positions = results["positions"]
    phases = results["phases"]
    good_place = results["good_place"]
    directions = results["directions"]
    co_ords = {}
    co_ords["north"] = np.nonzero(
        (45 <= directions) &
        (directions < 135))
    co_ords["south"] = np.nonzero(
        (225 <= directions) &
        (directions <= 315))
    if (phases.size != 0) and good_place:
        for direction in "north", "south":
            dim_pos = positions[1][co_ords[direction]]
            directional_phases = phases[co_ords[direction]]
            fig, ax = plt.subplots()
            ax.scatter(dim_pos, directional_phases)
            # ax.hist2d(dim_pos, directional_phases, bins=[10, 90])
            parts = os.path.basename(spike_file[0]).split(".")
            end_name = (
                parts[0] + "_unit" + str(ndata.get_unit_no()) + "_" +
                direction + ".png")
            out_name = os.path.join(save_dir, end_name)
            make_dir_if_not_exists(out_name)
            fig.savefig(out_name)
            plt.close(fig)


def phase(dir, spike_name, pos_name, lfp_name):
    ndata = load_data(dir, spike_name, pos_name, lfp_name)
    results = ndata.phase_at_spikes(
        should_filter=True)
    boundary = results["boundary"]
    positions = results["positions"]
    phases = results["phases"]
    print(boundary)
    dim_pos = positions[1]
    plt.hist2d(dim_pos, phases, bins=[10, 180])
    plt.savefig("out.png")
    plt.scatter(dim_pos, phases)
    plt.savefig("outscatt.png")


def load_data(dir, spike_name, pos_name, lfp_name):
    spike_file = os.path.join(dir, spike_name)
    pos_file = os.path.join(dir, pos_name)
    lfp_file = os.path.join(dir, lfp_name)
    unit_no = 6
    ndata = NData()
    ndata.set_spike_file(spike_file)
    ndata.set_spatial_file(pos_file)
    ndata.set_lfp_file(lfp_file)
    ndata.load()
    ndata.set_unit_no(unit_no)
    return ndata


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\Recordings\LCA1'
    main(dir)
