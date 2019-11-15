"""Plot all spatial cells in an Axona directory."""
import os

from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_containeranalysis as nca
from neurochat.nc_clust import NClust
from neurochat.nc_utils import log_exception

import numpy as np


def compare_two(file1, file2, index):
    nclust1 = NClust()
    nclust2 = NClust()
    nclust1.load(file1, "Axona")
    nclust2.load(file2, "Axona")
    units1 = nclust1.get_unit_list()
    units2 = nclust2.get_unit_list()
    bc_matrix = np.zeros(shape=(len(units1), len(units2)), dtype=np.float32)
    hd_matrix = np.zeros(shape=(len(units1), len(units2)), dtype=np.float32)
    for i, unit1 in enumerate(units1):
        for j, unit2 in enumerate(units2):
            try:
                bc, hd = nclust1.cluster_similarity(nclust2, unit1, unit2)
                print("({}, {}): BC {:.2f}, HD {:.2f}".format(
                    i + 1, j + 1, bc, hd))
                bc_matrix[i, j] = bc
                hd_matrix[i, j] = hd
            except Exception as e:
                log_exception(e, "at ({}, {})".format(i, j))
                bc_matrix[i, j] = np.nan
                hd_matrix[i, j] = np.nan

    out_filename = "output{}.csv".format(index)

    with open(out_filename, "w") as f:
        # Save the BC
        # for i in range(len(units1)):
        #     out_str = ""
        #     for j in range(len(units2)):
        #         out_str += str(bc_matrix[i, j]) + ","
        #     out_str = out_str[:-1] + "\n"
        #     f.write(out_str)

        # Save the HD
        for i in range(len(units1)):
            out_str = ""
            for j in range(len(units2)):
                out_str += str(hd_matrix[i, j]) + ","
            out_str = out_str[:-1] + "\n"
            f.write(out_str)


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\Recordings\recording_example'
    file1 = os.path.join(dir, "010416b-LS3-50Hz10V5ms.1")
    file2 = os.path.join(dir, "010416b-LS3-50Hz10V5ms.2")
    compare_two(file1, file2, 1)
    file1 = os.path.join(dir, "010416b-LS3-50Hz10V5ms.1")
    file2 = os.path.join(dir, "010416b-LS3-50Hz10V5ms.2")
    compare_two(file1, file2, 2)
