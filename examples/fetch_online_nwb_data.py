"""Downloads an NWB file from neurochat's OSF."""
import urllib.request
import os

from neurochat.nc_data import NData


def main(url, file_name, verbose=False):
    """
    This demonstrates example hdf5 usage.

    url should be the download url of a hdf5 file.
    file_name is a local disk path to store that file in.
    if verbose is true, information about the hdf5 file is printed.

    """
    # Fetch a file from OSF if not available on disk
    if not os.path.exists(file_name):
        print("Downloading file from {} to {}".format(
            url, file_name))
        urllib.request.urlretrieve(url, file_name)
    else:
        print("Using {}".format(file_name))

    if verbose:
        from skm_pyutils.py_print import print_h5
        print_h5(file_name)

    # Set up the h5 paths
    spike_path = "/processing/Shank/7"
    pos_path = "/processing/Behavioural/Position"
    lfp_path = "/processing/Neural Continuous/LFP/eeg"

    # HDF requires filename + path_in_hdf5
    # This function just does that
    def to_hdf_path(x):
        return file_name + "+" + x

    # Load in that data
    ndata = NData()
    ndata.set_data_format("NWB")
    ndata.set_spatial_file(to_hdf_path(pos_path))
    ndata.set_spike_file(to_hdf_path(spike_path))
    ndata.set_lfp_file(to_hdf_path(lfp_path))
    ndata.load()

    # Choose the unit number from those available
    print("Units are:", ndata.get_unit_list())
    unit_no = int(input("Unit to use:\n").strip())
    ndata.set_unit_no(unit_no)
    print("Loaded:", ndata)

    # Perform analysis
    ndata.place()
    ndata.wave_property()
    # print(ndata.get_results()["Spatial Skaggs"])
    # print(ndata.get_results()["Mean Spiking Freq"])
    print(ndata.get_results(spaces_to_underscores=True))
    print(ndata.get_results(spaces_to_underscores=False))


if __name__ == "__main__":
    file_name = "example.hdf"
    url = 'https://osf.io/89t7g/download/'
    verbose = False
    main(url, file_name, verbose=verbose)
