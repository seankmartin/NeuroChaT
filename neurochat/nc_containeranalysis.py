"""
This module contains analysis functions for NDataContainer objects.

@author: Sean Martin; martins7 at tcd dot ie
"""
import gc
import logging
from itertools import compress
from math import floor, ceil
import os
import gc
import re
from collections.abc import Iterable

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_data import NData
from neurochat.nc_clust import NClust
from neurochat.nc_utils import smooth_1d, find_true_ranges
from neurochat.nc_utils import find_peaks
from neurochat.nc_utils import window_rms
from neurochat.nc_utils import distinct_window_rms
from neurochat.nc_utils import make_dir_if_not_exists
from neurochat.nc_utils import log_exception
from neurochat.nc_plot import print_place_cells

import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib.pyplot import savefig, close


def place_cell_summary(
        collection, dpi=150, out_dirname="nc_plots",
        filter_place_cells=False, filter_low_freq=False,
        opt_end="", base_dir=None, output_format="png",
        output=["Wave", "Path", "Place", "HD", "LowAC", "Theta", "HighISI"],
        isi_bound=350, isi_bin_length=2, fixed_color=None,
        save_data=False, point_size=None, color_isi=True,
        burst_thresh=5, hd_predict=False):
    """
    Quick Png spatial information summary of each cell in collection.

    Parameters
    ----------
    collection : NDataCollection
        The collection to plot summaries of.
    dpi : int, default 150
        Dpi of the output figures.
    out_dirname : str, default "nc_plots
        The relative name of the dir to save pngs to
    filter_place_cells: bool, default True
        Whether to filter out non spatial cells from the plots.
        Considered non spatial if shuffled Skaggs, Coherency or Sparsity
        is similar to actual values.
    filter_low_freq: bool, default True
        Filter out cells with spike freq less than 0.1Hz
    opt_end : str, default ""
        A string to append to the file output just before the extension
    base_dir : str, default None
        An optional directory to save the files to
    output_format : str, default png
        What format to save the output image in
    output : List of str,
        default ["Wave", "Path", "Place", "HD", "LowAC", "Theta", "HighISI"]
        Input should be some subset and/or permutation of these
    isi_bound: int, default 350
        How long in ms to plot the ISI to
    isi_bin_length: int, default 1
        How long in ms the ISI bins should be
    save_data: bool, default False
        Whether to save out the information used for the plot
    color_isi: bool, default True
        Whether the ISI should be black or blue
    burst_thresh: int, default 5
        How long in ms to consider the window for burst to be
    hd_predict: bool, default False
        Whether the head directional graph should be plotted with predicted HD.

    Returns
    -------
    None
    """

    def save_dicts_to_csv(filename, dicts_arr):
        """Saves the last element of each arr in dicts_arr to file"""
        with open(filename, "w") as f:
            for d_arr in dicts_arr:
                if d_arr is not None:
                    d = d_arr[-1]
                    for k, v in d.items():
                        out_str = k.replace(" ", "_")
                        if isinstance(v, Iterable):
                            if isinstance(v, np.ndarray):
                                v = v.flatten()
                            else:
                                v = np.array(v).flatten()
                            str_arr = [str(x) for x in v]
                            out_str = out_str + "," + ",".join(str_arr)
                        else:
                            out_str += "," + str(v)
                        f.write(out_str + "\n")

    good_placedata = []
    good_graphdata = []
    good_wavedata = []
    good_headdata = []
    good_thetadata = []
    good_isidata = []
    good_units = []
    bad_placedata = []
    bad_graphdata = []
    bad_wavedata = []
    bad_headdata = []
    bad_thetadata = []
    bad_isidata = []
    bad_units = []
    skipped = 0

    if point_size is None:
        point_size = dpi / 7
    for i, data in enumerate(collection):
        try:
            data_idx, unit_idx = collection._index_to_data_pos(i)
            print("Working on {}, {}".format(unit_idx, len(
                collection.get_units(data_idx)) - 1))
            filename = collection.get_file_dict()["Spike"][data_idx][0]
            unit_number = collection.get_units(data_idx)[unit_idx]
            print("Working on {} unit {}".format(
                filename, unit_number))

            count = data.spike.get_unit_spikes_count()
            # Skip very low count cells
            if count < 5:
                skipped += 1
                print("Skipping as only {} spikes".format(count))
            else:
                duration = data.spike.get_duration()
                good = True

                # Place cell filtering is based on
                # https://www.nature.com/articles/ncomms11824
                # Activity-plasticity of hippocampus place maps
                # Schoenenberger et al, 2016
                if filter_low_freq:
                    if (count / duration) < 0.25 or (count / duration) > 7:
                        print("Reject spike frequency {}".format(
                            count / duration))
                        good = False

                if good and filter_place_cells:
                    skaggs = data.loc_shuffle(nshuff=1)
                    bad_sparsity = skaggs['refSparsity'] >= 0.3
                    bad_cohere = skaggs['refCoherence'] <= 0.55
                    first_str_part = "Accept "

                    if bad_sparsity or bad_cohere:
                        good = False
                        first_str_part = "Reject "

                    print((
                        first_str_part +
                        "Skaggs {:2f}, " +
                        "Sparsity {:2f}, " +
                        "Coherence {:2f}").format(
                        skaggs['refSkaggs'], skaggs['refSparsity'],
                        skaggs['refCoherence']))

                if good:
                    good_units.append(unit_idx)
                    placedata = good_placedata
                    graphdata = good_graphdata
                    wavedata = good_wavedata
                    headdata = good_headdata
                    thetadata = good_thetadata
                    isidata = good_isidata
                else:
                    bad_units.append(unit_idx)
                    placedata = bad_placedata
                    graphdata = bad_graphdata
                    wavedata = bad_wavedata
                    headdata = bad_headdata
                    thetadata = bad_thetadata
                    isidata = bad_isidata

                if (
                    (len(bad_units) + len(good_units)) >
                        len(collection.get_units(data_idx))):
                    save_bad = bad_units
                    save_good = good_units
                    good_placedata = []
                    good_graphdata = []
                    good_wavedata = []
                    good_headdata = []
                    good_thetadata = []
                    good_isidata = []
                    good_units = []
                    bad_placedata = []
                    bad_graphdata = []
                    bad_wavedata = []
                    bad_headdata = []
                    bad_thetadata = []
                    bad_isidata = []
                    bad_units = []
                    skipped = 0
                    print("ERROR: Found too many units in the collection")
                    raise Exception(
                        "Accumlated more units than possible " +
                        "bad {} good {} total {}".format(
                            save_bad, save_good, collection.get_units(data_idx)))

                if "Place" in output:
                    placedata.append(data.place())
                else:
                    bad_placedata = None
                    good_placedata = None
                if "LowAC" in output:
                    graphdata.append(data.isi_corr(bins=1, bound=[-10, 10]))
                else:
                    bad_graphdata = None
                    good_graphdata = None
                if "Wave" in output:
                    wavedata.append(data.wave_property())
                else:
                    bad_wavedata = None
                    good_wavedata = None
                if "HD" in output:
                    headdata.append(data.hd_rate())
                else:
                    bad_headdata = None
                    good_headdata = None
                if "Theta" in output:
                    thetadata.append(
                        data.theta_index(
                            bins=2, bound=[-350, 350]))
                else:
                    bad_thetadata = None
                    good_thetadata = None
                if "HighISI" in output:
                    isidata.append(
                        data.isi(bins=int(isi_bound / isi_bin_length),
                                 bound=[0, isi_bound]))
                else:
                    bad_isidata = None
                    good_isidata = None

                if save_data:
                    try:
                        spike_name = os.path.basename(filename)
                        parts = spike_name.split(".")
                        f_dir = os.path.dirname(filename)

                        data_basename = (
                            parts[0] + "_" + parts[1] + "_" +
                            str(unit_number) + opt_end + ".csv")
                        if base_dir is not None:
                            main_dir = base_dir
                            out_base = f_dir[len(base_dir + os.sep):]
                            if len(out_base) != 0:
                                out_base = ("--").join(out_base.split(os.sep))
                                data_basename = out_base + "--" + data_basename
                        else:
                            main_dir = f_dir
                        out_name = os.path.join(
                            main_dir, out_dirname, "data", data_basename)
                        make_dir_if_not_exists(out_name)
                        save_dicts_to_csv(
                            out_name,
                            [placedata, graphdata, wavedata,
                                headdata, thetadata, isidata])
                    except Exception as e:
                        log_exception(
                            e, "Occurred during place cell data saving on" +
                            " {} unit {} name {} in {}".format(
                                data_idx, unit_number, spike_name, main_dir))

            # Save the accumulated information
            if unit_idx == len(collection.get_units(data_idx)) - 1:
                if ((len(bad_units) + len(good_units)) !=
                        len(collection.get_units(data_idx)) - skipped):
                    print("ERROR: Did not cover all units in the collection")
                    print("Good {}, Bad {}, Total {}".format(
                        bad_units, good_units, collection.get_units(data_idx)))
                spike_name = os.path.basename(filename)
                parts = spike_name.split(".")
                f_dir = os.path.dirname(filename)

                out_basename = (
                    parts[0] + "_" + parts[1] + opt_end + "." + output_format)

                if base_dir is not None:
                    main_dir = base_dir
                    out_base = f_dir[len(base_dir + os.sep):]
                    if len(out_base) != 0:
                        out_base = ("--").join(out_base.split(os.sep))
                        out_basename = out_base + "--" + out_basename
                else:
                    main_dir = f_dir

                if filter_place_cells:
                    named_units = [
                        collection.get_units(data_idx)[j]
                        for j in good_units]
                    bad_named_units = [
                        collection.get_units(data_idx)[j]
                        for j in bad_units]
                else:
                    named_units = collection.get_units(data_idx)
                    bad_named_units = []

                # Save figures one by one if using pdf or svg
                one_by_one = (output_format == "pdf") or (
                    output_format == "svg")

                if len(named_units) > 0:
                    if filter_place_cells:
                        print((
                            "Plotting summary for {} " +
                            "spatial units {}").format(
                            spike_name, named_units))
                    else:
                        print((
                            "Plotting summary for {} " +
                            "units {}").format(
                            spike_name, named_units))

                    fig = print_place_cells(
                        len(named_units), cols=len(output),
                        placedata=good_placedata, graphdata=good_graphdata,
                        wavedata=good_wavedata, headdata=good_headdata,
                        thetadata=good_thetadata, isidata=good_isidata,
                        size_multiplier=4, point_size=point_size,
                        units=named_units, fixed_color=fixed_color,
                        output=output, color_isi=color_isi,
                        burst_ms=burst_thresh, one_by_one=one_by_one,
                        raster=one_by_one, hd_predict=hd_predict)

                    if one_by_one:
                        for k, f in enumerate(fig):
                            unit_number = named_units[k]
                            iname = (
                                out_basename[:-4] + "_" +
                                str(unit_number) + out_basename[-4:])
                            if filter_low_freq or filter_place_cells:
                                out_name = os.path.join(
                                    main_dir, out_dirname, "good", iname)
                            else:
                                out_name = os.path.join(
                                    main_dir, out_dirname, iname)

                            print("Saving place cell figure to {}".format(
                                out_name))
                            make_dir_if_not_exists(out_name)
                            f.savefig(out_name, dpi=dpi, format=output_format)

                    else:
                        if filter_low_freq or filter_place_cells:
                            out_name = os.path.join(
                                main_dir, out_dirname, "good", out_basename)
                        else:
                            out_name = os.path.join(
                                main_dir, out_dirname, out_basename)
                        print("Saving place cell figure to {}".format(
                            out_name))
                        make_dir_if_not_exists(out_name)
                        fig.savefig(out_name, dpi=dpi, format=output_format)

                    close("all")
                    gc.collect()

                if len(bad_named_units) > 0:
                    print((
                        "Plotting bad summary for {} " +
                        "non-spatial units {}").format(
                        spike_name, bad_named_units))
                    fig = print_place_cells(
                        len(bad_named_units), cols=len(output),
                        placedata=bad_placedata, graphdata=bad_graphdata,
                        wavedata=bad_wavedata, headdata=bad_headdata,
                        thetadata=bad_thetadata, isidata=bad_isidata,
                        size_multiplier=4, point_size=point_size,
                        units=bad_named_units, fixed_color=fixed_color,
                        output=output, color_isi=color_isi,
                        burst_ms=burst_thresh, one_by_one=one_by_one,
                        raster=one_by_one, hd_predict=hd_predict)

                    if one_by_one:
                        for k, f in enumerate(fig):
                            unit_number = bad_named_units[k]
                            iname = (
                                out_basename[:-4] + "_" +
                                str(unit_number) + out_basename[-4:])
                            out_name = os.path.join(
                                main_dir, out_dirname, "bad", iname)

                            print("Saving place cell figure to {}".format(
                                out_name))
                            make_dir_if_not_exists(out_name)
                            f.savefig(out_name, dpi=dpi,
                                      format=output_format)

                    else:
                        out_name = os.path.join(
                            main_dir, out_dirname, "bad", out_basename)
                        print("Saving place cell figure to {}".format(
                            out_name))
                        make_dir_if_not_exists(out_name)
                        fig.savefig(out_name, dpi=dpi,
                                    format=output_format)

                    close("all")
                    gc.collect()

                good_placedata = []
                good_graphdata = []
                good_wavedata = []
                good_headdata = []
                good_thetadata = []
                good_isidata = []
                good_units = []
                bad_placedata = []
                bad_graphdata = []
                bad_wavedata = []
                bad_headdata = []
                bad_thetadata = []
                bad_isidata = []
                bad_units = []
                skipped = 0

        except Exception as e:
            log_exception(
                e, "Occurred during place cell summary on data" +
                   " {} unit {} name {} in {}".format(
                       data_idx, unit_number, spike_name, main_dir))
    return


def evaluate_clusters(collection, idx1, idx2, set_units=False):
    """
    Find which units are closest in terms of clustering.

    Uses the Hungarian (Munkres) cost optimisation based on Hellinger distance
    between the clusters.

    Parameters
    ----------
    collection : NDataCollection
        The collection to find similar cells in
    idx1 : int
        The first data point in the collection to consider
    idx2 : int
        The second data point in the collection to consider

    Returns
    -------
    dict
        For each unit in data[idx1] (key), a tuple consisting of the
        best matching unit from data[idx2] and the distance for this (val)

    """
    nclust1 = NClust()
    nclust2 = NClust()

    sub_col1 = collection.subsample(idx1)
    info1 = sub_col1.get_file_dict()["Spike"][0]
    nclust1.load(info1[0], info1[2])

    sub_col2 = collection.subsample(idx2)
    info2 = sub_col2.get_file_dict()["Spike"][0]
    nclust2.load(info2[0], info2[2])

    distance_shape = (
        len(sub_col1.get_units()[0]), len(sub_col2.get_units()[0]))
    distances = np.zeros(shape=distance_shape)
    # Build a matrix of distances for each unit
    for idx1, unit1 in enumerate(sub_col1.get_units()[0]):
        for idx2, unit2 in enumerate(sub_col2.get_units()[0]):
            _, dh = nclust1.cluster_similarity(nclust2, unit1, unit2)
            distances[idx1, idx2] = dh
            # print(
            #     "{} {}: Bhattacharyya {} Hellinger {}".format(
            #         unit1, unit2, bc, dh))

    # Solve the linear sum assignment problem based on the Hungarian method
    solution = linear_sum_assignment(distances)
    best_matches = {}
    for i, j in zip(solution[0], solution[1]):
        best_matches[sub_col1.get_units()[0][i]] = (
            sub_col2.get_units()[0][j], distances[i, j])

    print("Best assignment is {}".format(best_matches))

    if set_units:
        run_units = [key for key in best_matches.keys()]
        best_units = [val[0] for _, val in best_matches.items()]
        collection.set_units([run_units, best_units])
    return best_matches


def count_units_in_bins(
        collection, bin_length, in_range=None, multi_ranges=None):
    """
    Count the amount of units that fire in certain bins.

    Parameters
    ----------
    collection : NDataCollection
        The collection of units to count over
    bin_length : float
        The length of the bins in seconds
    in_range : tuple
        The range of time to count units over

    Returns
    -------
    list of tuples:
        (unit counts in bins, bin_centres) for each data object in the collection

    """
    result = []
    calc_range = (in_range is None) and (multi_ranges is None)

    for idx in range(collection.get_num_data()):
        if collection.get_num_data() > 1:
            sub_collection = collection.subsample(idx)
        else:
            sub_collection = collection

        if calc_range:
            in_range = (0, sub_collection.get_data(0).get_duration())
        elif multi_ranges:
            in_range = multi_ranges[idx]
        num_bins = ceil(float(in_range[1] - in_range[0]) / bin_length)

        arr = np.empty(shape=(len(collection), num_bins))
        for data_idx, data in enumerate(sub_collection):
            spikes = data.get_unit_stamps_in_ranges([in_range])
            # Check if the unit spikes in the bin
            hist_val, bins = np.histogram(
                spikes, bins=num_bins, range=in_range)
            hist_val = np.clip(hist_val, 0, 1)
            arr[data_idx] = hist_val
        bin_centres = [(bins[j + 1] + bins[j]) / 2 for j in range(num_bins)]
        mua_tuple = (np.sum(arr, axis=0), bin_centres)
        result.append(mua_tuple)

    return result


# def smooth_speeds(collection, allow_multiple=False):
#     """
#     Smooth all the speed data in the collection.

#     Parameters
#     ----------
#     collection : NDataContainer
#         Container to get the information from
#     allows_multiple : bool
#         Allow smoothing multiple times, default False

#     Returns
#     -------
#     None

#     """
#     if collection._smoothed_speed and not allow_multiple:
#         logging.warning(
#             "NDataContainer has already been speed smoothed, not smoothing")

#     for i in range(collection.get_num_data()):
#         data = collection.get_data(i)
#         data.smooth_speed()
#         collection._smoothed_speed = True


def spike_positions(collection, should_sort=True, mode="vertical"):
    """
    Get the spike positions for a number of units.

    Parameters
    ----------
    collection : NDataContainer or NData list or NData object


    Returns
    -------
    positions : list of positions of the rat when the cell spiked

    """
    if isinstance(collection, NDataContainer) and should_sort:
        collection.sort_units_spatially(mode=mode)

    if isinstance(collection, NData):
        positions = collection.get_event_loc(collection.get_unit_stamp())[1]
        if mode == "vertical":
            positions = positions[1]
        elif mode == "horizontal":
            positions = positions[0]
        else:
            logging.error("nca: mode only supports vertical or horizontal")
    else:
        positions = []
        for data in collection:
            position = data.get_event_loc(data.get_unit_stamp())[1]
            if mode == "vertical":
                position = position[1]
            elif mode == "horizontal":
                position = position[0]
            else:
                logging.error("nca: mode only supports vertical or horizontal")
            positions.append(position)

    return positions


def spike_times(collection, filter_speed=False, **kwargs):
    """
    Return a list of all spike times in the collection.

    Parameters
    ----------
    collection : NDataContainer or NData
        Either the container or data object to get spike times from
    filter_speed : bool
        If true, don't consider spike times when the rat is non moving
    kwargs
        should_smooth : bool
            Smooth the speed data if true
        ranges : list
            List of tuples indicating time ranges to get spikes in

    Returns
    -------
    list
        The list of spike times if collection is NData
        or a 2d list containing a list of times for each collection item

    """
    should_smooth = kwargs.get("should_smooth", False)
    ranges = kwargs.get("ranges", None)

    if isinstance(collection, NData):
        if ranges is not None:
            time_data = collection.get_unit_stamps_in_ranges(ranges)
        elif filter_speed:
            ranges = collection.non_moving_periods(**kwargs)
            time_data = collection.get_unit_stamps_in_ranges(ranges)
        else:
            times = collection.get_unit_stamp()

    else:
        if should_smooth:
            smooth_speeds(collection)
            kwargs["should_smooth"] = False

        times = []
        for data in collection:
            if ranges is not None:
                time_data = data.get_unit_stamps_in_ranges(ranges)
            elif filter_speed:
                ranges = data.non_moving_periods(**kwargs)
                time_data = data.get_unit_stamps_in_ranges(ranges)
            else:
                time_data = data.get_unit_stamp()
            times.append(time_data)
    return times

    # def multi_unit_activity(collection, time_range=None, strip=False, **kwargs):
    # """
    # For each recording in the collection, detect periods of MUA.

    # WORK IN PROGRESS, NEEDS TO BE MODIFIED BEFORE REAL USE
    # Do not pass ranges and filter speed, only pass one.

    # Parameters
    # ----------
    # collection : NDataContainer
    #     The collection of units to detect Muti unit activity.
    # time_range : tuple, default None
    #     Optional time range to consider for the MUA.
    # strip: bool, default False
    #     If working with one data object in the collection,
    #     remove the surrounding array in output dict.

    # kwargs
    # ------
    # mua_bin_length : float
    #     The length of bins for mua histogram calculation in seconds
    # filter_length : float
    #     The std_dev of the gaussian used for filtering in seconds
    # mua_mode : str
    #     "rms_peaks" - calculate rms window and find peaks in this
    #     or "raw" - calculate mua histogram, extract bins with all cells active
    #     or "high_activity" - calculate rms window and
    #                          look for sustained high activity in this
    # mua_length : float
    #     The length of a mua event in seconds
    # filter_mua : bool
    #     Should the mua histogram be filtered by a guassian
    # mua_percentile : float
    #     The percentile threshold for a mua peak

    # Returns
    # -------
    # dict
    #     hists, mua

    # """
    # mua_bin_length = kwargs.get("mua_bin_length", 0.001)
    # filter_length = kwargs.get("filter_length", 0.01)
    # mode = kwargs.get("mua_mode", "rms_peaks")
    # mua_length = kwargs.get("mua_length", 0.6)
    # filter_mua = kwargs.get("filter_mua", True)
    # mua_percentile = kwargs.get("mua_percentile", 99)

    # result = {"mua hists": [], "mua": []}

    # # Get mua histogram for each data point
    # for data_idx in range(collection.get_num_data()):
    #     if collection.get_num_data() > 1:
    #         sub_collection = collection.subsample(data_idx)
    #     else:
    #         sub_collection = collection
    #     sample_rate = sub_collection.get_data(0).lfp.get_sampling_rate()
    #     sigma = filter_length * sample_rate
    #     unit_hist = count_units_in_bins(
    #         collection, mua_bin_length, time_range)[0]
    #     if filter_mua:
    #         unit_hist = (
    #             smooth_1d(unit_hist[0], filttype='g', filtsize=sigma),
    #             unit_hist[1])
    #     result["mua hists"].append(unit_hist)

    # for i, hist in enumerate(result['mua hists']):
    #     # Look for long periods of high activity
    #     if mode == "high_activity":
    #         p95 = np.percentile(hist[0], 95)
    #         result['mua'].append(find_true_ranges(
    #             hist[1], hist[0] > p95, min_range=mua_length))

    #     # Look for peaks in the activity
    #     if mode == "rms_peaks":
    #         p_val = np.percentile(hist[0], mua_percentile)
    #         _, peaks = find_peaks(hist[0], thresh=p_val)
    #         corresponding_ranges = [
    #             (hist[1][peak] - mua_length * 0.5,
    #              hist[1][peak] + mua_length * 0.5)
    #             for peak in peaks]
    #         result['mua'].append(corresponding_ranges)

    #     # Get areas where the number of units active is maximal
    #     if mode == "raw":
    #         if collection.get_num_data() > 1:
    #             num_cells = len(collection.subsample(i))
    #         else:
    #             num_cells = len(collection)
    #         mua_indices = np.argwhere(hist[0] == num_cells)
    #         corresponding_ranges = [
    #             (hist[1][index] - mua_length * 0.5,
    #              hist[1][index] + mua_length * 0.5)
    #             for index in mua_indices.flatten()]
    #         result["mua"].append(corresponding_ranges)

    # if strip and collection.get_num_data() == 1:
    #     result["mua hists"] = result["mua hists"][0]
    #     result["mua"] = result["mua"][0]

    # return result

    # def replay(collection, run_idx, sleep_idx, **kwargs):
    # """
    # Run and sleep session comparison.

    # Set the units of interest in the collection before running.

    # Parameters
    # ----------
    # collection : NDataContainer
    #     The collection of run and sleep data
    # run_idx : int
    #     The index in the collection for the run data
    # sleep_idx : int
    #     The index in the collection for the sleep data

    # kwargs
    # ------
    # sorting_mode : str
    #     "vertical" or "horizontal" order for spatial sorting
    # swr_window : float
    #     Lenth of SWR event around peak in seconds
    # match_clusters : bool
    #     If true, set the units being used in sleep to those
    #     most similar from run

    # kwargs are also passed into
    # nc_lfp.sharp_wave_ripples and
    # multi_unit_activity and
    # nc_spatial.non_moving_periods

    # Returns
    # -------
    # dict
    #     Graphical and numerical analysis results

    # See also
    # --------
    # nc_lfp.sharp_wave_ripples
    # nc_spatial.non_moving_periods
    # multi_unit_activity

    # """
    # results = {}

    # # Parse the kwargs
    # sorting_mode = kwargs.get("sorting_mode", "vertical")
    # swr_window = kwargs.get("swr_window", 0.2)
    # match_clusters = kwargs.get("match_clusters", True)

    # # Sort the run data spatially
    # truth_arr = [False for i in range(collection.get_num_data())]
    # truth_arr[run_idx] = True
    # collection.sort_units_spatially(truth_arr, mode=sorting_mode)

    # # Match up cells between the recordings
    # if match_clusters:
    #     evaluate_clusters(collection, run_idx, sleep_idx, set_units=True)

    # # Find the longest period of continuous sleep
    # sleep = collection.get_data(sleep_idx)
    # sleep_subsample = collection.subsample(sleep_idx)
    # sample_rate = sleep.lfp.get_sampling_rate()
    # non_moving_periods = np.array(
    #     sleep.non_moving_periods(**kwargs)) * sample_rate

    # # Could take multiple periods instead of just the longest
    # sorted_periods = sorted(
    #     non_moving_periods, key=lambda x: x[1] - x[0], reverse=True)
    # longest_sleep_period = sorted_periods[0]
    # raw_spike_times = spike_times(
    #     sleep_subsample,
    #     ranges=[longest_sleep_period / sample_rate])

    # # Estimate SWR
    # result_swr = sleep.lfp.sharp_wave_ripples(
    #     in_range=longest_sleep_period / sample_rate, **kwargs)

    # # Estimate MUA bursts
    # result_mua = multi_unit_activity(
    #     sleep_subsample, longest_sleep_period / sample_rate,
    #     strip=True, **kwargs)

    # results.update(result_swr)
    # results.update(result_mua)
    # results["spike times"] = raw_spike_times
    # results["num cells"] = len(sleep_subsample)

    # # Get the overlapping ranges of SWR and MUA
    # def swr_interval(peak):
    #     return (peak - 0.5 * swr_window, peak + 0.5 * swr_window)

    # def overlapping_swr_mua(mua, swr_peak):
    #     swr_range = swr_interval(swr_peak)
    #     overlapping = (
    #         (swr_range[0] < mua[0] < swr_range[1]) or
    #         (swr_range[0] < mua[1] < swr_range[1])
    #     )
    #     return overlapping

    # overlap = [
    #     mua_range for mua_range in results["mua"]
    #     if any(overlapping_swr_mua(mua_range, peak)
    #            for peak in results["swr times"])
    # ]

    # results["overlap swr mua"] = overlap

    # return results
