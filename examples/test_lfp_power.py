import sys
import os
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import entropy

try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def plot_lfp_signal(
        lfp, lower, upper, out_name,
        filt=True, nsamples=None, offset=0,
        nsplits=3, figsize=(32, 4), ylim=(-0.4, 0.4)):
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    fig, axes = plt.subplots(nsplits, 1, figsize=figsize)
    for i in range(nsplits):
        start = int(offset + i * (nsamples // nsplits))
        end = int(offset + (i + 1) * (nsamples // nsplits))
        if nsplits == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(
            lfp.get_timestamp()[start:end],
            filtered_lfp[start:end], color='k')
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_name, dpi=400)
    plt.close(fig)
    return filtered_lfp


def raw_lfp_power(lfp_samples, fs, splits, lower, upper, prefilt=False):
    """
    This can be used to get the power before splitting up the signal.

    Minor differences between this and filtering after splitting
    """

    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    results = OrderedDict()
    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        power = np.sum(np.square(sample)) / sample.size
        results["Raw power {}".format(i)] = power
    return results


def lfp_power(new_data, i, max_f, in_dir, prefilt=False, should_plot=True):
    # 1.6 or 2 give similar
    filtset = [10, 1.5, max_f, 'bandpass']

    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 1.6, band_total=prefilt,
        first_name="Theta", second_name="Delta",
        totalband=filtset[1:3])

    if should_plot:
        graphData = new_data.spectrum(
            window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
            filtset=filtset, fmax=max_f, db=False, tr=False)
        fig = nc_plot.lfp_spectrum(graphData)
        fig.savefig(os.path.join(in_dir, "spec" + str(i) + ".png"))

        graphData = new_data.spectrum(
            window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
            filtset=filtset, fmax=max_f, db=True, tr=True)
        fig = nc_plot.lfp_spectrum_tr(graphData)
        fig.savefig(os.path.join(in_dir, "spec_tr" + str(i) + ".png"))

    return new_data.get_results()


def lfp_entropy(
        lfp_samples, fs, splits, lower, upper, prefilt=False, etype="sample"):
    results = OrderedDict()

    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        if etype == "svd":
            et = entropy.svd_entropy(sample, order=3, delay=1)
        elif etype == "spectral":
            et = entropy.spectral_entropy(
                sample, 100, method='welch', normalize=True)
        elif etype == "sample":
            et = entropy.sample_entropy(sample, order=3)
        elif etype == "perm":
            et = entropy.perm_entropy(sample, order=3, normalize=True)
        else:
            print("Error: unrecognised entropy type {}".format(
                etype
            ))
            exit(-1)
        results["Entropy {}".format(i)] = et

    return results


def lfp_distribution(
        filename, upper, out_dir, splits,
        prefilt=False, get_entropy=False,
        get_theta=False, return_all=False):
    lfp_s_array = []
    lfp_t_array = []
    data = NData()
    avg = OrderedDict()
    ent = OrderedDict()

    if return_all:
        power_arr = np.zeros(shape=(32, len(splits)))
    for j in range(len(splits)):
        avg["Avg power {}".format(j)] = 0
        if get_entropy:
            ent["Avg entropy {}".format(j)] = 0

    for i in range(32):
        end = str(i + 1)
        if end == "1":
            load_loc = filename
        else:
            load_loc = filename + end
        data.lfp.load(load_loc)
        lfp_samples = data.lfp.get_samples()
        fs = data.lfp.get_sampling_rate()
        if prefilt:
            lfp_samples = butter_filter(
                lfp_samples, fs, 10, 1.5, upper, 'bandpass')
        lfp_s_array.append(lfp_samples)
        lfp_t_array.append(data.lfp.get_timestamp())

        p_result = raw_lfp_power(
            lfp_samples, fs, splits, 1.5, upper, prefilt=False)
        for j in range(len(splits)):
            avg["Avg power {}".format(j)] += (
                p_result["Raw power {}".format(j)] / 32)
            power_arr[i, j] = p_result["Raw power {}".format(j)]
        if get_entropy:
            p_result = lfp_entropy(
                lfp_samples, fs, splits, 1.5, upper, prefilt=False)
            for j in range(len(splits)):
                ent["Avg entropy {}".format(j)] += (
                    p_result["Entropy {}".format(j)] / 32)

    samples = np.concatenate(lfp_s_array)
    times = np.concatenate(lfp_t_array)

    fig, ax = plt.subplots()
    h = ax.hist2d(times, samples, bins=100)
    fig.colorbar(h[3])
    fig.savefig(os.path.join(out_dir, "dist.png"))

    if return_all:
        return avg, ent, power_arr
    return avg, ent


def lfp_theta_dist(filename, max_f, splits, prefilt=False):
    data = NData()
    filtset = [10, 1.5, max_f, 'bandpass']
    # This is for theta and delta power
    power_arr = np.zeros(shape=(6, 32, len(splits)))
    for i in range(32):
        end = str(i + 1)
        if end == "1":
            load_loc = filename
        else:
            load_loc = filename + end
        data.lfp.load(load_loc)
        for j, split in enumerate(splits):
            new_data = data.subsample(split)
            new_data.bandpower_ratio(
                [5, 11], [1.5, 4], 1.33, band_total=prefilt,
                first_name="Theta", second_name="Delta",
                totalband=filtset[1:3])
            t_result = new_data.get_results()
            power_arr[0, i, j] = t_result["Theta Power"]
            power_arr[1, i, j] = t_result["Delta Power"]
            power_arr[2, i, j] = t_result["Theta Delta Power Ratio"]
            power_arr[3, i, j] = t_result["Theta Power (Relative)"]
            power_arr[4, i, j] = t_result["Delta Power (Relative)"]
            power_arr[5, i, j] = t_result["Total Power"]

    return power_arr


def result_to_csv(result, out_dir, out_name="results.csv"):
    def arr_to_str(name, arr):
        out_str = name
        for val in arr:
            if isinstance(val, str):
                out_str = "{},{}".format(out_str, val)
            else:
                out_str = "{},{:2f}".format(out_str, val)
        return out_str

    out_loc = os.path.join(out_dir, out_name)
    make_dir_if_not_exists(out_loc)
    with open(out_loc, "w") as f:
        for key, val in result.items():
            if isinstance(val, dict):
                out_str = arr_to_str(key, val.values())
            elif isinstance(val, np.ndarray):
                out_str = arr_to_str(key, val.flatten())
            elif isinstance(val, list):
                out_str = arr_to_str(key, val)
            else:
                print("Unrecognised type {} quitting".format(
                    type(val)
                ))
                exit(-1)
            f.write(out_str + "\n")


def main(parsed, opt_merge=None):
    # Extract parsed args
    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    loc = parsed.loc
    eeg_num = parsed.eeg_num
    split_s = parsed.splits
    out_loc = parsed.out_loc
    every_min = parsed.every_min
    recording_dur = parsed.recording_dur
    get_entropy = parsed.get_entropy
    return_all = parsed.g_all

    # Do setup
    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    if every_min:
        splits = [(60 * i, 60 * (i + 1)) for i in range(recording_dur)]
        splits.append((0, 600))
        splits.append((600, 1200))
        splits.append((1200, 1800))

    else:
        splits = []
        for i in range(len(split_s) // 2):
            splits.append((split_s[i * 2], split_s[i * 2 + 1]))

    # Always include the full recording in this
    splits.append((0, recording_dur * 60))
    if eeg_num != "1":
        load_loc = loc + eeg_num
    else:
        load_loc = loc

    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)
    out_dir = os.path.join(in_dir, out_loc)

    if ndata.lfp.get_duration() == 0:
        print("Failed to correctly load lfp at {}".format(
            load_loc))
        exit(-1)

    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(os.path.join(out_dir, "dummy.txt"))

    # Plot signals
    out_name = os.path.join(out_dir, "full_signal.png")
    plot_lfp_signal(
        ndata.lfp, 1.5, max_lfp, out_name, filt=False)
    out_name = os.path.join(
        in_dir, out_dir, "full_signal_filt.png")
    filtered_lfp = plot_lfp_signal(
        ndata.lfp, 1.5, max_lfp, out_name, filt=True)
    if not filt:
        filtered_lfp = ndata.lfp

    # Calculate measures on this tetrode
    fs = ndata.lfp.get_sampling_rate()
    p_results = raw_lfp_power(
        filtered_lfp, fs, splits, 1.5, max_lfp, prefilt=False)

    if get_entropy:
        e_results = lfp_entropy(
            filtered_lfp, fs, splits[-4:], 1.5, max_lfp, prefilt=False)

    # Calculate measures over the dist
    d_result = lfp_distribution(
        loc, max_lfp, out_dir, splits[-4:],
        prefilt=filt, get_entropy=get_entropy, return_all=return_all)

    if get_entropy:
        results = {
            "power": p_results,
            "entropy": e_results,
            "avg_power": d_result[0],
            "avg_entropy": d_result[1]
        }
    else:
        results = {
            "power": p_results,
            "avg_power": d_result[0]
        }

    # Output the results
    result_to_csv(results, out_dir)

    t_results = lfp_theta_dist(
        loc, max_lfp, splits, prefilt=filt)

    return results, d_result[-1], t_results


def plot_sample_of_signal(
        load_loc, out_dir=None, name=None, offseta=0, length=50):
    """Plot a small filtered sample of the LFP signal."""
    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)

    if out_dir is None:
        out_loc = "nc_signal"
        out_dir = os.path.join(in_dir, out_loc)

    if name is None:
        name = "full_signal_filt.png"

    out_name = os.path.join(out_dir, name)
    make_dir_if_not_exists(out_name)
    plot_lfp_signal(
        ndata.lfp, 5, 11, out_name, filt=True,
        offset=ndata.lfp.get_sampling_rate() * offseta,
        nsamples=ndata.lfp.get_sampling_rate() * length,
        nsplits=1, ylim=(-0.3, 0.3),
        figsize=(20, 8))


def main_plot():
    """Main control of plotting through python options."""
    root = r"C:\Users\smartin5\Recordings\ER"
    name = "29082019-bt2\\29082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    plot_sample_of_signal(
        load_loc, out_dir="nc_results", name="Sal", offseta=400)
    name = "30082019-bt2\\30082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    plot_sample_of_signal(
        load_loc, out_dir="nc_results", name="Ser", offseta=400)


def main_cfg():
    """Main control through cmd options."""
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--nofilt", "-nf", action="store_true",
        help="Should not pre filter lfp before power and spectral analysis")
    parser.add_argument(
        "--max_freq", "-mf", type=int, default=40,
        help="The maximum lfp frequency to consider"
    )
    parser.add_argument(
        "--loc", type=str, help="Lfp file location"
    )
    parser.add_argument(
        "--eeg_num", "-en", type=str, help="EEG number", default="1"
    )
    parser.add_argument(
        "--splits", "-s", nargs="*", type=int, help="Splits",
        default=[0, 600, 600, 1200, 1200, 1800]
    )
    parser.add_argument(
        "--out_loc", "-o", type=str, default="nc_results",
        help="Relative name of directory to store results in"
    )
    parser.add_argument(
        "--every_min", "-em", action="store_true",
        help="Calculate lfp every minute"
    )
    parser.add_argument(
        "--recording_dur", "-d", type=int, default=30,
        help="How long in minutes the recording lasted"
    )
    parser.add_argument(
        "--get_entropy", "-e", action="store_true",
        help="Calculate entropy"
    )
    parser.add_argument(
        "--g_all", "-a", action="store_true",
        help="Get all values instead of just average"
    )
    parsed = parser.parse_args()

    main(parsed)


def main_py():
    """Main control through python options."""
    root = "C:\\Users\\smartin5\\Recordings\\ER\\"
    from types import SimpleNamespace
    ent = False
    arr_raw_pow = []
    arr_band_pow = []

    names_list = [
        ("29072019-bt\\29072019-bt-1st30min-LFP.eeg", "firstt", "bt1_1_sal"),
        ("30072019-bt\\30072019-bt-1st30min-LFP-DSER.eeg", "firstt", "bt1_1_ser"),
        ("29072019-bt\\29072019-bt-last30min-LFP.eeg", "lastt", "bt1_2_sal"),
        ("30072019-bt\\30072019-bt-last30min-LFP-DSER.eeg", "lastt", "bt1_2_ser"),
        ("29082019-nt2\\29082019-nt2-LFP-1st-Saline.eeg", "firstt", "nt2_1_sal"),
        ("30082019-nt2\\30082019-nt2-LFP-1st.eeg", "firstt", "nt2_1_ser"),
        ("29082019-nt2\\29082019-nt2-LFP-2nd-Saline.eeg", "lastt", "nt2_2_sal"),
        ("30082019-nt2\\30082019-nt2-LFP-2nd.eeg", "lastt", "nt2_2_ser"),
        ("29082019-bt2\\29082019-bt2-1st-LFP1_MERGE_29082019-bt2-1st-LFP2.eeg",
         "firstt", "bt2_1_sal"),
        ("30082019-bt2\\30082019-bt2-1st-LFP.eeg", "firstt", "bt2_1_ser"),
        ("29082019-bt2\\29082019-bt2-2nd-LFP.eeg", "lastt", "bt2_2_sal"),
        ("30082019-bt2\\30082019-bt2-2nd-LFP.eeg", "lastt", "bt2_2_ser")
    ]

    for name in names_list:
        args = SimpleNamespace(
            max_freq=40,
            nofilt=False,
            loc=os.path.join(root, name[0]),
            eeg_num="13",
            splits=[],
            out_loc=name[1],
            every_min=False,
            recording_dur=1800,
            get_entropy=ent,
            g_all=True
        )
        _, all1, band1 = main(args)
        arr_raw_pow.append(all1)
        arr_band_pow.append(band1)

    for i in range(0, len(arr_raw_pow), 2):
        difference = arr_raw_pow[i + 1] - arr_raw_pow[i]
        print("Mean difference is {:4f}".format(np.mean(difference)))
        print("Std deviation is {:4f}".format(np.std(difference)))

    _results = OrderedDict()
    _results["tetrodes"] = [i + 1 for i in range(32)]

    for (name, arr) in zip(names_list, arr_raw_pow):
        key_name = name[2]
        _results[key_name] = arr

    band_names = ["theta", "delta", "ratio", "theta rel", "delta rel", "total"]
    for i, bname in enumerate(band_names):
        for (name, arr) in zip(names_list, arr_band_pow):
            key_name = bname + " " + name[2]
            _results[key_name] = arr[i]

    # Some t_tests
    from scipy import stats
    _all = arr_raw_pow
    serine_list = [_all[i].flatten() for i in range(3, 12, 4)]
    saline_list = [_all[i].flatten() for i in range(2, 12, 4)]
    final1 = np.concatenate(serine_list)[3::4]
    final2 = np.concatenate(saline_list)[3::4]
    t_res = stats.ttest_rel(final1, final2)

    headers = [
        "Mean in Saline", "Mean in Serine",
        "Std Error in Saline", "Std Error in Serine",
        "T-test stat", "P-Value"]
    _results["Summary Stats"] = headers

    out_vals = [
        final2.mean(), final1.mean(),
        stats.sem(final2, ddof=1), stats.sem(final1, ddof=1),
        t_res[0], t_res[1]
    ]
    _results["Stats Vals"] = out_vals
    result_to_csv(_results, "nc_results", "power_results.csv")

    return


if __name__ == "__main__":
    # main_cfg()
    main_py()
    # main_plot()
