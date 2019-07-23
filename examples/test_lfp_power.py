import sys
import os
import argparse
import matplotlib.pyplot as plt

sys.path.insert(1, r'C:\Users\smartin5\Repos\myNeurochat')
try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def plot_lfp_signal(lfp, lower, upper, out_name, filt=True, nsamples=None):
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    plt.plot(
        lfp.get_timestamp()[0:nsamples],
        filtered_lfp[0:nsamples], color='k')
    plt.savefig(out_name)
    plt.close()


def lfp_power(new_data, i, max_f, in_dir, prefilt=False):
    # 1.6 or 2 give similar
    filtset = [10, 1.5, max_f, 'bandpass']

    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 1.6, prefilt=prefilt,
        first_name="Theta", second_name="Delta",
        filtset=filtset)

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


def main(parsed):
    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    loc = parsed.loc
    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    in_dir = os.path.dirname(loc)
    ndata = NData()
    ndata.lfp.load(loc)
    out_dir = os.path.join(in_dir, "nc_results")

    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(os.path.join(out_dir, "dummy.txt"))

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        out_name = os.path.join(out_dir, "full_signal.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=False)
        out_name = os.path.join(
            in_dir, "nc_results", "full_signal_filt.png")
        plot_lfp_signal(
            ndata.lfp, 1.5, max_lfp, out_name, filt=True)

        splits = [
            (0, 600), (600, 1200),
            (1200, ndata.lfp.get_duration()),
            (0, ndata.lfp.get_duration())]

        for i, split in enumerate(splits):
            new_data = ndata.subsample(split)
            results = lfp_power(
                new_data, i, max_lfp, out_dir, prefilt=filt)
            print("For {} results are {}".format(i, results))
            f.write("{}: {}\n".format(i, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--nofilt", "-nf", action="store_true",
        help="Should pre filter lfp before power and spectral analysis")
    parser.add_argument(
        "--max_freq", "-mf", type=int, default=40,
        help="The maximum lfp frequency to consider"
    )
    parser.add_argument(
        "--loc", type=str, help="Lfp file location"
    )
    parsed = parser.parse_args()

    main(parsed)
