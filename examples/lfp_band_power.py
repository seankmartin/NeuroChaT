from load_example_data import load_data
import matplotlib.pyplot as plt


def main():
    ndata = load_data()
    print(ndata.lfp.get_sampling_rate())
    print(ndata.lfp.get_total_samples())
    print(ndata.lfp.get_duration() * ndata.lfp.get_sampling_rate())
    theta = ndata.lfp.bandpower([4, 8], relative=False, window_sec=4)
    delta = ndata.lfp.bandpower([0.5, 4], relative=False, window_sec=4)
    full = ndata.lfp.bandpower([0, 125], relative=True, window_sec=4)
    full1 = ndata.lfp.bandpower([0, 125], relative=False, window_sec=4)
    print(full, full1)
    ratio = ndata.lfp.bandpower_ratio(
        [4, 8], [0.5, 4], 4, relative=False
    )
    print(ratio)
    print(theta, delta, theta / delta)
    plt.plot(ndata.lfp.get_timestamp(), ndata.lfp.get_samples())
    plt.savefig("out.png")


if __name__ == "__main__":
    main()
