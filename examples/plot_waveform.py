"""Simple example to plot the waveform of a cell."""
from load_example_data import load_data

import matplotlib.pyplot as plt
import neurochat.nc_plot as nc_plot


def main():
    ndata = load_data()
    results = ndata.wave_property()
    nc_plot.wave_property(results)
    plt.savefig("out.png")
    print(ndata.get_results())


if __name__ == "__main__":
    main()
