from itm_schema.ml_pipeline import (
    KDMAMeasurement,
    KDMAProfile,
)

import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import random




def plot_kdma_measurement(km: KDMAMeasurement, ax=None):
    if ax is None:
        ax = plt.figure().gca()
    if km.hist:
        # histogram
        ax.bar(km.hist.bin_edges[:-1], km.hist.bin_values, width=0.1)
    # value
    ax.axvline(km.value, c='red', lw=4)
    if km.kde:
        # kde
        X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
        log_dens = km.kde.score_samples(X_plot)
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens)*10,
            color='green',
            lw=2,
            linestyle="-",
            label="kernel = 'gaussian'",
        )
    ax.set_title(km.kdma_id.name)

# providing a path to a file name writes your graph to that file
def plot_kdma_profile(kdma_profile: KDMAProfile, ax=None, fig=None, file_name=None):
    if ax is None:
        ax = plt.figure().gca()
    if fig is None:
        fig = plt.figure()

    X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
    for kdma_id, kdma_mes in kdma_profile.kdma_measurements.items():
        log_dens = kdma_mes.kde.score_samples(X_plot)
        color = random.choice(list(mcolors.XKCD_COLORS.values()))
        ax.plot(
            X_plot[:, 0],
            np.exp(log_dens)*10,
            color=color,
            lw=1,
            linestyle="-",
            label=kdma_mes.kdma_id.name,
        )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f'KDMAProfile for {kdma_profile.dm_id}')
    
    # Save the figure for each profile if a filename was given
    if file_name != None:
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        plt.close()  # Close the figure to release resources