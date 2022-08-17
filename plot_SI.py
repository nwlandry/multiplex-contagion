import os
import random
import shelve
from fractions import Fraction

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

dataFolder = "Data"

datasetFolder = "CO90"
inputFilename = "CO90_data_gc_SI_timeseries_numseeds=1_long_run"
outputFilename = "CO90_data_gc_SI_plot_numseeds=1_long_run"

outputFolder = "Figures"
inputPath = os.path.join(dataFolder, datasetFolder, inputFilename)
outputPath = os.path.join(outputFolder, outputFilename)

with shelve.open(inputPath) as data:
    pValues = data["p-values"]
    t = data["t"]
    ts1 = data["ts1"]
    ts2 = data["ts2"]
    ts3 = data["ts3"]
    ts4 = data["ts4"]

transparency = 0.1
numTrajectories = 100
numBins = 15
maxTime = 1000
maxTime = min(t[-1], maxTime)
t = t[:maxTime]
ts1 = ts1[:, :, :maxTime]
ts2 = ts2[:, :, :maxTime]
ts3 = ts3[:, :, :maxTime]
ts4 = ts4[:, :, :maxTime]

fig = plt.figure(figsize=(4, 6))
plt.axis("off")
outer = gridspec.GridSpec(len(pValues), 1)

lims = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

for index in range(len(pValues)):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[index], wspace=0.3, hspace=0.0
    )

    # plot trajectories
    ax1 = fig.add_subplot(inner[:2])
    if index != len(pValues) - 1:
        ax1.set_xticklabels([])

    ax1.set_ylim(lims[index])
    ax1.text(
        1,
        0.8 * lims[index][1],
        r"$\beta=$" + str(Fraction(pValues[index]).limit_denominator(2000)),
        fontsize=10,
    )

    ax1.plot(
        t, np.mean(ts1[index, :, :], axis=0), "b", linewidth=1, label="sex network"
    )
    for i in random.sample(range(np.size(ts1, axis=1)), numTrajectories):
        ax1.plot(t, ts1[index, i, :], "b", linewidth=0.5, alpha=transparency)

    ax1.plot(
        t, np.mean(ts2[index, :, :], axis=0), "r", linewidth=1, label="drug network"
    )
    for i in random.sample(range(np.size(ts2, axis=1)), numTrajectories):
        ax1.plot(t, ts2[index, i, :], "r", linewidth=0.5, alpha=transparency)

    ax1.plot(
        t, np.mean(ts3[index, :, :], axis=0), "g", linewidth=1, label="combined uniplex"
    )
    for i in random.sample(range(np.size(ts3, axis=1)), numTrajectories):
        ax1.plot(t, ts3[index, i, :], "g", linewidth=0.5, alpha=transparency)

    ax1.plot(
        t, np.mean(ts4[index, :, :], axis=0), "k", linewidth=1, label="multiplexed data"
    )
    for i in random.sample(range(np.size(ts4, axis=1)), numTrajectories):
        ax1.plot(t, ts4[index, i, :], "k", linewidth=0.5, alpha=transparency)

    # plot uniplex histogram
    ax2 = fig.add_subplot(inner[2], sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    bothSexAndDrugs = ts1[index, :, -1] + ts2[index, :, -1] - ts3[index, :, -1]
    onlySex = ts1[index, :, -1] - bothSexAndDrugs
    onlyDrugs = ts2[index, :, -1] - bothSexAndDrugs

    bins = np.linspace(lims[index][0], lims[index][1], numBins)
    width = 0.5 * (bins[1] - bins[0])

    vals1, bins1 = np.histogram(onlySex, bins=bins)
    vals2, bins2 = np.histogram(bothSexAndDrugs, bins=bins)
    vals3, bins3 = np.histogram(onlyDrugs, bins=bins)

    ax2.barh(0.5 * (bins[:-1] + bins[1:]), vals1, width, color="blue")
    ax2.barh(0.5 * (bins[:-1] + bins[1:]), vals2, width, left=vals1, color="purple")
    ax2.barh(
        0.5 * (bins[:-1] + bins[1:]), vals3, width, left=vals1 + vals2, color="red"
    )

    ax2.set_xlim([0, 3000])

    # multiplexed data
    ax3 = fig.add_subplot(inner[3], sharey=ax1)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    if index == len(pValues) - 1:
        ax1.set_xlabel("time")
        ax1.set_ylabel("epidemic extent")

        ax2.set_xlabel("trajectories")
        ax3.set_xlabel("trajectories")
    else:
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])

    multiplexedSexAndDrugs = ts4[index, :, -1]

    vals1, bins1 = np.histogram(multiplexedSexAndDrugs, bins=bins)
    ax3.barh(0.5 * (bins[:-1] + bins[1:]), vals1, width, color="black")

    ax3.set_xlim([0, 1000])

plt.savefig(outputPath + ".png", dpi=1000)
plt.savefig(outputPath + ".pdf", dpi=1000)
plt.show()
