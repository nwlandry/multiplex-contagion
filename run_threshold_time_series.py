import os
import shelve

import multiplex_contagion

# JOAPP DATA
data_folder = "Data"
dataset_folder = "JOAPP"
filename = "JOAPP_data_gc"
full_file_path = os.path.join(data_folder, dataset_folder, filename)

network1_key = "foursquare-network"
network2_key = "twitter-network"
combined_network_key = "combined-network"


# Load data
with shelve.open(full_file_path) as data:
    network1 = data[network1_key]
    network2 = data[network2_key]
    combined_network = data[combined_network_key]

num_processes = len(os.sched_getaffinity(0))
print(str(num_processes) + " cores")

num_infected = 1
num_simulations = 1000
tmax = 50

# twitter-foursquare threshold_values for paper
# zoom
threshold_values = [1 / 8, 1 / 10, 1 / 12, 1 / 15]

# full
threshold_values = [1 / 2, 1 / 4, 1 / 20, 1 / 100]
(
    timeseries1,
    timeseries2,
    timeseries3,
    timeseries4,
) = multiplex_contagion.generateTimeSeriesForThresholdContagionInParallel(
    network1,
    network2,
    combined_network,
    threshold_values,
    num_simulations,
    num_infected,
    tmax,
    num_processes,
)

timeseries_filename = filename + "_threshold_timeseries_numseeds=" + str(num_infected)
timeseries_full_file_path = os.path.join(
    data_folder, dataset_folder, timeseries_filename
)

with shelve.open(timeseries_full_file_path) as data:
    data["threshold-values"] = threshold_values
    data["t"] = range(tmax + 1)
    data["ts1"] = timeseries1
    data["ts2"] = timeseries2
    data["ts3"] = timeseries3
    data["ts4"] = timeseries4
