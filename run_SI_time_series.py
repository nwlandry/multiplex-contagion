import multiplex_contagion
import shelve
import os

# COLORADO 90 DATA
data_folder = "Data"
dataset_folder = "CO90"
filename = "CO90_data"
full_file_path = os.path.join(data_folder, dataset_folder, filename)

network1_key = "sex-network"
network2_key = "drug-network"
combined_network_key = "combined-network"

# CO90 DATA
with shelve.open(full_file_path) as data:
    network1 = data[network1_key]
    network2 = data[network2_key]
    combined_network = data[combined_network_key]

num_processes = len(os.sched_getaffinity(0))
print(str(num_processes) + " cores")

num_infected = 1
num_simulations = 1000
tmax = 500

# CO90 p-values
# Zoomed
p_values = [1 / 75, 1 / 50, 1 / 30, 1 / 20]

# full
p_values = [1 / 1000, 1 / 100, 1 / 10, 1 / 2]

(
    timeseries1,
    timeseries2,
    timeseries3,
    timeseries4,
) = multiplex_contagion.generateTimeSeriesForSIContagionInParallel(
    network1,
    network2,
    combined_network,
    p_values,
    num_simulations,
    num_infected,
    tmax,
    num_processes,
)

newFilename = filename + "_SI_timeseries_numseeds=" + str(num_infected)
newfull_file_path = os.path.join(data_folder, dataset_folder, newFilename)

with shelve.open(newfull_file_path) as data:
    data["p-values"] = p_values
    data["t"] = range(tmax + 1)
    data["ts1"] = timeseries1
    data["ts2"] = timeseries2
    data["ts3"] = timeseries3
    data["ts4"] = timeseries4
