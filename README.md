# Multiplex Contagion

This repository accompanies the article ["On limitations of uniplex networks for modeling multiplex contagion"](https://doi.org/10.1371/journal.pone.0279345) by Nicholas Landry and jimi adams and provides all scripts necessary to reproduce all results and figures.

This repository provides the bases for a project seeking to demonstrate the limitations of modeling multiplex diffusion processes from uniplex network data. We do so by: starting from multiplex data sources, and simulating diffusion processes over those networks. We then decompose each network into uniplex versions, and re-simulate the same diffusion processes over those. We then examine how the addition of the component uniplex simulations correspond to the composite simulation process.

---

What's here:

**Data** (in the Data folder)

- The import_CO90_data_full.py and import_CO90_data_GC.py file convert the raw data in the ICPSR_22140 folder from "Project 90" in Colorado Springs ([original data](https://opr.princeton.edu/archive/p90/)) into NetworkX objects and saves using the Python "shelve" module. They load the full data and the giant component respectively. The output objects can be retrieved with the following keys:
    - *sex-network* - NetworkX object of the sex network
    - *drug-network* - NetworkX object of the drug network (both drugs and needles)
    - *combined-network* - NetworkX object of the multiplexed sex and drug networks

- The import_JOAPP_data_full.py and import_JOAPP_data_GC.py files convert the raw data in the Twitter-Foursquare folder ([original data](https://doi.org/10.6084/m9.figshare.4585270.v1)) into NetworkX objects and saves using the Python "shelve" module. They load the full data and the giant component respectively. The output objects can be retrieved with the following keys:
    - *foursquare-network* - NetworkX object of the Foursquare network
    - *twitter-network* - NetworkX object of the Twitter network
    - *combined-network* - NetworkX object of the multiplexed Foursquare and Twitter networks

**Libraries**

- multiplex_contagion.py is a collection of functions to simulate contagion processes on multiplexed and uniplex data
- plot_utilities.py contains the function used to plot the multiplex networks in both Jupyter notebooks.

**Scripts**

- run_SI_time_series.py runs an ensemble of SI contagion simulations for a list of p values and outputs the time series for each realization and parameter value into a shelve file.
- run_threshold_time_series.py runs an ensemble of threshold contagion simulations for a list of threshold values and outputs the time series for each realization and parameter value into a shelve file.
- plot_SI_time_series.py plots the result from run_SI_time_series.py
- plot_threshold_time_series.py plots the result from run_threshold_time_series.py

**Notebooks**

- plot_CO90_dataset.ipynb generates a multiplex visualization of the Project 90 dataset based on [this script](https://github.com/jkbren/matplotlib-multilayer-network)
- plot_JOAPP_dataset.ipynb generates a multiplex visualization of the JOAPP dataset based on [this script](https://github.com/jkbren/matplotlib-multilayer-network)

**Getting Started**

Clone this repository and within the main directory, make a directory titled "Data". Download the [Project 90 dataset](https://opr.princeton.edu/archive/p90/) and the [JOAPP dataset](https://doi.org/10.6084/m9.figshare.4585270.v1). Within the Data folder, copy the unzipped `ICPSR_22140` and `rsos160863_si_001` folders from the Project 90 and JOAPP datasets respectively into the Data folder. All scripts should work once this is complete.
