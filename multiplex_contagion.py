"""
The purpose of this module is to run SI and threshold contagion on NetworkX graphs as individual simulations as well as ensembles of simulations for a range of parameter values
"""
import numpy as np
import random
import multiprocessing as mp

# SI Process
def SIContagion(G, initialInfected, p, tmin=0, tmax=100):
    """
    Parameters
    ----------
    G : NetworkX.Graph
    initialInfected : A set() of the nodes that are initially infected
    p : float between 0 and 1 that is the probability of infection
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    t : numpy.array of the times
    S : numpy.array of the number of susceptible at each time
    I : numpy.array of the number of infected at each time
    """
    n = G.number_of_nodes()
    infecteds = initialInfected.copy()
    newInfecteds = infecteds.copy()
    t = tmin
    times = [tmin]
    I = [len(infecteds)]
    S = [n - I[0]]

    while t < tmax:
        for node in G.nodes():
            if node not in infecteds:
                for nbr in G.neighbors(node):
                    if nbr in infecteds and random.random() <= p:
                        newInfecteds.add(node)
                        break
        t += 1
        infecteds = newInfecteds.copy()
        times.append(t)
        I.append(len(infecteds))
        S.append(n - I[-1])
    return np.array(times), np.array(S), np.array(I)


def SIContagionTwoNetworks(G1, G2, initialInfected, p, tmin=0, tmax=100):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    initialInfected : A set() of the nodes that are initially infected
    p : float between 0 and 1 that is the probability of infection
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    t : numpy.array of the times
    S : numpy.array of the number of susceptible (union between G1 and G2) at each time
    I : numpy.array of the number of infected (union between G1 and G2) at each time

    Notes
    -------
    - G1 and G2 must have the same number of nodes
    """
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    if n1 != n2:
        print(n1)
        print(n2)
        return
    infecteds1 = initialInfected.copy()
    infecteds2 = initialInfected.copy()
    newInfecteds1 = infecteds1.copy()
    newInfecteds2 = infecteds2.copy()
    t = tmin
    times = [tmin]
    I = [len(initialInfected)]
    S = [n1 - I[0]]

    while t < tmax:
        newInfecteds1 = infecteds1.copy()
        newInfecteds2 = infecteds2.copy()
        for node in G1.nodes():
            if node not in infecteds1:
                for nbr in G1.neighbors(node):
                    if nbr in infecteds1 and random.random() <= p:
                        newInfecteds1.add(node)
                        break

        for node in G2.nodes():
            if node not in infecteds2:
                for nbr in G2.neighbors(node):
                    if nbr in infecteds2 and random.random() <= p:
                        newInfecteds2.add(node)
                        break
        t += 1
        infecteds1 = newInfecteds1.copy()
        infecteds2 = newInfecteds2.copy()
        times.append(t)
        I.append(len(infecteds1.union(infecteds2)))
        S.append(n1 - I[-1])
    return np.array(times), np.array(S), np.array(I)


# Threshold process
def thresholdContagion(G, initialInfected, threshold, tmin=0, tmax=100):
    """
    Parameters
    ----------
    G : NetworkX.Graph
    initialInfected : A set() of the nodes that are initially infected
    threshold : float between 0 and 1 that is the fraction of infected neighbors needed for adoption
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    t : numpy.array of the times
    S : numpy.array of the number of susceptible at each time
    I : numpy.array of the number of infected at each time
    """
    n = G.number_of_nodes()
    infecteds = initialInfected.copy()
    newInfecteds = infecteds.copy()
    t = tmin
    times = [tmin]
    I = [len(infecteds)]
    S = [n - I[0]]
    while t < tmax:
        for node in G.nodes():
            if node not in infecteds:
                try:
                    fractionInfectedNeighbors = len(
                        [len for label in G.neighbors(node) if label in infecteds]
                    ) / len(list(G.neighbors(node)))
                except:
                    fractionInfectedNeighbors = 0
                if fractionInfectedNeighbors >= threshold:
                    newInfecteds.add(node)
        t += 1
        infecteds = newInfecteds.copy()
        times.append(t)
        I.append(len(infecteds))
        S.append(n - I[-1])
    return np.array(times), np.array(S), np.array(I)


def thresholdContagionTwoNetworks(G1, G2, initialInfected, threshold, tmin=0, tmax=100):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    initialInfected : A set() of the nodes that are initially infected
    threshold : float between 0 and 1 that is the fraction of infected neighbors needed for adoption
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    t : numpy.array of the times
    S : numpy.array of the number of susceptible (union between G1 and G2) at each time
    I : numpy.array of the number of infected (union between G1 and G2) at each time

    Notes
    -------
    - G1 and G2 must have the same number of nodes
    """
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    if n1 != n2:
        print(n1)
        print(n2)
        return
    infecteds1 = initialInfected.copy()
    infecteds2 = initialInfected.copy()
    newInfecteds1 = infecteds1.copy()
    newInfecteds2 = infecteds2.copy()
    t = tmin

    times = [tmin]
    I = [len(initialInfected)]
    S = [n1 - I[0]]

    while t < tmax:
        for node in G1.nodes():
            if node not in infecteds1:
                try:
                    fractionInfectedNeighbors = len(
                        [len for label in G1.neighbors(node) if label in infecteds1]
                    ) / len(list(G1.neighbors(node)))
                except:
                    fractionInfectedNeighbors = 0
                if fractionInfectedNeighbors >= threshold:
                    newInfecteds1.add(node)

        for node in G2.nodes():
            if node not in infecteds2:
                try:
                    fractionInfectedNeighbors = len(
                        [len for label in G2.neighbors(node) if label in infecteds2]
                    ) / len(list(G2.neighbors(node)))
                except:
                    fractionInfectedNeighbors = 0
                if fractionInfectedNeighbors >= threshold:
                    newInfecteds2.add(node)
        t += 1
        infecteds1 = newInfecteds1.copy()
        infecteds2 = newInfecteds2.copy()
        times.append(t)
        I.append(len(infecteds1.union(infecteds2)))
        S.append(n1 - I[-1])
    return np.array(times), np.array(S), np.array(I)


### Methods for generating data for plotting
def generateEquilibriaForThresholdContagion(
    G1, G2, G3, thresholds, numSimulations, numInfected, tmax
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    thresholds : A list() of floats between 0 and 1 that is the fraction of infected neighbors needed for adoption
    numSimulations : int which is the number of simulations to run per threshold value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1
    equilibrium2 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G2
    equilibrium3 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1 and G2 separately and unioned
    equilibrium4 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    equilibrium1 = np.zeros([len(thresholds), numSimulations])
    equilibrium2 = np.zeros([len(thresholds), numSimulations])
    equilibrium3 = np.zeros([len(thresholds), numSimulations])
    equilibrium4 = np.zeros([len(thresholds), numSimulations])

    nodeList = G3.nodes()
    n = len(nodeList)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            t, S, I1 = thresholdContagion(G1, initialInfected, threshold, tmax=tmax)
            t, S, I2 = thresholdContagion(G2, initialInfected, threshold, tmax=tmax)
            t, S, I3 = thresholdContagionTwoNetworks(
                G1, G2, initialInfected, threshold, tmin=0, tmax=tmax
            )
            t, S, I4 = thresholdContagion(G3, initialInfected, threshold, tmax=tmax)

            equilibrium1[i, sim] = I1[-1] / n
            equilibrium2[i, sim] = I2[-1] / n
            equilibrium3[i, sim] = I3[-1] / n
            equilibrium4[i, sim] = I4[-1] / n
    return equilibrium1, equilibrium2, equilibrium3, equilibrium4


def generateEquilibriaForSIContagion(
    G1, G2, G3, pValues, numSimulations, numInfected, tmax
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    pValues : A list() of floats between 0 and 1 that is the probability of pairwise infection
    numSimulations : int which is the number of simulations to run per threshold value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1
    equilibrium2 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G2
    equilibrium3 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1 and G2 separately and unioned
    equilibrium4 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    equilibrium1 = np.zeros([len(pValues), numSimulations])
    equilibrium2 = np.zeros([len(pValues), numSimulations])
    equilibrium3 = np.zeros([len(pValues), numSimulations])
    equilibrium4 = np.zeros([len(pValues), numSimulations])

    nodeList = G3.nodes()
    n = len(nodeList)
    for i in range(len(pValues)):
        p = pValues[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            t, S, I1 = SIContagion(G1, initialInfected, p, tmax=tmax)
            t, S, I2 = SIContagion(G2, initialInfected, p, tmax=tmax)
            t, S, I3 = SIContagionTwoNetworks(G1, G2, initialInfected, p, tmax=tmax)
            t, S, I4 = SIContagion(G3, initialInfected, p, tmax=tmax)

            equilibrium1[i, sim] = I1[-1] / n
            equilibrium2[i, sim] = I2[-1] / n
            equilibrium3[i, sim] = I3[-1] / n
            equilibrium4[i, sim] = I4[-1] / n
    return equilibrium1, equilibrium2, equilibrium3, equilibrium4


# same as above but in parallel
def generateEquilibriaForThresholdContagionInParallel(
    G1,
    G2,
    G3,
    thresholds,
    numSimulations,
    numInfected,
    tmax,
    numProcesses,
    verbose=True,
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    thresholds : A list() of floats between 0 and 1 that is the fraction of infected neighbors needed for adoption
    numSimulations : int which is the number of simulations to run per threshold value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1
    equilibrium2 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G2
    equilibrium3 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1 and G2 separately and unioned
    equilibrium4 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    equilibrium1 = np.zeros([len(thresholds), numSimulations])
    equilibrium2 = np.zeros([len(thresholds), numSimulations])
    equilibrium3 = np.zeros([len(thresholds), numSimulations])
    equilibrium4 = np.zeros([len(thresholds), numSimulations])

    nodeList = G3.nodes()
    argList = list()
    n = len(nodeList)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            argList.append((G1, G2, G3, threshold, initialInfected, tmax, verbose))

    with mp.Pool(processes=numProcesses) as pool:
        equilibria = pool.starmap(thresholdEquilibriumFunction, argList)

    line = 0
    for i in range(len(thresholds)):
        for sim in range(numSimulations):
            equilibrium1[i, sim] = equilibria[line][0] / n
            equilibrium2[i, sim] = equilibria[line][1] / n
            equilibrium3[i, sim] = equilibria[line][2] / n
            equilibrium4[i, sim] = equilibria[line][3] / n
            line += 1

    return equilibrium1, equilibrium2, equilibrium3, equilibrium4


# packages the functions together so we can use the starmap function to parallelize
def thresholdEquilibriumFunction(G1, G2, G3, threshold, initialInfected, tmax, verbose):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    threshold : float between 0 and 1 that is the fraction of infected neighbors needed for adoption
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : float with the equilibrium for a given threshold value run on G1
    equilibrium2 : float with the equilibrium for a given threshold value run on G2
    equilibrium3 : float with the equilibrium for a given threshold value run on G1 and G2 separately and unioned
    equilibrium4 : float with the equilibrium for a given threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    t, S, I1 = thresholdContagion(G1, initialInfected, threshold, tmax=tmax)
    t, S, I2 = thresholdContagion(G2, initialInfected, threshold, tmax=tmax)
    t, S, I3 = thresholdContagionTwoNetworks(
        G1, G2, initialInfected, threshold, tmin=0, tmax=tmax
    )
    t, S, I4 = thresholdContagion(G3, initialInfected, threshold, tmax=tmax)
    if verbose:
        print(
            str(I1[-1])
            + ", "
            + str(I2[-1])
            + ", "
            + str(I3[-1])
            + ", "
            + str(I4[-1])
            + ": threshold = "
            + str(threshold),
            flush=True,
        )
    return I1[-1], I2[-1], I3[-1], I4[-1]


# same as above but in parallel
def generateEquilibriaForSIContagionInParallel(
    G1, G2, G3, pValues, numSimulations, numInfected, tmax, numProcesses, verbose=True
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    pValues : A list() of floats between 0 and 1 that is the probability of pairwise infection
    numSimulations : int which is the number of simulations to run per p value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1
    equilibrium2 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G2
    equilibrium3 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G1 and G2 separately and unioned
    equilibrium4 : numpy.array with dimensions numSimulations x length(thresholds) with the equilibria for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    equilibrium1 = np.zeros([len(pValues), numSimulations])
    equilibrium2 = np.zeros([len(pValues), numSimulations])
    equilibrium3 = np.zeros([len(pValues), numSimulations])
    equilibrium4 = np.zeros([len(pValues), numSimulations])

    nodeList = G3.nodes()
    argList = list()
    n = len(nodeList)
    for i in range(len(pValues)):
        p = pValues[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            argList.append((G1, G2, G3, p, initialInfected, tmax, verbose))
    with mp.Pool(processes=numProcesses) as pool:
        equilibria = pool.starmap(SIEquilibriumFunction, argList)

    line = 0
    for i in range(len(pValues)):
        for sim in range(numSimulations):
            equilibrium1[i, sim] = equilibria[line][0] / n
            equilibrium2[i, sim] = equilibria[line][1] / n
            equilibrium3[i, sim] = equilibria[line][2] / n
            equilibrium4[i, sim] = equilibria[line][3] / n
            line += 1
    return equilibrium1, equilibrium2, equilibrium3, equilibrium4


# packages the functions together so we can use the starmap function to parallelize
def SIEquilibriumFunction(G1, G2, G3, p, initialInfected, tmax, verbose):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    p : float between 0 and 1 that is the probability of pairwise infection
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    equilibrium1 : numpy.array with dimensions numSimulations x 1 with the equilibria for each simulation and threshold value run on G1
    equilibrium2 : numpy.array with dimensions numSimulations x 1 with the equilibria for each simulation and threshold value run on G2
    equilibrium3 : numpy.array with dimensions numSimulations x 1 with the equilibria for each simulation and threshold value run on G1 and G2 separately and unioned
    equilibrium4 : numpy.array with dimensions numSimulations x 1 with the equilibria for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    t, S, I1 = SIContagion(G1, initialInfected, p, tmax=tmax)
    t, S, I2 = SIContagion(G2, initialInfected, p, tmax=tmax)
    t, S, I3 = SIContagionTwoNetworks(G1, G2, initialInfected, p, tmax=tmax)
    t, S, I4 = SIContagion(G3, initialInfected, p, tmax=tmax)
    if verbose:
        print(
            str(I1[-1])
            + ", "
            + str(I2[-1])
            + ", "
            + str(I3[-1])
            + ", "
            + str(I4[-1])
            + ": p = "
            + str(p),
            flush=True,
        )
    return I1[-1], I2[-1], I3[-1], I4[-1]


### Data for plotting, but now it's the whole time-series
# same as above but in parallel
def generateTimeSeriesForThresholdContagionInParallel(
    G1,
    G2,
    G3,
    thresholds,
    numSimulations,
    numInfected,
    tmax,
    numProcesses,
    verbose=True,
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    thresholds : A list() of floats between 0 and 1 that is the fraction of infected neighbors needed for adoption
    numSimulations : int which is the number of simulations to run per threshold value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    timeseries1 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and threshold value run on G1
    timeseries2 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and threshold value run on G2
    timeseries3 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and threshold value run on G1 and G2 separately and unioned
    timeseries4 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and threshold value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    timeseries1 = np.zeros([len(thresholds), numSimulations, tmax + 1])
    timeseries2 = np.zeros([len(thresholds), numSimulations, tmax + 1])
    timeseries3 = np.zeros([len(thresholds), numSimulations, tmax + 1])
    timeseries4 = np.zeros([len(thresholds), numSimulations, tmax + 1])

    nodeList = G3.nodes()
    argList = list()
    n = len(nodeList)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            argList.append((G1, G2, G3, threshold, initialInfected, tmax, verbose))

    with mp.Pool(processes=numProcesses) as pool:
        equilibria = pool.starmap(thresholdTimeSeriesFunction, argList)

    line = 0
    for i in range(len(thresholds)):
        for sim in range(numSimulations):
            timeseries1[i, sim, :] = equilibria[line][0] / n
            timeseries2[i, sim, :] = equilibria[line][1] / n
            timeseries3[i, sim, :] = equilibria[line][2] / n
            timeseries4[i, sim, :] = equilibria[line][3] / n
            line += 1

    return timeseries1, timeseries2, timeseries3, timeseries4


# packages the functions together so we can use the starmap function to parallelize
def thresholdTimeSeriesFunction(G1, G2, G3, threshold, initialInfected, tmax, verbose):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    threshold : float between 0 and 1 that is the fraction of infected neighbors needed for adoption
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    I1 : numpy.array with length tmax with the timeseries for the given p value run on G1
    I2 : numpy.array with length tmax with the timeseries for the given p value run on G2
    I3 : numpy.array with length tmax with the timeseries for the given p value run on G1 and G2 separately and unioned
    I4 :numpy.array with length tmax with the timeseries for the given p value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    t, S, I1 = thresholdContagion(G1, initialInfected, threshold, tmax=tmax)
    t, S, I2 = thresholdContagion(G2, initialInfected, threshold, tmax=tmax)
    t, S, I3 = thresholdContagionTwoNetworks(
        G1, G2, initialInfected, threshold, tmin=0, tmax=tmax
    )
    t, S, I4 = thresholdContagion(G3, initialInfected, threshold, tmax=tmax)
    if verbose:
        print(
            str(I1[-1])
            + ", "
            + str(I2[-1])
            + ", "
            + str(I3[-1])
            + ", "
            + str(I4[-1])
            + ": threshold = "
            + str(threshold),
            flush=True,
        )
    return I1, I2, I3, I4


# same as above but in parallel
def generateTimeSeriesForSIContagionInParallel(
    G1, G2, G3, pValues, numSimulations, numInfected, tmax, numProcesses, verbose=True
):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    pValues : A list() of floats between 0 and 1 that is the probability of pairwise infection
    numSimulations : int which is the number of simulations to run per p value
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    timeseries1 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and p value run on G1
    timeseries2 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and p value run on G2
    timeseries3 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and p value run on G1 and G2 separately and unioned
    timeseries4 : numpy.array with dimensions numSimulations x length(thresholds) x tmax with the time series for each simulation and p value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    timeseries1 = np.zeros([len(pValues), numSimulations, tmax + 1])
    timeseries2 = np.zeros([len(pValues), numSimulations, tmax + 1])
    timeseries3 = np.zeros([len(pValues), numSimulations, tmax + 1])
    timeseries4 = np.zeros([len(pValues), numSimulations, tmax + 1])

    nodeList = G3.nodes()
    argList = list()
    n = len(nodeList)
    for i in range(len(pValues)):
        p = pValues[i]
        for sim in range(numSimulations):
            initialInfected = numberOfInfectedNodes(nodeList, numInfected)
            argList.append((G1, G2, G3, p, initialInfected, tmax, verbose))
    with mp.Pool(processes=numProcesses) as pool:
        equilibria = pool.starmap(SITimeSeriesFunction, argList)

    line = 0
    for i in range(len(pValues)):
        for sim in range(numSimulations):
            timeseries1[i, sim, :] = equilibria[line][0] / n
            timeseries2[i, sim, :] = equilibria[line][1] / n
            timeseries3[i, sim, :] = equilibria[line][2] / n
            timeseries4[i, sim, :] = equilibria[line][3] / n
            line += 1
    return timeseries1, timeseries2, timeseries3, timeseries4


# packages the functions together so we can use the starmap function to parallelize
def SITimeSeriesFunction(G1, G2, G3, p, initialInfected, tmax, verbose):
    """
    Parameters
    ----------
    G1 : NetworkX.Graph (layer 1)
    G2 : NetworkX.Graph (layer 2)
    G3 : NetworkX.Graph (multiplex of layers 1 and 2)
    p : float between 0 and 1 that is the probability of pairwise infection
    tmin : integer of the initial time
    tmax : integer of the final time

    Returns
    -------
    I1 : numpy.array with length tmax with the timeseries for the given p value run on G1
    I2 : numpy.array with length tmax with the timeseries for the given p value run on G2
    I3 : numpy.array with length tmax with the timeseries for the given p value run on G1 and G2 separately and unioned
    I4 :numpy.array with length tmax with the timeseries for the given p value run on G3

    Notes
    -------
    - G1, G2, and G3 must have the same number of nodes
    """
    t, S, I1 = SIContagion(G1, initialInfected, p, tmax=tmax)
    t, S, I2 = SIContagion(G2, initialInfected, p, tmax=tmax)
    t, S, I3 = SIContagionTwoNetworks(G1, G2, initialInfected, p, tmax=tmax)
    t, S, I4 = SIContagion(G3, initialInfected, p, tmax=tmax)
    if verbose:
        print(
            str(I1[-1])
            + ", "
            + str(I2[-1])
            + ", "
            + str(I3[-1])
            + ", "
            + str(I4[-1])
            + ": p = "
            + str(p),
            flush=True,
        )
    return I1, I2, I3, I4


def fractionOfInfectedNodes(nodeList, fractionInfected):
    """
    Parameters
    ----------
    nodeList : list() of node ids
    fractionInfected : a float between 0 and 1 indicating the fraction of nodes to randomly infect

    Returns
    -------
    infecteds : a set() of initially infected nodes
    """
    infecteds = set()
    for node in nodeList:
        if random.random() <= fractionInfected:
            infecteds.add(node)
    return infecteds


def numberOfInfectedNodes(nodeList, numInfected):
    """
    Parameters
    ----------
    nodeList : list() of node ids
    numInfected : a int between 1 and the number of nodes indicating the number of nodes to randomly infect

    Returns
    -------
    infecteds : a set() of initially infected nodes
    """
    infecteds = random.sample(list(nodeList), numInfected)
    return set(infecteds)
