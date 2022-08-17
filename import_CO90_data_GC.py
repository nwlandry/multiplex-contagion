import os
import shelve

import networkx as nx
import numpy as np
import pandas as pd

data_folder = "Data"
dataset_folder = "CO90"
filename = "CO90_data_gc"

directory1 = "Data"
directory2 = "ICPSR_22140"
directory3node = "DS0001"
directory3edge = "DS0002"
edge_directory = os.path.join(directory1, directory2, directory3edge)
node_directory = os.path.join(directory1, directory2, directory3node)
node_file = "22140-0001-Data.tsv"
edge_file = "22140-0002-Data.tsv"

df_nodes = pd.read_csv(
    os.path.join(node_directory, node_file), delimiter="\t", low_memory=False
)

df_edges = pd.read_csv(
    os.path.join(edge_directory, edge_file), delimiter="\t", low_memory=False
)

# parse by study
node_data = df_nodes[df_nodes["STUDYNUM"] == 1]
edge_data = df_edges[df_edges["STUDYNUM"] == 1]

sex_edges = edge_data[edge_data["TIETYPE"] == 3]
drug_edges = edge_data[edge_data["TIETYPE"] == 2]
needle_edges = edge_data[edge_data["TIETYPE"] == 4]
drug_and_needle_edges = pd.concat(
    [drug_edges, needle_edges]
)  # drug and needle together
combined_edges = edge_data[edge_data["TIETYPE"] > 1]

combined_network = nx.from_pandas_edgelist(combined_edges, source="ID1", target="ID2")
connected_nodes = max(nx.connected_components(combined_network), key=len)
combined_network = combined_network.subgraph(connected_nodes).copy()

node_list = connected_nodes

sex_network = nx.from_pandas_edgelist(sex_edges, source="ID1", target="ID2")
sex_network = sex_network.subgraph(connected_nodes).copy()
sex_network.add_nodes_from(node_list)


drug_network = nx.from_pandas_edgelist(
    drug_and_needle_edges, source="ID1", target="ID2"
)
drug_network = drug_network.subgraph(connected_nodes).copy()
drug_network.add_nodes_from(node_list)

print(f"The size of all the networks is {len(combined_network)}")

print(
    f"The mean degree of the combined network is {2 * len(combined_network.edges) / len(combined_network)}"
)
print(
    f"The mean degree of the sex network is {2 * len(sex_network.edges) / len(sex_network)}"
)
print(
    f"The mean degree of the drug network is {2 * len(drug_network.edges) / len(drug_network)}"
)

print(
    f"The average random component size of the combined network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(combined_network))}"
)
print(
    f"The average random component size of the sex network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(sex_network))}"
)
print(
    f"The average random component size of the drug network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(drug_network))}"
)

print(
    f"The average clustering coefficient of the combined network is {nx.average_clustering(combined_network)}"
)
print(
    f"The average clustering coefficient of the sex network is {nx.average_clustering(sex_network)}"
)
print(
    f"The average clustering coefficient of the drug network is {nx.average_clustering(drug_network)}"
)

A1 = nx.adjacency_matrix(combined_network).todense()
A2 = nx.adjacency_matrix(sex_network).todense()
A3 = nx.adjacency_matrix(drug_network).todense()

overlap = len(np.where(A2 + A3 == 2)[0])

print(
    f"The average kappa statistic with respect to the combined network is {overlap/np.count_nonzero(A1)}"
)
print(
    f"The average kappa statistic with respect to the sex network is {overlap/np.count_nonzero(A2)}"
)
print(
    f"The average kappa statistic with respect to the drug network is {overlap/np.count_nonzero(A3)}"
)

with shelve.open(os.path.join(data_folder, dataset_folder, filename)) as data:
    data["sex-network"] = sex_network
    data["drug-network"] = drug_network
    data["combined-network"] = combined_network
