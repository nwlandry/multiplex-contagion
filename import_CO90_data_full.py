import networkx as nx
import os
import pandas as pd
import shelve
import matplotlib.pyplot as plt
import numpy as np

data_folder = "Data"
dataset_folder = "CO90"
filename = "CO90_data_full"

directory1 = "Data"
directory2 = "ICPSR_22140"
directory3node = "DS0001"
directory3edge = "DS0002"
edge_directory = os.path.join(directory1, directory2, directory3edge)
node_directory = os.path.join(directory1, directory2, directory3node)
node_file = "22140-0001-Data.tsv"
edge_file = "22140-0002-Data.tsv"

df_nodes = pd.read_csv(
    os.path.join(node_directory, node_file
), delimiter="\t", low_memory=False
)
df_edges = pd.read_csv(
    os.path.join(edge_directory
, edge_file), delimiter="\t", low_memory=False
)
# parse by study
node_data = df_nodes[df_nodes["STUDYNUM"] == 1]
edge_data = df_edges[df_edges["STUDYNUM"] == 1]

sex_edges = edge_data[edge_data["TIETYPE"] == 3]
drug_edges = edge_data[edge_data["TIETYPE"] == 2]
needle_edges = edge_data[edge_data["TIETYPE"] == 4]
drug_and_needle_edges = pd.concat([drug_edges, needle_edges])  # drug and needle together
combined_edges = edge_data[edge_data["TIETYPE"] > 1]

node_list = list(node_data["RID"])

combined_network = nx.from_pandas_edgelist(combined_edges, source="ID1", target="ID2")
combined_network.add_nodes_from(node_list)

node_list = combined_network.nodes()
sex_network = nx.from_pandas_edgelist(sex_edges, source="ID1", target="ID2")

sex_network.add_nodes_from(node_list)

drug_network = nx.from_pandas_edgelist(drug_and_needle_edges, source="ID1", target="ID2")
drug_network.add_nodes_from(node_list)

print("The size of all the networks is " + str(len(combined_network)))

print("The mean degree of the combined network is " + str(2 * len(combined_network.edges) / len(combined_network)))
print("The mean degree of the sex network is " + str(2 * len(sex_network.edges) / len(sex_network)))
print("The mean degree of the drug network is " + str(2 * len(drug_network.edges) / len(drug_network)))

with shelve.open(os.path.join(data_folder, dataset_folder, filename)) as data:
    data["sex-network"] = sex_network
    data["drug-network"] = drug_network
    data["combined-network"] = combined_network
