import csv
import os
import shelve

import networkx as nx
import numpy as np

data_folder = "Data"
dataset_folder = "JOAPP"
filename = "JOAPP_data_full"

directory1 = "Data"
directory2 = "rsos160863_si_001"
directory = os.path.join(directory1, directory2)

uid_file = "twitter_foursquare_mapper.dat"
twitter_edge_file = "tedges.txt"
foursquare_edge_file = "fedges.txt"

# This assumes that there are 0 overlapping platform-specific IDs. This is true
# for these data but may not be true in general.
platform_ID_to_UID = dict()
uids = set()
with open(os.path.join(directory, uid_file)) as uid_data:
    reader = csv.reader(uid_data, delimiter=",")
    next(reader)

    for line in reader:
        uid = int(line[0])
        twitter_id = line[1]
        foursquare_id = line[2]
        platform_ID_to_UID[twitter_id] = uid
        platform_ID_to_UID[foursquare_id] = uid
        uids.add(uid)

# import twitter edges
twitter_edge_list = list()
with open(os.path.join(directory, twitter_edge_file)) as twitter_data:
    reader = csv.reader(twitter_data, delimiter=" ")
    for line in reader:
        # If the platform ID doesn't exist in the id-UID conversion dictionary, append a UID.
        try:
            uid1 = platform_ID_to_UID[line[0]]
        except:
            uid1 = max(platform_ID_to_UID.values()) + 1
            uids.add(uid1)
            platform_ID_to_UID[line[0]] = uid1

        try:
            uid2 = platform_ID_to_UID[line[1]]
        except:
            uid2 = max(platform_ID_to_UID.values()) + 1
            uids.add(uid2)
            platform_ID_to_UID[line[1]] = uid1

        twitter_edge_list.append((uid1, uid2))


# import foursquare edges
foursquare_edge_list = list()
with open(os.path.join(directory, foursquare_edge_file)) as foursquareData:
    reader = csv.reader(foursquareData, delimiter=" ")
    for line in reader:
        try:
            uid1 = platform_ID_to_UID[line[0]]
        except:
            uid1 = max(platform_ID_to_UID.values()) + 1
            uids.add(uid1)
            platform_ID_to_UID[line[0]] = uid1

        try:
            uid2 = platform_ID_to_UID[line[1]]
        except:
            uid2 = max(platform_ID_to_UID.values()) + 1
            uids.add(uid2)
            platform_ID_to_UID[line[1]] = uid1

        foursquare_edge_list.append((uid1, uid2))


twitter_network = nx.Graph()
twitter_network.add_edges_from(twitter_edge_list)
twitter_network.add_nodes_from(uids)

foursquare_network = nx.Graph()
foursquare_network.add_edges_from(foursquare_edge_list)
foursquare_network.add_nodes_from(uids)


combined_edge_list = twitter_edge_list + foursquare_edge_list
combined_network = nx.Graph()
combined_network.add_edges_from(combined_edge_list)
combined_network.add_nodes_from(uids)

print(f"The size of all the networks is {len(combined_network)}")

print(
    f"The mean degree of the combined network is {2 * len(combined_network.edges) / len(combined_network)}"
)
print(
    f"The mean degree of the foursquare network is {2 * len(foursquare_network.edges) / len(foursquare_network)}"
)
print(
    f"The mean degree of the twitter network is {2 * len(twitter_network.edges) / len(twitter_network)}"
)

print(
    f"The average random component size of the combined network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(combined_network))}"
)
print(
    f"The average random component size of the foursquare network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(foursquare_network))}"
)
print(
    f"The average random component size of the twitter network is {sum(len(cc)**2/len(combined_network) for cc in nx.connected_components(twitter_network))}"
)

print(
    f"The average clustering coefficient of the combined network is {nx.average_clustering(combined_network)}"
)
print(
    f"The average clustering coefficient of the foursquare network is {nx.average_clustering(foursquare_network)}"
)
print(
    f"The average clustering coefficient of the twitter network is {nx.average_clustering(twitter_network)}"
)

A1 = nx.adjacency_matrix(combined_network).todense()
A2 = nx.adjacency_matrix(foursquare_network).todense()
A3 = nx.adjacency_matrix(twitter_network).todense()

overlap = len(np.where(A2 + A3 == 2)[0])

print(
    f"The average kappa statistic with respect to the combined network is {overlap/np.count_nonzero(A1)}"
)
print(
    f"The average kappa statistic with respect to the foursquare network is {overlap/np.count_nonzero(A2)}"
)
print(
    f"The average kappa statistic with respect to the twitter network is {overlap/np.count_nonzero(A3)}"
)

with shelve.open(os.path.join(data_folder, dataset_folder, filename)) as data:
    data["foursquare-network"] = foursquare_network
    data["twitter-network"] = twitter_network
    data["combined-network"] = combined_network
