import networkx as nx
import os
import csv
import shelve

data_folder = "Data"
dataset_folder = "JOAPP"
filename = "JOAPP_data_gc"

directory1 = "Data"
directory2 = "rsos160863_si_001"
directory = os.path.join(directory1, directory2)

uid_file = "twitter_foursquare_mapper.dat"
twitter_edge_file = "tedges.txt"
foursquare_edge_file = "fedges.txt"

# This assumes that there are 0 overlapping platform-specific IDs. This is true
# for these data but may not be true in general.
twitter_to_UID = dict()
foursquare_to_UID = dict()

with open(os.path.join(directory, uid_file)) as uid_data:
    reader = csv.reader(uid_data, delimiter=",")
    next(reader)

    for line in reader:
        uid = int(line[0])
        twitter_id = line[1]
        foursquare_id = line[2]
        twitter_to_UID[twitter_id] = uid
        foursquare_to_UID[foursquare_id] = uid

# import twitter edges
twitter_edge_list = list()
with open(os.path.join(directory, twitter_edge_file)) as twitter_data:
    reader = csv.reader(twitter_data, delimiter=" ")
    for line in reader:
        # If the platform ID doesn't exist in the id-UID conversion dictionary, append a UID.
        try:
            uid1 = twitter_to_UID[line[0]]
            uid2 = twitter_to_UID[line[1]]
            twitter_edge_list.append((uid1, uid2))
        except:
            pass

# import foursquare edges
foursquare_edge_list = list()
with open(os.path.join(directory, foursquare_edge_file)) as foursquareData:
    reader = csv.reader(foursquareData, delimiter=" ")
    for line in reader:
        try:
            uid1 = foursquare_to_UID[line[0]]
            uid2 = foursquare_to_UID[line[1]]
            foursquare_edge_list.append((uid1, uid2))
        except:
            pass
        foursquare_edge_list.append((uid1, uid2))

combined_edge_list = twitter_edge_list + foursquare_edge_list
combined_network = nx.Graph()
combined_network.add_edges_from(combined_edge_list)
connected_nodes = max(nx.connected_components(combined_network), key=len)
combined_network = combined_network.subgraph(connected_nodes).copy()

node_list = connected_nodes

twitter_network = nx.Graph()
twitter_network.add_edges_from(twitter_edge_list)
twitter_network = twitter_network.subgraph(node_list).copy()
twitter_network.add_nodes_from(node_list)

foursquare_network = nx.Graph()
foursquare_network.add_edges_from(foursquare_edge_list)
foursquare_network = foursquare_network.subgraph(node_list).copy()
foursquare_network.add_nodes_from(node_list)


print("The size of all the networks is " + str(len(combined_network)))

print("The mean degree of the combined network is " + str(2 * len(combined_network.edges) / len(combined_network)))
print("The mean degree of the foursquare network is " + str(2 * len(foursquare_network.edges) / len(foursquare_network)))
print("The mean degree of the twitter network is " + str(2 * len(twitter_network.edges) / len(twitter_network)))


with shelve.open(os.path.join(data_folder, dataset_folder, filename)) as data:
    data["foursquare-network"] = foursquare_network
    data["twitter-network"] = twitter_network
    data["combined-network"] = combined_network
