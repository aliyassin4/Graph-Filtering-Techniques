import pickle
import networkx as nx

def save_obj(obj, path, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path, name):
    with open(path  + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def convert_label_to_integers(G):
    g = nx.convert_node_labels_to_integers(G)
    return g, G.nodes()


def relabel_nodes(G, old_labels):
    return nx.relabel_nodes(G, dict(zip(G.nodes(), old_labels)))