import community.community_louvain as community

import collections
import heapq
import operator
import scipy.stats as stats

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags


def orderCommunities(c):
    i = 0
    keys_partition = list()
    for j in c:
        keys_partition.append(i)
        i = i + 1

    partition = dict()
    for i in keys_partition:
        partition[i] = []


    i = 0
    for j in c: 
        for k in c[j]: 
            partition[i].append(k) 
        i = i + 1

    return partition


def communityInfo(c, partition):
    print('Number of partitions: ', len(partition))
    l = list()
    for i in c: 
        for j in c[i]: 
            l.append(j)  
    print('Number of nodes in the communities detected: ', len(l))

    s = set(l)  
    print('Number of repetitions: ', len(l) - len(s))
    print()
    print()




def getSparseA(g):
#     edges = g.edges()
#     sources, targets = list(zip(*edges))
#     if nx.is_weighted(g):
#         weights = np.array([G[u][v]['weight'] for u,v in G.edges()], dtype=float)
#     else:
#         weights = np.ones(len(sources))
#     self_loop_inds = (np.array(sources) == np.array(targets))
#     weights[self_loop_inds] = weights[self_loop_inds] / 2
#     weights = list(weights)
#     A = csr_matrix((weights + weights, (sources + targets, targets + sources)), shape=(len(g), len(g)))
    return nx.to_scipy_sparse_matrix(g)


def getGroupIndicator(g, membership, rows=None):
    if not rows:
        rows = list(range(g.vcount()))
    cols = membership
    vals = np.ones(len(cols))
    group_indicator_mat = csr_matrix((vals, (rows, cols)),
                                     shape=(len(g), max(membership) + 1))
    return group_indicator_mat


def getDegMat(node_deg_by_group, rows, cols):
    degrees = node_deg_by_group.sum(1)
    degrees = np.array(degrees).flatten()
    deg_mat = csr_matrix((degrees, (rows, cols)),
                         shape=node_deg_by_group.shape)
    degrees = degrees[:, np.newaxis]
    return degrees, deg_mat


def newMods(g, part):
    #if g.is_weighted():
    #    weight_key = 'weight'
    #else:
    weight_key = None
    index = list(range(len(g)))
    membership = part.membership_list # Steph: "part" is an instance of a class that has a "membership attribute"

    m = sum([g.degree(node, weight=weight_key) for node in g.nodes()])/2

    A = getSparseA(g)
    self_loops = A.diagonal().sum()
    group_indicator_mat = getGroupIndicator(g, membership, rows=index)
    node_deg_by_group = A * group_indicator_mat

    internal_edges = (node_deg_by_group[index, membership].sum() + self_loops) / 2

    degrees, deg_mat = getDegMat(node_deg_by_group, index, membership)
    node_deg_by_group += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = node_deg_by_group[index, membership].transpose() - degrees

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    # expanding out (group_degs - node_deg_by_group)^2 is slightly faster:
    expected_impact = np.power(group_degs, 2).sum() - 2 * (node_deg_by_group * group_degs.transpose()) +\
        node_deg_by_group.multiply(node_deg_by_group).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)
    q1s = q1_links - q1_degrees
    q1s = np.array(q1s).flatten()
    return q1s


def modularity_vitality(g, modularity, part):
    q0 = modularity
    q1s = newMods(g, part)
    vitalities = (q0 - q1s).tolist()
    return vitalities


def mappingAndRelabeling(g):
    # Mapping
    g_nx=g.copy()
    l_nodes = g_nx.nodes()
    taille=len(l_nodes)
    dict_graph = dict ()  # nodes in the key and themselves
    for i in l_nodes:
        dict_graph[i] = [i]
    index = 0
    for i in dict_graph:
        for j in dict_graph[i]:
            dict_graph[i] = index
            index = index + 1
            
    # Relabling: Construct a new graph with those mappings now
    mapping = dict_graph
    g_relabled = nx.relabel_nodes(g, mapping, copy=True)
    
    return g_relabled

def flip_nodes_and_communities(dict_nodes_communities):
    # Step 1: initialize communities as keys
    new_dict = {}
    for k, v in dict_nodes_communities.items():
        new_dict[v]=[]
    
    # Step 2: Fill in nodes
    for kk,vv in new_dict.items():
        for k,v in dict_nodes_communities.items():
            if dict_nodes_communities[k] == kk: # If the community number (value) in `best` is the same as new_dict key (key), append the node (key) in `best`
            #print(k,v)
                new_dict[kk].append(k)
    
    return new_dict

class communityInformation:
    def __init__(self, modularity_value, communities):
        self.modularity = modularity_value
        self.membership = communities
        self.membership_list = list()
        for i in self.membership:
            self.membership_list.append(self.membership[i])
            
# Returns a list of the top_k nodes and their centralities, and heap (list) of top k nodes --> heap will be used for removal
def get_top_k_best_nodes(dict_centrality, k):
    
    # The sorted() function returns a sorted list of the specified iterable object
    top_k = sorted(dict_centrality.items(), key=operator.itemgetter(1), reverse=True)[:k]
    first_nodes = heapq.nlargest(k, dict_centrality, key=dict_centrality.get)
    
    return top_k, first_nodes

def modularity_backbone(g, k):
    
    g1 = g.copy()
    
    communities = community.best_partition(g1, random_state=123)

    modularity_value = community.modularity(communities, g1)

    infomap_communities = flip_nodes_and_communities(communities)
    infomap_communities_organized = orderCommunities(infomap_communities)

    communities_instance = communityInformation(modularity_value, communities)

    list_modv = modularity_vitality(g1, communities_instance.modularity, communities_instance)

    dict_original_modv_absolute = {}
    for i, node in enumerate(g1.nodes()):
        dict_original_modv_absolute[node] = abs(list_modv[i])
    
    
        
    top_y, top_x = get_top_k_best_nodes(dict_original_modv_absolute, len(g1))
    
    nodes_removed = []
    modularity_at_each_node_removal = []
    modularity_at_each_node_removal.append(community.modularity(communities, g)) # Intiial modularity
    communities_flipped_prunned = {}
   
    
    for i in range(k):
        last_element = top_x.pop() # Get the node to be removed
        

        # Working on Q1        
        g1.remove_node(last_element) # Remove it from the network
        communities.pop(last_element) # Remove it from the communities
        modularity_value_after_removal = community.modularity(communities, g1)
        modularity_at_each_node_removal.append(modularity_value_after_removal)
        
                
        # Working on Q3       
        nodes_removed.append(last_element) 
        
        
    # Working on Q2
    for k,v in infomap_communities_organized.items():
        communities_flipped_prunned[k] = []
        for node1 in v:
            if node1 in nodes_removed:
                continue
            else:
                communities_flipped_prunned[k].append(node1)
        
    return g1, modularity_at_each_node_removal, communities_flipped_prunned, nodes_removed, top_x