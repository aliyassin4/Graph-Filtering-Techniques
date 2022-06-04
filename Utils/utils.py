import numpy as np
import pandas as pd
import networkx as nx
import community.community_louvain as community
from collections import Counter
from Utils.portrait_divergence import portrait_divergence, portrait_divergence_weighted


import math

def degree_distribution(G, weight_value):
    vk = dict(G.degree(weight=weight_value))
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

def weight_distribution(G, weight_value):
    vk = [w for u, v, w in G.edges(data=weight_value)]
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk


def shannon_entropy_for_weights(G, weight_value = None):
    k,Pk = weight_distribution(G, weight_value)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

def shannon_entropy(G, weight_value = None):
    k,Pk = degree_distribution(G, weight_value)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H


def extract_measures(G):

    #initialize the dictionary measures for the results
    graph_measures = dict()


    #--------------------------------------------
    #calculate the fraction of nodes preserved
    graph_measures['nodes_fraction'] = 1

    #--------------------------------------------
    #calculate the fraction of edges preserved
    graph_measures['edge_fraction'] = 1
    
    #--------------------------------------------
    #calculate the number of connected components
    graph_measures['nb_connected_components'] = nx.number_connected_components(G)
    
    #--------------------------------------------
    #calculate the diameter
    #graph_measures['diameter'] = nx.diameter(G)
        
    #--------------------------------------------
    #calculate the average weighted degree 
    average_weighted_degree = sum([G.degree(node, weight='weight') for node in G.nodes()])/len(G.nodes())
    graph_measures['average_weighted_degree']= round(average_weighted_degree, 2)

    #--------------------------------------------
    #calculate the average link weight 
    average_link_weight = sum([G.edges()[edge]['weight'] for edge in G.edges()])/len(G.edges())
    graph_measures['average_link_weight'] = round(average_link_weight, 2)

    #--------------------------------------------
    #calculate the average betwenness
    average_betweeness = sum(nx.edge_betweenness_centrality(G, weight='weight', normalized=False).values())/len(G.nodes())
    graph_measures['average_betweeness'] = round(average_betweeness, 2)


    #--------------------------------------------
    #calculate the density
    density = round((2*len(G.edges()))/(len(G.nodes())*(len(G.nodes())-1)), 3)
    graph_measures['density'] = round(density, 3)


    #--------------------------------------------
    #calculate the entropy
    entropy = round(shannon_entropy(G), 3)
    graph_measures['degree_entropy'] = entropy
    
    
    #--------------------------------------------
    #calculate the entropy
    entropy = round(shannon_entropy(G, weight_value='weight'), 3)
    graph_measures['weighted_degree_entropy'] = entropy

    #--------------------------------------------
    #calculate the entropy
    entropy = round(shannon_entropy_for_weights(G, weight_value='weight'), 3)
    graph_measures['weightes_entropy'] = entropy
    
    
    #--------------------------------------------
    #calculate the weighted modulartiy
    communities = community.best_partition(G, random_state=1)
    weighted_modularity = round(community.modularity(communities, G, weight='weight'), 3)
    graph_measures['weighted_modularity'] = weighted_modularity
    
    #--------------------------------------------
    #calculate the portrait divergence
    portrait_divergence_distance = portrait_divergence_weighted(G, G)
    graph_measures['portrait_divergence_distance'] = portrait_divergence_distance
        
    
    #--------------------------------------------
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    graph_measures['degree_assortativity'] = degree_assortativity
        
    #--------------------------------------------
    weighted_degree_assortativity = nx.degree_assortativity_coefficient(G, weight='weight')
    graph_measures['weighted_degree_assortativity'] = weighted_degree_assortativity
        
        
    #--------------------------------------------
    degrees = list(dict(G.degree()).values())
    degree_cv = np.var(degrees)/(np.average(degrees)**2)
    graph_measures['degree_cv'] = degree_cv
        
    #--------------------------------------------
    weighted_degrees = list(dict(G.degree(weight='weight')).values())
    weighted_degree_cv = np.var(weighted_degrees)/(np.average(weighted_degrees)**2)
    graph_measures['weighted_degree_cv'] = weighted_degree_cv
        
    #--------------------------------------------
    weights = list(nx.get_edge_attributes(G, 'weight').values())
    weight_cv = np.var(weights)/(np.average(weights)**2)
    graph_measures['weight_cv'] = weight_cv
        
        
    #--------------------------------------------
    unweighted_average_clustering = nx.average_clustering(G)
    graph_measures['unweighted_average_clustering'] = unweighted_average_clustering
        
    #--------------------------------------------
    weighted_average_clustering = nx.average_clustering(G, weight='weight')
    graph_measures['weighted_average_clustering'] = weighted_average_clustering
    
    #--------------------------------------------
    graph_measures['coverage'] = len(G)/len(G)
                                                            
    #--------------------------------------------
    graph_measures['jaccard_similarity'] = jaccard_similarity(get_edges(G), get_edges(G))
      
    #--------------------------------------------
    graph_measures['overlap_coefficient'] = overlap_coefficient(get_edges(G), get_edges(G))
    
    #--------------------------------------------
    graph_measures['average_shortest_path_length'] = average_shortest_path_length(G)
    
    
    #--------------------------------------------
    #calculate the degree distribution
#     degrees = [d for n,d in G.degree()]
#     degree_distribution = Counter(degrees) 
#     graph_measures['degree_distribution'] = degree_distribution
    
    

    return graph_measures
    
    
    
def extract_backbones_and_measures(original_graph, full_backbone, fractions, criteria):
    
    G = original_graph
    N = len(original_graph.nodes())
    E = len(original_graph.edges())
    
    #initialize the dictionary that will save all the extracted backbones from different sizes with the evaluation measures
    backbones_dict = dict()

    #initialize the dataframe measures for the network backbones
    measures = ['nodes_fraction', 'edge_fraction', 'average_weighted_degree', 'average_link_weight', 'average_betweeness', 'density', 'degree_entropy', 'weighted_degree_entropy', 'weightes_entropy', 'weighted_modularity', 'threshold', 'nb_connected_components', 'average_components_size', 'diameter', 'degree_assortativity', 'weighted_degree_assortativity', 'portrait_divergence_distance', 'degree_cv', 'weighted_degree_cv', 'weight_cv', 'unweighted_average_clustering', 'weighted_average_clustering', 'coverage', 'jaccard_similarity', 'overlap_coefficient', 'average_shortest_path_length']#, 'degree_distribution']
    backbone_measures = pd.DataFrame(columns=measures, index=fractions)


    #loop through the fractions and extract the backbone with this fraction of nodes
    for fraction in fractions:

        #calculate the number of nodes to preserve
        nodes_fraction = int(fraction*N)
        edge_fraction = int(fraction*E)
        
        #initialize the graph to save the backbone
        backbone = nx.Graph()

        #loop through the dataframe backbone and add rows until the we reach the target number of nodes
#         if criteria == 'Nodes':
#             for row in full_backbone.iterrows():
#                 source = row[1]['source']
#                 target = row[1]['target']
#                 weight = row[1]['weight']
#                 threshold = row[1]['threshold']

#                 if len(backbone.nodes()) >= nodes_fraction:
#                     backbone_measures['threshold'][fraction] = round(threshold, 3)
#                     break
#                 backbone.add_edge(source, target, weight=weight, threshold=threshold)
                
        if criteria == 'Nodes':
            for row in full_backbone.iterrows():
                node = row[1]['node']
                score = row[1]['threshold']


                if len(backbone.nodes()) >= nodes_fraction:
                    backbone_measures['threshold'][fraction] = round(threshold, 3)
                    break
                backbone.add_edge(source, target, weight=weight, threshold=threshold)
                
        if criteria == 'Edges':   
            backbone_list = full_backbone[0:edge_fraction]
            edge_attr = list(backbone_list.columns)
            edge_attr.remove('source')
            edge_attr.remove('target')
            backbone = nx.from_pandas_edgelist(backbone_list, edge_attr=edge_attr, create_using=nx.Graph())
            
        #take only the largest connected component
        largest_cc = max(nx.connected_components(backbone), key=len)
        backbone_lcc = backbone.subgraph(largest_cc).copy()
        
        #add the backbone that extraxts this fraction of nodes to the dictionary
        backbones_dict[fraction] = backbone

        #--------------------------------------------
        #calculate the fraction of nodes preserved
        nodes_fraction = len(backbone.nodes())/len(G.nodes())
        backbone_measures['nodes_fraction'][fraction] = round(nodes_fraction, 3)
        
        #--------------------------------------------
        #calculate the fraction of edges preserved
        edge_fraction = len(backbone.edges())/len(G.edges())
        backbone_measures['edge_fraction'][fraction] = round(edge_fraction, 3)
        
        #--------------------------------------------
        #calculate the number of connected components, the average size, and the standard deviation   
        component_sizes = []
        for component in nx.connected_components(backbone):
            comp = backbone.subgraph(component).copy()
            component_sizes.append(len(comp))
        
        backbone_measures['average_components_size'][fraction] = (np.average(component_sizes), np.var(component_sizes))
        backbone_measures['nb_connected_components'][fraction] = nx.number_connected_components(backbone)

        #--------------------------------------------
        #calculate the diameter
        #backbone_measures['diameter'][fraction] = nx.diameter(backbone_lcc)
                
        #--------------------------------------------
        #calculate the average weighted degree 
        average_weighted_degree = sum([backbone.degree(node, weight='weight') for node in backbone.nodes()])/len(backbone.nodes())
        backbone_measures['average_weighted_degree'][fraction] = round(average_weighted_degree, 2)

        #--------------------------------------------
        #calculate the average link weight 
        average_link_weight = sum([backbone.edges()[edge]['weight'] for edge in backbone.edges()])/len(backbone.edges())
        backbone_measures['average_link_weight'][fraction] = round(average_link_weight, 2)

        #--------------------------------------------
        #calculate the average betwenness
        average_betweeness = sum(nx.edge_betweenness_centrality(backbone_lcc, weight='weight', normalized=False).values())/len(backbone.nodes())
        backbone_measures['average_betweeness'][fraction] = round(average_betweeness, 2)


        #--------------------------------------------
        #calculate the density
        density = round((2*len(backbone.edges()))/(len(backbone.nodes())*(len(backbone.nodes())-1)), 3)
        backbone_measures['density'][fraction] = round(density, 3)


        #--------------------------------------------
        #calculate the entropy
        entropy = round(shannon_entropy(backbone), 3)
        backbone_measures['degree_entropy'][fraction] = entropy
                                    
        #--------------------------------------------
        #calculate the entropy
        entropy = round(shannon_entropy(backbone, weight_value='weight'), 3)
        backbone_measures['weighted_degree_entropy'][fraction] = entropy
        
        #--------------------------------------------
        #calculate the entropy
        entropy = round(shannon_entropy_for_weights(backbone, weight_value='weight'), 3)
        backbone_measures['weightes_entropy'][fraction] = entropy
        
    
        #--------------------------------------------
        #calculate the weighted modulartiy
        communities = community.best_partition(backbone_lcc, random_state=1)
        weighted_modularity = round(community.modularity(communities, backbone_lcc, weight='weight'), 3)
        backbone_measures['weighted_modularity'][fraction] = weighted_modularity
        
        
        #--------------------------------------------
        #calculate the portrait divergence
        portrait_divergence_distance = portrait_divergence_weighted(original_graph, backbone_lcc)
        backbone_measures['portrait_divergence_distance'][fraction] = portrait_divergence_distance
       
        #--------------------------------------------
        degree_assortativity = nx.degree_assortativity_coefficient(backbone)
        backbone_measures['degree_assortativity'][fraction] = degree_assortativity
        
        #--------------------------------------------
        weighted_degree_assortativity = nx.degree_assortativity_coefficient(backbone, weight='weight')
        backbone_measures['weighted_degree_assortativity'][fraction] = weighted_degree_assortativity
        
        #--------------------------------------------
        degrees = list(dict(backbone.degree()).values())
        degree_cv = np.var(degrees)/(np.average(degrees)**2)
        backbone_measures['degree_cv'][fraction] = degree_cv
        
        #--------------------------------------------
        weighted_degrees = list(dict(backbone.degree(weight='weight')).values())
        weighted_degree_cv = np.var(weighted_degrees)/(np.average(weighted_degrees)**2)
        backbone_measures['weighted_degree_cv'][fraction] = weighted_degree_cv
        
        #--------------------------------------------
        weights = list(nx.get_edge_attributes(backbone, 'weight').values())
        weight_cv = np.var(weights)/(np.average(weights)**2)
        backbone_measures['weight_cv'][fraction] = weight_cv
        
        
        #--------------------------------------------
        unweighted_average_clustering = nx.average_clustering(backbone)
        backbone_measures['unweighted_average_clustering'][fraction] = unweighted_average_clustering
        
        #--------------------------------------------
        weighted_average_clustering = nx.average_clustering(backbone, weight='weight')
        backbone_measures['weighted_average_clustering'][fraction] = weighted_average_clustering
        
        
        #--------------------------------------------
        backbone_measures['coverage'][fraction] = len(backbone)/len(G)
        
        
        #--------------------------------------------
        backbone_measures['jaccard_similarity'][fraction] = jaccard_similarity(get_edges(backbone), get_edges(G))
        
        #--------------------------------------------
        backbone_measures['overlap_coefficient'][fraction] = overlap_coefficient(get_edges(backbone), get_edges(G))
        
        
        #--------------------------------------------
        backbone_measures['average_shortest_path_length'][fraction] = average_shortest_path_length(backbone)
        
        
        #--------------------------------------------
        #calculate the degree distribution
#         degrees = [d for n,d in G.degree()]
#         degree_distribution = Counter(degrees) 
#         backbone_measures['degree_distribution'][fraction] = degree_distribution
    

    return backbones_dict, backbone_measures
    
    

 

def average_shortest_path_length(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    largest_cc = graph.subgraph(largest_cc).copy()
    return nx.average_shortest_path_length(largest_cc, weight='weight')

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

def overlap_coefficient(g, h):
    i = set(g).intersection(h)
    return round(len(i) / min(len(g), len(h)),3)

def get_edges(graph):
    graph = nx.to_pandas_edgelist(graph)
    graph['edge'] = graph.apply(lambda x: "%s-%s" % (min(x["source"], x["target"]), max(x["source"], x["target"])), axis = 1)
    return graph['edge']