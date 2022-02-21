import numpy as np
import pandas as pd
import networkx as nx
import community.community_louvain as community


import math

def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

def shannon_entropy(G):
    k,Pk = degree_distribution(G)
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
    graph_measures['diameter'] = nx.diameter(G)
        
    #--------------------------------------------
    #calculate the average weighted degree 
    average_weighted_degree = sum([G.degree(node, weight='weight') for node in G.nodes()])/len(G.nodes())
    graph_measures['average_weighted_degree']= round(average_weighted_degree, 2)

    #--------------------------------------------
    #calculate the average link weight 
    average_link_weight = sum([G.edges()[edge]['weight'] for edge in G.edges()])/len(G.nodes())
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
    graph_measures['entropy'] = entropy


    #--------------------------------------------
    #calculate the weighted modulartiy
    communities = community.best_partition(G, random_state=1)
    weighted_modularity = round(community.modularity(communities, G, weight='weight'), 3)
    graph_measures['weighted_modularity'] = weighted_modularity
    

    return graph_measures
    
    
    
def extract_backbones_and_measures(original_graph, full_backbone, fractions, criteria):
    
    G = original_graph
    N = len(original_graph.nodes())
    E = len(original_graph.edges())
    
    #initialize the dictionary that will save all the extracted backbones from different sizes with the evaluation measures
    backbones_dict = dict()

    #initialize the dataframe measures for the network backbones
    measures = ['nodes_fraction', 'edge_fraction', 'average_weighted_degree', 'average_link_weight', 'average_betweeness', 'density', 'entropy', 'weighted_modularity', 'threshold', 'nb_connected_components', 'diameter']
    backbone_measures = pd.DataFrame(columns=measures, index=fractions)


    #loop through the fractions and extract the backbone with this fraction of nodes
    for fraction in fractions:

        #calculate the number of nodes to preserve
        nodes_fraction = int(fraction*N)
        edge_fraction = int(fraction*E)
        
        #initialize the graph to save the backbone
        backbone = nx.Graph()

        #loop through the dataframe backbone and add rows until the we reach the target number of nodes
        if criteria == 'Nodes':
            for row in full_backbone.iterrows():
                source = row[1]['source']
                target = row[1]['target']
                weight = row[1]['weight']
                threshold = row[1]['threshold']

                if len(backbone.nodes()) >= nodes_fraction:
                    backbone_measures['threshold'][fraction] = round(threshold, 3)
                    break
                backbone.add_edge(source, target, weight=weight)
        if criteria == 'Edges':   
            backbone_list = full_backbone[0:edge_fraction]
            backbone = nx.from_pandas_edgelist(backbone_list, edge_attr='weight', create_using=nx.Graph())
            
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
        #calculate the number of connected components
        backbone_measures['nb_connected_components'][fraction] = nx.number_connected_components(backbone)

        #--------------------------------------------
        #calculate the diameter
        backbone_measures['diameter'][fraction] = nx.diameter(backbone_lcc)
                
        #--------------------------------------------
        #calculate the average weighted degree 
        average_weighted_degree = sum([backbone.degree(node, weight='weight') for node in backbone.nodes()])/len(backbone.nodes())
        backbone_measures['average_weighted_degree'][fraction] = round(average_weighted_degree, 2)

        #--------------------------------------------
        #calculate the average link weight 
        average_link_weight = sum([backbone.edges()[edge]['weight'] for edge in backbone.edges()])/len(backbone.nodes())
        backbone_measures['average_link_weight'][fraction] = round(average_link_weight, 2)

        #--------------------------------------------
        #calculate the average betwenness
        average_betweeness = sum(nx.edge_betweenness_centrality(backbone, weight='weight', normalized=False).values())/len(backbone.nodes())
        backbone_measures['average_betweeness'][fraction] = round(average_betweeness, 2)


        #--------------------------------------------
        #calculate the density
        density = round((2*len(backbone.edges()))/(len(backbone.nodes())*(len(backbone.nodes())-1)), 3)
        backbone_measures['density'][fraction] = round(density, 3)


        #--------------------------------------------
        #calculate the entropy
        entropy = round(shannon_entropy(backbone), 3)
        backbone_measures['entropy'][fraction] = entropy


        #--------------------------------------------
        #calculate the weighted modulartiy
        communities = community.best_partition(backbone, random_state=1)
        weighted_modularity = round(community.modularity(communities, backbone, weight='weight'), 3)
        backbone_measures['weighted_modularity'][fraction] = weighted_modularity



    return backbones_dict, backbone_measures
    
    

    
