import numpy as np
import networkx as nx
from collections import Counter
import community.community_louvain as community
import matplotlib.pyplot as plt
import powerlaw as pl



def get_diameter(G):
    return nx.diameter(get_lcc(G))

def get_nodes_number(G):
    return len(G)

def get_edges_number(G):
    return len(G.edges())

def get_node_fraction(G, orginal_graph):
    return round(len(G)/len(orginal_graph), 3)

def get_edge_fraction(G, orginal_graph):
    return round(len(G.edges())/len(orginal_graph.edges()), 3)

def get_density(G):
    return round(nx.density(G), 4)

def get_degree_assortativity(G):
    return round(nx.degree_assortativity_coefficient(G), 3)

def get_weighted_degree_assortativity(G):
    return round(nx.degree_assortativity_coefficient(G, weight='weight'), 3)

def get_average_clustering(G):
    return round(nx.average_clustering(G), 3)

def get_weighted_average_clustering(G):
    return round(nx.average_clustering(G, weight='weight'), 3)

def get_weights(G):
    return list(nx.get_edge_attributes(G, 'weight').values())

def get_degrees(G):
    return list(dict(G.degree()).values())

def get_clustering_coefficient(G):
    return list(dict(nx.clustering(G)).values())

def get_weighted_clustering_coefficient(G):
    return list(dict(nx.clustering(G, weight='weight')).values())

def get_weighted_degrees(G):
    return list(dict(G.degree(weight='weight')).values())

def get_edge_betweenness(G):
    return list(nx.edge_betweenness_centrality(G, normalized=True).values())

def get_weighted_edge_betweenness(G):
    return list(nx.edge_betweenness_centrality(G, weight='weight', normalized=True).values())

def get_average(values):
    return sum(values)/len(values)

def get_average_edge_weight(G):
    return get_average(get_weights(G))

def get_average_degree(G):
    return get_average(get_degrees(G))

def get_average_weighted_degree(G):
    return get_average(get_weighted_degrees(G))

def get_average_edge_betweenness(G):
    return get_average(get_edge_betweenness(G))

def get_average_weighted_edge_betweenness(G):
    return get_average(get_weighted_edge_betweenness(G))

def get_average_shortest_path_length(G):
    return nx.average_shortest_path_length(get_lcc(G))

def get_weighted_average_shortest_path_length(G):
    return nx.average_shortest_path_length(get_lcc(G), weight='weight')

def get_weight_distribution(G):
    values = get_weights(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G.edges())))
    return dist, values

def get_degree_distribution(G):
    values = get_degrees(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

def get_weighted_degree_distribution(G):
    values = get_weighted_degrees(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values


def get_clustering_coefficient_distribution(G):
    values = get_clustering_coefficient(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

def get_weighted_clustering_coefficient_distribution(G):
    values = get_weighted_clustering_coefficient(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

def get_edge_betweenness_distribution(G):
    values = get_edge_betweenness(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

def get_weighted_edge_betweenness_distribution(G):
    values = get_weighted_edge_betweenness(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

def get_edge_jaccard_coefficient(G, H):
    return len(nx.intersection(G, H).edges())/len(nx.compose(G, H).edges())

def get_edge_overlap_coefficient(G, H):
    return len(nx.intersection(G, H).edges()) / min(len(G.edges()), len(H.edges()))

def get_relative_difference(R, ref):
    return (R-ref)/ref

def get_lcc(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def get_lcc_percentage(G):
    return (len(get_lcc(G))/len(G))*100
    

def get_number_connected_components(G):
    return nx.number_connected_components(G)

def get_connected_components_sizes(G):
    return [len(component) for component in nx.connected_components(G)]

def get_connected_components_sizes_wrt_edges(G):
    return [len(G.subgraph(c).edges()) for c in nx.connected_components(G)]
    
def get_weighted_modularity(G):
    lcc= get_lcc(G)
    communities = community.best_partition(lcc, random_state=1)
    return round(community.modularity(communities, lcc, weight='weight'), 3)

def get_graph_difference(G, H):
    dif = nx.Graph()
    for edge in G.edges():
        if not H.has_edge(edge[0], edge[1]):
            dif.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]['weight'])
    return dif
    

def plot_power_law_pdf(fit, color='r', linewidth='2'):
    fit.power_law.plot_pdf(color=color, linewidth=linewidth)
    
def plot_truncated_power_law_pdf(fit, color='r', linewidth='2'):
    fit.truncated_power_law.plot_pdf(color=color, linewidth=linewidth)

def plot_lognormal_pdf(fit, color='r', linewidth='2'):
    fit.lognormal.plot_pdf(color=color, linewidth=linewidth)

def plot_stretched_exponential_pdf(fit, color='r', linewidth='2'):
    fit.stretched_exponential.plot_pdf(color=color, linewidth=linewidth)

def plot_exponential_pdf(fit, color='r', linewidth='2'):
    fit.exponential.plot_pdf(color=color, linewidth=linewidth)
    

def fit_distribution(dist, title):
    plt.figure(figsize=(5,5))
    fit = pl.Fit(dist, discrete=True, fit_method='KS', xmin=1)
    ks_dict = dict()
    pl_params = dict()
    ks_dict['power_law'] = fit.power_law.KS()
    pl_params['power_law'] = {'alpha': fit.power_law.parameter1}

    ks_dict['truncated_power_law'] = fit.truncated_power_law.KS()
    pl_params['truncated_power_law'] = {'alpha':fit.truncated_power_law.parameter1, 'lambda' : fit.truncated_power_law.parameter2}

    ks_dict['lognormal'] = fit.lognormal.KS()
    pl_params['lognormal'] = {'mu' : fit.lognormal.parameter1, 'sigma':fit.lognormal.parameter2}

    ks_dict['stretched_exponential'] = fit.stretched_exponential.KS()
    pl_params['stretched_exponential'] = {'lambda' : fit.stretched_exponential.parameter1, 'beta' : fit.stretched_exponential.parameter2}

    ks_dict['exponential'] = fit.exponential.KS()
    pl_params['exponential'] = {'lambda' : fit.exponential.parameter1}


    best_fit = min(list(ks_dict.values()))
    key = [k for k, v in ks_dict.items() if v == best_fit][0]

    globals()[f'plot_{key}_pdf'](fit)#, color='r', linewidth=2)


    x1, y1 = pl.pdf(dist, linear_bins=True)
    ind1 = y1>0
    y1 = y1[ind1]
    x1 = x1[:-1]
    x1 = x1[ind1]
    plt.scatter(x1, y1, color='k', s=5, label="Data")
    plt.title(title + " Distribution")
    plt.ylim(ymin = 0.0001)
    plt.xlim(xmax = 1900)
    #plt.xlabel(r'$\bf{title}$', {'fontsize':12})
    plt.ylabel(r'$\bf{Frequency}$', {'fontsize':12})
    plt.legend((key, r'data'),
            shadow=False, loc=(0.5, 0.7), handlelength=1.5, fontsize=8)

    return ks_dict, {'best_fit':key, 'KS': best_fit, 'params': pl_params[key]}


def get_rbo(l1, l2, p = 0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score. 
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []
    
    sl,ll = sorted([(len(l1), l1),(len(l2),l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l 
    # (the longer of the two lists)
    ss = set([]) # contains elements from the smaller list till depth i
    ls = set([]) # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1
        
        # if two elements are same then 
        # we don't need to add to either of the set
        if x == y: 
            x_d[d] = x_d[d-1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else: 
            ls.add(x) 
            if y != None: ss.add(y)
            x_d[d] = x_d[d-1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)     
        #calculate average overlap
        sum1 += x_d[d]/d * pow(p, d)
        
    sum2 = 0.0
    for i in range(l-s):
        d = s+i+1
        sum2 += x_d[d]*(d-s)/(d*s)*pow(p,d)

    sum3 = ((x_d[l]-x_d[s])/l+x_d[s]/s)*pow(p,l)

    # Equation 32
    rbo_ext = (1-p)/p*(sum1+sum2)+sum3
    return rbo_ext