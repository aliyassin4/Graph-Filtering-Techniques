import networkx as nx

def lans(G):
    G = G.copy()
    for u, v, w in G.edges(data='weight'):
        u_degree = G.degree(u, weight='weight')
        puv = w/u_degree
        u_n = G[u]
        count = len([n for n in u_n if u_n[n]['weight']/u_degree <= puv])
        u_pval = 1-count/len(u_n)

        v_degree = G.degree(v, weight='weight')
        pvu = w/v_degree
        v_n = G[v]
        count = len([n for n in v_n if v_n[n]['weight']/v_degree <= pvu])
        v_pval = 1-count/len(v_n)

        G[u][v]['p-value'] = min(v_pval, u_pval)

    return G