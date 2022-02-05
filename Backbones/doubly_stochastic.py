import sys, warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import binom

warnings.filterwarnings('ignore')

def read(filename, column_of_interest, triangular_input = False, consider_self_loops = True, undirected = False, drop_zeroes = True, sep = "\t"):
   """Reads a field separated input file into the internal backboning format (a Pandas Dataframe).
   The input file should have three or more columns (default separator: tab).
   The input file must have a one line header with the column names.
   There must be two columns called 'source' and 'target', indicating the origin and destination of the interaction.
   All other columns must contain integer or floats, indicating the edge weight.
   In case of undirected network, the edges have to be present in both directions with the same weights, or set triangular_input to True.

   Args:
   filename (str): The path to the file containing the edges.
   column_of_interest (str): The column name identifying the weight that will be used for the backboning.

   KWArgs:
   triangular_input (bool): Is the network undirected and are the edges present only in one direction? default: False
   consider_self_loops (bool): Do you want to consider self loops when calculating the backbone? default: True
   undirected (bool): Is the network undirected? default: False
   drop_zeroes (bool): Do you want to keep zero weighted connections in the network? Important: it affects methods based on degree, like disparity_filter. default: False
   sep (char): The field separator of the inout file. default: tab

   Returns:
   The parsed network data, the number of nodes in the network and the number of edges.
   """
   table = pd.read_csv(filename, sep = sep)
   table = table[["source", "target", column_of_interest]]
   table.rename(columns = {column_of_interest: "weight"}, inplace = True)
   if drop_zeroes:
      table = table[table["weight"] > 0]
   if not consider_self_loops:
      table = table[table["source"] != table["target"]]
   if triangular_input:
      table2 = table.copy()
      table2["new_source"] = table["target"]
      table2["new_target"] = table["source"]
      table2.drop("source", 1, inplace = True)
      table2.drop("target", 1, inplace = True)
      table2 = table2.rename(columns = {"new_source": "source", "new_target": "target"})
      table = pd.concat([table, table2], axis = 0)
      table = table.drop_duplicates(subset = ["source", "target"])
   original_nodes = len(set(table["source"]) | set(table["target"]))
   original_edges = table.shape[0]
   if undirected:
      return table, original_nodes, original_edges / 2
   else:
      return table, original_nodes, original_edge


def doubly_stochastic(table, undirected = False, return_self_loops = False):
   table = table.copy()
   table2 = table.copy()
   original_nodes = len(set(table["source"]) | set(table["target"]))
   table = pd.pivot_table(table, values = "weight", index = "source", columns = "target", aggfunc = "sum", fill_value = 0) + .0001
   row_sums = table.sum(axis = 1)
   attempts = 0
   while np.std(row_sums) > 1e-12:
      table = table.div(row_sums, axis = 0)
      col_sums = table.sum(axis = 0)
      table = table.div(col_sums, axis = 1)
      row_sums = table.sum(axis = 1)
      attempts += 1
      if attempts > 1000:
         warnings.warn("Matrix could not be reduced to doubly stochastic. See Sec. 3 of Sinkhorn 1964", RuntimeWarning)
         return pd.DataFrame()
   table = pd.melt(table.reset_index(), id_vars = "source")
   table = table[table["source"] < table["target"]]
   table = table[table["value"] > 0].sort_values(by = "value", ascending = False)
   table = table.merge(table2[["source", "target", "weight"]], on = ["source", "target"])
   i = 0
   if undirected:
      G = nx.Graph()
      while nx.number_connected_components(G) != 1 or nx.number_of_nodes(G) < original_nodes or nx.is_connected(G) == False:
         edge = table.iloc[i]
         G.add_edge(edge["source"], edge["target"], weight = edge["value"])
         table.loc[table.loc[(table['source'] == edge["source"]) & (table['target'] == edge["target"])].index[0],'ds_backbone'] = True 
         i += 1
   else:
      G = nx.DiGraph()
      while nx.number_weakly_connected_components(G) != 1 or nx.number_of_nodes(G) < original_nodes or nx.is_connected(G) == False:
         edge = table.iloc[i]
         G.add_edge(edge["source"], edge["target"], weight = edge["value"])
         table.loc[table.loc[(table['source'] == edge["source"]) & (table['target'] == edge["target"])].index[0],'ds_backbone'] = True 
         i += 1
   #table = pd.melt(nx.to_pandas_adjacency(G).reset_index(), id_vars = "index")
   table = table[table["value"] >= 0]
   table.rename(columns = {"index": "source", "variable": "target", "value": "score"}, inplace = True)
   table = table.fillna(False) 
   if not return_self_loops:
      table = table[table["source"] != table["target"]]
   if undirected:
      table = table[table["source"] <= table["target"]]
   #nx.draw(G)     
   return table[["source", "target", "weight", "score", "ds_backbone"]]


#another method
#from Backbones.doubly_stochastic import read#, doubly_stochastic as ds

# original_table, nnodes, nnedges = read("../Datasets/lesmis.csv", "weight", sep=',', consider_self_loops=False, triangular_input = True, undirected=True) 

# original_nodes = len(set(original_table["source"]) | set(original_table["target"]))
# table = pd.pivot_table(original_table, values = "weight", index = "source", columns = "target", aggfunc = "sum", fill_value = 0) + 0.0001

# from sinkhorn_knopp import sinkhorn_knopp as skp
# sk = skp.SinkhornKnopp()
# table_ds = sk.fit(table)
# table = pd.DataFrame(data=table_ds, index=table.index, columns=table.columns)
# table = table.stack().reset_index()
# table.rename(columns = {'source':'source','target':'target',0:'score'}, inplace=True)
# table = table[table['source'] != table['target']]
# table = table[table['source'] < table['target']]

# table = table.merge(original_table, on = ["source", "target"])


# table = table.sort_values(by = 'score', ascending = False)

# i=0
# G = nx.Graph()
# while True:
#         edge = table.iloc[i]
#         G.add_edge(edge["source"], edge["target"], weight = edge["score"])
#         #table.loc[table.loc[(table['source'] == edge["source"]) & (table['target'] == edge["target"])].index[0],'ds_backbone'] = True
#         i = i+1
#         if nx.is_connected(G) and nx.number_of_nodes(G) == original_nodes:
#             break
# nx.to_pandas_edgelist(G)