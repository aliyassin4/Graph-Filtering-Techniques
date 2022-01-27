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
   table.rename(columns = {column_of_interest: "nij"}, inplace = True)
   if drop_zeroes:
      table = table[table["nij"] > 0]
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


def high_salience_skeleton(table, undirected = False, return_self_loops = False):
   #sys.stderr.write("Calculating HSS score...\n")
   table = table.copy()
   table["distance"] = 1.0 / table["nij"]
   nodes = set(table["source"]) | set(table["target"])
   G = nx.from_pandas_edgelist(table, source = "source", target = "target", edge_attr = "distance", create_using = nx.DiGraph())
   cs = defaultdict(float)
   for s in nodes:
      pred = defaultdict(list)
      dist = {t: float("inf") for t in nodes}
      dist[s] = 0.0
      Q = defaultdict(list)
      for w in dist:
         Q[dist[w]].append(w)
      S = []
      while len(Q) > 0:
         v = Q[min(Q.keys())].pop(0)
         S.append(v)
         for _, w, l in G.edges(nbunch = [v,], data = True):
            new_distance = dist[v] + l["distance"]
            if dist[w] > new_distance:
               Q[dist[w]].remove(w)
               dist[w] = new_distance
               Q[dist[w]].append(w)
               pred[w] = []
            if dist[w] == new_distance:
               pred[w].append(v)
         while len(S) > 0:
            w = S.pop()
            for v in pred[w]:
               cs[(v, w)] += 1.0
         Q = defaultdict(list, {k: v for k, v in Q.items() if len(v) > 0})
   table["score"] = table.apply(lambda x: cs[(x["source"], x["target"])] / len(nodes), axis = 1)
   if not return_self_loops:
      table = table[table["source"] != table["target"]]
   if undirected:
      table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["source"], x["target"]), max(x["source"], x["target"])), axis = 1)
      table_maxscore = table.groupby(by = "edge")["score"].sum().reset_index()
      table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
      table = table.drop_duplicates(subset = ["edge"])
      table = table.drop("edge", 1)
      table = table.drop("score_min", 1)
      table["score"] = table["score"] #/ 2.0
   return table[["source", "target", "nij", "score"]]

