import sys, warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import binom



def noise_corrected(table, undirected = False, return_self_loops = False, calculate_p_value = False):
   #sys.stderr.write("Calculating NC score...\n")
   table = table.copy()
   source_sum = table.groupby(by = "source").sum()[["weight"]]
   table = table.merge(source_sum, left_on = "source", right_index = True, suffixes = ("", "_source_sum"))
   target_sum = table.groupby(by = "target").sum()[["weight"]]
   table = table.merge(target_sum, left_on = "target", right_index = True, suffixes = ("", "_target_sum"))
   table.rename(columns = {"weight_source_sum": "ni.", "weight_target_sum": "n.j"}, inplace = True)
   table["n.."] = table["weight"].sum()
   table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
   if calculate_p_value:
      table["score"] = binom.cdf(table["weight"], table["n.."], table["mean_prior_probability"])
      return table[["source", "target", "weight", "score"]]
   table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
   table["score"] = ((table["kappa"] * table["weight"]) - 1) / ((table["kappa"] * table["weight"]) + 1)
   table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
   table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
   table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
   table["alpha_post"] = table["alpha_prior"] + table["weight"]
   table["beta_post"] = table["n.."] - table["weight"] + table["beta_prior"]
   table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
   table["variance_weight"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
   table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
   table["variance_cij"] = table["variance_weight"] * (((2 * (table["kappa"] + (table["weight"] * table["d"]))) / (((table["kappa"] * table["weight"]) + 1) ** 2)) ** 2) 
   table["sdev_cij"] = table["variance_cij"] ** .5
   if not return_self_loops:
      table = table[table["source"] != table["target"]]
   if undirected:
      table = table[table["source"] <= table["target"]]
   return table#[["source", "target", "weight", "score", "sdev_cij"]]
