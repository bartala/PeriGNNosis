import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

# Load edge lists (exported with elementId)
cb_edges = pd.read_csv("/.../ptsd.csv")
no_edges = pd.read_csv("/.../no_cbptsd.csv")


import networkx as nx
import numpy as np

def normalized_kcore(G):
    # Work on undirected copy (k-core is undirected)
    H = G.to_undirected().copy()

    # Remove self-loops
    H.remove_edges_from(nx.selfloop_edges(H))

    # Compute k-core numbers
    core = nx.core_number(H)
    core_vals = np.array(list(core.values()))

    # Normalize
    return core_vals / core_vals.max()


def centrality_summary(G):
    N = G.number_of_nodes()

    # Normalized degree
    deg = np.array([d / (N - 1) for _, d in G.degree()])

    # Normalized betweenness
    bet = np.array(list(
        nx.betweenness_centrality(G, normalized=True).values()
    ))

    # Normalized closeness
    clo = np.array(list(
        nx.closeness_centrality(G).values()
    ))

    # PageRank (already normalized)
    pr = np.array(list(
        nx.pagerank(G).values()
    ))

    # k-core (normalized, self-loops removed)
    core_norm = normalized_kcore(G)

    return {
        "degree": deg,
        "betweenness": bet,
        "closeness": clo,
        "pagerank": pr,
        "kcore": core_norm
    }

cb_cent = centrality_summary(G_cb)
no_cent = centrality_summary(G_no)


for key in cb_cent.keys():
    cb_vals = cb_cent[key]
    no_vals = no_cent[key]

    ks_D, ks_p = ks_2samp(cb_vals, no_vals)

    print(f"\n{key.upper()}")
    print(f"CB-PTSD mean: {cb_vals.mean():.6g}")
    print(f"No-CB-PTSD mean: {no_vals.mean():.6g}")
    print(f"KS D: {ks_D:.4f}, p-value: {ks_p:.3e}")
