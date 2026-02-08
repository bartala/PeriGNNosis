import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from pathlib import Path
import os
import matplotlib.ticker as ticker

# =========================
# USER SETTINGS
# =========================

OUT_PDF = Path("Figure2_lollipop_standard_grayscale.pdf")

# Only show types with >10 occurrences
MIN_FREQ = 10

# Figure size (Adjusted to fit labels on the left)
# Narrower width as requested
FIGSIZE = (7.5, 6)

# Font sizes
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

WRAP_WIDTH = 20

# =========================
# LOAD + PREP
# =========================
# Ensure paths are defined if not in global scope
if 'CSV_NODES' not in locals():
    PTH = "..."
    CSV_NODES = os.path.join(PTH, "nodes.csv")
    CSV_EDGES = os.path.join(PTH, "edges.csv")

nodes = pd.read_csv(CSV_NODES)
edges = pd.read_csv(CSV_EDGES)

nodes["freq"] = pd.to_numeric(nodes["freq"])
edges["freq"] = pd.to_numeric(edges["freq"])

nodes = nodes[nodes["freq"] > MIN_FREQ].copy()
edges = edges[edges["freq"] > MIN_FREQ].copy()

# Combine 'CAUSE' and 'CAUSES' in edges
cause_mask = edges["relType"].isin(["CAUSE", "CAUSES"])
if cause_mask.any():
    combined_freq = edges.loc[cause_mask, "freq"].sum()
    # Remove original entries
    edges = edges[~cause_mask].copy()
    # Add new combined entry
    combined_entry = pd.DataFrame([{"relType": "CAUSE / CAUSES", "freq": combined_freq}])
    edges = pd.concat([edges, combined_entry], ignore_index=True)

# Log transform
nodes["logfreq"] = np.log10(nodes["freq"])
edges["logfreq"] = np.log10(edges["freq"])

# Clean labels
nodes["label"] = nodes["nodeType"].astype(str).str.replace("_", " ", regex=False)
edges["label"] = edges["relType"].astype(str).str.replace("_", " ", regex=False)

def wrap_labels(series, width=WRAP_WIDTH):
    return ["\n".join(textwrap.wrap(str(s), width=width, break_long_words=False)) for s in series]

nodes["label"] = wrap_labels(nodes["label"])
edges["label"] = wrap_labels(edges["label"])

# Strip leading/trailing whitespace from labels
nodes["label"] = nodes["label"].str.strip()

# Remove "Entity" label from nodes
nodes = nodes[nodes["label"] != "Entity"].copy()

# --- VERIFICATION STEP ---
print("Unique Node Labels after filtering 'Entity':")
print(nodes['label'].unique())
# --- END VERIFICATION STEP ---

# Sort for plotting (bottom to top)
nodes = nodes.sort_values("freq", ascending=True)
edges = edges.sort_values("freq", ascending=True)

# Shared X limit (or per panel)
xmax = max(nodes["logfreq"].max(), edges["logfreq"].max()) * 1.05

# =========================
# PLOTTING
# =========================
def lollipop_standard(ax, ylabels, xvals, xmax_val):
    y = np.arange(len(ylabels))
    # Horizontal lines (start from 0.5)
    ax.hlines(y=y, xmin=0.5, xmax=xvals, color='#1f77b4', linewidth=1.5)
    # Dots
    ax.plot(xvals, y, "o", color='#1f77b4', markerfacecolor='#1f77b4', markeredgewidth=1.5, markersize=6)

    # Only set y-ticks; labels will be handled outside this function
    ax.set_yticks(y)
    ax.tick_params(axis="y", length=0) # Hide tick marks

    # Styling (start x-axis at 0.5)
    ax.set_xlim(0.5, xmax_val)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)) # Set ticks every 0.5
    ax.grid(axis="x", linestyle=":", alpha=0.5, color='gray')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Create figure with white background (fixes dark mode visibility issues)
# Removed constrained_layout=True to use tight_layout with padding
fig = plt.figure(figsize=FIGSIZE, facecolor='white')

# Adjusted width ratios to make Figure (a) narrower
gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 1])

# (a) Nodes
ax1 = fig.add_subplot(gs[0, 0])
# Increased limit to 2.6 to avoid cutting off dots near 2.5 (e.g. Woman)
lollipop_standard(ax1, nodes["label"], nodes["logfreq"], 3.0) # Extended xmax to fit labels

# Place labels on top of the lollipop bars for Figure (a)
for i, (label, x_val) in enumerate(zip(nodes["label"], nodes["logfreq"])):
    ax1.text(x_val + 0.2, i, label, va='center', ha='left', fontsize=10)
ax1.set_xlabel("Log10(Frequency)")
#ax1.set_title("Node Types", loc="left", fontweight="bold")
ax1.text(-0.05, 1.02, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")
ax1.set_yticklabels([]) # Ensure y-labels are hidden for ax1

# (b) Edges
ax2 = fig.add_subplot(gs[0, 1])

# Filter out the least frequent relation type for Figure (b)
edges_filtered = edges.iloc[7:].copy() # Exclude the first TWO rows (least frequent) for more space

lollipop_standard(ax2, edges_filtered["label"], edges_filtered["logfreq"], xmax)
ax2.set_xlabel("Log10(Frequency)")
#ax2.set_title("Relationship Types", loc="left", fontweight="bold")
ax2.text(-0.05, 1.02, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

# Place labels on top of the lollipop bars for Figure (b)
# The largest lollipop is the last one in the sorted edges_filtered DataFrame
for i, (label, x_val) in enumerate(zip(edges_filtered["label"], edges_filtered["logfreq"])):
    if i == len(edges_filtered) - 1: # Check if this is the last item (largest freq)
        # Place label on the bar, inside, aligned right
        ax2.text(x_val - 2.0, 33.0, label, va='center', ha='right', fontsize=10, color='black')
    else:
        # Default placement for others
        ax2.text(x_val + 0.1, i, label, va='center', ha='left', fontsize=10)
ax2.set_yticklabels([]) # Ensure y-labels are hidden for ax2

# Use tight_layout with moderate width padding to balance spacing and overlap
plt.tight_layout(w_pad=2.0)
plt.show()

# Save
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor='white')
print(f"Saved: {OUT_PDF.resolve()}")
