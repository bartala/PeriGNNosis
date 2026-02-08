# read graph from disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import gc
import os
import warnings
from tqdm.notebook import tqdm

data = torch.load(os.path.join(PTH, 'graphRAG/data/perignnosis_graph.pt'), weights_only=False)

# Check feature dimensions for 'Woman' nodes
# Should be [Num_Women, 782] (768 embedding + 13 PDI + 1 Complication)
if 'Woman' in data.node_types:
    print(f"Woman Feature Matrix shape: {data['Woman'].x.shape}")

# Check features for a random entity type (e.g., 'Concept' or 'Emotion')
# Should be [Num_Nodes, 768] (Embedding only)
sample_entity = [t for t in data.node_types if t != 'Woman'][0]
print(f"{sample_entity} Feature Matrix shape: {data[sample_entity].x.shape}")

# Extract Statistics
total_nodes = data.num_nodes
total_edges = data.num_edges
node_types = data.node_types
edge_types = data.edge_types

# Print Results
print(f"--- Graph Statistics ---")
print(f"Total Nodes: {total_nodes}")
print(f"Total Edges: {total_edges}")
print(f"Number of Node Types: {len(node_types)}")
print(f"Number of Edge Types: {len(edge_types)}")

print(f"\n--- Node Type Breakdown ---")
for nt in node_types:
    print(f"  - {nt}: {data[nt].num_nodes} nodes")

print(f"\n--- Edge Type Breakdown (Top 10) ---")
# Showing just top 10 to avoid flooding console if you have 700+ types
for i, et in enumerate(edge_types):
    if i >= 10:
        print(f"  ... and {len(edge_types) - 10} more.")
        break
    src, rel, dst = et
    print(f"  - {src} -[{rel}]-> {dst}: {data[et].num_edges} edges")


# ----- trian the GNN -----

CONFIG = {
    'SEED': 42,
    'INNER_EPOCHS': 5,
    'OUTER_EPOCHS': 15,
    'HIDDEN_CHANNELS': 64,
    'LR': 0.001,
    'WEIGHT_DECAY': 1e-4,
    'ALL_PDI': list(range(13)),
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG['SEED'])


def plot_combined_roc(results_dict, filename="roc_comparison.pdf"):
    plt.figure(figsize=(10, 8))
    styles = {
        'GNN': {'color': 'blue', 'label': 'PeriGNNosis (Graph)'},
        'SVM-Clinical': {'color': 'red', 'label': 'Clinical Only (SVM)'},
        'Text-MLP': {'color': 'green', 'label': 'Text Embeddings (MLP)'}
    }
    mean_fpr = np.linspace(0, 1, 100)
    for name, data_list in results_dict.items():
        tprs = []
        aucs = []
        for fold_res in data_list:
            if fold_res['y_true'] is None or len(fold_res['y_true']) == 0: continue
            fpr, tpr, _ = roc_curve(fold_res['y_true'], fold_res['y_prob'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(fold_res['AUC'])
        if not tprs: continue
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        style = styles.get(name, {'color': 'black', 'label': name})
        label_text = fr"{style['label']} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
        plt.plot(mean_fpr, mean_tpr, color=style['color'], label=label_text, lw=2, alpha=0.9)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=style['color'], alpha=0.1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Predictive Performance Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()


def to_native(x):
    if isinstance(x, dict): return {to_native(k): to_native(v) for k, v in x.items()}
    if isinstance(x, list): return [to_native(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.float32, np.float64)): return float(x)
    try: return to_native(x.cpu().numpy())
    except: return x

def validate_and_fix_edge_types(data):
    for edge_type in data.edge_types:
        if edge_type not in data.edge_index_dict:
             data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
             idx = data[edge_type].edge_index
             if not torch.is_tensor(idx): idx = torch.tensor(idx)
             idx = idx.long().contiguous()
             if idx.dim() == 1: idx = idx.view(2, -1)
             data[edge_type].edge_index = idx
    return data

def sanitize_and_fix_graph(data):
    temp_data = HeteroData()
    woman_key = 'Woman'
    node_map = {}
    for nt in data.node_types:
        if nt == '__Entity__': new_nt = 'Entity'
        else: new_nt = nt.replace(' ', '_').replace('-', '_')
        node_map[nt] = new_nt
        if nt == 'Woman': woman_key = new_nt
        temp_data[new_nt].num_nodes = data[nt].num_nodes
        if 'x' in data[nt]: temp_data[new_nt].x = data[nt].x
        if 'y' in data[nt]: temp_data[new_nt].y = data[nt].y
    edge_list = []
    for src, rel, dst in data.edge_types:
        new_src, new_dst = node_map[src], node_map[dst]
        edge_list.append((new_src, rel.replace(' ', '_'), new_dst, data[src, rel, dst]))
    active_nodes = set()
    for src, _, dst, _ in edge_list:
        active_nodes.add(src); active_nodes.add(dst)
    active_nodes.add(woman_key)
    final_data = HeteroData()
    for nt in active_nodes:
        final_data[nt].num_nodes = temp_data[nt].num_nodes
        if 'x' in temp_data[nt]: final_data[nt].x = temp_data[nt].x
        if 'y' in temp_data[nt]: final_data[nt].y = temp_data[nt].y
    for src, rel, dst, store in edge_list:
        if src in active_nodes and dst in active_nodes:
            for k, v in store.items(): final_data[src, rel, dst][k] = v
    max_idx = {nt: 0 for nt in active_nodes}
    for src, rel, dst in final_data.edge_types:
        idx = final_data[src, rel, dst].edge_index
        if idx.numel() > 0:
            max_idx[src] = max(max_idx[src], idx[0].max().item())
            max_idx[dst] = max(max_idx[dst], idx[1].max().item())
    for nt in active_nodes:
        if final_data[nt].num_nodes is None or final_data[nt].num_nodes <= max_idx[nt]:
            final_data[nt].num_nodes = max_idx[nt] + 1
    feat_dim = 768
    for nt in active_nodes:
        if 'x' not in final_data[nt] or final_data[nt].x is None:
            final_data[nt].x = torch.zeros((final_data[nt].num_nodes, feat_dim))
        elif final_data[nt].x.shape[0] < final_data[nt].num_nodes:
             old_x = final_data[nt].x
             pad = torch.zeros((final_data[nt].num_nodes - old_x.shape[0], old_x.shape[1]))
             final_data[nt].x = torch.cat([old_x, pad], dim=0)
    final_data = T.ToUndirected()(final_data)
    final_data = validate_and_fix_edge_types(final_data)
    return final_data, woman_key

def get_train_subgraph(data, train_idx, woman_key):
    node_mask = {nt: torch.ones(data[nt].num_nodes, dtype=torch.bool) for nt in data.node_types}
    woman_mask = torch.zeros(data[woman_key].num_nodes, dtype=torch.bool)
    woman_mask[train_idx] = True
    node_mask[woman_key] = woman_mask
    subg = data.subgraph(node_mask)
    subg = validate_and_fix_edge_types(subg)
    return subg

# For inner loop leakage prevention: Remove edges from test women
def remove_women_edges(data, blocked_idx, woman_key):
    data = data.clone()
    blocked_idx = np.asarray(blocked_idx, dtype=np.int64)
    blocked_mask_cpu = torch.zeros(data[woman_key].num_nodes, dtype=torch.bool)
    blocked_mask_cpu[blocked_idx] = True
    for src, rel, dst in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        if edge_index.size(1) == 0: continue
        dev = edge_index.device
        blocked_mask = blocked_mask_cpu.to(dev)
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=dev)
        if src == woman_key:
            keep_mask = keep_mask & (~blocked_mask[edge_index[0]])
        if dst == woman_key:
            keep_mask = keep_mask & (~blocked_mask[edge_index[1]])
        data[src, rel, dst].edge_index = edge_index[:, keep_mask]
    return data

def calculate_metrics(y_true, y_pred, y_probs):
    auc_score = 0.5
    try:
        if len(np.unique(y_true)) > 1: auc_score = roc_auc_score(y_true, y_probs)
    except: pass
    return {'AUC': auc_score, 'F1': f1_score(y_true, y_pred, pos_label=1), 'y_true': y_true, 'y_prob': y_probs}

def cleanup():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


# ----- MODELS

class TextMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, out_channels=2):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        return self.lin2(self.dropout(self.relu(self.lin1(x))))

class HeteroGraphSAGE(nn.Module):
    def __init__(self, metadata, woman_key, hidden_channels=64, out_channels=2):
        super().__init__()
        self.woman_key = woman_key
        node_types, edge_types = metadata
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr='mean')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr='mean')
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: self.relu(val) for key, val in x.items()}
        x = self.conv2(x, edge_index_dict)
        x = {key: self.relu(val) for key, val in x.items()}
        return self.lin(x[self.woman_key])

def apply_feats_inplace(g, pdi_idx, woman_key):
    x = g[woman_key].x
    emb = x[:, :768]
    comp = x[:, 781:]
    if pdi_idx is None or len(pdi_idx) == 0: # Use default (all or none logic handled by caller)
        # If explicitly None passed, we assume 'all' was meant, or handled outside.
        # But for Stage B selection, we need strict subsets.
        # Fallback: if pdi_idx is None, use everything (Stage A behavior)
        pdi = x[:, 768:781]
    else:
        pdi = x[:, [768 + i for i in pdi_idx]]
    g[woman_key].x = torch.cat([emb, pdi, comp], dim=1)
    return g


# --- FEATURE SELECTION LOGIC

def train_eval_inner_gnn(g_train, g_val_full, val_mask_cpu, pdi_indices, device, woman_key):
    train_data = apply_feats_inplace(g_train.clone(), pdi_indices, woman_key).to(device)
    eval_data  = apply_feats_inplace(g_val_full.clone(), pdi_indices, woman_key).to(device)
    model = HeteroGraphSAGE(g_val_full.metadata(), woman_key, CONFIG['HIDDEN_CHANNELS']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    model.train()
    for _ in range(CONFIG['INNER_EPOCHS']):
        optimizer.zero_grad()
        out = model(train_data.x_dict, train_data.edge_index_dict)
        loss = F.cross_entropy(out, train_data[woman_key].y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(eval_data.x_dict, eval_data.edge_index_dict)
        val_mask = val_mask_cpu.to(out.device)
        pred = out[val_mask].argmax(dim=1).cpu().numpy()
        y_true = eval_data[woman_key].y[val_mask].cpu().numpy()
    score = f1_score(y_true, pred, pos_label=1)
    del model, optimizer, train_data, eval_data, out
    cleanup()
    return score

def select_features_1se(data, train_idx, device, woman_key):
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG['SEED'])
    y_train = data[woman_key].y[train_idx].cpu().numpy()
    inner_splits = []
    for t_idx, v_idx in inner_skf.split(np.zeros(len(train_idx)), y_train):
        # Map inner indices back to global indices
        global_train = train_idx[t_idx]
        global_val   = train_idx[v_idx]
        g_inner_train = get_train_subgraph(data, global_train, woman_key)
        g_inner_val = data
        val_mask = torch.zeros(data[woman_key].num_nodes, dtype=torch.bool)
        val_mask[global_val] = True
        inner_splits.append((g_inner_train, g_inner_val, val_mask))

    current_features = list(range(13))
    history = []

    # Backward Elimination
    while len(current_features) > 0:
        scores = []
        for g_trn, g_val, v_msk in inner_splits:
            s = train_eval_inner_gnn(g_trn, g_val, v_msk, current_features, device, woman_key)
            scores.append(s)
        mean_s = np.mean(scores)
        se_s = np.std(scores) / np.sqrt(len(scores))
        history.append({'subset': list(current_features), 'score': mean_s, 'se': se_s})

        if len(current_features) == 1: break

        best_removal_score = -1
        worst_feat = -1
        for feat in current_features:
            temp_feats = [f for f in current_features if f != feat]
            temp_scores = []
            for g_trn, g_val, v_msk in inner_splits:
                ts = train_eval_inner_gnn(g_trn, g_val, v_msk, temp_feats, device, woman_key)
                temp_scores.append(ts)
            avg_ts = np.mean(temp_scores)
            if avg_ts > best_removal_score:
                best_removal_score = avg_ts
                worst_feat = feat
        current_features.remove(worst_feat)

    best_run = max(history, key=lambda x: x['score'])
    target = best_run['score'] - best_run['se']
    selected = best_run['subset']
    # 1-SE Logic: Find smallest subset within SE of best
    for run in reversed(history):
        if run['score'] >= target:
            selected = run['subset']
            break
    return selected


# --- MAIN

def main(path='perignnosis_graph.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = torch.load(path, weights_only=False)
    print("Sanitizing graph...")
    data, woman_key = sanitize_and_fix_graph(data)
    y_all = data[woman_key].y.cpu().numpy().astype(int)

    NUM_FOLDS = 5
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=CONFIG['SEED'])

    # --- STAGE A: 3-WAY COMPARISON ---
    print("\n=== STAGE A: 3-Way Model Comparison ===")
    roc_data = {'GNN': [], 'SVM-Clinical': [], 'Text-MLP': []}

    # We save the fold indices to reuse them in Stage B exactly
    folds_indices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        print(f"  Fold {fold+1}/{NUM_FOLDS}")
        folds_indices.append((train_idx, test_idx))

        # Data Prep
        train_graph = get_train_subgraph(data, train_idx, woman_key).to(device)
        test_graph  = data.clone().to(device)
        # Stage A uses ALL features
        apply_feats_inplace(train_graph, CONFIG['ALL_PDI'], woman_key)
        apply_feats_inplace(test_graph,  CONFIG['ALL_PDI'], woman_key)

        X_cpu = data[woman_key].x.cpu().numpy()
        y_cpu = data[woman_key].y.cpu().numpy()

        # GNN
        model = HeteroGraphSAGE(data.metadata(), woman_key, CONFIG['HIDDEN_CHANNELS']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
        model.train()
        for _ in range(CONFIG['OUTER_EPOCHS']):
            opt.zero_grad()
            out = model(train_graph.x_dict, train_graph.edge_index_dict)
            loss = F.cross_entropy(out, train_graph[woman_key].y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(test_graph.x_dict, test_graph.edge_index_dict)[test_idx]
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            roc_data['GNN'].append(calculate_metrics(y_all[test_idx], preds, probs))
        cleanup()

        # SVM
        X_clin = X_cpu[:, 768:]
        svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=CONFIG['SEED'])
        svm.fit(X_clin[train_idx], y_cpu[train_idx])
        probs = svm.predict_proba(X_clin[test_idx])[:, 1]
        preds = svm.predict(X_clin[test_idx])
        roc_data['SVM-Clinical'].append(calculate_metrics(y_cpu[test_idx], preds, probs))

        # MLP
        X_emb = torch.from_numpy(X_cpu[:, :768]).float().to(device)
        y_emb = torch.from_numpy(y_cpu).long().to(device)
        mlp = TextMLP().to(device)
        opt_mlp = torch.optim.Adam(mlp.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
        mlp.train()
        for _ in range(CONFIG['OUTER_EPOCHS']):
            opt_mlp.zero_grad()
            loss = F.cross_entropy(mlp(X_emb[train_idx]), y_emb[train_idx])
            loss.backward()
            opt_mlp.step()
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X_emb[test_idx])
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            roc_data['Text-MLP'].append(calculate_metrics(y_cpu[test_idx], preds, probs))
        cleanup()

    print("\nStage A Results (Mean AUC):")
    scores = {k: np.mean([r['AUC'] for r in v]) for k, v in roc_data.items()}
    for k, v in scores.items(): print(f"  {k}: {v:.3f}")
    plot_combined_roc(roc_data, filename="roc_comparison.pdf")

    # after running the aboce code GNN is the winner (target for optimization)
    winner = 'GNN'
  
    # --- FEATURE SELECTION ---
    selected_subsets = []
    stage_b_metrics = []
    STAGE_B_THRESHOLD = 0.39  # specific threshold selection

    for fold, (train_idx, test_idx) in enumerate(folds_indices):
        print(f"  Fold {fold+1}/{NUM_FOLDS} Selection...")

        # Leak-proof data for selection
        outer_safe_data = remove_women_edges(data, test_idx, woman_key)

        # Run Backward Elimination
        best_pdi = select_features_1se(outer_safe_data, train_idx, device, woman_key)
        selected_subsets.append(best_pdi)
        print(f"    Selected {len(best_pdi)} items: {best_pdi}")

        # Final Retrain on Outer Fold
        train_graph = get_train_subgraph(data, train_idx, woman_key).to(device)
        test_graph  = data.clone().to(device)

        apply_feats_inplace(train_graph, best_pdi, woman_key)
        apply_feats_inplace(test_graph,  best_pdi, woman_key)

        model = HeteroGraphSAGE(data.metadata(), woman_key, CONFIG['HIDDEN_CHANNELS']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)

        model.train()
        for _ in range(CONFIG['OUTER_EPOCHS']):
            opt.zero_grad()
            out = model(train_graph.x_dict, train_graph.edge_index_dict)
            loss = F.cross_entropy(out, train_graph[woman_key].y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(test_graph.x_dict, test_graph.edge_index_dict)[test_idx]
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # --- THRESHOLD MODIFICATION ---
            # Instead of argmax (which is threshold 0.5), we use 0.39
            preds = (probs >= STAGE_B_THRESHOLD).astype(int)
            # ------------------------------

            stage_b_metrics.append(calculate_metrics(y_all[test_idx], preds, probs))
        cleanup()

    print("\nStage B Final Results (Optimized GNN):")
    print(f"  Mean AUC: {np.mean([r['AUC'] for r in stage_b_metrics]):.3f}")
    print(f"  Mean F1:  {np.mean([r['F1'] for r in stage_b_metrics]):.3f}")

    # Save Everything
    full_results = {
        'stage_a': roc_data,
        'stage_b_metrics': stage_b_metrics,
        'selected_subsets': selected_subsets
    }
    with open(os.path.join(PTH,'graphRAG/data/full_results.json'), 'w') as f:
        json.dump(to_native(full_results), f, indent=2)
    print("Full results saved to full_results.json")

if __name__ == "__main__":
    if 'PTH' in globals():
        main(os.path.join(PTH, 'graphRAG/data/perignnosis_graph.pt'))
    else:
        main('perignnosis_graph.pt')
