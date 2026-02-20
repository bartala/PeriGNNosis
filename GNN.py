import os
import gc
import json
import random
import numpy as np
import pandas as pd
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, LayerNorm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)


PTH ='/.../Postpartum_depression/'

# ==============================================================================
# CONFIG (Stable Base)
# ==============================================================================
CONFIG = {
    "SEED": 42,
    "OUTER_FOLDS": 5,
    "INNER_FOLDS": 3,

    # training
    "INNER_EPOCHS": 10,
    "OUTER_EPOCHS": 25,
    "HIDDEN_CHANNELS": 64,   # Reverted to 64 for stability
    "LR": 5e-4,
    "WEIGHT_DECAY": 1e-2,    # Reverted to 1e-2
    "GAT_HEADS": 4,

    # baselines
    "C_GRID": [0.1, 1, 10],
}

# --- Clinical feature layout (Woman node only) ---
EMB_DIM = 768   # narrative embedding dim
PDI_DIM = 13    # number of PDI items in your stored graph features
COMP_DIM = 1    # complication flag
TOTAL_DIM = EMB_DIM + PDI_DIM + COMP_DIM  # expected woman feature width

PDI_LABELS = [
    "PDI1", "PDI2", "PDI3", "PDI4", "PDI5", "PDI6", "PDI7",
    "PDI8", "PDI9", "PDI10", "PDI11", "PDI12", "PDI13"
]

# ==============================================================================
# UTILS & STATS
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))

def to_jsonable(obj):
    import numpy as _np
    if isinstance(obj, (_np.integer,)): return int(obj)
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.ndarray,)): return obj.tolist()
    if isinstance(obj, dict): return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_jsonable(x) for x in obj]
    return obj

def find_threshold_from_oof(y_true, p, method="f1") -> float:
    y_true, p = np.asarray(y_true).astype(int), np.asarray(p).astype(float)
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    if thresholds.size == 0: return 0.5
    if method == "f1":
        f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        return float(thresholds[int(np.argmax(f1s))])
    if method == "high_sens":
        best = np.argmax(recall[:-1])
        return float(thresholds[int(best)])
    raise ValueError(f"Unknown threshold method: {method}")

def summarize_at_threshold(y_true, p, thr: float) -> dict:
    y_true, p = np.asarray(y_true).astype(int), np.asarray(p).astype(float)
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    return {
        "threshold": float(thr), "auc": safe_auc(y_true, p), "f1": float(f1_score(y_true, y_hat)),
        "sensitivity": float(sens), "specificity": float(spec), "precision": float(prec), "recall": float(sens),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def bootstrap_auc_difference(y_true, p1, p2, name1="Model 1", name2="Model 2", n_bootstraps=2000):
    y_true, p1, p2 = np.asarray(y_true).astype(int), np.asarray(p1).astype(float), np.asarray(p2).astype(float)
    auc1, auc2 = roc_auc_score(y_true, p1), roc_auc_score(y_true, p2)
    diff = auc1 - auc2
    
    print("\n" + "="*60)
    print("STEP A.1: STATISTICAL SIGNIFICANCE (DeLong Surrogate)")
    print("="*60)
    print(f"{name1:<12} AUC: {auc1:.4f}")
    print(f"{name2:<12} AUC: {auc2:.4f}")
    print(f"Difference:       {diff:.4f}")

    rng = np.random.RandomState(42)
    boot_diffs = []
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        sub_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[sub_idx])) < 2: continue
        boot_diffs.append(roc_auc_score(y_true[sub_idx], p1[sub_idx]) - roc_auc_score(y_true[sub_idx], p2[sub_idx]))
    
    std_err = np.std(boot_diffs)
    z_score = diff / (std_err + 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"P-value:          {p_value:.4f} (Z-score: {z_score:.4f}, SE: {std_err:.5f})")
    if p_value < 0.05:
        print(f">>> RESULT: SIGNIFICANT (p < 0.05). {name1 if diff > 0 else name2} is superior.")
    else:
        print(">>> RESULT: NOT SIGNIFICANT (p >= 0.05). Performance gap is statistical noise.")
    print("="*60 + "\n")

# ==============================================================================
# GRAPH SANITIZATION + LEAKAGE BLOCK
# ==============================================================================
def validate_and_fix_edge_index(data: HeteroData) -> HeteroData:
    for et in data.edge_types:
        if "edge_index" not in data[et] or data[et].edge_index is None:
            data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
            continue
        ei = data[et].edge_index
        if not torch.is_tensor(ei): ei = torch.as_tensor(ei)
        ei = ei.long().contiguous()
        if ei.dim() == 1:
            if ei.numel() % 2 != 0: ei = ei[:-1]
            ei = ei.view(2, -1)
        data[et].edge_index = ei
    return data

def sanitize_and_fix_graph(raw: HeteroData):
    raw = raw.cpu()
    node_map = {nt: (nt.replace(" ", "_").replace("-", "_") if nt != "__Entity__" else "Entity") for nt in raw.node_types}
    tmp = HeteroData()

    for nt, new_nt in node_map.items():
        if "x" in raw[nt] and raw[nt].x is not None:
            tmp[new_nt].x = raw[nt].x
            tmp[new_nt].num_nodes = int(raw[nt].x.size(0))
        else:
            tmp[new_nt].num_nodes = int(raw[nt].num_nodes) if getattr(raw[nt], "num_nodes", None) else 0
        if "y" in raw[nt] and raw[nt].y is not None:
            tmp[new_nt].y = raw[nt].y

    woman_key = node_map["Woman"]
    for (src, rel, dst) in raw.edge_types:
        new_src, new_dst, new_rel = node_map[src], node_map[dst], rel.replace(" ", "_").replace("-", "_")
        if "edge_index" in raw[src, rel, dst]:
            tmp[new_src, new_rel, new_dst].edge_index = raw[src, rel, dst].edge_index

    data = T.ToUndirected()(tmp)
    data = validate_and_fix_edge_index(data)

    for nt in data.node_types:
        target_dim = TOTAL_DIM if nt == woman_key else EMB_DIM
        if "x" not in data[nt] or data[nt].x is None:
            data[nt].x = torch.zeros((data[nt].num_nodes, target_dim), dtype=torch.float32)
        else:
            x = data[nt].x
            if x.dim() != 2: raise ValueError(f"{nt}.x must be 2D, got shape {tuple(x.shape)}")
            curr_dim = x.size(1)
            if curr_dim < target_dim:
                pad = torch.zeros((data[nt].num_nodes, target_dim - curr_dim), dtype=x.dtype, device=x.device)
                data[nt].x = torch.cat([x, pad], dim=1)
            elif curr_dim > target_dim:
                raise ValueError(f"CRITICAL: {nt} node has {curr_dim} features but expected {target_dim}.")

    for (src, rel, dst) in data.edge_types:
        ei = data[src, rel, dst].edge_index
        if ei is None or ei.numel() == 0: continue
        mask = ((ei[0] >= 0) & (ei[0] < data[src].num_nodes) & (ei[1] >= 0) & (ei[1] < data[dst].num_nodes))
        data[src, rel, dst].edge_index = ei[:, mask]

    return data, woman_key

def block_women_outgoing_edges(data: HeteroData, blocked_idx, woman_key: str) -> HeteroData:
    data = data.clone()
    blocked = torch.zeros(data[woman_key].num_nodes, dtype=torch.bool)
    blocked[torch.as_tensor(blocked_idx, dtype=torch.long)] = True

    for (src, rel, dst) in data.edge_types:
        if src != woman_key: continue
        ei = data[src, rel, dst].edge_index
        if ei is None or ei.numel() == 0: continue
        data[src, rel, dst].edge_index = ei[:, ~blocked[ei[0]]]

    return data

def apply_pdi_mask_inplace(data: HeteroData, woman_key: str, selected_pdis):
    if selected_pdis is None: return data
    selected_pdis = [int(i) for i in selected_pdis]
    x = data[woman_key].x
    
    pdi_start, pdi_end = EMB_DIM, EMB_DIM + PDI_DIM
    keep_mask = torch.zeros(PDI_DIM, dtype=torch.bool, device=x.device)
    keep_mask[torch.as_tensor(selected_pdis, dtype=torch.long, device=x.device)] = True

    data = data.clone()
    x = data[woman_key].x.clone()
    pdi_block = x[:, pdi_start:pdi_end]
    pdi_block[:, ~keep_mask] = 0.0
    x[:, pdi_start:pdi_end] = pdi_block
    data[woman_key].x = x
    return data

# ==============================================================================
# MODELS
# ==============================================================================
class HeteroGAT(nn.Module):
    """
    GNN: Preserves exact PDI scores while aggregating graph context.
    """
    def __init__(self, metadata, woman_key, hidden_channels=64, out_channels=2, heads=4):
        super().__init__()
        self.woman_key = woman_key
        node_types, edge_types = metadata

        self.encoder = nn.ModuleDict({
            nt: nn.Linear(TOTAL_DIM if nt == woman_key else EMB_DIM, hidden_channels)
            for nt in node_types
        })
        self.norms = nn.ModuleDict({nt: LayerNorm(hidden_channels) for nt in node_types})
        
        self.conv1 = HeteroConv({
            et: GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False)
            for et in edge_types
        }, aggr="sum")

        self.conv2 = HeteroConv({
            et: GATConv(hidden_channels, hidden_channels, heads=1, concat=False, add_self_loops=False)
            for et in edge_types
        }, aggr="sum")

        self.res_lin = nn.Linear(hidden_channels, hidden_channels)
        
        self.lin = nn.Linear(hidden_channels + PDI_DIM + COMP_DIM, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        # 1. Isolate the raw clinical features (PDI + Complication) before message passing
        # This will correctly capture the features *after* Step B has applied its masking
        raw_woman = x_dict[self.woman_key]
        clinical_features = raw_woman[:, EMB_DIM:] # Shape: [N, 14]

        # 2. Standard Encoding
        h_dict = {nt: self.norms[nt](self.encoder[nt](x)) for nt, x in x_dict.items()}
        h_initial = h_dict[self.woman_key]

        # 3. Message Passing (Learning the Graph Context)
        x1 = self.conv1(h_dict, edge_index_dict)
        x1 = {k: self.relu(x1.get(k, h_dict[k])) for k in h_dict.keys()}

        x2 = self.conv2(x1, edge_index_dict)
        x2_woman = self.relu(x2.get(self.woman_key, x1[self.woman_key]))

        # 4. Graph Context + Residual
        graph_context = x2_woman + self.res_lin(h_initial)

        # 5. LATE FUSION: Concatenate graph context with exact clinical features
        out = torch.cat([graph_context, clinical_features], dim=-1)
        
        return self.lin(out)

class WomanMLP(nn.Module):
    def __init__(self, woman_key, hidden, out=2):
        super().__init__()
        self.woman_key = woman_key
        self.net = nn.Sequential(
            nn.Linear(TOTAL_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out)
        )

    def forward(self, x_dict, edge_index_dict=None):
        return self.net(x_dict[self.woman_key])

def build_model(model_name, metadata, woman_key):
    if model_name == "GNN": return HeteroGAT(metadata, woman_key, CONFIG["HIDDEN_CHANNELS"], 2, CONFIG["GAT_HEADS"])
    if model_name == "FFNN": return WomanMLP(woman_key, CONFIG["HIDDEN_CHANNELS"], 2)
    raise ValueError(f"Unknown model_name: {model_name}")

# ==============================================================================
# TRAINING HELPERS
# ==============================================================================
def train_and_predict(model, g, train_idx, target_idx, y_torch, device, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    tr_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(g.x_dict, g.edge_index_dict)[tr_t], y_torch[tr_t])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        te_t = torch.as_tensor(target_idx, device=device, dtype=torch.long)
        return F.softmax(model(g.x_dict, g.edge_index_dict)[te_t], dim=1)[:, 1].detach().cpu().numpy()

# ==============================================================================
# STEP A – MODEL SELECTION
# ==============================================================================
def stepA_model_selection(data, woman_key, device, out_dir):
    y_all = data[woman_key].y.view(-1).cpu().numpy().astype(int)
    y_torch = torch.as_tensor(y_all, dtype=torch.long, device=device)
    X_cpu = data[woman_key].x.cpu().numpy()

    skf = StratifiedKFold(n_splits=CONFIG["OUTER_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])
    model_names = ["GNN", "FFNN", "LR-Fusion", "SVM-Fusion"]
    aucs = {m: [] for m in model_names}
    oof_probs = {m: np.zeros(len(y_all), dtype=float) for m in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(y_all, y_all)):
        fold_seed = CONFIG["SEED"] + fold
        set_seed(fold_seed)
        outer_safe = block_women_outgoing_edges(data, test_idx, woman_key)

        for m in ["GNN", "FFNN"]:
            g = outer_safe.clone().to(device)
            model = build_model(m, g.metadata(), woman_key).to(device)
            p_test = train_and_predict(model, g, train_idx, test_idx, y_torch, device, CONFIG["OUTER_EPOCHS"])
            aucs[m].append(safe_auc(y_all[test_idx], p_test))
            oof_probs[m][test_idx] = p_test
            del model, g; cleanup()

        for m in ["LR-Fusion", "SVM-Fusion"]:
            kind = "LR" if "LR" in m else "SVM"
            clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=fold_seed) if kind == "LR" else SVC(probability=True, class_weight="balanced", random_state=fold_seed)
            inner_cv = StratifiedKFold(n_splits=CONFIG["INNER_FOLDS"], shuffle=True, random_state=fold_seed)
            gs = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("clf", clf)]), {"clf__C": CONFIG["C_GRID"]}, scoring="roc_auc", cv=inner_cv, n_jobs=-1, refit=True).fit(X_cpu[train_idx], y_all[train_idx])
            p_test = gs.best_estimator_.predict_proba(X_cpu[test_idx])[:, 1]
            aucs[m].append(safe_auc(y_all[test_idx], p_test))
            oof_probs[m][test_idx] = p_test

    summary = pd.DataFrame({"model": model_names, "mean_auc": [float(np.mean(aucs[m])) for m in model_names]}).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    best_model = summary.loc[0, "model"]

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"y_true": y_all, **oof_probs}).to_csv(os.path.join(out_dir, "stepA_oof_predictions.csv"), index=False)

    return summary, best_model, oof_probs, y_all

# ==============================================================================
# STEP B – PDI SELECTION
# ==============================================================================
def rank_pdis_by_inner_train_auc(y_train, X_train_woman):
    ranks = []
    for j in range(PDI_DIM):
        col = X_train_woman[:, EMB_DIM + j]
        a = 0.5 if (np.std(col) < 1e-12 or len(np.unique(y_train)) < 2) else roc_auc_score(y_train, col)
        ranks.append((j, float(a)))
    ranks.sort(key=lambda t: t[1], reverse=True)
    return [int(i) for i, _ in ranks]

def pick_best_k_on_outer_train(best_model_name, outer_safe_graph, woman_key, train_idx, y_all, device, seed):
    inner = StratifiedKFold(n_splits=CONFIG["INNER_FOLDS"], shuffle=True, random_state=seed)
    k_scores = {k: [] for k in range(1, PDI_DIM + 1)}
    X_woman_cpu = outer_safe_graph[woman_key].x.cpu().numpy()

    for inner_id, (itr, ival) in enumerate(inner.split(train_idx, y_all[train_idx])):
        set_seed(seed + 100 + inner_id)
        tr_glob, val_glob = train_idx[itr], train_idx[ival]
        inner_safe = block_women_outgoing_edges(outer_safe_graph, val_glob, woman_key)
        pdi_rank = rank_pdis_by_inner_train_auc(y_all[tr_glob], X_woman_cpu[tr_glob])

        for k in range(1, PDI_DIM + 1):
            g = apply_pdi_mask_inplace(inner_safe, woman_key, pdi_rank[:k]).clone().to(device)
            model = build_model(best_model_name, g.metadata(), woman_key).to(device)
            p_val = train_and_predict(model, g, tr_glob, val_glob, torch.as_tensor(y_all, dtype=torch.long, device=device), device, CONFIG["INNER_EPOCHS"])
            k_scores[k].append(safe_auc(y_all[val_glob], p_val))
            del model, g; cleanup()

    mean_scores = {k: float(np.mean(v)) for k, v in k_scores.items()}
    best_k = max(mean_scores.keys(), key=lambda kk: mean_scores[kk])
    full_rank = rank_pdis_by_inner_train_auc(y_all[train_idx], X_woman_cpu[train_idx])
    
    return full_rank[:best_k], {"best_k": int(best_k), "mean_auc_by_k": {int(k): float(v) for k, v in mean_scores.items()}, "rank_full_outer_train": [int(i) for i in full_rank]}

def stepB_pdi_selection_and_oof(best_model_name, data, woman_key, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    y_all = data[woman_key].y.view(-1).cpu().numpy().astype(int)
    y_torch = torch.as_tensor(y_all, dtype=torch.long, device=device)
    skf = StratifiedKFold(n_splits=CONFIG["OUTER_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])
    
    oof_probs_selected = np.zeros_like(y_all, dtype=float)
    selected_per_fold, diagnostics_per_fold = {}, {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(y_all, y_all)):
        fold_seed = CONFIG["SEED"] + fold
        set_seed(fold_seed)
        outer_safe = block_women_outgoing_edges(data, test_idx, woman_key)

        selected_pdis, diag = pick_best_k_on_outer_train(best_model_name, outer_safe, woman_key, train_idx, y_all, device, fold_seed)
        selected_per_fold[f"fold_{fold+1}"] = [int(i) for i in selected_pdis]
        diagnostics_per_fold[f"fold_{fold+1}"] = to_jsonable(diag)
        print(f"[Step B] Fold {fold+1}/{CONFIG['OUTER_FOLDS']}: selected k={len(selected_pdis)} PDIs -> {selected_pdis}")

        g = apply_pdi_mask_inplace(outer_safe, woman_key, selected_pdis).clone().to(device)
        model = build_model(best_model_name, g.metadata(), woman_key).to(device)
        oof_probs_selected[test_idx] = train_and_predict(model, g, train_idx, test_idx, y_torch, device, CONFIG["OUTER_EPOCHS"])
        del model, g; cleanup()

    with open(os.path.join(out_dir, "stepB_selected_pdis_per_fold.json"), "w") as f: json.dump(to_jsonable(selected_per_fold), f, indent=2)
    with open(os.path.join(out_dir, "stepB_selection_diagnostics.json"), "w") as f: json.dump(to_jsonable(diagnostics_per_fold), f, indent=2)

    counts = np.zeros(PDI_DIM, dtype=int)
    for sel in selected_per_fold.values():
        for i in sel: counts[int(i)] += 1

    freq_df = pd.DataFrame({"pdi_index_0based": list(range(PDI_DIM)), "pdi_label": PDI_LABELS[:PDI_DIM], "selected_in_folds": counts, "selected_rate": counts / CONFIG["OUTER_FOLDS"]}).sort_values(["selected_in_folds", "pdi_index_0based"], ascending=[False, True])
    freq_df.to_csv(os.path.join(out_dir, "stepB_pdi_selection_frequency.csv"), index=False)

    return y_all, oof_probs_selected, selected_per_fold, freq_df

# ==============================================================================
# MAIN
# ==============================================================================
def main(graph_path, out_dir="final_pipeline_outputs", threshold_method="f1"):
    set_seed(CONFIG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw = torch.load(graph_path, map_location="cpu", weights_only=False)
    data, woman_key = sanitize_and_fix_graph(raw)

    # === STEP A ===
    stepA_table, best_model, stepA_probs, y_all_stepA = stepA_model_selection(data, woman_key, device, out_dir)
    print("\n=== STEP A: Model selection by mean outer-CV AUC (no thresholds) ===")
    print(stepA_table.to_string(index=False))
    print(f"\n[Step A] Best model by AUC: {best_model}")
    print(f"[Saved] Step A OOF predictions: {os.path.join(out_dir, 'stepA_oof_predictions.csv')}")

    # === STEP A.1: DELONG TEST ===
    top_2_models = stepA_table["model"].iloc[:2].tolist()
    if len(top_2_models) >= 2:
        m1, m2 = top_2_models
        bootstrap_auc_difference(y_all_stepA, stepA_probs[m1], stepA_probs[m2], name1=m1, name2=m2)

    # === STEP B ===
    print("\n=== STEP B: PDI selection (inner-CV on outer-train; leakage-safe) ===")
    y_true, oof_p_selected, selected_per_fold, freq_df = stepB_pdi_selection_and_oof(best_model, data, woman_key, device, out_dir)

    oof_path = os.path.join(out_dir, "stepB_oof_predictions_selected.csv")
    pd.DataFrame({"y_true": y_true.astype(int), "p_selected": oof_p_selected.astype(float)}).to_csv(oof_path, index=False)
    print(f"\n[Saved] OOF predictions: {oof_path}")
    print(f"[Saved] PDI frequency table: {os.path.join(out_dir, 'stepB_pdi_selection_frequency.csv')}")

    # === STEP C ===
    thr = find_threshold_from_oof(y_true, oof_p_selected, method=threshold_method)
    metrics = summarize_at_threshold(y_true, oof_p_selected, thr)

    metrics_path = os.path.join(out_dir, "stepC_threshold_and_metrics.json")
    with open(metrics_path, "w") as f: json.dump(to_jsonable(metrics), f, indent=2)

    print("\n=== STEP C: Threshold selection AFTER Step B (on OOF predictions) ===")
    print(f"Threshold method: {threshold_method}")
    print(f"Selected threshold: {metrics['threshold']:.4f}")
    print(f"AUC={metrics['auc']:.3f} | F1={metrics['f1']:.3f} | Sens={metrics['sensitivity']:.3f} | Spec={metrics['specificity']:.3f} | Prec={metrics['precision']:.3f}")
    print(f"[Saved] Threshold+metrics: {metrics_path}")

if __name__ == "__main__":
    path = os.path.join(PTH, "graphRAG/data/perignnosis_graph.pt") if "PTH" in globals() else "perignnosis_graph.pt"
    main(path, out_dir="final_pipeline_outputs", threshold_method="f1")
