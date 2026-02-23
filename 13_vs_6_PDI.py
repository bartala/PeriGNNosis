import torch
import torch.nn.functional as F
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# -----------------------------
# Paired permutation test on pooled OOF AUC
# -----------------------------
def paired_permutation_test_auc(y_true, preds_A, preds_B, n_permutations=10000, seed=42):
    y_true = np.asarray(y_true).astype(int).ravel()
    preds_A = np.asarray(preds_A).astype(float).ravel()
    preds_B = np.asarray(preds_B).astype(float).ravel()

    rng = np.random.default_rng(seed)

    auc_A = float(roc_auc_score(y_true, preds_A))
    auc_B = float(roc_auc_score(y_true, preds_B))
    obs_diff = auc_A - auc_B

    count_extreme = 0
    n = len(y_true)

    for _ in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_A = np.where(swap, preds_B, preds_A)
        perm_B = np.where(swap, preds_A, preds_B)
        perm_diff = roc_auc_score(y_true, perm_A) - roc_auc_score(y_true, perm_B)
        if abs(perm_diff) >= abs(obs_diff):
            count_extreme += 1

    # add-one correction
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {"auc_A": auc_A, "auc_B": auc_B, "diff_A_minus_B": obs_diff, "p_value": float(p_value)}

# -----------------------------
# -----------------------------
configs = {
    "PDI-13": list(range(13)),
    "PDI-6":  [0, 1, 2, 4, 11, 12],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting AUC-only direct comparison...\n" + "="*60)

oof_probs = {}  # store pooled OOF probs for each config

for config_name, pdi_indices in configs.items():
    # pooled OOF probabilities for all women (filled fold-by-fold)
    oof = np.full(len(y_all), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        torch.manual_seed(42 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + fold)

        train_graph = get_train_subgraph(data, train_idx, woman_key).to(device)
        test_graph = data.clone().to(device)

        apply_feats_inplace(train_graph, pdi_indices, woman_key)
        apply_feats_inplace(test_graph, pdi_indices, woman_key)

        model = HeteroGraphSAGE(data.metadata(), woman_key, hidden_channels=64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        model.train()
        for _ in range(15):
            opt.zero_grad()
            out = model(train_graph.x_dict, train_graph.edge_index_dict)
            loss = F.cross_entropy(out, train_graph[woman_key].y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(test_graph.x_dict, test_graph.edge_index_dict)[test_idx]
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        oof[test_idx] = probs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if np.isnan(oof).any():
        raise RuntimeError(f"{config_name}: OOF probs contain NaNs. Check indexing/splits.")

    auc = roc_auc_score(y_all, oof)
    oof_probs[config_name] = oof
    print(f"{config_name} pooled OOF AUC: {auc:.6f}")

print("\n" + "="*60)
perm = paired_permutation_test_auc(
    y_true=y_all,
    preds_A=oof_probs["PDI-13"],
    preds_B=oof_probs["PDI-6"],
    n_permutations=10000,
    seed=42
)

print("Paired permutation test on pooled OOF AUC (two-sided, 10k perms)")
print(f"AUC PDI-13: {perm['auc_A']:.6f}")
print(f"AUC PDI-6 : {perm['auc_B']:.6f}")
print(f"ΔAUC (13-6): {perm['diff_A_minus_B']:.6f}")
print(f"p-value    : {perm['p_value']:.6f}")
