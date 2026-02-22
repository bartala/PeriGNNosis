import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Configuration
PREDICTIONS_FILE = "final_benchmark_optimized/benchmark_predictions.csv"
THRESHOLD = 0.2246  # Your optimal threshold from Step C
TARGET_MODEL = "PeriGNNosis"

# Load Data
df = pd.read_csv(PREDICTIONS_FILE)
y_true = df['y_true'].values
models = [c for c in df.columns if c != 'y_true']

# Apply Threshold (Convert Probabilities to Binary Classifications)
binary_preds = {}
for m in models:
    binary_preds[m] = (df[m].values >= THRESHOLD).astype(int)

# Define McNemar Helper
def run_mcnemar(y_true, pred_A, pred_B):
    """
    Constructs the contingency table based on *correctness*:
               B Correct | B Incorrect
    A Correct |    a     |     b
    A Incorr  |    c     |     d
    """
    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    a = np.sum(correct_A & correct_B)
    b = np.sum(correct_A & ~correct_B) # A got it right, B missed it
    c = np.sum(~correct_A & correct_B) # A missed it, B got it right
    d = np.sum(~correct_A & ~correct_B)

    table = [[a, b],
             [c, d]]

    # Exact=True uses binomial distribution (better for small sample sizes like N=302)
    result = mcnemar(table, exact=True)
    return result.pvalue, b, c

# Run the Test
print(f"=== McNemar's Test (Threshold = {THRESHOLD}) ===")
print(f"Comparing {TARGET_MODEL} vs Baselines based on Correct Classifications:\n")

target_preds = binary_preds[TARGET_MODEL]

results_summary = []
for m in models:
    if m == TARGET_MODEL:
        continue

    p_val, target_won, baseline_won = run_mcnemar(y_true, target_preds, binary_preds[m])

    print(f"vs {m:<18}:")
    print(f"  - {TARGET_MODEL} was right & baseline was wrong: {target_won} patients")
    print(f"  - Baseline was right & {TARGET_MODEL} was wrong: {baseline_won} patients")
    print(f"  - p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("STATISTICALLY SIGNIFICANT DIFFERENCE (*)")
    print("-" * 50)
