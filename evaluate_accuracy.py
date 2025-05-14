# quick_eval.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from utils.dataset_utils    import load_reference_signs_from_images

# ─── CONFIG ─────────────────────────────────────────────
DATASET_DIR   = "data/dataset"
MAX_PER_CLASS = 500
MIN_CONF      = 0.3
# ───────────────────────────────────────────────────────

# 1) Build (or load) your reference library exactly as you do in main
reference_signs = load_reference_signs_from_images(
    dataset_dir=DATASET_DIR,
    min_conf=MIN_CONF,
    max_per_class=MAX_PER_CLASS
)

# 2) Extract the one-frame feature vector for each template
X, y = [], []
for _, row in reference_signs.iterrows():
    sm = row["sign_model"]
    if not sm.lh_embedding:           # skip any template with no left‐hand
        continue
    X.append(sm.lh_embedding[0])      # 441-dim angle vector
    y.append(row["name"])
X = np.vstack(X)                     # shape = (N_refs, 441)
y = np.array(y)

print(f"Loaded {X.shape[0]} templates for {len(np.unique(y))} letters.")

# 3) Leave-One-Out nearest neighbor
nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
dists, inds = nbrs.kneighbors(X)
# inds[:,0] is always the sample itself; we want the 2nd neighbor
pred_loo = y[inds[:,1]]
acc_loo  = np.mean(pred_loo == y)
print(f"LOO 1-NN accuracy: {acc_loo:.2%}")

# 4) 5-fold cross-validation with a 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1, algorithm="auto")
scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"5-fold CV (1-NN) accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
