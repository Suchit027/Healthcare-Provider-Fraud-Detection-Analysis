"""
evaluate.py
===========
Evaluation pipeline for all models on 10 random test splits.

Metrics computed per split:
  - Accuracy, Precision, Recall, F1 (binary)
  - PR-AUC (where applicable)

Aggregation across 10 splits:
  - Mean, Variance, Std Dev

Usage:
  python evaluate.py --data_dir ./  [--weights model_weights.pt] [--n_splits 10]

Directory structure expected:
  <data_dir>/xtrain.csv
  <data_dir>/ytrain.csv
  <data_dir>/xval.csv     ← pool from which test splits are drawn
  <data_dir>/yval.csv
  model_weights.pt        ← required for Neural Network (weights only, no training)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"

import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def compute_metrics(ytest, y_pred, y_scores=None):
    """Return dict of metrics. PR-AUC is None if y_scores not provided."""
    metrics = {
        "accuracy":  accuracy_score(ytest, y_pred),
        "precision": precision_score(ytest, y_pred, zero_division=0),
        "recall":    recall_score(ytest, y_pred, zero_division=0),
        "f1":        f1_score(ytest, y_pred, zero_division=0),
        "pr_auc":    None,
    }
    if y_scores is not None:
        prec_c, rec_c, _ = precision_recall_curve(ytest, y_scores)
        metrics["pr_auc"] = auc(rec_c, prec_c)
    return metrics


def aggregate(results: list[dict]) -> dict:
    """
    Given a list of per-split metric dicts, return a dict with
    mean / variance / std for each metric.
    """
    keys = [k for k in results[0] if results[0][k] is not None]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in results if r[k] is not None], dtype=float)
        agg[k] = {
            "mean": float(np.mean(vals)),
            "variance": float(np.var(vals)),
            "std": float(np.std(vals)),
            "values": vals.tolist(),
        }
    return agg


def make_splits(X_pool, y_pool, n_splits=10, test_size=0.5, seed=0):
    """
    Draw `n_splits` non-overlapping (random) test subsets from the val pool.
    Each split is (X_test, y_test).
    """
    splits = []
    for i in range(n_splits):
        _, X_test, _, y_test = train_test_split(
            X_pool, y_pool,
            test_size=test_size,
            random_state=seed + i,
            stratify=y_pool,
        )
        splits.append((X_test, y_test))
    return splits


def print_report(model_name: str, agg: dict):
    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")
    header = f"{'Metric':<12}  {'Mean':>10}  {'Std':>10}  {'Variance':>12}"
    print(header)
    print("─" * len(header))
    for metric, stats in agg.items():
        print(
            f"{metric:<12}  {stats['mean']:>10.4f}  {stats['std']:>10.4f}  {stats['variance']:>12.6f}"
        )


# ──────────────────────────────────────────────────────────────
# Model evaluators
# ──────────────────────────────────────────────────────────────

def run_logistic_regression(X_train, y_train, splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    results = []
    for X_test, y_test in splits:
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        results.append(compute_metrics(y_test, y_pred, y_prob))
    return results


def run_gradient_boosting(X_train, y_train, splits):
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    results = []
    for X_test, y_test in splits:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(compute_metrics(y_test, y_pred, y_prob))
    return results


def run_knn(X_train, y_train, splits):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    results = []
    for X_test, y_test in splits:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(compute_metrics(y_test, y_pred, y_prob))
    return results


def run_svc(X_train, y_train, splits):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(random_state=42, max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    results = []
    for X_test, y_test in splits:
        y_pred   = model.predict(X_test)
        y_scores = model.decision_function(X_test)   # no predict_proba for LinearSVC
        results.append(compute_metrics(y_test, y_pred, y_scores))
    return results


def run_isolation_forest(X_train, splits):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=200, contamination="auto",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train)

    results = []
    for X_test, y_test in splits:
        y_pred   = (model.predict(X_test) == -1).astype(int)
        y_scores = -model.decision_function(X_test)  # invert: higher = more anomalous
        results.append(compute_metrics(y_test, y_pred, y_scores))
    return results


def run_elliptic_envelope(X_train, splits):
    from sklearn.covariance import EllipticEnvelope

    model = EllipticEnvelope(contamination=0.1)
    model.fit(X_train)

    results = []
    for X_test, y_test in splits:
        y_pred   = (model.predict(X_test) == -1).astype(int)
        y_scores = -model.decision_function(X_test)
        results.append(compute_metrics(y_test, y_pred, y_scores))
    return results


def run_local_outlier_factor(X_train, splits):
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    model.fit(X_train)

    results = []
    for X_test, y_test in splits:
        y_pred   = (model.predict(X_test) == -1).astype(int)
        y_scores = -model.decision_function(X_test)
        results.append(compute_metrics(y_test, y_pred, y_scores))
    return results


def run_hbos(X_train, splits):
    try:
        from pyod.models.hbos import HBOS
    except ImportError:
        print("  [SKIP] pyod not installed — run: pip install pyod")
        return None

    model = HBOS()
    model.fit(X_train)

    results = []
    for X_test, y_test in splits:
        y_pred   = model.predict(X_test)
        y_scores = model.decision_function(X_test)
        results.append(compute_metrics(y_test, y_pred, y_scores))
    return results


def run_neural_network(splits, weights_path):
    """
    PyTorch MLP (LinearModel) from model.ipynb.
    Weights are loaded from `weights_path` — no training is performed.
    PR-AUC: computed from raw sigmoid output (continuous score).
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        print("  [SKIP] torch not installed.")
        return None

    if not os.path.exists(weights_path):
        print(f"  [SKIP] Neural Network: weights file not found at '{weights_path}'.")
        return None

    # ── Model definition (mirrors model.ipynb) ──────────────────
    class LinearModel(nn.Module):
        def __init__(self, p=0.2):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(65, 40), nn.ReLU(), nn.BatchNorm1d(40),
                nn.Linear(40, 35), nn.ReLU(), nn.BatchNorm1d(35),
                nn.Linear(35, 30), nn.ReLU(), nn.BatchNorm1d(30),
                nn.Dropout(p),
                nn.Linear(30, 25), nn.ReLU(), nn.BatchNorm1d(25),
                nn.Linear(25, 20), nn.ReLU(), nn.BatchNorm1d(20),
                nn.Linear(20, 10), nn.ReLU(), nn.BatchNorm1d(10),
                nn.Linear(10,  5), nn.ReLU(), nn.BatchNorm1d(5),
            )
            self.classifier = nn.Sequential(
                nn.Linear(5, 1), nn.Sigmoid()
            )
            self.feature.apply(self._init)

        def forward(self, x):
            return self.classifier(self.feature(x))

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LinearModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"  [NN] Loaded weights from '{weights_path}' — running inference only.")
    model.eval()

    results = []
    for X_test, y_test in splits:
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model(X_tensor).cpu().numpy().flatten()
        y_pred = (probs >= 0.5).astype(int)
        results.append(compute_metrics(y_test, y_pred, probs))
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on 10 test splits.")
    parser.add_argument("--data_dir",  default=".",          help="Directory containing CSV files")
    parser.add_argument("--weights",   default="model_weights.pt", help="Path to PyTorch weights")
    parser.add_argument("--n_splits",  type=int, default=10, help="Number of test splits")
    parser.add_argument("--test_size", type=float, default=0.5, help="Fraction of val pool per split")
    parser.add_argument("--seed",      type=int, default=0,  help="Base random seed")
    args = parser.parse_args()

    d = r"D:\Suchit\Deep-Learning\src"

    # -- Load data --
    def load_labels(path):
        """Load label CSV — handles int, float, or string-encoded numerics."""
        s = pd.read_csv(path, header=None).iloc[:, 0]
        return pd.to_numeric(s, errors="raise").values.ravel().astype(int)

    print("Loading data...")
    X_train = pd.read_csv(os.path.join(d, "xtrain.csv"), header=None).values
    y_train = load_labels(os.path.join(d, "ytrain.csv"))
    X_pool  = pd.read_csv(os.path.join(d, "xval.csv"),   header=None).values
    y_pool  = load_labels(os.path.join(d, "yval.csv"))

    # ── Build 10 test splits from val pool ───────────────────────
    print(f"Building {args.n_splits} test splits (test_size={args.test_size})…")
    splits = make_splits(X_pool, y_pool,
                         n_splits=args.n_splits,
                         test_size=args.test_size,
                         seed=args.seed)

    # ── Registry of models ───────────────────────────────────────
    # Each entry: (display_name, callable_returning_list_of_metric_dicts)
    models = {
        "Logistic Regression":  lambda: run_logistic_regression(X_train, y_train, splits),
        "Gradient Boosting":    lambda: run_gradient_boosting(X_train, y_train, splits),
        "KNN":                  lambda: run_knn(X_train, y_train, splits),
        "SVC (Linear)":         lambda: run_svc(X_train, y_train, splits),
        "Isolation Forest":     lambda: run_isolation_forest(X_train, splits),
        "Elliptic Envelope":    lambda: run_elliptic_envelope(X_train, splits),
        "Local Outlier Factor": lambda: run_local_outlier_factor(X_train, splits),
        "HBOS":                 lambda: run_hbos(X_train, splits),
        "Neural Network":       lambda: run_neural_network(splits, r"D:\Suchit\Deep-Learning\src\model_weights.pt"),
    }

    # ── Run & collect ────────────────────────────────────────────
    all_agg = {}
    for name, fn in models.items():
        print(f"\nRunning {name}…")
        try:
            results = fn()
            if results is None:
                continue
            agg = aggregate(results)
            all_agg[name] = agg
            print_report(name, agg)
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")

    # ── Save summary CSV ─────────────────────────────────────────
    rows = []
    for model_name, agg in all_agg.items():
        row = {"model": model_name}
        for metric, stats in agg.items():
            row[f"{metric}_mean"]     = round(stats["mean"],     4)
            row[f"{metric}_std"]      = round(stats["std"],      4)
            row[f"{metric}_variance"] = round(stats["variance"], 6)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = "evaluation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n\n{'═'*60}")
    print(f"  Summary saved to: {out_path}")
    print(f"{'═'*60}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()