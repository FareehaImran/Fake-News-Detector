#!/usr/bin/env python3
"""
train_model.py  –  Fake News Detection: Comparative Analysis
=============================================================
This script:
- Trains multiple ML classifiers on fake news dataset
- Evaluates performance using standard metrics
- Generates explainability outputs using LIME
- Saves trained models and evaluation results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)

# LIME for explainability 
from lime.lime_text import LimeTextExplainer

# ─── Constants ───────────────────────────────────────────────────────────────
LABELS      = ("REAL", "FAKE")
RANDOM_SEED = 42


# ─── Utilities ───────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV with UTF-8 fallback to latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def build_combined_text(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """Concatenate title + body for richer features."""
    title = df["title"].fillna("") if "title" in df.columns else pd.Series([""] * len(df))
    body  = df[text_col].fillna("") if text_col in df.columns else pd.Series([""] * len(df))
    return (title + " " + body).str.strip()


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc_all(roc_data: Dict[str, Tuple], out: Path) -> None:
    """Overlay ROC curves for all classifiers on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Classifiers", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_comparison_bar(metrics_dict: Dict[str, Dict], out: Path) -> None:
    """Grouped bar chart comparing all classifiers across key metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    classifiers = list(metrics_dict.keys())
    x = np.arange(len(metric_keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, clf_name in enumerate(classifiers):
        vals = [metrics_dict[clf_name].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=clf_name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x + width * (len(classifiers) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Classifier Comparison – All Metrics",
                                          fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_training_time(time_dict: Dict[str, float], out: Path) -> None:
    """Bar chart of training time (computational efficiency – proposal criterion 5)."""
    names = list(time_dict.keys())
    times = [time_dict[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, times, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=11)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Computational Efficiency – Training Time per Classifier",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ─── LIME Explainability ──────────────────────────────────────────────────────
def generate_lime_explanations(
    pipeline: Pipeline,
    X_test_raw: list,
    y_test: np.ndarray,
    clf_name: str,
    out_dir: Path,
    n_samples: int = 3,
) -> None:
    """
    Generate LIME explanations for n_samples predictions.
    Addresses proposal objective: explainability and interpretability comparison.
    """
    explainer = LimeTextExplainer(class_names=list(LABELS))

    # pick one correct FAKE and one correct REAL for demonstration
    indices_to_explain = []
    for label in [1, 0]:          # FAKE then REAL
        candidates = [i for i, y in enumerate(y_test) if y == label][:n_samples]
        indices_to_explain.extend(candidates[:1])

    def predict_proba_fn(texts):
        return pipeline.predict_proba(texts)

    for idx in indices_to_explain:
        text    = X_test_raw[idx]
        true_lbl = LABELS[y_test[idx]]
        exp = explainer.explain_instance(text, predict_proba_fn, num_features=10, labels=[0, 1])

        fig = exp.as_pyplot_figure(label=y_test[idx])
        fig.suptitle(
            f"LIME Explanation – {clf_name}\nTrue Label: {true_lbl}",
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout()
        safe_name = clf_name.replace(" ", "_")
        fig.savefig(out_dir / f"lime_{safe_name}_{true_lbl}_{idx}.png", dpi=150)
        plt.close(fig)
        print(f"  [LIME] Saved explanation for index {idx} (True: {true_lbl})")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fake News Detection – Comparative ML Analysis"
    )
    ap.add_argument("--real",     required=True, help="Path to True.csv")
    ap.add_argument("--fake",     required=True, help="Path to Fake.csv")
    ap.add_argument("--text-col", default="text", help="Text column name (default: text)")
    ap.add_argument("--outdir",   default="outputs", help="Output directory")
    ap.add_argument("--lime",     action="store_true", default=True,
                    help="Generate LIME explanations (default: True)")
    ap.add_argument("--no-lime",  dest="lime", action="store_false",
                    help="Skip LIME explanations (faster)")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")
    lime_dir = ensure_dir(outdir / "lime_explanations")

    print("=" * 65)
    print("  Fake News Detection – Comparative Analysis")
    print("  Comparative Machine Learning Training Pipeline")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1/6] Loading datasets...")
    df_real = read_csv_safe(Path(a.real))
    df_fake = read_csv_safe(Path(a.fake))
    print(f"  Real news articles : {len(df_real):,}")
    print(f"  Fake news articles : {len(df_fake):,}")

    # ── 2. Build combined text ────────────────────────────────────────
    df_real["combined_text"] = build_combined_text(df_real, a.text_col)
    df_fake["combined_text"] = build_combined_text(df_fake, a.text_col)

    X_raw = pd.concat(
        [df_real["combined_text"], df_fake["combined_text"]], ignore_index=True
    ).tolist()
    y = np.array([0] * len(df_real) + [1] * len(df_fake))  # 0=REAL, 1=FAKE

    # ── 3. Train / test split (80/20 stratified) ──────────────────────
    print("\n[2/6] Splitting dataset (80% train / 20% test, stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

    # ── 4. Shared TF-IDF vectorizer ──────────────────────────────────
    print("\n[3/6] Fitting TF-IDF vectorizer (1–3 grams, max 20k features)...")
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.80,
        min_df=3,
        max_features=20_000,
    )
    X_train_vec = tfidf.fit_transform(X_train_raw)
    X_test_vec  = tfidf.transform(X_test_raw)
    print(f"  Vocabulary size : {len(tfidf.vocabulary_):,} features")

    # ── 5. Define classifiers (proposal comparison table) ─────────────
    # LinearSVC doesn't support predict_proba natively → wrap with CalibratedClassifierCV
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED)
        ),
        "Passive Aggressive": CalibratedClassifierCV(
            SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1',
                          eta0=1.0, max_iter=1000, random_state=RANDOM_SEED)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_SEED
        ),
    }

    # ── 6. Train, evaluate, and compare all classifiers ───────────────
    print("\n[4/6] Training and evaluating all classifiers...")
    print("-" * 65)

    all_metrics   = {}
    roc_data      = {}
    training_times= {}
    best_pipeline = None
    best_f1       = -1.0
    best_name     = ""
    cv            = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, clf in classifiers.items():
        print(f"\n  ▶ {name}")

        # Build pipeline for cross-validation (vectorizer + classifier)
        pipe_cv = Pipeline([("tfidf", tfidf), ("clf", clf)])

        # 5-fold cross-validation F1 (on training data)
        cv_f1 = cross_val_score(pipe_cv, X_train_raw, y_train,
                                cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"    CV F1 (5-fold)  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # Fit on training split and measure time
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        elapsed = time.time() - t0
        training_times[name] = elapsed
        print(f"    Training time   : {elapsed:.2f}s")

        # Predict on held-out test set
        y_prob = clf.predict_proba(X_test_vec)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        ap   = average_precision_score(y_test, y_prob)

        print(f"    Accuracy        : {acc:.4f}")
        print(f"    Precision       : {prec:.4f}")
        print(f"    Recall          : {rec:.4f}")
        print(f"    F1-Score        : {f1:.4f}")
        print(f"    ROC-AUC         : {auc:.4f}")
        print(f"    Avg Precision   : {ap:.4f}")

        # Store metrics
        all_metrics[name] = {
            "accuracy"      : round(acc,  4),
            "precision"     : round(prec, 4),
            "recall"        : round(rec,  4),
            "f1"            : round(f1,   4),
            "roc_auc"       : round(auc,  4),
            "avg_precision" : round(ap,   4),
            "cv_f1_mean"    : round(float(cv_f1.mean()), 4),
            "cv_f1_std"     : round(float(cv_f1.std()),  4),
            "training_time_sec": round(elapsed, 2),
        }

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        # Individual confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        safe = name.replace(" ", "_")
        plot_confusion_matrix(cm, f"Confusion Matrix – {name}",
                              charts / f"cm_{safe}.png")

        # Track best model (by F1)
        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            # Build a full pipeline for the best model (used for LIME + saving)
            best_pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
            best_pipeline.fit(X_train_raw, y_train)   # refit on raw text

    print(f"\n  ★ Best classifier by F1: {best_name} ({best_f1:.4f})")

    # ── 7. Save comparison charts ──────────────────────────────────────
    print("\n[5/6] Saving comparison charts...")
    plot_roc_all(roc_data,         charts / "roc_all_classifiers.png")
    plot_comparison_bar(all_metrics, charts / "metrics_comparison.png")
    plot_training_time(training_times, charts / "training_time_comparison.png")
    print("  Saved: roc_all_classifiers.png")
    print("  Saved: metrics_comparison.png")
    print("  Saved: training_time_comparison.png")

    # ── 8. LIME Explanations ──────────────────────────────────────────
    if a.lime:
        print("\n[6/6] Generating LIME explanations for best classifier...")
        print(f"  Classifier: {best_name}")
        try:
            generate_lime_explanations(
                pipeline   = best_pipeline,
                X_test_raw = X_test_raw,
                y_test     = y_test,
                clf_name   = best_name,
                out_dir    = lime_dir,
                n_samples  = 2,
            )
            # Also generate LIME for Logistic Regression (most interpretable)
            if best_name != "Logistic Regression":
                lr_pipeline = Pipeline([
                    ("tfidf", tfidf),
                    ("clf", classifiers["Logistic Regression"])
                ])
                lr_pipeline.fit(X_train_raw, y_train)
                print("  Also generating LIME for Logistic Regression (most interpretable)...")
                generate_lime_explanations(
                    pipeline   = lr_pipeline,
                    X_test_raw = X_test_raw,
                    y_test     = y_test,
                    clf_name   = "Logistic Regression",
                    out_dir    = lime_dir,
                    n_samples  = 2,
                )
        except Exception as e:
            print(f"  [LIME] Warning: {e}")
    else:
        print("\n[6/6] LIME explanations skipped (--no-lime flag set).")

    # ── 9. Save artifacts ──────────────────────────────────────────────
    print("\n  Saving model artifacts...")
    joblib.dump(best_pipeline, outdir / "pipeline.joblib")
    joblib.dump(tfidf,         outdir / "vectorizer.joblib")
    joblib.dump(best_pipeline.named_steps["clf"], outdir / "model.joblib")

    # Save full comparison results
    results = {
        "best_classifier" : best_name,
        "best_f1"         : round(best_f1, 4),
        "comparison"      : all_metrics,
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # ── 10. Print summary table ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARATIVE RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Classifier':<26} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Time':>7}"
    print(header)
    print("-" * 65)
    for name, m in all_metrics.items():
        print(
            f"{name:<26} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f} "
            f"{m['training_time_sec']:>6.1f}s"
        )
    print("=" * 65)
    print(f"\n  ★ Best model: {best_name} saved to outputs/pipeline.joblib")
    print(f"  ★ Full metrics: outputs/metrics.json")
    print(f"  ★ Charts: outputs/charts/")
    print(f"  ★ LIME explanations: outputs/lime_explanations/")
    print("\n  Tip: Use a threshold < 0.50 to increase FAKE recall.")
    print("=" * 65)


if __name__ == "__main__":
    main()
