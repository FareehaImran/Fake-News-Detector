#!/usr/bin/env python3
"""
detect.py  –  CLI fake news detection with interpretability info
================================================================
Project: Fake News Detection – Comparative Analysis
Author : Fareeha Imran | CSC-25S-014 | SMIU

Changes from original:
  - Added --explain flag for LIME word-level explanation in the terminal
  - Shows interpretability rating alongside prediction
  - Supports listing stored comparison metrics via --show-metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from text_clean import clean_text


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path):
    if pipeline_path and Path(pipeline_path).exists():
        pipe = joblib.load(pipeline_path)
        return pipe, None, None
    if not (model_path and vectorizer_path):
        raise ValueError("Provide --pipeline OR both --model and --vectorizer.")
    clf = joblib.load(model_path)
    vec = joblib.load(vectorizer_path)
    return None, clf, vec


def get_metrics_path(pipeline_path: str | None) -> Path | None:
    if pipeline_path:
        return Path(pipeline_path).parent / "metrics.json"
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fake News Detector CLI  |  Fareeha Imran, CSC-25S-014, SMIU"
    )
    ap.add_argument("--pipeline",     help="Path to pipeline.joblib (preferred).")
    ap.add_argument("--model",        help="Path to model.joblib.")
    ap.add_argument("--vectorizer",   help="Path to vectorizer.joblib.")
    ap.add_argument("--text",         help="Headline or article text to classify.")
    ap.add_argument("--threshold",    type=float, default=0.50,
                    help="Decision threshold for FAKE (default: 0.50).")
    ap.add_argument("--explain",      action="store_true",
                    help="Show LIME word-level explanation for this prediction.")
    ap.add_argument("--show-metrics", action="store_true",
                    help="Print comparative metrics for all trained classifiers.")
    args = ap.parse_args()

    # ── Show metrics table ────────────────────────────────────────────
    if args.show_metrics:
        metrics_path = get_metrics_path(args.pipeline)
        if metrics_path and metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"\n  Best classifier : {data['best_classifier']}")
            print(f"  Best F1-Score   : {data['best_f1']}\n")
            header = f"  {'Classifier':<26} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Time':>7}"
            print(header)
            print("  " + "-" * 62)
            for name, m in data["comparison"].items():
                print(
                    f"  {name:<26} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
                    f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f} "
                    f"{m['training_time_sec']:>6.1f}s"
                )
            print()
        else:
            print("No metrics.json found. Run train_model.py first.")
        if not args.text:
            return

    # ── Require text for prediction ───────────────────────────────────
    if not args.text:
        ap.error("--text is required for prediction (or use --show-metrics).")

    pipe, clf, vec = load_pipeline_or_parts(args.pipeline, args.model, args.vectorizer)

    cleaned = clean_text(args.text)

    if pipe is not None:
        prob = float(pipe.predict_proba([cleaned])[0, 1])
    else:
        X = vec.transform([cleaned])
        prob = float(clf.predict_proba(X)[0, 1])

    label = "FAKE" if prob >= args.threshold else "REAL"

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Label     : {label:<28}│")
    print(f"  │  Fake prob : {prob:.4f} ({prob:.1%}){'':<14}│")
    print(f"  │  Threshold : {args.threshold:.2f}{'':<27}│")
    print(f"  └─────────────────────────────────────────┘")

    if prob >= 0.80:
        print("  ⚠  High confidence FAKE.")
    elif prob >= args.threshold:
        print("  ⚠  Likely FAKE – verify with additional sources.")
    elif prob >= 0.35:
        print("  ℹ  Borderline – consider fact-checking.")
    else:
        print("  ✓  High confidence REAL.")

    # ── LIME explanation (optional) ───────────────────────────────────
    if args.explain:
        try:
            from lime.lime_text import LimeTextExplainer
            print("\n  Generating LIME explanation (top 10 features)...")
            explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])

            def predict_fn(texts):
                if pipe is not None:
                    return pipe.predict_proba(texts)
                from text_clean import clean_text as ct
                return clf.predict_proba(vec.transform([ct(t) for t in texts]))

            exp  = explainer.explain_instance(cleaned, predict_fn, num_features=10, labels=[0, 1])
            feats = exp.as_list(label=1)

            print(f"\n  {'Word/Phrase':<30} {'FAKE Weight':>12}  Direction")
            print("  " + "-" * 55)
            for word, weight in sorted(feats, key=lambda x: abs(x[1]), reverse=True):
                direction = "→ FAKE" if weight > 0 else "→ REAL"
                bar = "█" * int(abs(weight) * 30)
                print(f"  {word:<30} {weight:>+12.4f}  {direction}  {bar}")
            print()
        except ImportError:
            print("  [LIME] Install lime: pip install lime --break-system-packages")
        except Exception as e:
            print(f"  [LIME] Error: {e}")


if __name__ == "__main__":
    main()
