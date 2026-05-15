#!/usr/bin/env python3
"""
app.py  –  Fake News & Misinformation Detector
===============================================
Project: Fake News Detection – Comparative Analysis
Author : Fareeha Imran | CSC-25S-014 | SMIU
Course : Artificial Intelligence, 3rd Semester

Changes from original:
  - Shows comparative metrics for ALL trained classifiers (proposal objective 3)
  - Displays LIME explanation chart for best classifier prediction
  - Shows interpretability level and computational efficiency info per classifier
  - Loads metrics.json for the comparison dashboard tab
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


import joblib
import streamlit as st

LABELS = ("REAL", "FAKE")

# ─── Text cleaning (mirrors text_clean.py) ───────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Path helpers ─────────────────────────────────────────────────────────────
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths():
    out = project_root() / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
        "metrics": out / "metrics.json",
        "charts": out / "charts",
        "lime": out / "lime_explanations",
    }


def load_artifacts(pipeline_path, model_path, vectorizer_path):
    if pipeline_path.exists():
        return joblib.load(pipeline_path), None, None
    if model_path.exists() and vectorizer_path.exists():
        return None, joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None, None


# ─── Interpretability metadata (proposal Section 7 evaluation criteria) ───────
CLASSIFIER_INFO = {
    "Logistic Regression": {
        "interpretability": "⭐⭐⭐⭐⭐ Very High",
        "efficiency": "⭐⭐⭐⭐⭐ Very Fast",
        "description": "Feature weights directly show which words drive predictions. Best for explainability.",
    },
    "Support Vector Machine": {
        "interpretability": "⭐⭐⭐ Medium",
        "efficiency": "⭐⭐⭐⭐ Fast",
        "description": "Strong accuracy on text tasks. Less interpretable than LR but supports LIME explanations.",
    },
    "Passive Aggressive": {
        "interpretability": "⭐⭐⭐ Medium",
        "efficiency": "⭐⭐⭐⭐⭐ Very Fast",
        "description": "Online learning algorithm – updates weights on each sample. Good for streaming data.",
    },
    "Random Forest": {
        "interpretability": "⭐⭐ Low",
        "efficiency": "⭐⭐ Slower",
        "description": "Ensemble of decision trees. High accuracy but less transparent; requires LIME for explanations.",
    },
}


# ─── Main app ─────────────────────────────────────────────────────────────────
def main():
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()
    metrics_path = dp["metrics"]
    charts_path = dp["charts"]
    lime_path = dp["lime"]

    # ── Page config ───────────────────────────────────────────────────
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide",
    )

    # ── Sidebar: artifact status ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📁 Model Artifacts")
        st.code(
            f"pipeline  : {pipeline_path}\n"
            f"model     : {model_path}\n"
            f"vectorizer: {vectorizer_path}"
        )
        pipeline_ok = pipeline_path.exists()
        model_ok = model_path.exists()
        vectorizer_ok = vectorizer_path.exists()
        st.write(
            f"pipeline: **{'✅' if pipeline_ok else '❌'}**  |  "
            f"model: **{'✅' if model_ok else '❌'}**  |  "
            f"vectorizer: **{'✅' if vectorizer_ok else '❌'}**"
        )
        st.markdown("---")
        st.markdown(
            "**Project:** Fake News Detection: Comparative Review  \n"
            "**Author:** Fareeha Imran (CSC-25S-014)  \n"
            "**Course:** Artificial Intelligence  \n"
            "**University:** SMIU"
        )

    # ── Load model ────────────────────────────────────────────────────
    pipe, clf, vec = load_artifacts(pipeline_path, model_path, vectorizer_path)
    if pipe is None and (clf is None or vec is None):
        st.error(
            "**Model artifacts not found.**\n\n"
            "Run training first:\n"
            "```\n"
            "python src/train_model.py --real data/True.csv --fake data/Fake.csv\n"
            "```"
        )
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detect Fake News",
        "📊 Classifier Comparison",
        "🧠 LIME Explanations",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: Detection
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        st.title("📰 Fake News & Misinformation Detector")
        st.caption(
            "TF-IDF feature extraction + comparative ML classifiers  |  "
            "Fareeha Imran, CSC-25S-014, SMIU"
        )

        txt = st.text_area(
            "Paste headline or article text:",
            height=220,
            placeholder="e.g. 'Scientists discover new vaccine that prevents all cancers...'"
        )
        threshold = st.slider(
            "FAKE decision threshold",
            min_value=0.05, max_value=0.95, value=0.50, step=0.01,
            help="Lower = catches more fake news (higher recall). Higher = fewer false alarms (higher precision)."
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            analyze = st.button("🔍 Analyze", use_container_width=True)

        if analyze and txt.strip():
            cleaned = clean_text(txt)
            if pipe is not None:
                prob_fake = float(pipe.predict_proba([cleaned])[0, 1])
            else:
                X = vec.transform([cleaned])
                prob_fake = float(clf.predict_proba(X)[0, 1])

            label = "FAKE" if prob_fake >= threshold else "REAL"

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                color = "🔴" if label == "FAKE" else "🟢"
                st.metric("Prediction", f"{color} {label}")
            with col_b:
                st.metric("Fake Probability", f"{prob_fake:.1%}")
            with col_c:
                st.metric("Threshold", f"{threshold:.2f}")

            # Probability bar
            st.progress(
                prob_fake,
                text=f"Fake probability: {prob_fake:.1%}  (threshold: {threshold:.2f})"
            )

            # Confidence interpretation
            if prob_fake >= 0.80:
                st.warning("⚠️ High confidence this is **FAKE** news.")
            elif prob_fake >= threshold:
                st.warning("⚠️ Moderately likely to be **FAKE** news.")
            elif prob_fake >= 0.35:
                st.info("ℹ️ Borderline – verify with additional sources.")
            else:
                st.success("✅ High confidence this is **REAL** news.")

            st.caption(
                "💡 Tip: Lower the threshold to catch more fake news "
                "(increases recall, may increase false positives)."
            )

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: Classifier Comparison (proposal objective 3)
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.title("📊 Comparative Analysis – All Classifiers")
        st.caption(
            "Comparing Traditional ML approaches across: Accuracy, Precision, "
            "Recall, F1-Score, ROC-AUC, and Computational Efficiency"
        )

        # Load metrics.json if available
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            comparison = metrics_data.get("comparison", {})
            best_clf = metrics_data.get("best_classifier", "N/A")

            st.success(f"✅ Best classifier: **{best_clf}** (F1 = {metrics_data.get('best_f1', 'N/A')})")

            # Metrics table
            import pandas as pd
            rows = []
            for name, m in comparison.items():
                rows.append({
                    "Classifier": name,
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1']:.4f}",
                    "ROC-AUC": f"{m['roc_auc']:.4f}",
                    "CV F1 (5-fold)": f"{m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}",
                    "Train Time": f"{m['training_time_sec']:.1f}s",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                chart = charts_path / "metrics_comparison.png"
                if chart.exists():
                    st.image(str(chart), caption="Metrics Comparison – All Classifiers",
                             use_container_width=True)
            with col2:
                chart = charts_path / "roc_all_classifiers.png"
                if chart.exists():
                    st.image(str(chart), caption="ROC Curves – All Classifiers",
                             use_container_width=True)

            chart = charts_path / "training_time_comparison.png"
            if chart.exists():
                st.image(str(chart), caption="Computational Efficiency – Training Time",
                         use_container_width=True)

        else:
            st.warning(
                "No metrics.json found. Run training first:\n"
                "```\npython src/train_model.py --real data/True.csv --fake data/Fake.csv\n```"
            )

        # Interpretability info cards (always shown)
        st.markdown("---")
        st.subheader("🔍 Interpretability & Efficiency Summary")
        cols = st.columns(2)
        for i, (name, info) in enumerate(CLASSIFIER_INFO.items()):
            with cols[i % 2]:
                with st.expander(f"**{name}**", expanded=True):
                    st.write(f"**Interpretability:** {info['interpretability']}")
                    st.write(f"**Efficiency:** {info['efficiency']}")
                    st.write(info['description'])

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: LIME Explanations (proposal mentions LIME & MAPX)
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.title("🧠 LIME Explainability")
        st.caption(
            "LIME (Local Interpretable Model-Agnostic Explanations) shows "
            "which words most influenced each prediction – addressing the "
            "interpretability objective from the project proposal."
        )

        st.info(
            "**How LIME works:** For a given prediction, LIME perturbs the input text "
            "by removing words and observes how the model's output changes. "
            "Words highlighted in **green** push toward REAL; "
            "words in **red** push toward FAKE."
        )

        if lime_path.exists():
            lime_imgs = sorted(lime_path.glob("*.png"))
            if lime_imgs:
                # Group by classifier
                by_clf = {}
                for img in lime_imgs:
                    parts = img.stem.split("_")
                    clf_name = " ".join(parts[1:-2])  # lime_<clf_name>_<label>_<idx>
                    by_clf.setdefault(clf_name, []).append(img)

                for clf_name, imgs in by_clf.items():
                    st.subheader(f"Classifier: {clf_name}")
                    cols = st.columns(len(imgs))
                    for col, img in zip(cols, imgs):
                        label = "FAKE" if "FAKE" in img.name else "REAL"
                        with col:
                            st.image(str(img),
                                     caption=f"True Label: {label}",
                                     use_container_width=True)
                    st.markdown("---")
            else:
                st.warning("No LIME explanation images found in outputs/lime_explanations/")
        else:
            st.warning(
                "LIME explanations not yet generated. Run training with the --lime flag:\n"
                "```\npython src/train_model.py --real data/True.csv --fake data/Fake.csv --lime\n```"
            )

        # LIME on live input
        st.markdown("---")
        st.subheader("🔴 Live LIME Explanation")
        live_txt = st.text_area(
            "Paste text for live LIME explanation:",
            height=150,
            key="lime_input",
            placeholder="Paste any news text here to see which words influence the prediction..."
        )
        if st.button("Generate LIME Explanation") and live_txt.strip():
            try:
                from lime.lime_text import LimeTextExplainer
                import matplotlib.pyplot as plt

                explainer = LimeTextExplainer(class_names=list(LABELS))
                cleaned = clean_text(live_txt)

                def predict_fn(texts):
                    if pipe is not None:
                        return pipe.predict_proba(texts)
                    cleaned_texts = [clean_text(t) for t in texts]
                    return clf.predict_proba(vec.transform(cleaned_texts))

                with st.spinner("Generating LIME explanation..."):
                    exp = explainer.explain_instance(
                        cleaned, predict_fn, num_features=12, labels=[0, 1]
                    )
                    fig = exp.as_pyplot_figure(label=1)
                    fig.suptitle("LIME Explanation – FAKE class", fontweight="bold")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Show top features as table
                feats = exp.as_list(label=1)
                import pandas as pd
                df_lime = pd.DataFrame(feats, columns=["Word/Phrase", "FAKE Weight"])
                df_lime["Direction"] = df_lime["FAKE Weight"].apply(
                    lambda x: "→ FAKE" if x > 0 else "→ REAL"
                )
                st.dataframe(df_lime, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"LIME error: {e}")


if __name__ == "__main__":
    main()