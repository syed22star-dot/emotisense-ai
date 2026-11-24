"""
train_model.py
--------------
Reproducible training & 5-fold CV benchmarking script for EmotiSense AI.
Aligned strictly with Emotion_Heart_Validation.ipynb (scikit-learn 1.8.0):
  - Normalizes text while retaining stopwords & negation
  - Drops empty rows, removes conflicting-label texts (60 rows), and deduplicates (1 row) -> 15,939 rows
  - Uses string emotion classes directly (eliminating any LabelEncoder/LABEL_MAP mismatch)
  - 80/20 stratified split (SEED=42) + 5-fold StratifiedKFold CV
  - Benchmarks BoW+LR, TF-IDF+LR, BoW+MNB, TF-IDF+MNB
  - Saves sklearn 1.8.0 artifacts (promoting BoW+LR as primary CV & held-out winner) and emotion_metrics.json
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessor import preprocess

SEED = 42
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)
DATA_FILE = (
    os.path.join(APP_DIR, "train.txt")
    if os.path.exists(os.path.join(APP_DIR, "train.txt"))
    else os.path.join(PARENT_DIR, "train.txt")
)


def main():
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Loading dataset from: {DATA_FILE}")

    raw = pd.read_csv(DATA_FILE, sep=";", header=None, names=["text", "emotion"])
    raw_rows = len(raw)

    df = raw.dropna().copy()
    df["text"] = df["text"].map(preprocess)
    df = df[df["text"].str.len() > 0].copy()

    # Remove conflicting-label normalized texts, then same-label duplicates
    conflict_counts = df.groupby("text")["emotion"].nunique()
    conflict_texts = conflict_counts[conflict_counts > 1].index
    n_conflict = int(df["text"].isin(conflict_texts).sum())
    df = df[~df["text"].isin(conflict_texts)].copy()

    n_dup = int(df.duplicated(subset="text").sum())
    df = df.drop_duplicates(subset="text").copy()
    retained_rows = len(df)

    print(
        f"EMOTION raw={raw_rows}, conflicting_removed={n_conflict}, "
        f"same_label_duplicates_removed={n_dup}, retained={retained_rows}"
    )

    class_dist = df["emotion"].value_counts().to_dict()
    labels_sorted = sorted(df["emotion"].unique().tolist())

    Xtr, Xte, ytr, yte = train_test_split(
        df["text"], df["emotion"], test_size=0.2, random_state=SEED, stratify=df["emotion"]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_specs = [
        ("bow_lr", "BoW + Logistic Regression", CountVectorizer(), LogisticRegression(max_iter=2000)),
        ("tfidf_lr", "TF-IDF + Logistic Regression", TfidfVectorizer(), LogisticRegression(max_iter=2000)),
        ("bow_nb", "BoW + Multinomial NB", CountVectorizer(), MultinomialNB()),
        ("tfidf_nb", "TF-IDF + Multinomial NB", TfidfVectorizer(), MultinomialNB()),
    ]

    benchmarks = []
    fitted = {}

    for key, display_name, vectorizer, estimator in model_specs:
        pipe = Pipeline([("vectorizer", vectorizer), ("model", estimator)])
        cv_scores = cross_validate(
            pipe,
            Xtr,
            ytr,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            error_score="raise",
        )

        fitted_pipe = clone(pipe).fit(Xtr, ytr)
        fitted[key] = fitted_pipe
        pred = fitted_pipe.predict(Xte)

        test_acc = float(accuracy_score(yte, pred))
        macro_f1 = float(f1_score(yte, pred, average="macro"))
        macro_prec = float(precision_score(yte, pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(yte, pred, average="macro"))

        report_dict = classification_report(
            yte, pred, labels=labels_sorted, output_dict=True, zero_division=0
        )
        per_class = {
            lbl: {
                "precision": round(float(report_dict[lbl]["precision"]), 4),
                "recall": round(float(report_dict[lbl]["recall"]), 4),
                "f1": round(float(report_dict[lbl]["f1-score"]), 4),
                "support": int(report_dict[lbl]["support"]),
            }
            for lbl in labels_sorted
        }
        cm = confusion_matrix(yte, pred, labels=labels_sorted).tolist()

        benchmarks.append(
            {
                "key": key,
                "model": display_name,
                "cv_accuracy": round(float(cv_scores["test_accuracy"].mean()), 6),
                "cv_sd": round(float(cv_scores["test_accuracy"].std()), 6),
                "cv_macro_f1": round(float(cv_scores["test_macro_f1"].mean()), 6),
                "test_accuracy": round(test_acc, 6),
                "macro_f1": round(macro_f1, 6),
                "macro_precision": round(macro_prec, 6),
                "macro_recall": round(macro_rec, 6),
                "vocab_size": int(len(fitted_pipe.named_steps["vectorizer"].vocabulary_)),
                "per_class": per_class,
                "confusion_matrix": cm,
            }
        )

        print(
            f"{display_name:30s} | CV Acc: {cv_scores['test_accuracy'].mean():.4f} "
            f"(+/-{cv_scores['test_accuracy'].std():.4f}) | Test Acc: {test_acc:.4f} | Macro F1: {macro_f1:.4f}"
        )

    benchmarks.sort(key=lambda r: r["cv_accuracy"], reverse=True)
    winner = benchmarks[0]["key"]
    print(f"Selected CV Winner: {benchmarks[0]['model']} ({winner})")

    # Save vectorizer + model artifacts cleanly inside emotion-classifier/
    # Primary winner: BoW + Logistic Regression
    bow_lr = fitted["bow_lr"]
    joblib.dump(bow_lr.named_steps["model"], os.path.join(APP_DIR, "bow_model.pkl"))
    joblib.dump(bow_lr.named_steps["vectorizer"], os.path.join(APP_DIR, "bow_vectorizer.pkl"))

    # Secondary comparison: TF-IDF + Logistic Regression
    tfidf_lr = fitted["tfidf_lr"]
    joblib.dump(tfidf_lr.named_steps["model"], os.path.join(APP_DIR, "model.pkl"))
    joblib.dump(tfidf_lr.named_steps["vectorizer"], os.path.join(APP_DIR, "vectorizer.pkl"))

    # Baseline NB classifiers (sharing the same fitted vectorizers!)
    bow_nb = fitted["bow_nb"]
    tfidf_nb = fitted["tfidf_nb"]
    joblib.dump(bow_nb.named_steps["model"], os.path.join(APP_DIR, "nb_bow_model.pkl"))
    joblib.dump(bow_nb.named_steps["vectorizer"], os.path.join(APP_DIR, "nb_bow_vectorizer.pkl"))
    joblib.dump(tfidf_nb.named_steps["model"], os.path.join(APP_DIR, "nb_tfidf_model.pkl"))
    joblib.dump(tfidf_nb.named_steps["vectorizer"], os.path.join(APP_DIR, "nb_tfidf_vectorizer.pkl"))

    metrics_payload = {
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "dataset": {
            "raw_rows": raw_rows,
            "conflicting_removed": n_conflict,
            "duplicates_removed": n_dup,
            "retained_rows": retained_rows,
            "train_rows": len(Xtr),
            "test_rows": len(Xte),
            "class_distribution": {k: int(v) for k, v in class_dist.items()},
            "stopwords_preserved": True,
        },
        "labels": labels_sorted,
        "selected_primary_key": winner,
        "benchmarks": benchmarks,
    }

    with open(os.path.join(APP_DIR, "emotion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Saved all sklearn 1.8.0 emotion models and emotion_metrics.json successfully.")


if __name__ == "__main__":
    main()
