"""
train_model.py
--------------
Reproducible training, 5-fold CV benchmarking, and NLP lexical research export
for EmotiSense AI (scikit-learn 1.8.0), aligned with Emotion_Heart_Validation.ipynb.
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


def extract_top_tokens(pipe, labels_sorted, top_n=12):
    vec = pipe.named_steps["vectorizer"]
    clf = pipe.named_steps["model"]
    vocab = np.array(vec.get_feature_names_out())
    result = {}
    if hasattr(clf, "coef_"):
        for idx, lbl in enumerate(clf.classes_):
            coefs = clf.coef_[idx]
            pos_idx = np.argsort(coefs)[::-1][:top_n]
            neg_idx = np.argsort(coefs)[:top_n]
            result[str(lbl)] = {
                "positive": [{"word": str(vocab[i]), "weight": round(float(coefs[i]), 3)} for i in pos_idx],
                "negative": [{"word": str(vocab[i]), "weight": round(float(coefs[i]), 3)} for i in neg_idx],
            }
    return result


def main():
    print(f"scikit-learn version: {sklearn.__version__}")
    raw = pd.read_csv(DATA_FILE, sep=";", header=None, names=["text", "emotion"])
    raw_rows = len(raw)

    df = raw.dropna().copy()
    df["text"] = df["text"].map(preprocess)
    df = df[df["text"].str.len() > 0].copy()

    conflict_counts = df.groupby("text")["emotion"].nunique()
    conflict_texts = conflict_counts[conflict_counts > 1].index
    n_conflict = int(df["text"].isin(conflict_texts).sum())
    df = df[~df["text"].isin(conflict_texts)].copy()

    n_dup = int(df.duplicated(subset="text").sum())
    df = df.drop_duplicates(subset="text").copy()
    retained_rows = len(df)

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
                "cv_fold_accuracies": [round(float(v), 4) for v in cv_scores["test_accuracy"]],
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

    benchmarks.sort(key=lambda r: r["cv_accuracy"], reverse=True)
    winner = benchmarks[0]["key"]

    bow_lr = fitted["bow_lr"]
    joblib.dump(bow_lr.named_steps["model"], os.path.join(APP_DIR, "bow_model.pkl"))
    joblib.dump(bow_lr.named_steps["vectorizer"], os.path.join(APP_DIR, "bow_vectorizer.pkl"))

    tfidf_lr = fitted["tfidf_lr"]
    joblib.dump(tfidf_lr.named_steps["model"], os.path.join(APP_DIR, "model.pkl"))
    joblib.dump(tfidf_lr.named_steps["vectorizer"], os.path.join(APP_DIR, "vectorizer.pkl"))

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
        "top_tokens": {
            "bow_lr": extract_top_tokens(bow_lr, labels_sorted, top_n=12),
            "tfidf_lr": extract_top_tokens(tfidf_lr, labels_sorted, top_n=12),
        },
    }

    with open(os.path.join(APP_DIR, "emotion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("EmotiSense v2.1 research metrics saved successfully.")


if __name__ == "__main__":
    main()
