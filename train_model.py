"""
train_model.py
--------------
Senior NLP research, multi-granularity feature extraction & scaling ablation,
Control vs Hybrid Champion Ensemble benchmarking, McNemar + Bootstrap A/B testing,
and Adversarial Negation Stress-Testing for EmotiSense AI (scikit-learn 1.8.0).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
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
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from core.features import build_hybrid_feature_union, preprocess
from core.research_stats import (
    compute_mcnemar_test,
    compute_multiclass_bootstrap,
    compute_multiclass_calibration,
)

SEED = 42
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)
DATA_FILE = (
    os.path.join(APP_DIR, "train.txt")
    if os.path.exists(os.path.join(APP_DIR, "train.txt"))
    else os.path.join(PARENT_DIR, "train.txt")
)

STRESS_SUITE = [
    {"id": "neg_relief", "text": "i did not feel angry or upset at all i actually felt relieved and grateful", "expected": "joy", "category": "Negation Flip"},
    {"id": "neg_sad_joy", "text": "i was not happy with the delay and felt deeply disappointed all evening", "expected": "sadness", "category": "Negated Positive"},
    {"id": "contrast_pivot", "text": "i was terrified before the surgery but ended up thrilled and smiling afterward", "expected": "joy", "category": "Contrast Pivot"},
    {"id": "neg_fear_calm", "text": "i am not scared anymore i feel warm caring and deeply thankful for you", "expected": "love", "category": "Negation + Affection"},
    {"id": "elongation_oov", "text": "i feel sooooo happpppy and super thankful for my friends today", "expected": "joy", "category": "Character Elongation"},
    {"id": "typo_angry", "text": "i am genuinely furious and totally irritated by that disrespectful lie", "expected": "anger", "category": "High-Arousal Anger"},
    {"id": "rare_surprise", "text": "i was completely stunned and speechless when the unexpected letter arrived", "expected": "surprise", "category": "Minority Class"},
    {"id": "rare_love", "text": "i adore how tender affectionate and compassionate my partner has been lately", "expected": "love", "category": "Affectionate Lexicon"},
    {"id": "subtle_fear", "text": "my heart is racing because i feel uneasy tense and worried about tomorrow", "expected": "fear", "category": "Somatic Anxiety"},
    {"id": "subtle_sadness", "text": "sitting in this quiet room makes me feel empty lonely and heartbroken", "expected": "sadness", "category": "Latent Grief"},
]


def extract_top_tokens(vectorizer, clf, top_n=12):
    vocab = np.array(vectorizer.get_feature_names_out())
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


def run_ablation_study(Xtr, ytr, Xte, yte, cv):
    """Compare progressive NLP feature representations & scaling choices on LinearSVC (converges cleanly & fast)."""
    svc_ab = lambda c=0.25: CalibratedClassifierCV(LinearSVC(C=c, random_state=SEED), cv=3, method="sigmoid")
    configs = [
        ("word_bow", "1. Unigram BoW (Raw Counts)", Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=2000))])),
        ("word_tfidf", "2. Unigram TF-IDF (L2 Norm)", Pipeline([("vec", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=2000))])),
        ("word_12_sub", "3. Scaled BoW(1,2) + Sublinear TF-IDF(1,2)", Pipeline([("vec", build_hybrid_feature_union(True, True, (1, 2), False, False)), ("clf", svc_ab(0.25))])),
        ("word_char", "4. Word(1,2) + Char-WB(3,5) Subwords", Pipeline([("vec", build_hybrid_feature_union(True, True, (1, 2), True, False)), ("clf", svc_ab(0.25))])),
        ("hybrid_unscaled", "5. Word+Char + AffectiveSyntax (Unscaled)", Pipeline([("vec", build_hybrid_feature_union(True, True, (1, 2), True, True, "none")), ("clf", svc_ab(0.25))])),
        ("hybrid_std", "6. Word+Char + Syntax (StandardScaler w/o mean)", Pipeline([("vec", build_hybrid_feature_union(True, True, (1, 2), True, True, "std_nomean")), ("clf", svc_ab(0.25))])),
        ("hybrid_maxabs", "7. Word+Char + Syntax (MaxAbsScaler)", Pipeline([("vec", build_hybrid_feature_union(True, True, (1, 2), True, True, "maxabs")), ("clf", svc_ab(0.25))])),
    ]
    rows = []
    for key, label, pipe in configs:
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, n_jobs=-1, scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}, error_score="raise")
        fitted = clone(pipe).fit(Xtr, ytr)
        pred = fitted.predict(Xte)
        vec = fitted.named_steps["vec"]
        n_feats = len(vec.get_feature_names_out())
        print(f"   [Ablation] {label} -> CV {scores['test_accuracy'].mean()*100:.2f}% | Test {accuracy_score(yte, pred)*100:.2f}%", flush=True)
        rows.append({
            "key": key,
            "stage": label,
            "feature_dim": int(n_feats),
            "cv_accuracy": round(float(scores["test_accuracy"].mean() * 100.0), 2),
            "cv_macro_f1": round(float(scores["test_macro_f1"].mean() * 100.0), 2),
            "test_accuracy": round(float(accuracy_score(yte, pred) * 100.0), 2),
            "test_macro_f1": round(float(f1_score(yte, pred, average="macro") * 100.0), 2),
        })
    return rows


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

    # 1. Progressive Feature Engineering & Scaling Ablation
    ablation_rows = run_ablation_study(Xtr, ytr, Xte, yte, cv)
    print("Ablation study complete:")
    for r in ablation_rows:
        print(f" - {r['stage']}: CV Acc={r['cv_accuracy']}% | MacroF1={r['cv_macro_f1']}% | Test Acc={r['test_accuracy']}% (dim={r['feature_dim']})")

    # 2. Champion & Challenger Suite (Legacy 4 + Engineered Hybrid 3)
    hybrid_union_lr = build_hybrid_feature_union(True, True, (1, 2), True, True, "maxabs")
    hybrid_union_svc = build_hybrid_feature_union(True, True, (1, 2), True, True, "maxabs")
    hybrid_union_ens = build_hybrid_feature_union(True, True, (1, 2), True, True, "maxabs")

    ens_voter = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=2.0, max_iter=800)),
            ("svc", CalibratedClassifierCV(LinearSVC(C=0.30, random_state=SEED), cv=3, method="sigmoid")),
            ("cnb", ComplementNB(alpha=0.25)),
        ],
        voting="soft",
        weights=[2, 3, 1],
    )

    model_specs = [
        ("bow_lr", "BoW + Logistic Regression", "Control Baseline", Pipeline([("vectorizer", CountVectorizer()), ("model", LogisticRegression(max_iter=2000))])),
        ("tfidf_lr", "TF-IDF + Logistic Regression", "Normalized Baseline", Pipeline([("vectorizer", TfidfVectorizer()), ("model", LogisticRegression(max_iter=2000))])),
        ("bow_nb", "BoW + Multinomial NB", "Generative Baseline", Pipeline([("vectorizer", CountVectorizer()), ("model", MultinomialNB())])),
        ("tfidf_nb", "TF-IDF + Multinomial NB", "TF-IDF Baseline", Pipeline([("vectorizer", TfidfVectorizer()), ("model", MultinomialNB())])),
        ("hybrid_lr", "Hybrid BoW+TFIDF+Char+Syntax + LR", "Challenger v3", Pipeline([("vectorizer", hybrid_union_lr), ("model", LogisticRegression(C=2.0, max_iter=800))])),
        ("hybrid_svc", "Hybrid BoW+TFIDF+Char+Syntax + Calibrated SVC", "Margin Challenger", Pipeline([("vectorizer", hybrid_union_svc), ("model", CalibratedClassifierCV(LinearSVC(C=0.25, random_state=SEED), cv=3, method="sigmoid"))])),
        ("hybrid_ensemble", "Hybrid Soft Ensemble (LR+SVC+ComplementNB)", "Variant B Champion", Pipeline([("vectorizer", hybrid_union_ens), ("model", ens_voter)])),
    ]

    benchmarks = []
    fitted = {}
    heldout_preds = {}
    heldout_probs = {}

    for key, display_name, badge, pipe in model_specs:
        cv_scores = cross_validate(pipe, Xtr, ytr, cv=cv, n_jobs=-1, scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}, error_score="raise")
        fitted_pipe = clone(pipe).fit(Xtr, ytr)
        print(f"   [Benchmark] {key} fitted (CV {cv_scores['test_accuracy'].mean()*100:.2f}%)", flush=True)
        fitted[key] = fitted_pipe

        pred = fitted_pipe.predict(Xte)
        probs = fitted_pipe.predict_proba(Xte)
        heldout_preds[key] = pred
        # Align probability columns cleanly to labels_sorted
        cls_idx = [list(fitted_pipe.classes_).index(lbl) for lbl in labels_sorted]
        probs_aligned = probs[:, cls_idx]
        heldout_probs[key] = probs_aligned

        cal = compute_multiclass_calibration(yte, probs_aligned, labels_sorted, n_bins=10)
        test_acc = float(accuracy_score(yte, pred))
        macro_f1 = float(f1_score(yte, pred, average="macro"))
        macro_prec = float(precision_score(yte, pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(yte, pred, average="macro"))

        report_dict = classification_report(yte, pred, labels=labels_sorted, output_dict=True, zero_division=0)
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
        vec = fitted_pipe.named_steps["vectorizer"]
        vocab_size = int(len(vec.vocabulary_)) if hasattr(vec, "vocabulary_") else int(len(vec.get_feature_names_out()))

        benchmarks.append({
            "key": key,
            "model": display_name,
            "badge": badge,
            "cv_accuracy": round(float(cv_scores["test_accuracy"].mean()), 6),
            "cv_sd": round(float(cv_scores["test_accuracy"].std()), 6),
            "cv_fold_accuracies": [round(float(v), 4) for v in cv_scores["test_accuracy"]],
            "cv_macro_f1": round(float(cv_scores["test_macro_f1"].mean()), 6),
            "test_accuracy": round(test_acc, 6),
            "macro_f1": round(macro_f1, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "brier_score": cal["multiclass_brier"],
            "ece": cal["ece"],
            "reliability_bins": cal["reliability_bins"],
            "vocab_size": vocab_size,
            "per_class": per_class,
            "confusion_matrix": cm,
        })

    benchmarks.sort(key=lambda r: (r["cv_accuracy"], r["cv_macro_f1"]), reverse=True)
    winner = benchmarks[0]["key"]

    # 3. Statistical A/B Tests (Control BoW+LR vs Variant B Hybrid Ensemble & Hybrid LR & Hybrid SVC)
    y_test_list = yte.tolist()
    ab_tests = {
        "bow_vs_hybrid_ensemble": {
            "control_key": "bow_lr",
            "control_name": "BoW + Logistic Regression (Control)",
            "variant_key": "hybrid_ensemble",
            "variant_name": "Hybrid Soft Ensemble (LR+SVC+CNB)",
            "mcnemar": compute_mcnemar_test(y_test_list, heldout_preds["bow_lr"], heldout_preds["hybrid_ensemble"]),
            "bootstrap": compute_multiclass_bootstrap(y_test_list, heldout_probs["bow_lr"], heldout_probs["hybrid_ensemble"], labels_sorted, n_boot=600, seed=SEED),
        },
        "bow_vs_hybrid_lr": {
            "control_key": "bow_lr",
            "control_name": "BoW + Logistic Regression (Control)",
            "variant_key": "hybrid_lr",
            "variant_name": "Hybrid Word+Char+Syntax + LR",
            "mcnemar": compute_mcnemar_test(y_test_list, heldout_preds["bow_lr"], heldout_preds["hybrid_lr"]),
            "bootstrap": compute_multiclass_bootstrap(y_test_list, heldout_probs["bow_lr"], heldout_probs["hybrid_lr"], labels_sorted, n_boot=600, seed=SEED),
        },
        "bow_vs_hybrid_svc": {
            "control_key": "bow_lr",
            "control_name": "BoW + Logistic Regression (Control)",
            "variant_key": "hybrid_svc",
            "variant_name": "Hybrid Word+Char+Syntax + Calibrated SVC",
            "mcnemar": compute_mcnemar_test(y_test_list, heldout_preds["bow_lr"], heldout_preds["hybrid_svc"]),
            "bootstrap": compute_multiclass_bootstrap(y_test_list, heldout_probs["bow_lr"], heldout_probs["hybrid_svc"], labels_sorted, n_boot=600, seed=SEED),
        },
    }

    # 4. Adversarial Negation & Typo Stress Suite Evaluation
    stress_results = []
    for item in STRESS_SUITE:
        norm_t = preprocess(item["text"])
        p_ctrl = fitted["bow_lr"].predict([norm_t])[0]
        pr_ctrl = float(np.max(fitted["bow_lr"].predict_proba([norm_t])[0]) * 100.0)
        p_var = fitted["hybrid_ensemble"].predict([norm_t])[0]
        pr_var = float(np.max(fitted["hybrid_ensemble"].predict_proba([norm_t])[0]) * 100.0)
        stress_results.append({
            "id": item["id"],
            "category": item["category"],
            "text": item["text"],
            "expected": item["expected"],
            "control_pred": str(p_ctrl),
            "control_conf": round(pr_ctrl, 1),
            "control_pass": bool(str(p_ctrl) == item["expected"]),
            "variant_pred": str(p_var),
            "variant_conf": round(pr_var, 1),
            "variant_pass": bool(str(p_var) == item["expected"]),
        })

    # 5. Persist models cleanly (both legacy artifacts + v3.0 full pipelines)
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

    joblib.dump(fitted["hybrid_lr"], os.path.join(APP_DIR, "hybrid_lr_pipeline.pkl"))
    joblib.dump(fitted["hybrid_svc"], os.path.join(APP_DIR, "hybrid_svc_pipeline.pkl"))
    joblib.dump(fitted["hybrid_ensemble"], os.path.join(APP_DIR, "hybrid_ensemble_pipeline.pkl"))

    hlr = fitted["hybrid_lr"]
    metrics_payload = {
        "version": "3.0.0",
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
        "ablation": ablation_rows,
        "ab_tests": ab_tests,
        "stress_suite": {
            "control_pass_count": sum(1 for r in stress_results if r["control_pass"]),
            "variant_pass_count": sum(1 for r in stress_results if r["variant_pass"]),
            "total": len(stress_results),
            "cases": stress_results,
        },
        "top_tokens": {
            "bow_lr": extract_top_tokens(bow_lr.named_steps["vectorizer"], bow_lr.named_steps["model"], top_n=12),
            "tfidf_lr": extract_top_tokens(tfidf_lr.named_steps["vectorizer"], tfidf_lr.named_steps["model"], top_n=12),
            "hybrid_lr": extract_top_tokens(hlr.named_steps["vectorizer"], hlr.named_steps["model"], top_n=12),
        },
    }

    with open(os.path.join(APP_DIR, "emotion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("EmotiSense v3.0 research metrics & models saved:")
    for b in benchmarks:
        print(f" - {b['key']:15s} | CV Acc={b['cv_accuracy']*100:.2f}% | CV F1={b['cv_macro_f1']*100:.2f}% | Test Acc={b['test_accuracy']*100:.2f}% | MacroF1={b['macro_f1']*100:.2f}% | Brier={b['brier_score']:.4f} | ECE={b['ece']:.4f}")


if __name__ == "__main__":
    main()
