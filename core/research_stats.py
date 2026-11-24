"""
core/research_stats.py
----------------------
Statistical validation helpers for multi-class NLP benchmarking:
  - McNemar's paired significance test on held-out classification decisions
  - Vectorized 500/1,000-resample paired bootstrap confidence intervals (Accuracy, Macro F1, Brier)
  - 10-bin confidence calibration (ECE, Multiclass Brier Score, Reliability Curve)
"""

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score


def compute_mcnemar_test(y_true, pred_a, pred_b):
    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)

    ca = pa == yt
    cb = pb == yt
    both_right = int(np.sum(ca & cb))
    a_only = int(np.sum(ca & (~cb)))
    b_only = int(np.sum((~ca) & cb))
    both_wrong = int(np.sum((~ca) & (~cb)))

    n_disc = a_only + b_only
    if n_disc == 0:
        stat, p_val, method = 0.0, 1.0, "exact_binomial"
    elif n_disc < 25:
        p_val = float(binomtest(min(a_only, b_only), n=n_disc, p=0.5, alternative="two-sided").pvalue)
        stat = float(((abs(a_only - b_only) - 1) ** 2) / n_disc)
        method = "exact_binomial"
    else:
        stat = float(((abs(a_only - b_only) - 1) ** 2) / n_disc)
        p_val = float(chi2.sf(stat, df=1))
        method = "continuity_corrected_chi2"

    return {
        "method": method,
        "contingency": {
            "both_correct": both_right,
            "control_only_correct": a_only,
            "variant_only_correct": b_only,
            "both_wrong": both_wrong,
        },
        "discordant_pairs": n_disc,
        "net_wins_for_variant": int(b_only - a_only),
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant_alpha_05": bool(p_val < 0.05),
    }


def _fast_macro_f1(y_int, pred_int, n_classes):
    cm = np.bincount(y_int * n_classes + pred_int, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1s))


def compute_multiclass_calibration(y_true, probs, labels, n_bins=10):
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_int = np.array([lbl_to_idx[y] for y in y_true], dtype=np.int32)
    probs_arr = np.asarray(probs, dtype=np.float64)
    n_classes = len(labels)

    one_hot = np.eye(n_classes, dtype=np.float64)[y_int]
    brier = float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))

    top_conf = np.max(probs_arr, axis=1)
    top_pred = np.argmax(probs_arr, axis=1)
    correct = (top_pred == y_int).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    curve = []
    n = len(y_int)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (top_conf >= lo) & (top_conf <= hi if b == n_bins - 1 else top_conf < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        conf = float(np.mean(top_conf[mask]))
        acc = float(np.mean(correct[mask]))
        ece += (cnt / n) * abs(acc - conf)
        curve.append({
            "bin": f"{int(lo*100)}-{int(hi*100)}%",
            "count": cnt,
            "mean_conf": round(conf, 4),
            "empirical_acc": round(acc, 4),
        })

    return {
        "multiclass_brier": round(brier, 5),
        "ece": round(float(ece), 5),
        "reliability_bins": curve,
    }


def compute_multiclass_bootstrap(y_true, probs_a, probs_b, labels, n_boot=600, seed=42):
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_int = np.array([lbl_to_idx[y] for y in y_true], dtype=np.int32)
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    pred_a = np.argmax(pa, axis=1)
    pred_b = np.argmax(pb, axis=1)

    n = len(y_int)
    n_classes = len(labels)
    one_hot = np.eye(n_classes, dtype=np.float64)[y_int]
    brier_vec_a = np.sum((pa - one_hot) ** 2, axis=1)
    brier_vec_b = np.sum((pb - one_hot) ** 2, axis=1)

    rng = np.random.default_rng(seed)
    acc_a_s, acc_b_s, acc_d_s = [], [], []
    f1_a_s, f1_b_s, f1_d_s = [], [], []
    br_a_s, br_b_s, br_d_s = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y_int[idx]
        pai = pred_a[idx]
        pbi = pred_b[idx]

        aa = float(np.mean(pai == yi))
        ab = float(np.mean(pbi == yi))
        acc_a_s.append(aa)
        acc_b_s.append(ab)
        acc_d_s.append(ab - aa)

        fa = _fast_macro_f1(yi, pai, n_classes)
        fb = _fast_macro_f1(yi, pbi, n_classes)
        f1_a_s.append(fa)
        f1_b_s.append(fb)
        f1_d_s.append(fb - fa)

        bra = float(np.mean(brier_vec_a[idx]))
        brb = float(np.mean(brier_vec_b[idx]))
        br_a_s.append(bra)
        br_b_s.append(brb)
        br_d_s.append(brb - bra)

    def _ci(arr, pct=True):
        mult = 100.0 if pct else 1.0
        lo, med, hi = np.percentile(arr, [2.5, 50.0, 97.5])
        return {
            "mean": round(float(np.mean(arr) * mult), 3 if pct else 4),
            "ci_low": round(float(lo * mult), 3 if pct else 4),
            "ci_high": round(float(hi * mult), 3 if pct else 4),
            "median": round(float(med * mult), 3 if pct else 4),
        }

    return {
        "n_bootstrap": n_boot,
        "accuracy_control_pct": _ci(acc_a_s, pct=True),
        "accuracy_variant_pct": _ci(acc_b_s, pct=True),
        "accuracy_delta_pct": _ci(acc_d_s, pct=True),
        "macro_f1_control_pct": _ci(f1_a_s, pct=True),
        "macro_f1_variant_pct": _ci(f1_b_s, pct=True),
        "macro_f1_delta_pct": _ci(f1_d_s, pct=True),
        "brier_control": _ci(br_a_s, pct=False),
        "brier_variant": _ci(br_b_s, pct=False),
        "brier_delta": _ci(br_d_s, pct=False),
        "prob_variant_beats_control_acc": round(float(np.mean(np.array(acc_d_s) > 0) * 100.0), 1),
        "prob_variant_beats_control_f1": round(float(np.mean(np.array(f1_d_s) > 0) * 100.0), 1),
    }
