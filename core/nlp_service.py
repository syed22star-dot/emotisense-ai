"""
core/nlp_service.py
-------------------
Multi-model NLP inference service for EmotiSense AI v3.0:
  - 4 legacy models (bow_lr, tfidf_lr, bow_nb, tfidf_nb)
  - 3 engineered hybrid models (hybrid_lr, hybrid_svc, hybrid_ensemble)
  - Per-token linear attribution + Affective Syntax feature telemetry + Live A/B Comparator
"""

import csv
import io
import json
import os
import re
from collections import Counter
import joblib
import numpy as np

from core.features import analyze_syntax_signals, preprocess


EMOTION_META = {
    "anger": {"emoji": "😠", "color": "#dc2626"},
    "fear": {"emoji": "😨", "color": "#7c3aed"},
    "joy": {"emoji": "😄", "color": "#d97706"},
    "love": {"emoji": "❤️", "color": "#db2777"},
    "sadness": {"emoji": "😢", "color": "#2563eb"},
    "surprise": {"emoji": "😲", "color": "#0d9488"},
}

LEGACY_SPECS = {
    "bow_lr": {"name": "BoW + Logistic Regression", "model_file": "bow_model.pkl", "vec_file": "bow_vectorizer.pkl", "badge": "Control Baseline"},
    "tfidf_lr": {"name": "TF-IDF + Logistic Regression", "model_file": "model.pkl", "vec_file": "vectorizer.pkl", "badge": "Normalized Baseline"},
    "bow_nb": {"name": "BoW + Multinomial NB", "model_file": "nb_bow_model.pkl", "vec_file": "nb_bow_vectorizer.pkl", "badge": "Probabilistic Baseline"},
    "tfidf_nb": {"name": "TF-IDF + Multinomial NB", "model_file": "nb_tfidf_model.pkl", "vec_file": "nb_tfidf_vectorizer.pkl", "badge": "TF-IDF Baseline"},
}

PIPELINE_SPECS = {
    "hybrid_lr": {"name": "Hybrid Word+Char+Syntax + LR", "pipe_file": "hybrid_lr_pipeline.pkl", "badge": "Engineered Linear"},
    "hybrid_svc": {"name": "Hybrid Word+Char+Syntax + Calibrated SVC", "pipe_file": "hybrid_svc_pipeline.pkl", "badge": "Margin Challenger"},
    "hybrid_ensemble": {"name": "Hybrid Soft Ensemble (LR+SVC+CNB)", "pipe_file": "hybrid_ensemble_pipeline.pkl", "badge": "Variant B Champion"},
}


class EmotionNLPService:
    """Encapsulates EmotiSense AI v3.0 models, XAI extraction, A/B comparator, and batch screening."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.metrics_path = os.path.join(base_dir, "emotion_metrics.json")
        self.loaded_models = {}
        self.metrics_cache = {}
        self.reload()

    def reload(self):
        self.loaded_models = {}
        self.metrics_cache = {}
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics_cache = json.load(f)
            except Exception as exc:
                print(f"[EmotionNLPService] Warning reading emotion_metrics.json: {exc}")

        for key, cfg in LEGACY_SPECS.items():
            m_path = os.path.join(self.base_dir, cfg["model_file"])
            v_path = os.path.join(self.base_dir, cfg["vec_file"])
            if os.path.exists(m_path) and os.path.exists(v_path):
                try:
                    clf = joblib.load(m_path)
                    vec = joblib.load(v_path)
                    feat_names = vec.get_feature_names_out()
                    feat_to_idx = {str(name): idx for idx, name in enumerate(feat_names)}
                    bench = next((b for b in self.metrics_cache.get("benchmarks", []) if b["key"] == key), {})
                    self.loaded_models[key] = {
                        "key": key,
                        "kind": "pair",
                        "name": cfg["name"],
                        "badge": cfg["badge"],
                        "model": clf,
                        "vectorizer": vec,
                        "classes": [str(c) for c in clf.classes_],
                        "feat_to_idx": feat_to_idx,
                        "vocab_size": len(feat_names),
                        "cv_accuracy": bench.get("cv_accuracy"),
                        "test_accuracy": bench.get("test_accuracy"),
                        "macro_f1": bench.get("macro_f1"),
                        "brier_score": bench.get("brier_score"),
                    }
                except Exception as exc:
                    print(f"[EmotionNLPService] Warning loading {key}: {exc}")

        for key, cfg in PIPELINE_SPECS.items():
            p_path = os.path.join(self.base_dir, cfg["pipe_file"])
            if os.path.exists(p_path):
                try:
                    pipe = joblib.load(p_path)
                    vec = pipe.named_steps["vectorizer"]
                    clf = pipe.named_steps["model"]
                    feat_names = vec.get_feature_names_out()
                    feat_to_idx = {str(name): idx for idx, name in enumerate(feat_names)}
                    bench = next((b for b in self.metrics_cache.get("benchmarks", []) if b["key"] == key), {})
                    self.loaded_models[key] = {
                        "key": key,
                        "kind": "pipeline",
                        "pipeline": pipe,
                        "name": cfg["name"],
                        "badge": cfg["badge"],
                        "model": clf,
                        "vectorizer": vec,
                        "classes": [str(c) for c in clf.classes_],
                        "feat_to_idx": feat_to_idx,
                        "vocab_size": len(feat_names),
                        "cv_accuracy": bench.get("cv_accuracy"),
                        "test_accuracy": bench.get("test_accuracy"),
                        "macro_f1": bench.get("macro_f1"),
                        "brier_score": bench.get("brier_score"),
                    }
                except Exception as exc:
                    print(f"[EmotionNLPService] Warning loading {key}: {exc}")

    @property
    def is_ready(self) -> bool:
        return bool(self.loaded_models)

    def get_primary_key(self) -> str:
        k = self.metrics_cache.get("selected_primary_key", "hybrid_ensemble")
        if k in self.loaded_models:
            return k
        return "hybrid_ensemble" if "hybrid_ensemble" in self.loaded_models else ("bow_lr" if "bow_lr" in self.loaded_models else next(iter(self.loaded_models), "bow_lr"))

    def _predict_probs(self, entry: dict, clean_text: str):
        if entry["kind"] == "pipeline":
            return entry["pipeline"].predict_proba([clean_text])[0]
        X_vec = entry["vectorizer"].transform([clean_text])
        return entry["model"].predict_proba(X_vec)[0]

    def _extract_word_attribution(self, entry: dict, raw_text: str, clean_text: str, pred_label: str):
        classes = entry["classes"]
        if pred_label not in classes:
            return [], [], []
        cls_idx = classes.index(pred_label)

        clf = entry["model"]
        feat_to_idx = entry["feat_to_idx"]
        prefix = "word__" if entry["kind"] == "pipeline" else ""

        coef_vec = None
        if hasattr(clf, "coef_"):
            coef_vec = clf.coef_[cls_idx]
        elif hasattr(clf, "estimators_") and clf.estimators_:
            inner = clf.estimators_[0]
            if hasattr(inner, "coef_"):
                coef_vec = inner.coef_[cls_idx]

        tokens_seen = set(clean_text.split())
        bigrams = [f"{a} {b}" for a, b in zip(clean_text.split(), clean_text.split()[1:])]
        candidates = list(tokens_seen) + bigrams

        scored_tokens = {}
        X_vec = entry["vectorizer"].transform([clean_text])
        prefixes = ["word__", "bow__"] if entry["kind"] == "pipeline" else [""]
        if coef_vec is not None:
            for tok in candidates:
                total_w = 0.0
                matched = False
                for pfx in prefixes:
                    idx = feat_to_idx.get(pfx + tok)
                    if idx is not None and X_vec[0, idx] > 0:
                        total_w += float(coef_vec[idx] * X_vec[0, idx])
                        matched = True
                if matched:
                    scored_tokens[tok] = round(total_w, 4)
        elif hasattr(clf, "feature_log_prob_"):
            target_lp = clf.feature_log_prob_[cls_idx]
            other_lp = np.delete(clf.feature_log_prob_, cls_idx, axis=0).mean(axis=0)
            llr = target_lp - other_lp
            for tok in tokens_seen:
                idx = feat_to_idx.get(prefix + tok)
                if idx is not None and X_vec[0, idx] > 0:
                    scored_tokens[tok] = round(float(llr[idx]), 4)

        unigram_scores = {k: v for k, v in scored_tokens.items() if " " not in k}
        top_pos = sorted(
            [{"word": k, "score": v} for k, v in scored_tokens.items() if v > 0.01],
            key=lambda x: x["score"],
            reverse=True,
        )[:8]
        top_neg = sorted(
            [{"word": k, "score": v} for k, v in scored_tokens.items() if v < -0.01],
            key=lambda x: x["score"],
        )[:6]

        sentence_tokens = []
        for raw_tok in re.findall(r"\S+", raw_text):
            norm = preprocess(raw_tok)
            norm_parts = norm.split()
            w = sum(unigram_scores.get(np_tok, 0.0) for np_tok in norm_parts)
            impact = "positive" if w > 0.02 else ("negative" if w < -0.02 else "oov")
            sentence_tokens.append({
                "word": raw_tok,
                "normalized": norm,
                "score": round(w, 3),
                "impact": impact,
            })
        return top_pos, top_neg, sentence_tokens

    def analyze_single(self, raw_text: str, model_key=None, compare_all=True) -> dict:
        if not raw_text or not str(raw_text).strip():
            raise ValueError("Please enter text to analyze.")
        clean_text = preprocess(raw_text)
        if not clean_text:
            raise ValueError("Text contains no recognizable alphabetic words after cleaning.")

        chosen = str(model_key or self.get_primary_key())
        if chosen not in self.loaded_models:
            chosen = self.get_primary_key()

        entry = self.loaded_models[chosen]
        probs_arr = self._predict_probs(entry, clean_text)
        classes = entry["classes"]
        probs_map = {lbl: round(float(probs_arr[i]) * 100.0, 2) for i, lbl in enumerate(classes)}
        sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)
        pred_label, pred_pct = sorted_probs[0]
        runner_label, runner_pct = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)

        top_pos, top_neg, sentence_tokens = self._extract_word_attribution(entry, raw_text, clean_text, pred_label)
        syntax_diag = analyze_syntax_signals(raw_text)
        test_acc_str = f"{round(float(entry['test_accuracy']) * 100.0, 2)}%" if entry.get("test_accuracy") else "N/A"

        arena_obj = None
        arena_list = []
        if compare_all:
            for k, m_entry in self.loaded_models.items():
                m_probs = self._predict_probs(m_entry, clean_text)
                m_classes = m_entry["classes"]
                m_top_idx = int(np.argmax(m_probs))
                m_lbl = m_classes[m_top_idx]
                meta_lbl = EMOTION_META.get(m_lbl, {"emoji": "🎭", "color": "#6d28d9"})
                arena_list.append({
                    "key": k,
                    "name": m_entry["name"],
                    "badge": m_entry["badge"],
                    "emotion": m_lbl,
                    "emoji": meta_lbl["emoji"],
                    "color": meta_lbl["color"],
                    "confidence": round(float(m_probs[m_top_idx]) * 100.0, 2),
                    "cv_accuracy": round(float(m_entry["cv_accuracy"]) * 100.0, 2) if m_entry.get("cv_accuracy") else None,
                    "test_accuracy": round(float(m_entry["test_accuracy"]) * 100.0, 2) if m_entry.get("test_accuracy") else None,
                })
            arena_list.sort(key=lambda a: (a.get("test_accuracy") or 0.0), reverse=True)
            top_emo, top_cnt = Counter([a["emotion"] for a in arena_list]).most_common(1)[0]
            arena_obj = {
                "models": arena_list,
                "consensus_emotion": top_emo,
                "consensus_count": top_cnt,
                "total_models": len(arena_list),
                "unanimous": top_cnt == len(arena_list),
            }

        ctrl_item = next((a for a in arena_list if a["key"] == "bow_lr"), None)
        var_item = next((a for a in arena_list if a["key"] == "hybrid_ensemble"), None) or next((a for a in arena_list if a["key"] == "hybrid_lr"), None)

        return {
            "original_text": raw_text,
            "cleaned_text": clean_text,
            "model_key": chosen,
            "model_name": entry["name"],
            "model_badge": entry["badge"],
            "accuracy": test_acc_str,
            "emotion": pred_label,
            "emoji": EMOTION_META.get(pred_label, {}).get("emoji", "🎭"),
            "color": EMOTION_META.get(pred_label, {}).get("color", "#7c3aed"),
            "confidence": pred_pct,
            "runner_up": {"emotion": runner_label, "confidence": runner_pct},
            "margin_pct": round(pred_pct - runner_pct, 2),
            "all_probabilities": probs_map,
            "probabilities": probs_map,
            "syntax_signals": {k: v for k, v in syntax_diag.items() if k != "vector"},
            "token_importance": top_pos,
            "opposing_tokens": top_neg,
            "top_positive_words": top_pos,
            "top_negative_words": top_neg,
            "sentence_tokens": sentence_tokens,
            "arena": arena_obj,
            "arena_consensus": f"{arena_obj['consensus_count']}/{arena_obj['total_models']} Agree" if arena_obj else "",
            "ab_live": {
                "control": ctrl_item,
                "variant": var_item,
                "decision_flip": bool(ctrl_item and var_item and ctrl_item["emotion"] != var_item["emotion"]),
            },
        }

    def run_ab_compare(self, raw_text: str, control_key="bow_lr", variant_key="hybrid_ensemble") -> dict:
        c_key = control_key if control_key in self.loaded_models else "bow_lr"
        v_key = variant_key if variant_key in self.loaded_models else self.get_primary_key()
        ctrl_res = self.analyze_single(raw_text, model_key=c_key, compare_all=True)
        var_res = self.analyze_single(raw_text, model_key=v_key, compare_all=False)
        stat_key = "bow_vs_hybrid_ensemble" if v_key == "hybrid_ensemble" else ("bow_vs_hybrid_lr" if v_key == "hybrid_lr" else "bow_vs_hybrid_svc")
        return {
            "text": raw_text,
            "control": ctrl_res,
            "variant": var_res,
            "decision_flip": bool(ctrl_res["emotion"] != var_res["emotion"]),
            "confidence_delta": round(var_res["confidence"] - ctrl_res["confidence"], 2),
            "syntax_signals": ctrl_res["syntax_signals"],
            "cohort_statistics": (self.metrics_cache.get("ab_tests") or {}).get(stat_key, {}),
        }

    def run_batch_lines(self, entries: list) -> dict:
        primary_key = self.get_primary_key()
        ctrl_entry = self.loaded_models.get("bow_lr") or self.loaded_models[primary_key]
        var_entry = self.loaded_models[primary_key]

        dist = Counter()
        rows = []
        agree_cnt = 0
        flip_cnt = 0
        word_counts = Counter()

        for raw in entries[:2000]:
            clean = preprocess(raw)
            if not clean:
                continue
            p_var = self._predict_probs(var_entry, clean)
            idx_v = int(np.argmax(p_var))
            emo_v = var_entry["classes"][idx_v]
            conf_v = round(float(p_var[idx_v]) * 100.0, 1)

            p_ctrl = self._predict_probs(ctrl_entry, clean)
            idx_c = int(np.argmax(p_ctrl))
            emo_c = ctrl_entry["classes"][idx_c]

            agree = emo_v == emo_c
            if agree:
                agree_cnt += 1
            else:
                flip_cnt += 1
            dist[emo_v] += 1

            pos_t, _, _ = self._extract_word_attribution(var_entry, raw, clean, emo_v)
            for item in pos_t[:3]:
                word_counts[item["word"]] += 1

            rows.append({
                "row_id": len(rows) + 1,
                "text": raw[:140] + ("..." if len(raw) > 140 else ""),
                "emotion": emo_v,
                "emoji": EMOTION_META.get(emo_v, {}).get("emoji", "🎭"),
                "confidence": conf_v,
                "control_emotion": emo_c,
                "tfidf_emotion": emo_c,
                "models_agree": agree,
                "top_triggers": ", ".join(item["word"] for item in pos_t[:3]),
            })

        total = len(rows)
        if total == 0:
            raise ValueError("No valid sentences found in batch input.")
        dominant_emo, dominant_cnt = dist.most_common(1)[0]

        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["row_id", "text", "emotion", "confidence", "control_emotion", "models_agree", "top_triggers"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in ["row_id", "text", "emotion", "confidence", "control_emotion", "models_agree", "top_triggers"]})

        return {
            "summary": {
                "total_entries": total,
                "primary_model": var_entry["name"],
                "control_model": ctrl_entry["name"],
                "dominant_emotion": dominant_emo,
                "dominant_emoji": EMOTION_META.get(dominant_emo, {}).get("emoji", "🎭"),
                "dominant_pct": round((dominant_cnt / total) * 100.0, 1),
                "agreement_rate": round((agree_cnt / total) * 100.0, 1),
                "ab_flip_count": flip_cnt,
                "distribution": {emo: dist.get(emo, 0) for emo in sorted(EMOTION_META.keys())},
                "top_trigger_words": [{"word": w, "count": c} for w, c in word_counts.most_common(15)],
            },
            "rows": rows,
            "csv_export": csv_buf.getvalue(),
        }
