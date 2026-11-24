"""
app.py
------
Production Flask backend for EmotiSense AI v2.2.
Features:
  - 4 validated sklearn 1.8.0 pipelines (BoW+LR 89.84% Winner, TF-IDF+LR 84.97%, BoW+MNB, TF-IDF+MNB)
  - Autonomous 2-Pass Tool-Calling NLP LLM Agent (Gemini 4 Hydrogen / OpenAI):
      * Executes real NLP backend tools (analyze_text, run_batch_journal, inspect_vocabulary)
      * Dispatches live UI automation commands (`ui_commands`) to drive the website on screen
      * Includes natural-language autopilot fallback when no LLM API key is configured yet
"""

import csv
import io
import json
import logging
import os
import re
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import requests
import sklearn
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from preprocessor import preprocess

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

APP_START_TIME = time.time()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

EMOTION_META = {
    "anger": {"emoji": "😠", "color": "#dc2626"},
    "fear": {"emoji": "😨", "color": "#7c3aed"},
    "joy": {"emoji": "😄", "color": "#d97706"},
    "love": {"emoji": "❤️", "color": "#db2777"},
    "sadness": {"emoji": "😢", "color": "#2563eb"},
    "surprise": {"emoji": "😲", "color": "#0d9488"},
}

MODEL_REGISTRY = {
    "bow_lr": {"name": "BoW + Logistic Regression", "model_file": "bow_model.pkl", "vec_file": "bow_vectorizer.pkl", "badge": "CV & Test Winner"},
    "tfidf_lr": {"name": "TF-IDF + Logistic Regression", "model_file": "model.pkl", "vec_file": "vectorizer.pkl", "badge": "Normalized Linear"},
    "bow_nb": {"name": "BoW + Multinomial NB", "model_file": "nb_bow_model.pkl", "vec_file": "nb_bow_vectorizer.pkl", "badge": "Probabilistic Baseline"},
    "tfidf_nb": {"name": "TF-IDF + Multinomial NB", "model_file": "nb_tfidf_model.pkl", "vec_file": "nb_tfidf_vectorizer.pkl", "badge": "TF-IDF Baseline"},
}

loaded_models = {}
metrics_cache = {}


def load_assets():
    global metrics_cache
    try:
        with open(os.path.join(BASE_DIR, "emotion_metrics.json"), "r", encoding="utf-8") as f:
            metrics_cache = json.load(f)
    except Exception as exc:
        logger.warning("Could not load emotion_metrics.json: %s", exc)

    for key, cfg in MODEL_REGISTRY.items():
        m_path = os.path.join(BASE_DIR, cfg["model_file"])
        v_path = os.path.join(BASE_DIR, cfg["vec_file"])
        if os.path.exists(m_path) and os.path.exists(v_path):
            try:
                clf = joblib.load(m_path)
                vec = joblib.load(v_path)
                feat_names = vec.get_feature_names_out()
                feat_to_idx = {name: idx for idx, name in enumerate(feat_names)}
                bench = next((b for b in metrics_cache.get("benchmarks", []) if b["key"] == key), {})
                loaded_models[key] = {
                    "key": key, "name": cfg["name"], "badge": cfg["badge"],
                    "model": clf, "vectorizer": vec,
                    "classes": [str(c) for c in clf.classes_],
                    "feat_to_idx": feat_to_idx,
                    "cv_accuracy": bench.get("cv_accuracy"),
                    "test_accuracy": bench.get("test_accuracy"),
                    "macro_f1": bench.get("macro_f1"),
                }
            except Exception as exc:
                logger.error("Failed loading %s: %s", key, exc)


load_assets()


def _token_score_map(bundle, X_vec, clean_text, pred_idx):
    clf = bundle["model"]
    feat_to_idx = bundle["feat_to_idx"]
    words = clean_text.split()
    x_row = X_vec.toarray()[0]
    word_scores = {}
    for w in set(words):
        idx = feat_to_idx.get(w)
        if idx is None:
            continue
        x_val = float(x_row[idx])
        if hasattr(clf, "coef_"):
            weight = float(clf.coef_[pred_idx, idx])
            score = weight * x_val
        elif hasattr(clf, "feature_log_prob_"):
            log_probs = clf.feature_log_prob_[:, idx]
            weight = float(log_probs[pred_idx] - float(np.max(np.delete(log_probs, pred_idx))))
            score = weight * x_val
        else:
            weight, score = 0.0, 0.0
        word_scores[w] = {
            "word": w, "coef": round(weight, 4), "score": round(score, 4),
            "count": words.count(w),
            "impact": "positive" if score > 0 else ("negative" if score < 0 else "neutral"),
        }
    positive = sorted([v for v in word_scores.values() if v["score"] > 0], key=lambda x: x["score"], reverse=True)
    opposing = sorted([v for v in word_scores.values() if v["score"] < 0], key=lambda x: x["score"])
    sentence_tokens = [
        {"word": w, "score": word_scores[w]["score"] if w in word_scores else 0.0, "impact": word_scores[w]["impact"] if w in word_scores else "oov"}
        for w in words
    ]
    return positive[:10], opposing[:6], sentence_tokens


def _infer_single_bundle(bundle, clean_text):
    X_vec = bundle["vectorizer"].transform([clean_text])
    probs = bundle["model"].predict_proba(X_vec)[0]
    classes = bundle["classes"]
    pred_idx = int(np.argmax(probs))
    return X_vec, pred_idx, classes[pred_idx], float(probs[pred_idx]) * 100.0, {classes[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(classes))}


def predict_emotion(text: str, primary_key: str = "bow_lr", compare_models: bool = True):
    if primary_key not in loaded_models:
        primary_key = "bow_lr" if "bow_lr" in loaded_models else next(iter(loaded_models))
    clean = preprocess(text)
    if not clean:
        raise ValueError("Please include recognizable alphabetical words.")
    primary_bundle = loaded_models[primary_key]
    X_vec, pred_idx, emotion, confidence, all_probs = _infer_single_bundle(primary_bundle, clean)
    pos_tokens, neg_tokens, sentence_tokens = _token_score_map(primary_bundle, X_vec, clean, pred_idx)
    sorted_probs = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = sorted_probs[1] if len(sorted_probs) > 1 else (emotion, 0.0)
    meta = EMOTION_META.get(emotion, {"emoji": "🤖", "color": "#6d28d9"})

    result = {
        "emotion": emotion, "emoji": meta["emoji"], "color": meta["color"],
        "confidence": round(confidence, 2),
        "runner_up": {"emotion": runner_up[0], "confidence": runner_up[1]},
        "margin_pct": round(confidence - runner_up[1], 2),
        "all_probabilities": all_probs, "original_text": text, "clean_text": clean,
        "token_importance": pos_tokens, "opposing_tokens": neg_tokens, "sentence_tokens": sentence_tokens,
        "model_key": primary_key, "model_name": primary_bundle["name"],
        "accuracy": f"{round((primary_bundle.get('test_accuracy') or 0.8984) * 100, 2)}%",
        "cv_accuracy": f"{round((primary_bundle.get('cv_accuracy') or 0.8795) * 100, 2)}%",
    }
    if compare_models:
        arena = []
        votes = Counter()
        for key in ["bow_lr", "tfidf_lr", "bow_nb", "tfidf_nb"]:
            b = loaded_models.get(key)
            if not b:
                continue
            _, _, m_emo, m_conf, m_probs = _infer_single_bundle(b, clean)
            m_meta = EMOTION_META.get(m_emo, {"emoji": "🤖", "color": "#6d28d9"})
            votes[m_emo] += 1
            arena.append({
                "key": key, "name": b["name"], "badge": b["badge"],
                "emotion": m_emo, "emoji": m_meta["emoji"], "color": m_meta["color"],
                "confidence": round(m_conf, 2),
                "test_accuracy": round((b.get("test_accuracy") or 0) * 100, 2),
                "cv_accuracy": round((b.get("cv_accuracy") or 0) * 100, 2),
                "all_probabilities": m_probs,
            })
        top_emo, top_cnt = votes.most_common(1)[0]
        result["arena"] = {"models": arena, "consensus_emotion": top_emo, "consensus_count": top_cnt, "total_models": len(arena), "unanimous": top_cnt == len(arena)}
    return result


def run_batch_lines(entries):
    bow_bundle = loaded_models["bow_lr"]
    tfidf_bundle = loaded_models.get("tfidf_lr")
    valid_pairs = [(t, preprocess(t)) for t in entries[:2000] if preprocess(t)]
    if not valid_pairs:
        raise ValueError("No valid alphabetical sentences found.")
    clean_texts = [p[1] for p in valid_pairs]
    X_bow = bow_bundle["vectorizer"].transform(clean_texts)
    bow_probs = bow_bundle["model"].predict_proba(X_bow)
    bow_classes = bow_bundle["classes"]
    bow_preds = np.argmax(bow_probs, axis=1)

    tfidf_probs = tfidf_preds = None
    if tfidf_bundle:
        X_tf = tfidf_bundle["vectorizer"].transform(clean_texts)
        tfidf_probs = tfidf_bundle["model"].predict_proba(X_tf)
        tfidf_classes = tfidf_bundle["classes"]
        tfidf_preds = np.argmax(tfidf_probs, axis=1)

    dist_counts, trigger_counter = Counter(), Counter()
    agree_cnt = 0
    feat_to_idx = bow_bundle["feat_to_idx"]
    coef_mat = bow_bundle["model"].coef_
    rows = []
    for i, (orig, cln) in enumerate(valid_pairs):
        idx_b = int(bow_preds[i])
        emo_b = bow_classes[idx_b]
        conf_b = round(float(bow_probs[i, idx_b]) * 100.0, 1)
        dist_counts[emo_b] += 1
        emo_t, conf_t = emo_b, conf_b
        if tfidf_preds is not None:
            idx_t = int(tfidf_preds[i])
            emo_t, conf_t = tfidf_classes[idx_t], round(float(tfidf_probs[i, idx_t]) * 100.0, 1)
        if emo_b == emo_t:
            agree_cnt += 1
        top_word, best_w = "", -999.0
        for w in set(cln.split()):
            fidx = feat_to_idx.get(w)
            if fidx is not None and float(coef_mat[idx_b, fidx]) > best_w:
                best_w, top_word = float(coef_mat[idx_b, fidx]), w
        if top_word and best_w > 0:
            trigger_counter[top_word] += 1
        meta = EMOTION_META.get(emo_b, {"emoji": "🤖", "color": "#6d28d9"})
        rows.append({
            "row_id": i + 1, "text": orig, "emotion": emo_b, "emoji": meta["emoji"], "color": meta["color"],
            "confidence": conf_b, "tfidf_emotion": emo_t, "tfidf_confidence": conf_t, "agree": emo_b == emo_t, "top_trigger": top_word,
        })

    out_buf = io.StringIO()
    writer = csv.writer(out_buf)
    writer.writerow(["row_id", "text", "bow_emotion", "bow_confidence_pct", "tfidf_emotion", "tfidf_confidence_pct", "agree", "top_trigger"])
    for r in rows:
        writer.writerow([r["row_id"], r["text"], r["emotion"], r["confidence"], r["tfidf_emotion"], r["tfidf_confidence"], r["agree"], r["top_trigger"]])

    total = len(rows)
    dominant = dist_counts.most_common(1)[0][0]
    return {
        "summary": {
            "total_entries": total, "dominant_emotion": dominant, "dominant_emoji": EMOTION_META.get(dominant, {}).get("emoji", "🤖"),
            "agreement_rate": round((agree_cnt / total) * 100.0, 1),
            "distribution": {lbl: int(dist_counts.get(lbl, 0)) for lbl in sorted(EMOTION_META.keys())},
            "top_triggers": [{"word": w, "count": c} for w, c in trigger_counter.most_common(10)],
        },
        "rows": rows, "csv_export": out_buf.getvalue(),
    }


# ---------------------------------------------------------------------------
# Autonomous 2-Pass Emotion LLM Agent + Smart Autopilot
# ---------------------------------------------------------------------------

def _extract_gemini_text(payload):
    if not isinstance(payload, dict):
        return ""
    texts = []
    for step in payload.get("steps") or []:
        if step.get("type") == "model_output":
            for part in step.get("content") or []:
                if part.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])
    if texts:
        return "\n".join(texts).strip()
    return str(payload.get("output_text") or "").strip()


def _parse_json_envelope(raw_text):
    cleaned = raw_text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    else:
        b1, b2 = cleaned.find("{"), cleaned.rfind("}")
        if b1 != -1 and b2 > b1:
            cleaned = cleaned[b1:b2 + 1]
    return json.loads(cleaned)


def call_llm_raw(provider, api_key, system_prompt, prompt_text):
    provider = (provider or "").lower().strip()
    if provider == "gemini":
        model_name = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key, "Api-Revision": "2026-05-20"}
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1/interactions",
            headers=headers,
            json={"model": model_name, "system_instruction": system_prompt, "input": prompt_text},
            timeout=45,
        )
        if not resp.ok:
            raise RuntimeError(f"Gemini error ({resp.status_code}): {resp.text[:300]}")
        return _extract_gemini_text(resp.json()), model_name
    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]},
            timeout=45,
        )
        if not resp.ok:
            raise RuntimeError(f"OpenAI error ({resp.status_code}): {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"].strip(), model_name
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_llm_credentials(req_data):
    req_provider = (req_data.get("provider") or os.environ.get("LLM_PROVIDER") or "").lower().strip()
    session_key = (request.headers.get("X-LLM-API-Key") or req_data.get("api_key") or "").strip()
    env_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
    env_openai = os.environ.get("OPENAI_API_KEY", "").strip()
    if req_provider == "openai":
        return "openai", (session_key or env_openai)
    if req_provider == "gemini":
        return "gemini", (session_key or env_gemini)
    if session_key:
        return ("openai" if session_key.startswith("sk-") else "gemini"), session_key
    if env_gemini:
        return "gemini", env_gemini
    if env_openai:
        return "openai", env_openai
    return "gemini", ""


def execute_emotion_tools(tool_calls, default_text, default_model):
    ui_commands = []
    tool_outputs = {}
    for tc in tool_calls:
        name = str(tc.get("tool", "")).strip()
        args = tc.get("args") or {}

        if name == "switch_tab":
            tab = args.get("tab", "studio")
            ui_commands.append({"type": "switch_tab", "tab": tab})

        elif name == "analyze_text":
            txt = str(args.get("text") or default_text).strip()
            m_key = str(args.get("model") or default_model)
            res = predict_emotion(txt, primary_key=m_key, compare_models=True)
            ui_commands.append({"type": "switch_tab", "tab": "studio"})
            ui_commands.append({"type": "analyze_text", "text": txt, "model": m_key})
            res_copy = dict(res)
            res_copy.pop("sentence_tokens", None)
            tool_outputs["analysis"] = res_copy

        elif name == "run_batch_journal":
            lines = [str(x).strip() for x in (args.get("lines") or []) if str(x).strip()]
            if lines:
                b_res = run_batch_lines(lines)
                ui_commands.append({"type": "switch_tab", "tab": "batch"})
                ui_commands.append({"type": "run_batch_lines", "lines": lines})
                tool_outputs["batch_summary"] = b_res["summary"]
            else:
                ui_commands.append({"type": "switch_tab", "tab": "batch"})
                ui_commands.append({"type": "run_batch_demo"})
                tool_outputs["batch_demo"] = {"triggered": True}

        elif name == "inspect_vocabulary":
            emo = args.get("emotion", "joy")
            m_key = args.get("model", "bow_lr")
            ui_commands.append({"type": "switch_tab", "tab": "research"})
            ui_commands.append({"type": "inspect_vocab", "emotion": emo, "model": m_key})
            tok_data = metrics_cache.get("top_tokens", {}).get(m_key, {}).get(emo, {})
            tool_outputs["vocab"] = {"emotion": emo, "model": m_key, "top_positive": tok_data.get("positive", [])[:6]}

    return ui_commands, tool_outputs


def emotion_autopilot_fallback(message, current_text, active_model):
    q = message.lower().strip()
    if any(k in q for k in ["research lab", "confusion matrix", "vocabulary", "weights", "tokens for"]):
        emo = next((e for e in EMOTION_META if e in q), "joy")
        ui_cmds, outs = execute_emotion_tools([{"tool": "inspect_vocabulary", "args": {"emotion": emo, "model": "bow_lr"}}], current_text, active_model)
        pos = ", ".join(f"`{t['word']}` (`+{t['weight']}`)" for t in outs.get("vocab", {}).get("top_positive", [])[:5])
        return (
            f"I opened the **NLP Research Lab** and selected **{emo.title()}** weights on screen!\n"
            f"- **Top predictive words:** {pos}\n\n"
            "*(Connect a Gemini key in `🔑` anytime for open-ended NLP research chat.)*",
            ui_cmds,
            ["Run 10-entry batch journal demo", "Back to Emotion Studio"],
        )

    if any(k in q for k in ["batch", "journal demo", "analyze all"]):
        ui_cmds, _ = execute_emotion_tools([{"tool": "run_batch_journal", "args": {"lines": []}}], current_text, active_model)
        return (
            "I switched to **Batch Journal** and ran the 10-entry emotion screening live on screen!",
            ui_cmds,
            ["Inspect fear vocabulary weights", "Back to Emotion Studio"],
        )

    # Check if user gave a sentence directly to analyze
    m_quote = re.search(r"['\"]([^'\"]{6,})['\"]", message)
    candidate = m_quote.group(1) if m_quote else (message if len(message.split()) >= 4 and not message.lower().startswith(("hi", "hello", "hey", "why", "explain", "help")) else "")
    if candidate:
        ui_cmds, outs = execute_emotion_tools([{"tool": "analyze_text", "args": {"text": candidate, "model": active_model}}], candidate, active_model)
        an = outs.get("analysis", {})
        toks = ", ".join(f"`{t['word']}`" for t in an.get("token_importance", [])[:4])
        return (
            f"I loaded that sentence into **Emotion Studio** and analyzed all 4 models on screen:\n"
            f"- **Detected:** **{an.get('emotion', '').title()} ({an.get('confidence', 0)}%)**\n"
            f"- **Key Triggers:** {toks or 'overall phrasing'}\n"
            f"- **4-Model Consensus:** {an.get('arena', {}).get('consensus_count', 1)}/{an.get('arena', {}).get('total_models', 4)} models agree\n\n"
            "*(Add a Gemini key in `🔑` for live empathetic rewriting & open conversation!)*",
            ui_cmds,
            ["Inspect top vocabulary weights", "Run batch journal demo"],
        )

    return (
        "Hi! Share any sentence, message, or journal lines (e.g., *\"Analyze: 'I feel relieved and excited about tomorrow!'\"*) and I will load it into the Studio, highlight the trigger words, and show all 4 models live on screen!\n\n"
        "*(For full conversational EmotiCoach LLM intelligence & custom tone rewriting, add a `GEMINI_API_KEY` via `🔑` above.)*",
        [],
        [
            "Analyze: 'I did not feel angry at all, I felt relieved and grateful!'",
            "Run 10-entry batch journal demo",
            "Open NLP Research Lab for anger",
        ],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
@app.route("/api/ping", methods=["GET"])
def health():
    env_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    env_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    primary = loaded_models.get("bow_lr")
    return jsonify({
        "status": "healthy" if primary is not None else "degraded",
        "service": "EmotiSense AI", "version": "2.2.0", "sklearn_version": sklearn.__version__,
        "primary_model": primary["name"] if primary else None,
        "loaded_models": list(loaded_models.keys()),
        "classes": primary["classes"] if primary else list(EMOTION_META.keys()),
        "vocab_size": len(primary["feat_to_idx"]) if primary else 0,
        "llm_agent": {"configured": env_gemini or env_openai, "autonomous_tools": True},
        "uptime_seconds": round(time.time() - APP_START_TIME, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/robots.txt", methods=["GET"])
def robots_txt():
    base = request.url_root.rstrip("/")
    return Response(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n", mimetype="text/plain")


@app.route("/sitemap.xml", methods=["GET"])
def sitemap_xml():
    base = request.url_root.rstrip("/")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Response(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <url><loc>{base}/</loc><lastmod>{today}</lastmod><priority>1.0</priority></url>\n</urlset>\n',
        mimetype="application/xml",
    )


@app.route("/api/emotions", methods=["GET"])
def list_emotions():
    return jsonify({"emotions": [{"label": lbl, "emoji": meta["emoji"], "color": meta["color"]} for lbl, meta in EMOTION_META.items()]})


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    return jsonify(metrics_cache)


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if not loaded_models:
        return jsonify({"error": "No ML models loaded."}), 500
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter a sentence or reflection."}), 400
    if len(text) > 3000:
        return jsonify({"error": "Text too long (max 3000 chars)."}), 400
    try:
        return jsonify(predict_emotion(text, primary_key=str(data.get("model", "bow_lr")), compare_models=bool(data.get("compare_models", True))))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    if "bow_lr" not in loaded_models:
        return jsonify({"error": "Primary model not loaded."}), 500
    try:
        entries = []
        if "file" in request.files:
            file_obj = request.files["file"]
            raw_bytes = file_obj.read().decode("utf-8", errors="ignore")
            if (file_obj.filename or "").lower().endswith(".csv"):
                reader = csv.DictReader(io.StringIO(raw_bytes))
                fields = [f.strip() for f in (reader.fieldnames or [])]
                col = next((c for c in fields if c.lower() in ("text", "sentence", "message", "content", "entry")), fields[0] if fields else None)
                if not col:
                    return jsonify({"error": "CSV lacks a text column."}), 400
                entries = [str(row.get(col, "")).strip() for row in reader if str(row.get(col, "")).strip()]
            else:
                entries = [line.strip() for line in raw_bytes.splitlines() if line.strip()]
        else:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload.get("lines"), list):
                entries = [str(x).strip() for x in payload["lines"] if str(x).strip()]
            elif payload.get("raw_text"):
                entries = [line.strip() for line in str(payload["raw_text"]).splitlines() if line.strip()]
        if not entries:
            return jsonify({"error": "Provide at least one sentence, .txt, or .csv."}), 400
        return jsonify(run_batch_lines(entries))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    current_text = str(data.get("current_text", "")).strip()
    active_model = str(data.get("model", "bow_lr")).strip()
    history = data.get("history") or []
    provider, api_key = resolve_llm_credentials(data)

    if not api_key:
        reply_text, ui_commands, quick_replies = emotion_autopilot_fallback(message, current_text, active_model)
        return jsonify({
            "reply": reply_text, "mode": "autopilot-fallback", "provider": "local-parser",
            "model": "NLP Autopilot (Add Gemini Key for Full LLM)", "ui_commands": ui_commands, "quick_replies": quick_replies,
        })

    history_txt = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}" for m in history[-8:])
    planner_system = (
        "You are EmotiCoach AI, a warm, empathetic, autonomous Affective NLP & Website Automation Agent inside EmotiSense AI.\n"
        "NEVER sound scripted or canned. Respond naturally AND drive the website live with `tool_calls`!\n"
        "Available tools:\n"
        "1. `analyze_text`: args={\"text\": \"...\", \"model\": \"bow_lr\"|\"tfidf_lr\"|\"bow_nb\"|\"tfidf_nb\"}. Use whenever the user shares text to analyze, wants to test a message/example, or rewrites text to verify its emotion.\n"
        "2. `run_batch_journal`: args={\"lines\": [\"...\", \"...\"]}. Use when user gives multiple entries or asks to run batch screening (pass empty [] for 10-entry demo).\n"
        "3. `inspect_vocabulary`: args={\"emotion\": \"anger\"|\"fear\"|\"joy\"|\"love\"|\"sadness\"|\"surprise\", \"model\": \"bow_lr\"|\"tfidf_lr\"}.\n"
        "4. `switch_tab`: args={\"tab\": \"studio\"|\"research\"|\"batch\"}.\n\n"
        "Respond ONLY with valid JSON:\n"
        "{\"needs_tools\": true|false, \"tool_calls\": [{\"tool\": \"...\", \"args\": {...}}], \"direct_reply\": \"...\", \"follow_up_chips\": [\"...\"]}"
    )
    planner_input = f"Current Text in Studio: {current_text or '(empty)'}\n\nConversation:\n{history_txt}\n\nUser: {message}"

    try:
        raw_plan, model_used = call_llm_raw(provider, api_key, planner_system, planner_input)
        plan = _parse_json_envelope(raw_plan)
    except Exception as exc:
        reply_text, ui_commands, quick_replies = emotion_autopilot_fallback(message, current_text, active_model)
        return jsonify({
            "reply": f"{reply_text}\n\n_(LLM error `{exc}` — executed locally.)_",
            "mode": "autopilot-fallback", "provider": provider, "model": "Fallback",
            "ui_commands": ui_commands, "quick_replies": quick_replies,
        })

    tool_calls = plan.get("tool_calls") or []
    if not plan.get("needs_tools") or not tool_calls:
        return jsonify({
            "reply": plan.get("direct_reply") or raw_plan, "mode": "llm", "provider": provider,
            "model": model_used, "ui_commands": [], "quick_replies": plan.get("follow_up_chips") or [],
        })

    ui_commands, tool_outputs = execute_emotion_tools(tool_calls, current_text, active_model)

    synth_system = (
        "You are EmotiCoach AI. The NLP tools just ran live on screen with the exact results below.\n"
        "Explain the real computed emotion, trigger words, model consensus, and any requested tone rewrite warmly and naturally using ONLY the real tool outputs below.\n"
        "Respond with valid JSON: {\"reply\": \"...\", \"follow_up_chips\": [\"...\"]}"
    )
    synth_input = f"User Message: {message}\n\nExecuted UI Commands:\n{json.dumps(ui_commands)}\n\nReal NLP Outputs:\n{json.dumps(tool_outputs, indent=2)}"

    try:
        raw_final, model_used = call_llm_raw(provider, api_key, synth_system, synth_input)
        final_obj = _parse_json_envelope(raw_final)
        final_reply = final_obj.get("reply") or raw_final
        chips = final_obj.get("follow_up_chips") or plan.get("follow_up_chips") or []
    except Exception:
        an = tool_outputs.get("analysis", {})
        final_reply = f"I analyzed that on screen: **{an.get('emotion', '').title()} ({an.get('confidence', 0)}%)**."
        chips = plan.get("follow_up_chips") or []

    return jsonify({
        "reply": final_reply, "mode": "llm-autonomous", "provider": provider,
        "model": model_used, "ui_commands": ui_commands, "quick_replies": chips, "tool_outputs": tool_outputs,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
