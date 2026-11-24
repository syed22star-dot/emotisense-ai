"""
app.py
------
Production Flask backend for EmotiSense AI v2.1.
Features:
  - 4 validated sklearn 1.8.0 pipelines (BoW+LR 89.84% Winner, TF-IDF+LR 84.97%, BoW+MNB, TF-IDF+MNB)
  - Dynamic string label handling via estimator.classes_ + stopword/negation retention
  - Token-level XAI + inline sentence heatmap + top learned vocabulary explorer
  - Batch Journal / CSV Automation
  - Interactive 1-on-1 EmotiCoach AI:
      * Live Gemini (gemini-4-hydrogen) / OpenAI LLM when key is configured
      * Built-in Guided Emotion & NLP Explainer (works 100% out-of-the-box with clickable prompts!)
"""

import csv
import io
import json
import logging
import os
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
                    "key": key,
                    "name": cfg["name"],
                    "badge": cfg["badge"],
                    "model": clf,
                    "vectorizer": vec,
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
            other_max = float(np.max(np.delete(log_probs, pred_idx)))
            weight = float(log_probs[pred_idx] - other_max)
            score = weight * x_val
        else:
            weight, score = 0.0, 0.0

        word_scores[w] = {
            "word": w,
            "coef": round(weight, 4),
            "score": round(score, 4),
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
    emotion = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    all_probs = {classes[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(classes))}
    return X_vec, pred_idx, emotion, confidence, all_probs


def predict_emotion(text: str, primary_key: str = "bow_lr", compare_models: bool = True):
    if primary_key not in loaded_models:
        primary_key = "bow_lr" if "bow_lr" in loaded_models else next(iter(loaded_models))

    clean = preprocess(text)
    if not clean:
        raise ValueError("Please include recognizable alphabetical words so our NLP model can read it.")

    primary_bundle = loaded_models[primary_key]
    X_vec, pred_idx, emotion, confidence, all_probs = _infer_single_bundle(primary_bundle, clean)
    pos_tokens, neg_tokens, sentence_tokens = _token_score_map(primary_bundle, X_vec, clean, pred_idx)

    sorted_probs = sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = sorted_probs[1] if len(sorted_probs) > 1 else (emotion, 0.0)
    margin_pct = round(confidence - runner_up[1], 2)
    meta = EMOTION_META.get(emotion, {"emoji": "🤖", "color": "#7c3aed"})

    result = {
        "emotion": emotion,
        "emoji": meta["emoji"],
        "color": meta["color"],
        "confidence": round(confidence, 2),
        "runner_up": {"emotion": runner_up[0], "confidence": runner_up[1]},
        "margin_pct": margin_pct,
        "all_probabilities": all_probs,
        "original_text": text,
        "clean_text": clean,
        "token_importance": pos_tokens,
        "opposing_tokens": neg_tokens,
        "sentence_tokens": sentence_tokens,
        "model_key": primary_key,
        "model_name": primary_bundle["name"],
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
            m_meta = EMOTION_META.get(m_emo, {"emoji": "🤖", "color": "#7c3aed"})
            votes[m_emo] += 1
            arena.append(
                {
                    "key": key,
                    "name": b["name"],
                    "badge": b["badge"],
                    "emotion": m_emo,
                    "emoji": m_meta["emoji"],
                    "color": m_meta["color"],
                    "confidence": round(m_conf, 2),
                    "test_accuracy": round((b.get("test_accuracy") or 0) * 100, 2),
                    "cv_accuracy": round((b.get("cv_accuracy") or 0) * 100, 2),
                    "all_probabilities": m_probs,
                }
            )
        top_vote_emo, top_vote_cnt = votes.most_common(1)[0]
        result["arena"] = {
            "models": arena,
            "consensus_emotion": top_vote_emo,
            "consensus_count": top_vote_cnt,
            "total_models": len(arena),
            "unanimous": top_vote_cnt == len(arena),
        }
    return result


# ---------------------------------------------------------------------------
# Interactive EmotiCoach AI (Gemini/OpenAI LLM + Built-in NLP Concierge)
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
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    return ""


def call_llm_provider(provider, api_key, system_prompt, user_message, history, previous_interaction_id=None):
    provider = (provider or "").lower().strip()
    if provider == "gemini":
        model_name = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key, "Api-Revision": "2026-05-20"}
        url = "https://generativelanguage.googleapis.com/v1/interactions"

        def _post(inp, prev_id=None):
            body = {"model": model_name, "system_instruction": system_prompt, "input": inp}
            if prev_id:
                body["previous_interaction_id"] = prev_id
            return requests.post(url, headers=headers, json=body, timeout=45)

        if previous_interaction_id:
            resp = _post(user_message, prev_id=previous_interaction_id)
            if resp.ok:
                data = resp.json()
                txt = _extract_gemini_text(data)
                if txt:
                    return txt, data.get("id"), model_name
            elif resp.status_code not in (400, 404):
                raise RuntimeError(f"Gemini error ({resp.status_code}): {resp.text[:300]}")

        transcript = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in (history or [])[-8:]
        )
        full_inp = f"Recent conversation:\n{transcript}\n\nUser: {user_message}" if transcript else user_message
        resp = _post(full_inp)
        if not resp.ok:
            raise RuntimeError(f"Gemini error ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        txt = _extract_gemini_text(data)
        if not txt:
            raise RuntimeError("Gemini returned empty text.")
        return txt, data.get("id"), model_name

    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        messages = [{"role": "system", "content": system_prompt}]
        for m in (history or [])[-10:]:
            messages.append({"role": "assistant" if m.get("role") == "assistant" else "user", "content": str(m.get("content", ""))})
        messages.append({"role": "user", "content": user_message})
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "messages": messages},
            timeout=45,
        )
        if not resp.ok:
            raise RuntimeError(f"OpenAI error ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip(), None, model_name

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


def guided_emotion_coach_reply(message, analysis_snapshot):
    q = message.lower()
    actions = []
    quick_replies = []

    if any(w in q for w in ["what to write", "example", "not sure", "help me start", "new", "mood", "stuck", "feel"]):
        reply = (
            "**Not sure what to type? Pick what matches your mood right now** and I'll load a natural prompt into the analyzer so you can see how the AI reads tone & trigger words:"
        )
        actions = [
            {"label": "Overwhelmed / Stressed", "fill_text": "I feel overwhelmed with deadlines today and slightly worried about getting everything done."},
            {"label": "Grateful / Happy", "fill_text": "I feel genuinely relieved and happy about how smoothly things went today!"},
            {"label": "Frustrated / Annoyed", "fill_text": "I am really frustrated because nobody listened to my concern during the meeting."},
            {"label": "Test Negation ('Not Angry')", "fill_text": "I did not feel angry or humiliated at all; I felt calm and appreciative."},
        ]
    elif any(w in q for w in ["explain", "why", "trigger", "word", "bow", "tfidf", "tf-idf", "model", "negation"]):
        if analysis_snapshot and "emotion" in analysis_snapshot:
            emo = analysis_snapshot["emotion"]
            conf = analysis_snapshot["confidence"]
            toks = analysis_snapshot.get("token_importance", [])[:4]
            tok_txt = ", ".join(f"`{t['word']}` (`+{t['score']}`)" for t in toks) if toks else "overall word balance"
            arena = analysis_snapshot.get("arena") or {}
            agree_txt = f"{arena.get('consensus_count', 1)}/{arena.get('total_models', 4)} models agree on **{emo}**" if arena else ""
            reply = (
                f"**Why EmotiSense predicted {emo.title()} ({conf}%):**\n"
                f"- **Key Trigger Words:** {tok_txt}\n"
                f"- **Model Consensus:** {agree_txt}\n"
                f"- **Why BoW+LR (`89.84%`) is Primary:** Bag-of-Words preserves exact word frequency and negation pairs (`not`, `never`, `didnt`) instead of down-weighting common emotional anchors like TF-IDF sometimes does."
            )
            quick_replies = ["Reframe this message gently", "Try a negation test sentence"]
        else:
            reply = (
                "**How Our 4 NLP Models Work:**\n"
                "- **BoW + Logistic Regression (89.84% Winner):** Counts exact word presence including negations (`not`, `never`) and weights each word.\n"
                "- **TF-IDF + Logistic Regression (84.97%):** Scales down very frequent words; great for longer essays.\n\n"
                "Type any sentence on the left and click **Analyse Emotion** to inspect its word-by-word heatmap!"
            )
            actions = [
                {"label": "Try Sample Sentence", "fill_text": "I feel so grateful and excited about what we built together!"}
            ]
    elif any(w in q for w in ["reframe", "rewrite", "calm", "polite", "constructive"]):
        if analysis_snapshot and "emotion" in analysis_snapshot:
            emo = analysis_snapshot["emotion"]
            reply = (
                f"Your text currently carries strong **{emo}** signals ({analysis_snapshot['confidence']}%).\n"
                "**Quick Constructive Reframe Template:**\n"
                "- Acknowledge the feeling cleanly (*\"I noticed I felt frustrated when...\"*)\n"
                "- State the desired outcome calmly (*\"Going forward, it would help me if we could...\"*)\n\n"
                "Connect a **Gemini API Key** in Settings (`🔑`) anytime to let **Gemini 4 Hydrogen** generate custom word-for-word rewrites!"
            )
            actions = [
                {"label": "Test Calm Reframe Example", "fill_text": "I felt concerned about the timeline earlier, and I appreciate working together to keep things on track."}
            ]
        else:
            reply = "Analyze a sentence first, then ask me to help reframe its emotional tone!"
    else:
        reply = (
            "Hi! I'm **EmotiCoach AI**. I can help you test sentences, explain why certain words triggered **Joy / Sadness / Anger / Fear / Love / Surprise**, compare **BoW vs TF-IDF**, or help you articulate how you're feeling.\n\nWhat would you like to explore?"
        )
        quick_replies = [
            "Not sure what to write — give examples",
            "Explain why BoW beat TF-IDF",
            "Explain the current prediction",
        ]

    return reply, actions, quick_replies


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
    return (
        jsonify(
            {
                "status": "healthy" if primary is not None else "degraded",
                "service": "EmotiSense AI",
                "version": "2.1.0",
                "sklearn_version": sklearn.__version__,
                "primary_model": primary["name"] if primary else None,
                "loaded_models": list(loaded_models.keys()),
                "classes": primary["classes"] if primary else list(EMOTION_META.keys()),
                "vocab_size": len(primary["feat_to_idx"]) if primary else 0,
                "llm_agent": {"configured": env_gemini or env_openai, "guided_nlp_ready": True},
                "uptime_seconds": round(time.time() - APP_START_TIME, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ),
        200,
    )


@app.route("/robots.txt", methods=["GET"])
def robots_txt():
    base = request.url_root.rstrip("/")
    return Response(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n", mimetype="text/plain")


@app.route("/sitemap.xml", methods=["GET"])
def sitemap_xml():
    base = request.url_root.rstrip("/")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{base}/</loc><lastmod>{today}</lastmod><priority>1.0</priority></url>\n"
        "</urlset>\n"
    )
    return Response(xml, mimetype="application/xml")


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
    model_key = str(data.get("model", "bow_lr")).strip()
    compare_models = bool(data.get("compare_models", True))

    if not text:
        return jsonify({"error": "Please enter a sentence or reflection to analyze."}), 400
    if len(text) > 3000:
        return jsonify({"error": "Text is too long (max 3000 characters)."}), 400

    try:
        return jsonify(predict_emotion(text, primary_key=model_key, compare_models=compare_models))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        return jsonify({"error": "Internal prediction error."}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    bow_bundle = loaded_models.get("bow_lr")
    tfidf_bundle = loaded_models.get("tfidf_lr")
    if not bow_bundle:
        return jsonify({"error": "Primary BoW model not loaded."}), 500

    try:
        entries = []
        if "file" in request.files:
            file_obj = request.files["file"]
            fname = (file_obj.filename or "").lower()
            raw_bytes = file_obj.read().decode("utf-8", errors="ignore")
            if fname.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(raw_bytes))
                fieldnames = [f.strip() for f in (reader.fieldnames or [])]
                text_col = next((c for c in fieldnames if c.lower() in ("text", "sentence", "message", "content", "entry")), fieldnames[0] if fieldnames else None)
                if not text_col:
                    return jsonify({"error": "CSV lacks a text column."}), 400
                entries = [str(row.get(text_col, "")).strip() for row in reader if str(row.get(text_col, "")).strip()]
            else:
                entries = [line.strip() for line in raw_bytes.splitlines() if line.strip()]
        else:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload.get("lines"), list):
                entries = [str(x).strip() for x in payload["lines"] if str(x).strip()]
            elif payload.get("raw_text"):
                entries = [line.strip() for line in str(payload["raw_text"]).splitlines() if line.strip()]

        if not entries:
            return jsonify({"error": "Please provide at least one line, .txt, or .csv file."}), 400

        entries = entries[:2000]
        valid_pairs = [(t, preprocess(t)) for t in entries if preprocess(t)]
        if not valid_pairs:
            return jsonify({"error": "No valid alphabetical sentences found."}), 400

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

        dist_counts = Counter()
        agree_cnt = 0
        trigger_counter = Counter()
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
                emo_t = tfidf_classes[idx_t]
                conf_t = round(float(tfidf_probs[i, idx_t]) * 100.0, 1)
            if emo_b == emo_t:
                agree_cnt += 1

            top_word, best_w = "", -999.0
            for w in set(cln.split()):
                fidx = feat_to_idx.get(w)
                if fidx is not None and float(coef_mat[idx_b, fidx]) > best_w:
                    best_w = float(coef_mat[idx_b, fidx])
                    top_word = w
            if top_word and best_w > 0:
                trigger_counter[top_word] += 1

            meta = EMOTION_META.get(emo_b, {"emoji": "🤖", "color": "#7c3aed"})
            rows.append(
                {
                    "row_id": i + 1,
                    "text": orig,
                    "emotion": emo_b,
                    "emoji": meta["emoji"],
                    "color": meta["color"],
                    "confidence": conf_b,
                    "tfidf_emotion": emo_t,
                    "tfidf_confidence": conf_t,
                    "agree": emo_b == emo_t,
                    "top_trigger": top_word,
                }
            )

        out_buf = io.StringIO()
        writer = csv.writer(out_buf)
        writer.writerow(["row_id", "text", "bow_emotion", "bow_confidence_pct", "tfidf_emotion", "tfidf_confidence_pct", "agree", "top_trigger"])
        for r in rows:
            writer.writerow([r["row_id"], r["text"], r["emotion"], r["confidence"], r["tfidf_emotion"], r["tfidf_confidence"], r["agree"], r["top_trigger"]])

        total = len(rows)
        dominant = dist_counts.most_common(1)[0][0]
        return jsonify(
            {
                "summary": {
                    "total_entries": total,
                    "dominant_emotion": dominant,
                    "dominant_emoji": EMOTION_META.get(dominant, {}).get("emoji", "🤖"),
                    "agreement_rate": round((agree_cnt / total) * 100.0, 1),
                    "distribution": {lbl: int(dist_counts.get(lbl, 0)) for lbl in sorted(EMOTION_META.keys())},
                    "top_triggers": [{"word": w, "count": c} for w, c in trigger_counter.most_common(10)],
                },
                "rows": rows,
                "csv_export": out_buf.getvalue(),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Batch failed: {exc}"}), 400


@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    current_text = str(data.get("current_text", "")).strip()
    model_key = str(data.get("model", "bow_lr")).strip()
    session_summary = data.get("session_summary") or {}
    history = data.get("history") or []
    prev_id = data.get("previous_interaction_id")

    analysis_snapshot = {}
    if current_text and loaded_models:
        try:
            analysis_snapshot = predict_emotion(current_text, primary_key=model_key, compare_models=True)
            analysis_snapshot.pop("sentence_tokens", None)
        except Exception:
            pass

    provider, api_key = resolve_llm_credentials(data)
    if api_key:
        system_prompt = (
            "You are EmotiCoach AI, a warm, articulate 1-on-1 emotional reflection & NLP assistant inside EmotiSense AI v2.\n"
            "Explain emotion predictions clearly in everyday language, reference exact trigger tokens and BoW vs TF-IDF comparisons, "
            "and offer practical empathetic reframing.\n\n"
            f"ANALYZED TEXT TELEMETRY:\n{json.dumps(analysis_snapshot, indent=2)}\n\n"
            f"SESSION MOOD SUMMARY:\n{json.dumps(session_summary, indent=2)}"
        )
        try:
            reply_text, interaction_id, model_used = call_llm_provider(
                provider=provider,
                api_key=api_key,
                system_prompt=system_prompt,
                user_message=message,
                history=history,
                previous_interaction_id=prev_id,
            )
            _, actions, quick_replies = guided_emotion_coach_reply(message, analysis_snapshot)
            return jsonify(
                {
                    "reply": reply_text,
                    "mode": "llm",
                    "provider": provider,
                    "model": model_used,
                    "interaction_id": interaction_id,
                    "actions": actions,
                    "quick_replies": quick_replies,
                }
            )
        except Exception as exc:
            reply_text, actions, quick_replies = guided_emotion_coach_reply(message, analysis_snapshot)
            return jsonify(
                {
                    "reply": f"{reply_text}\n\n_(External LLM returned `{exc}` — switched to built-in EmotiCoach guide.)_",
                    "mode": "guided-nlp",
                    "provider": "built-in",
                    "model": "EmotiCoach Guided Engine",
                    "actions": actions,
                    "quick_replies": quick_replies,
                }
            )

    reply_text, actions, quick_replies = guided_emotion_coach_reply(message, analysis_snapshot)
    return jsonify(
        {
            "reply": reply_text,
            "mode": "guided-nlp",
            "provider": "built-in",
            "model": "EmotiCoach Guided Engine (Add Gemini Key for Open Chat)",
            "actions": actions,
            "quick_replies": quick_replies,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
