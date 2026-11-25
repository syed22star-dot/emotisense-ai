"""
app.py
------
Clean modular Flask entry point for VibeCheck v3.0.
Delegates NLP feature extraction, multi-model inference, A/B experimentation,
and autonomous LLM UI orchestration to `core/`.
"""

import csv
import io
import os
import time
import warnings
from datetime import datetime, timezone

import sklearn
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from core.agent_service import EmotionAgentService
from core.live_pulse_service import LivePulseService
from core.nlp_service import EMOTION_META, EmotionNLPService

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

APP_START_TIME = time.time()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

nlp_service = EmotionNLPService(BASE_DIR)
live_pulse_service = LivePulseService(nlp_service)
agent_service = EmotionAgentService(nlp_service, live_pulse_service=live_pulse_service)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
@app.route("/api/ping", methods=["GET"])
def health():
    env_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    env_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return jsonify({
        "status": "healthy" if nlp_service.is_ready else "degraded",
        "service": "VibeCheck",
        "version": "3.0.0",
        "sklearn_version": sklearn.__version__,
        "models_loaded": list(nlp_service.loaded_models.keys()),
        "primary_model": nlp_service.get_primary_key(),
        "llm_agent": {
            "configured": env_gemini or env_openai,
            "autonomous_tools": True,
            "default_gemini_model": os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen"),
            "fallback_models": os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash"),
        },
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
def get_emotions():
    return jsonify({"emotions": list(EMOTION_META.keys()), "metadata": EMOTION_META})


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    return jsonify(nlp_service.metrics_cache)


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    model_key = str(data.get("model") or nlp_service.get_primary_key()).strip()
    compare_all = bool(data.get("compare", data.get("compare_models", True)))
    try:
        return jsonify(nlp_service.analyze_single(text, model_key=model_key, compare_all=compare_all))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ab-compare", methods=["POST"])
def ab_compare():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Provide text for A/B comparison."}), 400
    try:
        return jsonify(nlp_service.run_ab_compare(
            text,
            control_key=data.get("control", "bow_lr"),
            variant_key=data.get("variant", nlp_service.get_primary_key()),
        ))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    entries = []
    try:
        if "file" in request.files:
            f = request.files["file"]
            raw_bytes = f.read().decode("utf-8", errors="ignore")
            if f.filename.lower().endswith(".csv"):
                reader = csv.DictReader(io.StringIO(raw_bytes))
                fields = [fld.strip() for fld in (reader.fieldnames or [])]
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
        return jsonify(nlp_service.run_batch_lines(entries))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/tone-guard", methods=["POST"])
def tone_guard():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    goal = str(data.get("goal", "calm")).strip()
    provider = str(data.get("provider", "")).strip()
    try:
        return jsonify(live_pulse_service.run_tone_guard(text, goal=goal, request_headers=request.headers, provider_override=provider))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/pulse/movie", methods=["POST"])
def pulse_movie():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    data = request.get_json(silent=True) or {}
    title = str(data.get("title") or data.get("query") or "").strip()
    model_key = str(data.get("model") or nlp_service.get_primary_key()).strip()
    try:
        return jsonify(live_pulse_service.analyze_movie_vibe(title, model_key=model_key))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/pulse/topic", methods=["POST"])
def pulse_topic():
    if not nlp_service.is_ready:
        return jsonify({"error": "NLP models not loaded."}), 500
    data = request.get_json(silent=True) or {}
    topic = str(data.get("topic") or data.get("query") or "").strip()
    model_key = str(data.get("model") or nlp_service.get_primary_key()).strip()
    try:
        return jsonify(live_pulse_service.analyze_topic_pulse(topic, model_key=model_key))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(agent_service.handle_chat(data, request.headers))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
