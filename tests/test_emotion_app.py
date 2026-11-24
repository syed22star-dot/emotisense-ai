"""Tests for EmotiSense AI v3.0 NLP feature extraction, A/B analytics, API routes, and agent."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from core.features import AffectiveSyntaxExtractor, analyze_syntax_signals, preprocess


def test_syntax_extractor_and_negation_guard():
    diag = analyze_syntax_signals("I didn't feel angry at all, actually I felt relieved!")
    assert "did not" in diag["normalized_text"]
    assert "not" in diag["negations_found"]
    assert "actually" in diag["pivot_cues"]
    assert len(diag["negated_negative_cues"]) >= 1
    assert diag["emotion_lex_hits"]["anger"] == 0.0
    assert diag["emotion_lex_hits"]["joy"] > 0.0

    mat = AffectiveSyntaxExtractor().fit_transform(["i am not angry", "soooo happy"])
    assert mat.shape == (2, 15)
    assert mat.min() >= 0.0


def test_health_emotions_benchmarks_and_ab_routes():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    hd = r.get_json()
    assert hd["status"] == "healthy"
    assert hd["version"] == "3.0.0"

    re_ = client.get("/api/emotions")
    assert re_.status_code == 200
    assert set(re_.get_json()["emotions"]) == {"anger", "fear", "joy", "love", "sadness", "surprise"}

    rb = client.get("/api/benchmarks")
    assert rb.status_code == 200
    bm = rb.get_json()
    assert len(bm["benchmarks"]) >= 6
    assert len(bm["ablation"]) >= 6
    assert "bow_vs_hybrid_ensemble" in bm["ab_tests"]
    assert len(bm["stress_suite"]["cases"]) == 10

    rp = client.post("/api/predict", json={"text": "I did not feel angry at all, actually I felt relieved and grateful.", "model": "hybrid_ensemble"})
    assert rp.status_code == 200
    pr = rp.get_json()
    assert pr["emotion"] == "joy"
    assert "ab_live" in pr
    assert "syntax_signals" in pr

    rab = client.post("/api/ab-compare", json={"text": "I am terrified of losing this chance", "control": "bow_lr", "variant": "hybrid_ensemble"})
    assert rab.status_code == 200
    assert rab.get_json()["variant"]["emotion"] == "fear"

    rbat = client.post("/api/batch", json={"lines": ["I am thrilled!", "I feel lonely tonight."]})
    assert rbat.status_code == 200
    assert rbat.get_json()["summary"]["total_entries"] == 2
