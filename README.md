# EmotiSense AI v3.0 — Affective NLP Studio, Feature Ablation & A/B Suite

EmotiSense AI v3.0 is a modular Flask + scikit-learn 1.8.0 multi-class emotion classification platform trained on **15,939 conflict-free texts** across 6 emotions (`anger`, `fear`, `joy`, `love`, `sadness`, `surprise`).

## Architecture & Package Layout

```text
emotion-classifier/
├── app.py                        # Clean Flask route controllers
├── train_model.py                # 7-stage ablation, 7-model CV, McNemar & Bootstrap, Stress Suite
├── preprocessor.py               # Backward-compatible re-export of core.features.preprocess
├── core/
│   ├── features.py               # Contraction normalization, AffectiveSyntaxExtractor (15 cues), FeatureUnion
│   ├── research_stats.py         # McNemar's test, vectorized paired bootstrap 95% CI, 10-bin ECE/Brier
│   ├── nlp_service.py            # Multi-model inference, token heatmap XAI, Live A/B, Batch Analyzer
│   ├── agent_service.py          # Autonomous 2-pass EmotiCoach AI driving UI & NLP tools
│   └── llm_client.py             # Resilient Gemini (4 Hydrogen -> 3.5 Flash Lite) / OpenAI client
└── tests/
    └── test_emotion_app.py       # Pytest unit & integration suite
```

## Key NLP & Data Science Upgrades in v3.0

- **Multi-Granularity Feature Extraction & Scaling (`FeatureUnion`):**
  - Combines scaled `(1,2)`-gram Bag-of-Words (`MaxAbsScaler`), sublinear `(1,2)`-gram TF-IDF, `(3,5)` character n-grams (`char_wb`) for elongation/typo resilience, and 15 `AffectiveSyntaxExtractor` features scaled via `MaxAbsScaler` (`+28` net discordant wins over Unigram BoW+LR).
  - Systematic 7-stage ablation confirms `MaxAbsScaler` (`88.80%` CV / `90.24%` Test) outperforms `StandardScaler(with_mean=False)` (`88.24%` CV / `89.59%` Test) on sparse NLP matrices.
- **Champion & Challenger Matrix (`n=3,188` held-out test set):**
  - **`hybrid_ensemble` (Soft Voting LR + CalibratedSVC + ComplementNB):** **`90.46%` Test Accuracy**, **`86.50%` Macro F1**, **`0.1471` Brier** (`97.0%` bootstrap accuracy win rate & strictly lower Brier CI `[-0.0264, -0.0104]` vs Control BoW+LR).
  - **`hybrid_svc` (Calibrated LinearSVC):** **`88.80%` 5-fold CV**, `90.24%` Test, lowest `0.1460` Brier.
  - **`bow_lr` (Control Baseline):** `88.06%` CV, `89.59%` Test.
- **Adversarial Stress Suite & Live A/B Arena (`/api/ab-compare`):**
  - Evaluates contrast pivots, negation flips, character elongations, and minority classes side-by-side (`8/10` Hybrid pass vs `6/10` BoW Baseline).

## Running Locally & Tests

```bash
pip install -r requirements.txt
pytest tests/test_emotion_app.py -v
python app.py   # http://127.0.0.1:5000
```

## Environment Variables

Configure via `.env` locally (never committed) or Render/Vercel env vars (see `.env.example`):
- `GEMINI_API_KEY` (uses `GEMINI_MODEL=gemini-4-hydrogen` with automatic `GEMINI_FALLBACK_MODELS`)
- `OPENAI_API_KEY` (optional alternative)
