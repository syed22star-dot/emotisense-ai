# EmotiSense AI v2.0 — Affective NLP Studio & Interactive AI Agent

EmotiSense AI is a production Flask + scikit-learn 1.8.0 emotion classification studio built on 16,000 raw records cleaned and deduplicated to **15,939 conflict-free samples** across 6 emotion categories (`anger`, `fear`, `joy`, `love`, `sadness`, `surprise`), following `Emotion_Heart_Validation.ipynb`.

## Key Upgrades in v2.0

- **Corrected & Re-Benchmarked NLP Pipeline (`sklearn 1.8.0`):**
  - Removes 60 conflicting-label rows and 1 duplicate (`15,939` retained).
  - Preserves stopwords and negation tokens (`not`, `never`, `no`, `didnt`) with zero NLTK runtime dependency.
  - Eliminates `LabelEncoder` index-mismatch risk by training directly on string class labels (`estimator.classes_`).
  - **Promoted Primary Winner (`BoW + Logistic Regression`):** `87.95%` 5-fold CV accuracy, **`89.84%`** held-out test accuracy, `0.8591` macro F1.
  - **4-Model Consensus Arena:** Compares `BoW+LR`, `TF-IDF+LR` (`84.97%`), `BoW+MultinomialNB` (`74.50%`), and `TF-IDF+MultinomialNB` (`62.80%`) side by side.
- **Inline XAI Sentence Heatmap & Token Weights:**
  - Highlights supporting and opposing tokens directly inside the normalized text sequence.
- **Batch Journal & CSV Automation (`/api/batch`):**
  - Upload `.txt`/`.csv` or paste multi-line journal logs (up to 2,000 entries) for batch classification, mood charting, trigger discovery, and CSV export.
- **1-on-1 EmotiAgent AI (Pure LLM):**
  - Grounded with live text, token weights, 4-model consensus, and session mood telemetry. Supports **Google Gemini** (`GEMINI_API_KEY`, `gemini-4-hydrogen` via Interactions API) and **OpenAI** (`OPENAI_API_KEY`).

## Running Locally

```bash
pip install -r requirements.txt
python train_model.py   # Optional: artifacts are pre-generated
python app.py           # http://127.0.0.1:5000
```

## Deployment & Uptime

- **Vercel:** Ready via `vercel.json` + `api/index.py` + `/api/ping` cron keep-alive.
- **Render:** Ready via `render.yaml` (`gunicorn app:app --bind 0.0.0.0:$PORT`) + `/health`.
- **GitHub Actions Uptime:** `.github/workflows/uptime.yml` pings `EMOTISENSE_HEALTH_URL` every 10 minutes.
