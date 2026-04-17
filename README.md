---
title: OncoProfilo
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🧬 OncoProfilo
**Breast Cancer Molecular Subtype Classification & Survival Risk Prediction**

Multi-task MLP trained on TCGA-BRCA RNA-seq data. Predicts PAM50 subtype (LumA / LumB / HER2 / Basal / Normal) and Cox survival risk score from gene expression profiles.

| Metric | Score |
|--------|-------|
| Test Accuracy | 66.2% |
| AUC-ROC | 0.8176 |
| C-index | 0.7231 |
| Macro F1 | 0.6328 |

## 📁 File Structure

```
oncoprofilo/
├── app.py                  # FastAPI backend
├── static/
│   └── index.html          # Frontend UI
├── models/
│   ├── best_model.pt
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── top_gene_indices.npy
│   └── selected_genes.npy
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── risk_stratification.png
│   └── shap_importance.png
├── test_inputs/
│   ├── basal_like.json
│   ├── her2_like.json
│   ├── luma_like.json
│   ├── lumb_like.json
│   └── normal_like.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space → SDK: **Docker**
2. Upload only the files listed in the structure above
3. Keep the model files inside the `models/` folder
4. Push — HF Spaces builds and runs automatically on port 7860

## Required For Hugging Face Space

These files are required for your deployed Space:

```text
README.md
Dockerfile
requirements.txt
app.py
static/index.html
models/best_model.pt
models/scaler.pkl
models/label_encoder.pkl
models/top_gene_indices.npy
models/selected_genes.npy
results/confusion_matrix.png
results/training_curves.png
results/risk_stratification.png
results/shap_importance.png
test_inputs/basal_like.json
test_inputs/her2_like.json
test_inputs/luma_like.json
test_inputs/lumb_like.json
test_inputs/normal_like.json
```

## Not Needed For Hugging Face Space

You can keep these locally, but they do not need to go to the Space:

```text
notebook/
models_v2/
logic/
walkthrough
walkthrough.md
docker-compose.yml
run_local.sh
.venv/
__pycache__/
.DS_Store
test_inputs/*.csv
test_inputs/expected_results.json
results/eda_overview.png
results/kaplan_meier.png
results/shap_per_class.png
results/v2_confusion_matrix.png
```

Without model files the app runs in **demo-only mode**.

## Run Locally With Docker

This repository already includes the app code, static frontend, and model artifacts in `models/`, so you can run it locally in Docker without opening the notebooks.

### 1. Install Docker Desktop

On macOS, install Docker Desktop and make sure it is running.

### 2. Open Terminal in this project

From the project root, run:

```bash
docker compose up --build
```

If you prefer plain Docker instead of Compose:

```bash
docker build -t oncoprofilo .
docker run --rm -p 7860:7860 oncoprofilo
```

### 3. Open it in Safari

Once the container starts, open:

```text
http://localhost:7860
```

The API and frontend are served from the same app, so Safari should work normally on that URL.

### 4. Stop the app

Press `Ctrl+C` in the terminal, or if you used Compose:

```bash
docker compose down
```

### Notes

- The current app loads model files from `models/` first.
- `models_v2/` exists in the repo, but `app.py` is currently wired to use `models/` unless you change the code.
- If Docker says the command is not found, install Docker Desktop first and then reopen Terminal.
- `.dockerignore` excludes local-only files from Docker builds so your Space upload stays cleaner.

---
*Built by Muhammed Panchla | Flowgenix AI*
