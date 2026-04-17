# 🧬 OncoProfilo
### Breast Cancer Molecular Subtype Classification & Survival Risk Prediction

> Multi-task deep learning on TCGA-BRCA RNA-seq data — predicts PAM50 molecular subtype and Cox survival risk score from tumor gene expression profiles.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **66.18%** |
| AUC-ROC | **0.8176** |
| C-index (Survival) | **0.7231** |
| Macro F1 | **0.6328** |
| Training Patients | **338** |

---

## 🔬 What is This?

Breast cancer is **not one disease** — it's 5 molecularly distinct subtypes with completely different prognoses and treatment strategies. Standard clinical tests to identify these subtypes (like IHC or PAM50 assays) can be expensive and slow.

OncoProfilo uses a **multi-task neural network** trained on real patient RNA-seq data from TCGA to:

1. **Classify** the tumor into one of 5 PAM50 molecular subtypes
2. **Predict** survival risk using a Cox Proportional Hazards head
3. **Explain** the prediction via SHAP gene importance scores

### PAM50 Subtypes

| Subtype | Biology | Prognosis |
|---------|---------|-----------|
| **Luminal A** | Hormone receptor +ve, low proliferation | ✅ Best |
| **Luminal B** | Hormone receptor +ve, higher proliferation | 🟡 Good |
| **HER2-enriched** | HER2 gene amplified, aggressive | 🟠 Moderate |
| **Basal-like** | Triple negative (ER−, PR−, HER2−) | 🔴 Worst |
| **Normal-like** | Resembles normal breast tissue | ⚪ Variable |

---

## 🏗️ Architecture

```
Input: Top 2,000 most variable genes (log2 FPKM-UQ normalized)
          ↓
   Shared Encoder (MLP)
   [2000 → 512 → 256 → 128]
   BatchNorm + Dropout + ReLU
          ↓
    ┌─────┴──────┐
    ↓            ↓
Classification  Survival
   Head          Head
[128 → 5]    [128 → 1]
 PAM50        Cox Risk
 Subtype       Score
```

**Loss function:** Joint training with `CrossEntropyLoss` (classification) + `Cox PH loss` (survival), weighted sum.

---

## 📁 Project Structure

```
oncoprofilo/
├── app.py                    # FastAPI backend
├── logic/                    # Inference logic modules
├── models/                   # Trained model artifacts
│   ├── best_model.pt         # Trained PyTorch model weights
│   ├── scaler.pkl            # StandardScaler for gene normalization
│   ├── label_encoder.pkl     # LabelEncoder for subtype classes
│   ├── top_gene_indices.npy  # Indices of selected top 2000 genes
│   └── selected_genes.npy    # Selected gene IDs
├── static/
│   └── index.html            # Frontend UI
├── notebook/
│   └── OncoProfilo.ipynb     # Full training notebook
├── test_inputs/              # Sample JSON inputs for testing
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Steps

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/oncoprofilo.git
cd oncoprofilo

# Build and run
docker compose up --build
```

Or using plain Docker:

```bash
docker build -t oncoprofilo .
docker run --rm -p 7860:7860 oncoprofilo
```

Then open **http://localhost:7860** in your browser.

---

## 🧪 Run the Notebook

The training notebook streams all data directly from the cloud via the **GDC (NIH) API** and **UCSC Xena** — no local data downloads needed.

### In VS Code
1. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
2. Open `notebook/OncoProfilo.ipynb`
3. Select a Python 3.8+ kernel
4. Run **Cell 1** first to install dependencies:
   ```
   %pip install lifelines shap gradio requests pandas numpy scikit-learn matplotlib seaborn torch
   ```
5. Run all cells top to bottom

### In Google Colab
Upload the notebook to [colab.research.google.com](https://colab.research.google.com) — it works identically.

---

## 🗂️ Data Pipeline

```
GDC API (NIH)                    UCSC Xena
     ↓                               ↓
TCGA-BRCA RNA-seq              PAM50 Subtype Labels
FPKM-UQ values                 (PAM50Call_RNAseq)
419 patients / 60,660 genes     956 labeled samples
     ↓                               ↓
          Inner join on case_id
                  ↓
           338 final patients
                  ↓
     Log2 normalization → Variance filtering
     → Top 2,000 genes → StandardScaler
                  ↓
         Train / Val / Test split
```

**Dataset:** TCGA-BRCA — The Cancer Genome Atlas Breast Cancer cohort (~1,100 patients, RNA-seq FPKM-UQ, clinical outcomes with survival days and vital status)

---

## 🔍 Explainability — SHAP Gene Importance

OncoProfilo uses **SHAP DeepExplainer** to identify which genes drove each prediction:

- Global feature importance across all subtypes
- Per-subtype SHAP gene rankings
- Per-sample top-10 contributing genes returned with every prediction

This allows clinicians and researchers to see *why* the model made a specific call, not just *what* it predicted.

---

## 📡 API Usage

The FastAPI backend exposes a `/predict` endpoint. Send a gene expression profile as JSON:

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"ENSG00000000003": 12.5, "ENSG00000000419": 22.4, ...}'
```

**Response:**
```json
{
  "predicted_subtype": "Basal",
  "confidence": "99.9%",
  "subtype_probabilities": {
    "Basal": "99.9%",
    "Her2": "0.0%",
    "LumA": "0.0%",
    "LumB": "0.0%",
    "Normal": "0.0%"
  },
  "survival_risk_score": "0.5017 (0=low risk, 1=high risk)",
  "risk_level": "HIGH",
  "top_contributing_genes": {
    "ENSG00000141510": 0.842,
    "ENSG00000012048": 0.731
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch |
| Survival Analysis | lifelines (Cox PH, Kaplan-Meier) |
| Explainability | SHAP DeepExplainer |
| Data | TCGA-BRCA via GDC API + UCSC Xena |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker → Hugging Face Spaces |

---

## 🗺️ Roadmap

- [x] v1 — Multi-task MLP (classification + survival) on RNA-seq
- [ ] v2 — Multi-omics fusion (RNA-seq + DNA methylation) with attention
- [ ] v3 — Transformer-based architecture for gene interactions
- [ ] Add TCGA pan-cancer support (not just BRCA)

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It is not a clinical diagnostic tool and should not be used to make medical decisions. Always consult qualified medical professionals.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by **Muhammed Panchla** | [Flowgenix AI](https://flowgenix.ai)*
