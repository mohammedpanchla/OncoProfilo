
<div align="center">

# 🧬 OncoProfilo

### Breast Cancer Subtype Classification & Survival Risk Prediction from RNA-seq

**TCGA-BRCA · PAM50 · Multi-Task MLP · Cox-PH · SHAP · GDC API**

*Built by [Muhammed Panchla](https://www.linkedin.com/in/flowgenix-ai-b51517278) · Flowgenix AI*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?style=flat-square&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=flat-square)

</div>

---

## Overview

**OncoProfilo** is an end-to-end clinical AI system that classifies breast cancer molecular subtypes (PAM50) and predicts patient survival risk from raw RNA-seq gene expression data — simultaneously, in a single forward pass.

The system is trained entirely on real patient data from **TCGA-BRCA** (The Cancer Genome Atlas Breast Invasive Carcinoma cohort), streamed live from the GDC API — no static files, no local downloads. A multi-task MLP with a shared encoder solves both classification and survival prediction from the same gene expression signal, and SHAP DeepExplainer provides per-patient gene-level explanations for every prediction.

The trained model is deployed as an interactive Gradio application with a demo mode (real TCGA test sample) and a custom JSON input mode for arbitrary gene expression profiles.

---

## Clinical Problem Statement

Breast cancer is not one disease — it is at least five molecularly distinct subtypes with dramatically different prognoses and treatment responses. Determining a patient's PAM50 subtype from their tumor's gene expression profile, and simultaneously estimating their survival risk from the same biological signal, is a core challenge in precision oncology.

| Subtype | Biological Profile | Prognosis |
|---|---|---|
| **Luminal A** | ER+, slow-growing, low proliferation | Best |
| **Luminal B** | ER+, faster-growing, higher proliferation | Good |
| **HER2-enriched** | HER2 gene amplified, targeted therapy eligible | Moderate |
| **Basal-like** | Triple-negative, most aggressive | Worst |
| **Normal-like** | Resembles normal breast tissue | Variable |

Each subtype has a unique gene expression signature. RNA-seq measures how active each gene is in a tumor. OncoProfilo learns to recognise these signatures and simultaneously output a survival risk score — all from the same 2,000 most variable genes.

---

## Architecture

```
TCGA-BRCA RNA-seq (GDC API — streamed, no local save)
              ↓
  419 patients × 60,660 genes (FPKM-UQ)
              ↓
  Top-2000 Variance Gene Selection
  Log2 Normalization → StandardScaler
              ↓
┌─────────────────────────────────────────────┐
│           OncoProfilo (Multi-Task MLP)       │
│                                             │
│  Input (2000)                               │
│      → Linear(2000, 512) → BN → ReLU       │
│      → Dropout(0.40)                        │
│      → Linear(512, 256)  → BN → ReLU       │
│      → Dropout(0.30)                        │
│      → Linear(256, 128)  → BN → ReLU       │
│                  ↓                          │
│         Shared Encoder (128-dim)            │
│              /            \                 │
│   Subtype Head          Survival Head       │
│  Linear → ReLU          Linear → ReLU      │
│  → Dropout(0.2)         → Dropout(0.2)     │
│  → Linear(64, 5)        → Linear(64, 1)    │
│  PAM50 Classes          → Sigmoid          │
│                          Cox Risk Score     │
└─────────────────────────────────────────────┘
         ↓                       ↓
   5-Class Prediction       Survival Risk
   (CrossEntropyLoss)       (Cox-PH Loss)
         ↓                       ↓
   SHAP DeepExplainer       Kaplan-Meier
   Gene Importance          Risk Stratification
```

### Model Components

| Component | Details |
|---|---|
| **Gene Selection** | Top-2,000 most variable genes from 60,660 ENSEMBL genes |
| **Preprocessing** | Log2 normalization → StandardScaler |
| **Shared Encoder** | 3-layer MLP (2000 → 512 → 256 → 128) · BatchNorm · ReLU · Dropout |
| **Subtype Head** | Linear(128→64) → ReLU → Dropout → Linear(64→5) — 5-class PAM50 |
| **Survival Head** | Linear(128→64) → ReLU → Dropout → Linear(64→1) → Sigmoid — Cox risk |
| **Total Parameters** | 1,207,430 (all trainable) |
| **Loss Function** | Total = 1.0 × CrossEntropy + 0.5 × Cox-PH |
| **Explainability** | SHAP DeepExplainer — global, per-subtype, and per-sample gene importance |

### Why This Architecture?

**Multi-task learning** — Subtype classification and survival prediction share the same underlying biological signal. A joint shared encoder forces the network to learn representations that are meaningful for both tasks simultaneously, acting as a natural regulariser on the limited TCGA training set size.

**Cox Proportional Hazards loss** — Standard regression asks how close the predicted number is to the actual number. Cox loss asks a biologically correct question: *did the model correctly rank patients by relative risk?* For a patient who died at time T, the Cox partial likelihood compares their predicted risk against all patients still alive at T. If higher risk was correctly assigned to the patient who died, loss is low. This rank-based objective is what the C-index directly measures.

**SHAP DeepExplainer** — Gene expression models are meaningless without knowing which genes drove each prediction. SHAP values decompose the model's output into per-gene contributions for each patient, making individual predictions auditable and biologically interpretable.

---

## Dataset

**TCGA-BRCA** — The Cancer Genome Atlas Breast Invasive Carcinoma

| Property | Value |
|---|---|
| Source | GDC API — STAR-Counts, FPKM-UQ (no local download) |
| Total files queried | 1,200 |
| Successfully downloaded | 419 patients |
| Genes per patient | 60,660 ENSEMBL genes |
| Genes used (after selection) | 2,000 (top variance) |
| PAM50 labels | UCSC Xena (TCGA pan-cancer clinical data) |
| Clinical data | GDC Cases API — survival days, vital status |
| Data split | 202 train / 68 val / 68 test |

---

## Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Max epochs | 80 |
| Early stopping | Patience = 15 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Batch size | 32 |
| Gradient clipping | max_norm = 1.0 |
| Classification loss weight (α) | 1.0 |
| Survival loss weight (β) | 0.5 |

Training completed with early stopping at **epoch 19**. Best validation loss: **1.7963**.

---

## Results

### Test Set Performance (68 patients, held out)

| Metric | Score |
|---|---|
| **Accuracy** | **66.2%** |
| **Macro F1 Score** | **0.6328** |
| **Macro AUC-ROC** | **0.8176** |
| **C-Index (Survival)** | **0.7231** |

> C-Index > 0.65 is considered clinically meaningful. 0.5 = random ordering.

### Per-Subtype Classification Report

| Subtype | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Basal | 0.88 | 0.78 | 0.82 | 9 |
| Her2 | 0.60 | 0.60 | 0.60 | 5 |
| LumA | 0.88 | 0.69 | 0.77 | 32 |
| LumB | 0.53 | 0.75 | 0.62 | 12 |
| Normal | 0.31 | 0.40 | 0.35 | 10 |

The most common confusion is **LumA ↔ LumB** — biologically expected, as these subtypes are transcriptomically adjacent and differ primarily in proliferation rate rather than receptor status. Basal-like achieves the highest F1 (0.82), reflecting its distinct triple-negative gene expression signature.

### Survival Risk Stratification

Patients split at the median predicted risk score into High/Low groups show separated Kaplan-Meier curves (log-rank p = 0.057). The C-Index of **0.7231** confirms the survival head correctly ranks 72.3% of patient pairs by relative risk — well above the 0.5 random baseline.

---

## ML Pipeline

```
GDC API Query (TCGA-BRCA STAR-Counts FPKM-UQ)
↓
Parallel Download — 20 workers, ThreadPoolExecutor
  419 / 500 files successful · 4.9 min total
↓
Expression Matrix Assembly
  419 patients × 60,660 genes
↓
PAM50 Label Fetch — UCSC Xena API
Clinical Metadata Fetch — GDC Cases API (survival days, vital status)
↓
Inner Join — expression + PAM50 + clinical → 338 matched patients
↓
Log2 Normalization → Top-2000 Variance Gene Selection → StandardScaler
↓
Stratified Split → 202 train / 68 val / 68 test
↓
Multi-Task MLP Training
  CrossEntropy (α=1.0) + Cox-PH Loss (β=0.5)
  AdamW · ReduceLROnPlateau · Early stopping at epoch 19
  Gradient clipping (max_norm=1.0)
↓
Evaluation
  Accuracy · Macro F1 · Macro AUC-ROC (classification)
  C-Index · Kaplan-Meier · Log-rank test (survival)
↓
SHAP DeepExplainer
  Global gene importance · Per-subtype attribution · Per-sample top-10 genes
↓
Gradio App — Demo mode (real TCGA sample) + Custom JSON input
```

---

## Explainability

SHAP DeepExplainer computes gene-level attribution values for every prediction. Three levels of explanation are produced:

**Global importance** — Mean absolute SHAP across all samples and classes. Identifies which of the 2,000 selected genes most consistently influence subtype predictions across the entire cohort.

**Per-subtype attribution** — Separate SHAP profiles for each PAM50 class, showing which genes drive Basal vs LumA vs HER2 predictions specifically.

**Per-sample top-10 genes** — For every individual prediction in the Gradio app, the top 10 genes contributing to the predicted subtype are returned with their SHAP magnitudes. This is the level of explanation required for any real clinical decision support tool.

---

## Gradio Application

The deployed app provides two inference modes:

**Demo Prediction** — One click runs a complete prediction on a real TCGA-BRCA test sample. Output includes predicted subtype, confidence, probability bar chart for all 5 classes, survival risk score and HIGH/LOW risk level, and the top 10 contributing genes with SHAP values.

**Custom Input (JSON)** — Paste any gene expression profile as a JSON dictionary mapping ENSEMBL gene IDs to FPKM values. The model normalises, selects the relevant 2,000 genes, and returns a structured JSON prediction.

```
Input format: {"ENSG00000000003": 12.5, "ENSG00000000005": 0.0, ...}
```

---

## Repository Structure

```
OncoProfilo/
├── app.py                           ← Gradio application
├── logic/                           ← Preprocessing + inference utilities
├── models/                          ← Trained model artifacts
│   ├── best_model.pt                ← PyTorch model weights (epoch 19)
│   ├── scaler.pkl                   ← Fitted StandardScaler
│   ├── label_encoder.pkl            ← PAM50 label encoder
│   └── selected_genes.npy           ← Top-2000 selected gene IDs
├── notebook/
│   └── OncoProfilo.ipynb            ← Full training notebook
├── results/                         ← Saved plots (EDA, training curves, SHAP, KM)
├── static/                          ← Frontend assets
├── test_inputs/                     ← Sample JSON inputs for validation
├── requirements.txt
└── README.md
```

---

## Running the Notebook

> **No data downloads required.** All TCGA-BRCA data is streamed live from the GDC and UCSC Xena APIs on every run. Nothing is saved to disk.

### Google Colab

Open `notebook/OncoProfilo.ipynb` in Colab and run all cells sequentially. Cell 1 installs all dependencies.

### VS Code

```bash
git clone https://github.com/muhammedpanchla/OncoProfilo.git
cd OncoProfilo
pip install -r requirements.txt
# Open notebook/OncoProfilo.ipynb
# Select Python 3.8+ kernel and run Cell 1 first
```

> The parallel GDC download cell takes approximately 5 minutes. Do not interrupt it.

### Running the App

```bash
python app.py
```

Open: `http://127.0.0.1:7860`

---

## Technologies Used

**Deep Learning**
- PyTorch — model architecture, custom Cox-PH loss, training loop

**Machine Learning**
- scikit-learn — variance gene selection, StandardScaler, stratified split, evaluation metrics

**Survival Analysis**
- lifelines — KaplanMeierFitter, log-rank test, concordance index

**Explainability**
- SHAP — DeepExplainer for global, per-subtype, and per-sample gene attribution

**Data**
- GDC API — TCGA-BRCA RNA-seq files (STAR-Counts FPKM-UQ)
- UCSC Xena API — PAM50 molecular subtype labels
- pandas / NumPy — expression matrix assembly and preprocessing

**Deployment**
- Gradio — interactive web application with demo and custom JSON input modes

---

## Limitations

- 338 matched patients after joining expression + PAM50 + clinical — a larger cohort would improve minority subtype performance, particularly Normal-like (F1 = 0.35)
- Top-variance gene selection is unsupervised — supervised selection methods (mutual information, ANOVA-F) may identify more informative gene sets
- Early stopping at epoch 19 suggests the model may benefit from stronger regularisation and longer training runs
- Not validated on external cohorts (METABRIC, SCAN-B)
- Not a certified medical device. All outputs are for research purposes only

---

## ⚠️ Disclaimer

> **Research use only.** OncoProfilo is not a certified medical device and has not undergone clinical validation. All model outputs — including PAM50 subtype predictions and survival risk scores — are for research and educational purposes only. No output from this system should be used in clinical decision-making without review by a qualified medical professional. The author accepts no liability for clinical misuse.

---

## Author

**Muhammed Panchla** — Flowgenix AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flowgenix_AI-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/flowgenix-ai-b51517278)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-muhammedpanchla-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/muhammedpanchla)
[![GitHub](https://img.shields.io/badge/GitHub-muhammedpanchla-181717?style=flat-square&logo=github)](https://github.com/muhammedpanchla)

---

<div align="center">
<sub>Built by Muhammed Panchla · Flowgenix AI · 2026</sub>
</div>
