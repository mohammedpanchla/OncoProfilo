<div align="center">

# 🧬 OncoProfilo v2

### Multi-Task Breast Cancer Subtype Classification + Survival Risk Prediction

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

**OncoProfilo v2** is a production-ready multi-task clinical AI system that classifies breast cancer molecular subtypes (PAM50) and predicts patient survival risk from raw RNA-seq gene expression data — simultaneously, in a single forward pass.

The system is trained entirely on real patient data from **TCGA-BRCA** (The Cancer Genome Atlas), streamed live from the GDC and UCSC Xena APIs — no static files, no local downloads. A leakage-safe preprocessing pipeline, class-weighted loss training, and 5-fold stratified cross-validation make this a credible machine learning system rather than an overfit demo.

v2 upgrades the original OncoProfilo architecture with a full baseline comparison sweep (Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM), smaller multi-task MLP variants with class-weighted cross-entropy, and a principled model selection cell that decides whether v2 outperforms the currently deployed model before saving any new artifacts.

---

## Clinical Problem Statement

Breast cancer is not one disease — it is at least five. The PAM50 molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like) have dramatically different prognoses and respond differently to treatment. Determining subtype from routine gene expression data, and simultaneously estimating survival risk from the same signal, is a core challenge in precision oncology.

OncoProfilo v2 demonstrates that a single multi-task neural network can solve both problems from raw FPKM expression values — and that class-weighted training closes the gap between majority and minority subtype performance.

---

## Architecture

```
TCGA-BRCA RNA-seq (GDC API)
         ↓
  460 patients × 60,660 genes (FPKM-UQ)
         ↓
  Top-k Variance Gene Selection (k = 500)
  Split-first — selected on training data only
         ↓
  Log1p Transform → StandardScaler
         ↓
┌────────────────────────────────────────┐
│        OncoProfiloV2 (Multi-Task MLP)  │
│                                        │
│  Input (500) → Linear → BN → GELU     │
│             → Linear → BN → GELU      │
│             → Linear → BN → GELU      │
│                    ↓                   │
│     Shared Encoder (256 → 128 → 64)   │
│           /              \             │
│  Subtype Head          Survival Head   │
│  Linear(64, 5)         Linear(64, 1)   │
│  PAM50 Classes         Cox Log-Risk    │
└────────────────────────────────────────┘
         ↓                    ↓
  5-Class Prediction    C-index / Risk Score
  (CrossEntropy)        (Cox-PH Loss)
```

### Model Components

| Module | Details |
|---|---|
| **Gene Selection** | Top-k variance selector — fit on training split only, no leakage |
| **Preprocessing** | Log1p transform → StandardScaler per split |
| **Shared Encoder** | 3-layer MLP (256 → 128 → 64) · BatchNorm · GELU · Dropout(0.35) |
| **Subtype Head** | Linear(64, 5) — PAM50 5-class classification |
| **Survival Head** | Linear(64, 1) — unconstrained Cox log-risk score |
| **Loss Function** | α × CrossEntropy(class_weighted) + β × CoxPH Loss |
| **Optimizer** | AdamW · ReduceLROnPlateau · Gradient clipping (max_norm=1.0) |
| **Explainability** | SHAP values on top selected genes |

### Why This Architecture?

**Multi-task learning** — Subtype classification and survival prediction share the same biological signal. A joint encoder forces the network to learn representations that are useful for both tasks, acting as a regulariser on the small TCGA training set.

**Class-weighted cross-entropy** — PAM50 labels are imbalanced (Luminal A dominates). Weighting each class inversely proportional to its frequency forces the network to treat minority subtypes as equally important during training, which directly improves Macro F1 over unweighted loss.

**Leakage-safe preprocessing** — Gene selection and scaling are fit exclusively on training data, then applied to val/test. This is the correct protocol. Many published notebooks get this wrong by selecting genes on the full dataset before splitting.

**Cox-PH loss** — The survival head is trained with a differentiable Cox partial likelihood that directly optimises rank concordance. The C-index is used only for monitoring — the loss optimises the right objective.

---

## Dataset

**TCGA-BRCA** — The Cancer Genome Atlas Breast Invasive Carcinoma

| Property | Value |
|---|---|
| Source | GDC API (streamed — no static files) |
| Expression workflow | STAR - Counts, FPKM-UQ |
| Total files queried | 1,200 |
| Successfully downloaded | 460 patients |
| Genes per patient | 60,660 ENSEMBL genes |
| PAM50 labels source | UCSC Xena (TCGA pan-cancer) |
| Clinical metadata | GDC Cases API (survival time, vital status) |
| Data split | 70 / 15 / 15 train / val / test |

### PAM50 Class Distribution

| Subtype | Biological Profile |
|---|---|
| Luminal A | Low-grade, ER+, best prognosis |
| Luminal B | ER+, higher proliferation |
| HER2-enriched | HER2 amplified, targeted therapy eligible |
| Basal-like | Triple-negative, aggressive, poorest prognosis |
| Normal-like | Resembles normal breast tissue |

The Luminal A dominance in TCGA-BRCA creates a class imbalance problem that standard accuracy obscures. This is why Macro F1 is used as the primary selection metric throughout v2.

---

## Experiment Design

v2 runs a structured three-stage experiment before deciding whether to promote new model artifacts:

```
Stage 1: Baseline Sweep
  LogReg_balanced / RandomForest_balanced / ExtraTrees_balanced
  XGBoost / LightGBM
  → 5-fold stratified CV + holdout evaluation

Stage 2: Multi-Task MLP Variants
  MLP_small_weighted  (256 → 128 → 64)  dropout=0.35
  MLP_tiny_weighted   (256 → 64)         dropout=0.40
  → Trained with class-weighted loss + Cox survival head

Stage 3: Model Selection
  → Compare all candidates on holdout Macro F1
  → Compare winner against current deployed app metrics
  → Save v2 artifacts only if v2 improves on v1
```

This structure means the notebook provides a portfolio signal regardless of outcome: if v2 wins, the upgrade is justified. If v2 does not win, the experiment itself demonstrates disciplined ML practice.

---

## Results

### Holdout Test Set (Final Model Comparison)

| Model | Family | Accuracy | Balanced Acc | Macro F1 | Macro AUC | C-Index |
|---|---|---|---|---|---|---|
| **MLP_small_weighted** | **MLP** | **0.608** | **0.585** | **0.564** | **0.830** | **0.874** |
| XGBoost | Baseline | 0.622 | 0.519 | 0.552 | 0.801 | — |
| ExtraTrees_balanced | Baseline | 0.635 | 0.524 | 0.545 | 0.736 | — |
| MLP_tiny_weighted | MLP | 0.581 | 0.544 | 0.532 | 0.798 | 0.797 |
| RandomForest_balanced | Baseline | 0.622 | 0.504 | 0.531 | 0.784 | — |
| LightGBM | Baseline | 0.622 | 0.514 | 0.530 | 0.754 | — |
| LogReg_balanced | Baseline | 0.554 | 0.471 | 0.506 | 0.764 | — |

**Best model: MLP_small_weighted** — highest Macro F1 (0.564) and Balanced Accuracy (0.585) across the full comparison. Accuracy is intentionally not used as the primary selector — Luminal A's frequency means a classifier that ignores minority subtypes can achieve high accuracy while being clinically useless.

**C-Index of 0.874** — the survival head achieves strong rank concordance, meaning the predicted risk scores correctly order patients by relative survival outcome in 87.4% of comparable pairs.

### Why MLP Wins on Macro F1 Despite Lower Raw Accuracy

Tree-based methods (Extra Trees: 63.5% accuracy) score higher on accuracy because they learn to exploit the Luminal A majority. The multi-task MLP with class-weighted loss distributes attention across all five subtypes, which hurts majority-class accuracy but substantially improves performance on Basal-like and HER2-enriched — the clinically critical minority classes.

---

## ML Pipeline

```
GDC API Query (TCGA-BRCA STAR-Counts FPKM-UQ)
↓
Parallel Download — 20 workers, ThreadPoolExecutor
  460 / 500 files successful · 5.2 min total
↓
Expression Matrix Assembly
  460 patients × 60,660 genes
↓
PAM50 Label Fetch — UCSC Xena API
Clinical Metadata Fetch — GDC Cases API
↓
Inner Join — expression + PAM50 + survival
↓
70 / 15 / 15 Stratified Split
↓
Log1p Transform (on train → apply to val/test)
Top-500 Variance Gene Selection (train only — no leakage)
StandardScaler (train only — no leakage)
↓
Stage 1: 5-Fold CV Baseline Sweep
  LogReg / RF / ExtraTrees / XGBoost / LightGBM
↓
Stage 2: Multi-Task MLP Training
  Class-weighted CrossEntropy + Cox-PH Loss
  AdamW · ReduceLROnPlateau · Early Stopping (patience=18)
↓
Stage 3: Holdout Evaluation + Model Selection
  Primary metric: Macro F1
↓
SHAP Explainability (top selected genes)
Kaplan-Meier Survival Curves by predicted subtype
↓
Artifact Export (models_v2/)
  best_model.pt / scaler.pkl / label_encoder.pkl / selected_genes.npy
```

---

## Key Engineering Decisions

### Leakage-Safe Preprocessing
Gene selection variance is computed **after** the train/val/test split, on the training set only. The same selected gene indices are then used to transform val and test. This is the correct protocol — and the one most commonly violated in published bioinformatics notebooks.

### Differential Loss Weighting
The combined loss `α × CrossEntropy + β × CoxPH` uses α=1.0 and β=0.25 for the best model. This weights classification as the primary objective while using survival as a regularising secondary task. The β parameter was swept across configurations.

### Parallel GDC Download
Sequential GDC downloads at ~1 file/sec would take ~8 hours for 500 files. The parallel implementation uses `ThreadPoolExecutor` with 20 workers and achieves 1.6 files/sec, completing in ~5 minutes. Thread-safe writes use a `threading.Lock`.

### Class Weight Computation
`sklearn.utils.compute_class_weight(class_weight='balanced')` computes per-class weights as `n_samples / (n_classes × class_count)`. These are passed directly to `nn.CrossEntropyLoss(weight=...)` and recomputed per fold to avoid leaking validation label distributions.

---

## Explainability

SHAP values are computed on the top selected genes to explain individual subtype predictions. This provides:

- Feature importance ranking for the 500 selected genes
- Per-patient explanation of why a specific PAM50 subtype was assigned
- Validation that the model is using biologically plausible gene signatures (ESR1 for Luminal, ERBB2 for HER2-enriched, BRCA1/BRCA2 for Basal-like)

Kaplan-Meier curves stratified by predicted PAM50 subtype are generated to verify that the model's risk scores produce statistically separable survival groups (log-rank test).

---

## Repository Structure

```
OncoProfilo/
├── app.py                           ← Gradio application (deployed model)
├── logic/                           ← Preprocessing + inference utilities
├── models/                          ← Deployed v1 model artifacts
│   ├── best_model.pt
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── selected_genes.npy
├── models_v2/                       ← v2 experiment artifacts (not deployed until promoted)
│   ├── best_model.pt
│   ├── model_config.json
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── selected_genes.npy
│   └── v2_experiment_summary.json
├── notebook/
│   └── OncoProfilo_v2_Model_Upgrade.ipynb   ← Full training + experiment notebook
├── results/                         ← Saved plots and metrics
├── static/                          ← Frontend assets
├── test_inputs/                     ← Sample inputs for validation
├── requirements.txt
└── README.md
```

---

## Running the Notebook

> **No data downloads required.** All data is streamed live from the GDC and UCSC Xena APIs on every run.

### Google Colab

Open `notebook/OncoProfilo_v2_Model_Upgrade.ipynb` in Colab and run all cells sequentially. Cell 1 installs all dependencies.

### VS Code

```bash
git clone https://github.com/muhammedpanchla/OncoProfilo.git
cd OncoProfilo
pip install -r requirements.txt
# Open notebook/OncoProfilo_v2_Model_Upgrade.ipynb
# Select Python 3.8+ kernel and run all cells
```

> The first cell (`%pip install ...`) must be run before any other cell. The GDC download cell takes approximately 5 minutes.

### Running the App

```bash
python app.py
```

Open: `http://127.0.0.1:7860`

---

## Technologies Used

**Deep Learning**
- PyTorch — model training, custom Cox-PH loss

**Machine Learning**
- scikit-learn — baselines, preprocessing pipeline, stratified CV
- XGBoost / LightGBM — gradient boosting baselines

**Survival Analysis**
- lifelines — Kaplan-Meier curves, log-rank test, concordance index

**Explainability**
- SHAP — feature attribution on selected gene set

**Data**
- GDC API — TCGA-BRCA RNA-seq expression files (STAR-Counts, FPKM-UQ)
- UCSC Xena API — PAM50 molecular subtype labels
- pandas / NumPy — data assembly and matrix operations

**Deployment**
- Gradio — web interface

---

## Limitations

- Trained on 460 TCGA-BRCA patients — a larger cohort would improve minority subtype performance
- Gene selection uses variance as a proxy for informativeness — supervised feature selection methods (e.g. mutual information) may yield better results
- The survival model uses normalized follow-up time, not raw days — this should be revisited for direct clinical use
- Not validated on external cohorts (METABRIC, SCAN-B)
- Not a certified medical device. All outputs are for research purposes only

---

## ⚠️ Disclaimer

> **Research use only.** OncoProfilo v2 is not a certified medical device and has not undergone clinical validation. All model outputs — including PAM50 subtype predictions and survival risk scores — are for research and educational purposes only. No output from this system should be used in clinical decision-making without review by a qualified medical professional. The author accepts no liability for clinical misuse.

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
