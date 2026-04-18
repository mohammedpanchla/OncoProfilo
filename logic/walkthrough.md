# OncoProfilo v2 — Notebook Walkthrough

This document explains the **OncoProfilo_v2_Model_Upgrade.ipynb** notebook from two angles: **the biology behind the project** and then **the technical (code) details cell by cell**.

---

## Part 1: The Biological Context (For Beginners)

### What is this project about?

OncoProfilo is a machine learning project that tries to solve a real-world **breast cancer** problem. Specifically, it takes **gene expression data** from breast cancer patients and uses it to do two things:

1. **Classify the cancer subtype** (PAM50 classification)
2. **Predict patient survival** (how long a patient is likely to live)

### Key Biological Concepts

#### 1. Breast Cancer & Gene Expression

Every cell in your body has DNA — a set of instructions. When a cell "reads" part of its DNA, it creates molecules called **RNA**, which then produce **proteins** that do the cell's work. The process of reading DNA into RNA is called **gene expression**.

In cancer, certain genes get "turned up" or "turned down" abnormally. By measuring how active each gene is (its **expression level**), scientists can learn about the cancer's behavior. The technology used here is called **RNA-seq** (RNA sequencing), which measures the expression of tens of thousands of genes simultaneously.

#### 2. TCGA-BRCA Dataset

**TCGA** stands for **The Cancer Genome Atlas** — a massive government-funded project that collected and analyzed cancer samples from thousands of patients. **BRCA** stands for **Breast Cancer**. This dataset contains:

- **RNA-seq gene expression data** for ~1,200 breast cancer patients
- **Clinical metadata**: whether the patient is alive or dead, how many days they survived
- **PAM50 subtype labels**: a classification system for breast cancer

#### 3. PAM50 Subtypes

**PAM50** is a clinical test that uses the expression of **50 specific genes** to classify breast cancers into 5 subtypes:

| Subtype | Behavior | Prognosis |
|---------|----------|-----------|
| **Luminal A (LumA)** | Hormone-receptor positive, slow-growing | Best prognosis |
| **Luminal B (LumB)** | Hormone-receptor positive, faster-growing | Good, but worse than LumA |
| **HER2-enriched (Her2)** | HER2 protein overexpressed | More aggressive |
| **Basal-like (Basal)** | Often "triple-negative" | Most aggressive |
| **Normal-like (Normal)** | Resembles normal breast tissue | Variable |

Knowing the subtype helps doctors decide which treatment to use (e.g., hormone therapy for Luminal types, targeted therapy for HER2).

#### 4. FPKM (Gene Expression Units)

Gene expression is measured in units called **FPKM** (Fragments Per Kilobase of transcript per Million mapped reads). Think of it as a "volume knob" for each gene — a higher number means the gene is more active in that patient's tumor.

#### 5. Survival Analysis & Concordance Index (C-Index)

- **Survival time**: How many days after diagnosis the patient lived (or was last seen alive)
- **Vital status**: "Dead" or "Alive" at the time of the study
- **Censored**: If a patient was still alive when the study ended, we know they survived *at least* X days, but not exactly how long
- **C-Index**: A number between 0 and 1 that measures how well a model predicts who lives longer. 0.5 = random guessing, 1.0 = perfect predictions

#### 6. Why Machine Learning?

With ~60,000 genes and only hundreds of patients, humans can't manually find patterns. ML models can:
- Identify which genes matter most for predicting cancer subtype
- Find complex patterns that differentiate subtypes
- Predict patient outcomes

---

## Part 2: Cell-by-Cell Technical Walkthrough

### Section 1: Install & Imports

#### Cell 1: Markdown — Title and Purpose

This introductory cell explains what the v2 notebook adds over the original:
- **Leakage-safe preprocessing**: Split data before selecting features (prevents the model from "cheating" by seeing test data during training)
- **Baseline comparisons**: Test simpler models alongside the neural network
- **Smaller MLP variants**: Reduce model size since there are few training samples
- **5-fold cross-validation**: More reliable evaluation than a single train/test split

#### Cell 2: Package Installation

```python
%pip install lifelines shap gradio requests pandas numpy scikit-learn matplotlib seaborn torch xgboost lightgbm -q
```
Installs all required Python libraries:
- `pandas`, `numpy`: Data handling
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Classical machine learning algorithms
- `torch` (PyTorch): Deep learning framework
- `lifelines`: Survival analysis (Kaplan-Meier curves, concordance index)
- `shap`: Model explainability
- `xgboost`, `lightgbm`: Gradient boosting classifiers (strong baselines)

#### Cell 3: Python Imports + Setup

This cell imports all needed libraries and sets up:
- **Reproducibility seed** (`SEED = 42`): Ensures results are repeatable
- **Device detection**: Checks if GPU (CUDA) is available; falls back to CPU
- **Optional boosting imports**: Tries to import XGBoost and LightGBM, gracefully skips if unavailable

---

### Section 2: Data Loading

#### Cell 4: GDC API Query

```python
GDC_FILES_URL = 'https://api.gdc.cancer.gov/files'
```
Queries the **Genomic Data Commons (GDC) API** — a public cancer data portal — for TCGA-BRCA RNA-seq files. The filter specifies:
- Project: `TCGA-BRCA` (breast cancer)
- Data type: Gene Expression Quantification
- Workflow: `STAR - Counts` (a specific RNA-seq processing pipeline)
- Format: TSV (tab-separated values)

Result: Finds **1,200** RNA-seq files

#### Cell 5: Data Volume Control

```python
MAX_EXPRESSION_FILES = 500
```
Limits downloads to 500 files to keep the prototype manageable. Set to `None` for the full dataset.

#### Cell 6: Parallel Download

Downloads individual gene expression files from GDC using **20 parallel threads** (via `ThreadPoolExecutor`). For each file:
1. Extracts the `case_id` (patient identifier like `TCGA-5L-AAT1`)
2. Downloads the TSV file
3. Parses it to get `gene_id → fpkm_unstranded` values
4. Only keeps Ensembl gene IDs (starting with `ENSG`)

Result: ~460 successful downloads in ~5 minutes

#### Cell 7: Expression Matrix Construction

Builds a big table (DataFrame) with:
- **Rows** = patients (460)
- **Columns** = genes (60,660)
- Each cell = the FPKM expression level of that gene in that patient

Missing values are filled with 0 (gene not detected = no expression).

#### Cell 8: Clinical Data Download

Downloads clinical metadata from GDC for each patient:
- `vital_status`: "Alive" or "Dead"
- `survival_days`: How many days the patient survived
- An `event` column: 1 if dead (uncensored), 0 if alive (censored)

#### Cell 9: PAM50 Subtype Labels

Downloads PAM50 subtype labels from **UCSC Xena** (a public genomics data browser). The labels classify each tumor into one of 5 subtypes: LumA, LumB, Her2, Basal, Normal.

If the download fails, falls back to simulated random subtypes for testing.

#### Cell 10: Merge All Data

Performs an **inner join** of expression data + clinical data + subtype labels. Only keeps patients present in all three datasets. Filters out missing survival data and unknown subtypes.

Result: **368 patients** with complete data.

---

### Section 3: Exploratory Data Analysis (EDA)

#### Cells 11-12: EDA Visualizations

Creates a 4-panel dashboard:

1. **Subtype Distribution** (bar chart): Shows LumA dominates (~176 patients), while Her2 has only ~26 — this **class imbalance** is a key challenge
2. **Survival Distribution** (histogram): Median survival ~1,100 days; some patients survive >5,000 days
3. **Vital Status** (pie chart): ~87% alive, ~13% dead at time of study
4. **Survival by Subtype** (boxplot): Shows survival differences between subtypes (Basal tends to have shorter survival)

Also generates a **Kaplan-Meier survival curve** per subtype, showing how survival probability decreases over time for each subtype. A **log-rank test** checks whether the survival differences between subtypes are statistically significant.

---

### Section 4: Leakage-Safe Preprocessing

> [!IMPORTANT]
> One of the most important improvements in v2 over v1!

#### What is "Data Leakage"?

Data leakage occurs when information from the test set accidentally influences the training process. If you select the top N variable genes using *all* data (including test samples), your model has already "seen" test data patterns — inflating accuracy estimates.

#### Cell 13: Train/Val/Test Split

Splits data into three sets:
- **Train**: 55% of patients (~202) — used to train models
- **Validation**: 25% (~92) — used to tune hyperparameters
- **Test**: 20% (~74) — used for final evaluation only

Uses **stratified splitting** so each subset has the same subtype proportions.

#### Cell 14: Feature Selection (Leakage-Safe)

Selects the top **N_GENES = 2000** most variable genes **using only training data**. This prevents leakage because the test set doesn't influence which genes are selected.

Uses `SelectKBest` with ANOVA F-test (`f_classif`) to rank genes by their ability to distinguish subtypes.

Also applies **StandardScaler** (z-score normalization) — fitted on training data, then applied to validation and test data.

---

### Section 5: Baseline Models

#### Why Baselines?

Before building a complex neural network, we should check: can simpler models already do the job? This is scientific discipline.

#### Cells 15-16: Define Baselines

Defines a set of classical ML classifiers:
| Model | Description |
|-------|-------------|
| Logistic Regression (balanced) | Simple linear model with class weights |
| Random Forest (balanced) | Ensemble of decision trees |
| Extra Trees (balanced) | Like Random Forest but more random |
| XGBoost | Gradient boosting (very powerful) |
| LightGBM | Fast gradient boosting |

All use `class_weight='balanced'` (or equivalent) to handle the imbalanced subtypes.

#### Cell 17: Cross-Validation

Runs **5-fold stratified cross-validation** on each baseline model. This means:
1. Split training data into 5 parts
2. Train on 4 parts, evaluate on the 5th
3. Repeat 5 times, rotating the evaluation part
4. Average the results

Reports: accuracy, balanced accuracy, **macro F1** (primary metric), and macro AUC.

#### Cell 18: Holdout Evaluation

Trains each baseline on the full training set and evaluates on the held-out test set. Results are ranked by macro F1 score.

---

### Section 6: Multi-Task Neural Network (MLP)

#### Cell 19: Model Architecture — `OncoProfiloV2`

Defines a PyTorch neural network that does **two tasks simultaneously**:

1. **Subtype Classification**: Predicts which of 5 PAM50 subtypes a tumor belongs to
2. **Survival Risk Prediction**: Predicts a Cox proportional hazards risk score

Architecture:
```
Input (2000 genes) → Linear → BatchNorm → GELU → Dropout
                   → Linear → BatchNorm → GELU → Dropout
                   → Linear → BatchNorm → GELU → Dropout
                   → Classification Head (5 outputs)
                   → Survival Risk Head (1 output)
```

Key features:
- **Smaller than v1**: Hidden layers are (256, 128, 64) instead of the original ~1.2M parameters — because with only ~200 training samples, a huge model overfits
- **Class-weighted cross-entropy**: Gives more importance to rare subtypes (Her2, Normal)
- **Cox PH loss**: A survival analysis loss that learns to rank patients by risk
- **Combined loss**: `α × classification_loss + β × survival_loss`

#### Cell 20: Training Function

The training loop includes:
- **AdamW optimizer** with weight decay (regularization)
- **ReduceLROnPlateau scheduler**: Reduces learning rate when validation metrics plateau
- **Gradient clipping**: Prevents exploding gradients
- **Early stopping** (patience=18): Stops training when validation doesn't improve for 18 epochs
- **Model selection**: Saves the checkpoint with the best validation score

#### Cell 21: Train MLP Variants

Trains two (or more in "full" mode) MLP configurations:
1. `MLP_small_weighted`: 3-layer (256→128→64), dropout=0.35
2. `MLP_tiny_weighted`: 2-layer (256→64), dropout=0.40

Each is evaluated on the held-out test set.

---

### Section 7: Model Selection & Decision

#### Cell 22: Compare All Models

Combines baseline and MLP results into one table, sorted by **Macro F1** score.

In the notebook's run, the results were:
| Model | Family | Macro F1 | C-Index |
|-------|--------|----------|---------|
| MLP_small_weighted | mlp | 0.5636 | 0.8741 |
| XGBoost | baseline | 0.5518 | N/A |
| ExtraTrees_balanced | baseline | 0.5455 | N/A |

The MLP wins on Macro F1, but both MLP and baselines achieve similar classification performance. The MLP's advantage is it also predicts survival risk (C-Index = 0.87 is quite good).

The notebook then compares against the **current deployed model's metrics** (`CURRENT_APP_METRICS`). If v2 improves Macro F1, it recommends upgrading; otherwise, it says "keep current deployed model for now."

#### Cell 23: Confusion Matrix

Prints a detailed classification report (precision, recall, F1 per subtype) and plots a confusion matrix showing where the model gets confused.

Key findings:
- **LumA** is classified well (F1 = 0.76) — largest class
- **Normal** is hardest (F1 = 0.24) — often confused with LumA
- **Basal** and **Her2** are reasonably well classified

#### Cell 24: Save Artifacts

Saves all experiment results for reproducibility:
- `v2_experiment_summary.json`: Full metrics and configuration
- Model weights (`best_model.pt` for MLP or `best_baseline.pkl` for classical models)
- `scaler.pkl`: The StandardScaler fitted on training data
- `label_encoder.pkl`: Maps subtype names ↔ numeric labels
- `selected_genes.npy`: Which 2000 genes were selected

> [!NOTE]
> The original `/models` folder is **not modified** — v2 artifacts go to `/models_v2`. This is a safety measure.

---

## Summary

This notebook demonstrates a **disciplined ML experimentation workflow** for a cancer genomics problem:

1. **Biology**: Uses real TCGA breast cancer data (RNA-seq + clinical) to classify PAM50 subtypes and predict survival
2. **Data integrity**: Downloads directly from public APIs (GDC + UCSC Xena)
3. **Leakage prevention**: Splits data before feature selection
4. **Baseline comparison**: Tests 5+ classical models before using deep learning
5. **Cross-validation**: 5-fold CV for more reliable estimates
6. **Multi-task learning**: Neural network jointly predicts subtype + survival
7. **Decision framework**: Only upgrades the deployed model if v2 improves on Macro F1

Even when v2 doesn't beat the current model, the notebook serves as evidence of **rigorous, disciplined experimentation** — which is itself a valuable portfolio signal.
