## The Biology First — Simply

You have PCB background so this will connect.

**What is RNA-seq data?**
You know DNA → RNA → Protein (central dogma). RNA-seq measures *how much* each gene is being expressed in a cell at a given moment. Think of it like a snapshot of which genes are "switched on" and how loudly.

In a cancer patient's tumor sample, certain genes are screaming loudly while others are silent. That pattern of gene activity is called the **gene expression profile** and it's unique to each cancer subtype.

**What are cancer subtypes?**
Breast cancer is not one disease. It's five molecularly distinct diseases that just happen to occur in the same organ:

| Subtype | Character |
|---|---|
| Luminal A | Slow growing, hormone receptor positive, best prognosis |
| Luminal B | Faster growing, still hormone positive |
| HER2-enriched | HER2 gene amplified, aggressive |
| Basal-like | Triple negative, most aggressive, hardest to treat |
| Normal-like | Resembles normal breast tissue |

Each subtype has a *different gene expression signature* — that's what our model learns to recognise.

**What is survival prediction?**
Every TCGA patient has clinical data attached — how many days they survived after diagnosis, and whether they're alive or dead. Our model learns to predict survival risk from the gene expression pattern. Not "will they die" — but "how high is their risk relative to others."

**So what are we actually doing biologically?**
We're teaching a neural network to look at a tumor's gene activity profile and answer two questions simultaneously:
- Which subtype is this tumor?
- How high is this patient's survival risk?

This is exactly what oncologists want — molecular subtyping + prognosis from a single biopsy.

---

## The Tech — Simply

**What is the data shape?**
Imagine a spreadsheet:
- Each **row** = one patient (around 1,100 patients)
- Each **column** = one gene (around 20,000 genes)
- Each **cell** = how active that gene is in that patient's tumor

So our input to the model is a vector of 20,000 numbers representing one patient's gene expression profile.

**Why do we reduce 20,000 genes to ~2,000?**
Most genes don't vary much between patients — they're just housekeeping genes doing the same thing in everyone. We keep only the genes that vary the most across patients, because those are the ones actually *distinguishing* the subtypes. This is called **variance filtering** — simple, biologically justified.

**What is the model architecture?**
A straightforward multi-task neural network:

```
Patient gene expression (2000 numbers)
            ↓
    Shared MLP Encoder
    (learns the pattern)
            ↓
    ┌───────┴────────┐
    ↓                ↓
Subtype          Survival
Classifier       Risk Score
(which of 5?)    (continuous number)
```

Both tasks learn together. The shared encoder learns a compressed biological representation that's useful for both questions simultaneously.

**What is Cox loss for survival?**
Normal regression loss compares predicted vs actual number. Survival data is tricky because many patients are still alive when data was collected — we don't know their final survival time yet. Cox loss handles this properly — it only asks "did the model correctly rank patients by risk?" not "did it predict the exact number of days?" This is the clinically standard approach.

**What is C-index?**
The metric for survival models. C-index of 0.5 = random. C-index of 1.0 = perfect ranking. Anything above 0.65 on real clinical data is meaningful.

**What is SHAP?**
After the model predicts, SHAP tells you *which genes pushed the prediction in which direction* for each patient. This is what makes it interpretable — you can say "for this patient, BRCA1 and ESR1 were the top contributing genes." That's clinically meaningful output.

---

## The Full Flow in One Picture

```
TCGA Database (cloud)
        ↓
Download RNA-seq + clinical data in Colab
        ↓
Preprocessing: normalize, filter to top 2000 genes
        ↓
PyTorch Dataset + DataLoader
        ↓
Train multi-task model (classifier + Cox survival)
        ↓
Evaluate: Accuracy, AUC, C-index, Kaplan-Meier curves
        ↓
SHAP: gene importance per patient
        ↓
Gradio App on Hugging Face:
Input gene expression → Output subtype + risk + top genes
```

---

## What The Single Notebook Looks Like (Sections)

```
OncoProfilo.ipynb
│
├── Section 1: Install & Imports
├── Section 2: Download TCGA-BRCA data (cloud, no local save)
├── Section 3: EDA — understand the data visually
├── Section 4: Preprocessing — normalize, filter, split
├── Section 5: PyTorch Dataset + DataLoader
├── Section 6: Model Architecture (multi-task MLP)
├── Section 7: Training Loop (classification + Cox loss)
├── Section 8: Evaluation (accuracy, AUC, C-index, KM curves)
├── Section 9: SHAP gene importance
└── Section 10: Gradio App (runs inside Colab, deploy to HF)
```

10 clean sections. One notebook. Everything visible, everything explained.

---

## What Makes This Genuinely Advanced

Just so you're clear on why this isn't beginner work:

- **Cox PH loss** — most ML courses never teach this
- **Multi-task learning** — two losses, one shared encoder, balanced training
- **20,000-feature SHAP** — computationally non-trivial
- **Kaplan-Meier stratified by predicted subtype** — clinical standard visualisation
- **Real clinical dataset** (TCGA is used in published papers)
- **Gradio deployment** with a genomics model — rare at your level

---

Are you clear on everything now? Any part of the biology or tech you want me to re-explain before we open the notebook and start writing Section 1?