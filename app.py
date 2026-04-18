import os, json, pickle, warnings, sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
warnings.filterwarnings("ignore")

# Compatibility shim for pickled artifacts saved with NumPy 2.x internals.
try:
    import numpy.core.numeric as _np_numeric
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

app = FastAPI(title="OncoProfilo API")

# ── Model globals ──────────────────────────────────────────────
MODEL        = None
SCALER       = None
LABEL_ENC    = None
TOP_INDICES  = None
SEL_GENES    = None
DEVICE       = None
MODEL_LOADED = False

MODEL_STATS = {
    "architecture": "Multi-task MLP",
    "data":         "TCGA-BRCA RNA-seq (338 patients)",
    "accuracy":     66.18,
    "macro_f1":     63.28,
    "auc_roc":      81.76,
    "c_index":      72.31,
    "n_genes":      2000,
    "n_classes":    5,
    "parameters":   "1,207,430",
    "epochs":       19,
    "classes":      ["Basal", "Her2", "LumA", "LumB", "Normal"],
    "class_stats": {
        "Basal":  {"precision": 0.88, "recall": 0.78, "f1": 0.82, "support": 9,  "risk": "HIGH"},
        "Her2":   {"precision": 0.60, "recall": 0.60, "f1": 0.60, "support": 5,  "risk": "HIGH"},
        "LumA":   {"precision": 0.88, "recall": 0.69, "f1": 0.77, "support": 32, "risk": "LOW"},
        "LumB":   {"precision": 0.53, "recall": 0.75, "f1": 0.62, "support": 12, "risk": "MODERATE"},
        "Normal": {"precision": 0.31, "recall": 0.40, "f1": 0.35, "support": 10, "risk": "LOW"},
    },
    "subtype_distribution": {"LumA": 161, "LumB": 60, "Normal": 48, "Basal": 45, "Her2": 24},
}

RISK_LABEL = {"Basal": "HIGH", "Her2": "HIGH", "LumB": "MODERATE", "LumA": "LOW", "Normal": "LOW"}
RISK_COLOR = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#10b981"}

DEMO_RESULT = {
    "predicted_subtype": "LumA",
    "confidence": 86.4,
    "probabilities": {"Basal": 2.9, "Her2": 1.9, "LumA": 86.4, "LumB": 6.1, "Normal": 2.7},
    "risk_score": 0.3458,
    "risk_level": "LOW",
    "top_genes": [
        {"gene": "ENSG00000160182.3", "shap": 0.003298},
        {"gene": "ENSG00000124935.4", "shap": 0.003153},
        {"gene": "ENSG00000159763.4", "shap": 0.003038},
        {"gene": "ENSG00000110484.7", "shap": 0.002985},
        {"gene": "ENSG00000244468.1", "shap": 0.002687},
    ],
    "source": "demo",
}


def try_load_model():
    global MODEL, SCALER, LABEL_ENC, TOP_INDICES, SEL_GENES, DEVICE, MODEL_LOADED
    try:
        import torch
        import torch.nn as nn

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class OncoProfilo(nn.Module):
            def __init__(self, input_dim, n_classes, dropout_rate=0.4):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                )
                self.subtype_head = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes)
                )
                self.survival_head = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid()
                )
            def forward(self, x):
                enc = self.encoder(x)
                return self.subtype_head(enc), self.survival_head(enc)

        model_dir = "models" if os.path.isdir("models") else "model"
        paths = {
            "model":   os.path.join(model_dir, "best_model.pt"),
            "scaler":  os.path.join(model_dir, "scaler.pkl"),
            "encoder": os.path.join(model_dir, "label_encoder.pkl"),
            "indices": os.path.join(model_dir, "top_gene_indices.npy"),
            "genes":   os.path.join(model_dir, "selected_genes.npy"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            print("⚠️  Model files not found — running in demo-only mode")
            return

        MODEL    = OncoProfilo(2000, 5).to(DEVICE)
        MODEL.load_state_dict(torch.load(paths["model"], map_location=DEVICE))
        MODEL.eval()

        with open(paths["scaler"],  "rb") as f: SCALER    = pickle.load(f)
        with open(paths["encoder"], "rb") as f: LABEL_ENC = pickle.load(f)
        TOP_INDICES = np.load(paths["indices"])
        SEL_GENES   = np.load(paths["genes"], allow_pickle=True)

        MODEL_LOADED = True
        print(f"✅ Model loaded on {DEVICE}")

    except Exception as e:
        print(f"⚠️  Model load failed: {e} — running in demo-only mode")


try_load_model()

# ── Serve static ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/test_inputs", StaticFiles(directory="test_inputs"), name="test_inputs")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/model-info")
async def model_info():
    return JSONResponse({"stats": MODEL_STATS, "loaded": MODEL_LOADED})


@app.post("/api/predict/demo")
async def predict_demo():
    """Return demo prediction (pre-computed from test set)."""
    return JSONResponse(DEMO_RESULT)


class CustomInput(BaseModel):
    gene_expression: dict  # gene_id → fpkm value


@app.post("/api/predict/custom")
async def predict_custom(payload: CustomInput):
    if not MODEL_LOADED:
        # Return realistic demo result when model not loaded
        return JSONResponse({**DEMO_RESULT, "source": "demo-fallback",
                             "note": "Model files not found. Showing demo output."})
    try:
        import torch
        expr = payload.gene_expression
        all_gene_ids = list(SEL_GENES)
        full_vec = np.array([float(expr.get(g, 0.0)) for g in all_gene_ids], dtype=np.float32)
        x_log  = np.log2(full_vec + 1)
        x_filt = x_log[TOP_INDICES[:2000]] if len(x_log) > 2000 else x_log
        x_sc   = SCALER.transform(x_filt.reshape(1, -1))
        tensor = torch.tensor(x_sc, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits, risk = MODEL(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            risk_score = float(risk.cpu().numpy()[0][0])

        pred_idx   = int(np.argmax(probs))
        pred_label = LABEL_ENC.classes_[pred_idx]
        prob_dict  = {LABEL_ENC.classes_[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}

        return JSONResponse({
            "predicted_subtype": pred_label,
            "confidence":        round(float(probs[pred_idx]) * 100, 1),
            "probabilities":     prob_dict,
            "risk_score":        round(risk_score, 4),
            "risk_level":        RISK_LABEL.get(pred_label, "UNKNOWN"),
            "top_genes":         [],
            "source":            "custom",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
