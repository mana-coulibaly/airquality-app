"""API de prédiction et d'analyse causale pour Air Quality."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
W_COLUMNS = [f"X{i}" for i in range(12)]
T_NAME = "NO2"
CLASS_LABELS = ["Bon", "Mauvais", "Moyen"]


class PredictPayload(BaseModel):
    W: dict[str, float] = Field(..., description="Les 12 variables de contexte.")
    T: float = Field(..., description="Valeur du traitement NO2.")


@lru_cache(maxsize=1)
def load_models() -> tuple[Any, Any, Any, Any]:
    """Charge les modèles une seule fois par instance serverless."""
    try:
        return (
            joblib.load(MODELS / "model_final_cls.pkl"),
            joblib.load(MODELS / "model_final.pkl"),
            joblib.load(MODELS / "scaler_W.pkl"),
            joblib.load(MODELS / "cf.pkl"),
        )
    except Exception as exc:  # pragma: no cover - message destiné à l'API
        raise RuntimeError(f"Impossible de charger les modèles : {exc}") from exc


def vectorize(payload: PredictPayload) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in W_COLUMNS if name not in payload.W]
    if missing:
        raise HTTPException(status_code=422, detail=f"Variables manquantes : {missing}")

    raw_w = np.array([[payload.W[name] for name in W_COLUMNS]], dtype=float)
    _, _, scaler, _ = load_models()
    scaled_w = scaler.transform(raw_w)
    treatment = np.array([[payload.T]], dtype=float)
    features = np.concatenate([scaled_w, treatment, np.zeros((1, 1))], axis=1)
    return raw_w, scaled_w, features


app = FastAPI(title="Air Quality API", version="1.0.0")
allowed_origins = [origin.strip() for origin in os.getenv("FRONTEND_URL", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict[str, str]:
    return {"service": "airquality-api", "status": "ok"}


@app.get("/config")
def config() -> dict[str, Any]:
    _, _, scaler, _ = load_models()
    return {
        "W_columns": W_COLUMNS,
        "T_name": T_NAME,
        "class_labels": CLASS_LABELS,
        "stats": {
            **{
                name: {
                    "min": float(mean - 3 * scale),
                    "max": float(mean + 3 * scale),
                    "mean": float(mean),
                }
                for name, mean, scale in zip(W_COLUMNS, scaler.mean_, scaler.scale_)
            },
            T_NAME: {"min": -100.0, "max": 100.0, "mean": 0.0},
        },
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> dict[str, Any]:
    _, scaled_w, features = vectorize(payload)
    classifier, regressor, _, causal_forest = load_models()
    try:
        probabilities = classifier.predict_proba(features)[0].tolist()
        class_pred = int(classifier.predict(features)[0])
        y_hat = float(regressor.predict(features)[0])
        cate = float(causal_forest.effect(scaled_w, T0=np.zeros(1), T1=np.ones(1))[0][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}") from exc

    return {
        "class_pred": class_pred,
        "class_label": CLASS_LABELS[class_pred],
        "proba": probabilities,
        "y_hat": y_hat,
        "cate": cate,
    }


@app.post("/whatif")
def what_if(payload: PredictPayload, pct: float = Query(-20, ge=-100, le=100)) -> dict[str, Any]:
    _, scaled_w, _ = vectorize(payload)
    _, _, _, causal_forest = load_models()
    current_t = np.array([payload.T], dtype=float)
    counterfactual_t = np.array([payload.T * (1 + pct / 100)], dtype=float)
    try:
        current_effect = causal_forest.effect(scaled_w, T0=np.zeros(1), T1=current_t)[0][0]
        counterfactual_effect = causal_forest.effect(scaled_w, T0=np.zeros(1), T1=counterfactual_t)[0][0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de simulation : {exc}") from exc

    return {
        "pct": pct,
        "current": {"treatment": float(current_t[0]), "cate": float(current_effect)},
        "counterfactual": {"treatment": float(counterfactual_t[0]), "cate": float(counterfactual_effect)},
    }
