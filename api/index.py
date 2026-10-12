"""API de prediction et d'analyse causale pour Air Quality."""

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
def load_scaler() -> Any:
    try:
        return joblib.load(MODELS / "scaler_W.pkl")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Impossible de charger le scaler : {exc}") from exc


@lru_cache(maxsize=1)
def load_classifier() -> Any:
    try:
        return joblib.load(MODELS / "model_final_cls.pkl")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Impossible de charger le classifieur : {exc}") from exc


@lru_cache(maxsize=1)
def load_regressor() -> Any:
    try:
        return joblib.load(MODELS / "model_final.pkl")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Impossible de charger le regresseur : {exc}") from exc


def vectorize(payload: PredictPayload) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in W_COLUMNS if name not in payload.W]
    if missing:
        raise HTTPException(status_code=422, detail=f"Variables manquantes : {missing}")
    raw_w = np.array([[payload.W[name] for name in W_COLUMNS]], dtype=float)
    scaled_w = load_scaler().transform(raw_w)
    return raw_w, scaled_w


def make_features(scaled_w: np.ndarray, treatment: float) -> np.ndarray:
    return np.concatenate(
        [scaled_w, np.array([[treatment]], dtype=float), np.zeros((1, 1))], axis=1
    )


def prediction_outputs(classifier: Any, regressor: Any, scaled_w: np.ndarray, treatment: float) -> dict[str, Any]:
    features = make_features(scaled_w, treatment)
    class_pred = int(classifier.predict(features)[0])
    return {
        "class_pred": class_pred,
        "class_label": CLASS_LABELS[class_pred],
        "proba": classifier.predict_proba(features)[0].tolist(),
        "y_hat": float(regressor.predict(features)[0]),
    }


def treatment_effect(regressor: Any, scaled_w: np.ndarray, treatment: float) -> float:
    """Estimate the local effect of one additional NO2 unit.

    The regression model is tree-based and can therefore return the same
    value over an interval. A zero result means that the prediction did not
    cross a model threshold between ``treatment`` and ``treatment + 1``;
    it does not mean that NO2 can never affect the prediction.
    """
    baseline = float(regressor.predict(make_features(scaled_w, treatment))[0])
    treated = float(regressor.predict(make_features(scaled_w, treatment + 1.0))[0])
    return treated - baseline


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
    scaler = load_scaler()
    return {
        "W_columns": W_COLUMNS,
        "T_name": T_NAME,
        "class_labels": CLASS_LABELS,
        "stats": {
            **{
                name: {"min": float(mean - 3 * scale), "max": float(mean + 3 * scale), "mean": float(mean)}
                for name, mean, scale in zip(W_COLUMNS, scaler.mean_, scaler.scale_)
            },
            T_NAME: {"min": -100.0, "max": 100.0, "mean": 0.0},
        },
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> dict[str, Any]:
    _, scaled_w = vectorize(payload)
    classifier = load_classifier()
    regressor = load_regressor()
    try:
        result = prediction_outputs(classifier, regressor, scaled_w, payload.T)
        result["cate"] = treatment_effect(regressor, scaled_w, payload.T)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {exc}") from exc
    return result


@app.post("/whatif")
def what_if(payload: PredictPayload, pct: float = Query(-20, ge=-100, le=100)) -> dict[str, Any]:
    _, scaled_w = vectorize(payload)
    classifier = load_classifier()
    regressor = load_regressor()
    current_t = float(payload.T)
    counterfactual_t = current_t * (1 + pct / 100)
    try:
        cate = treatment_effect(regressor, scaled_w, current_t)
        current = prediction_outputs(classifier, regressor, scaled_w, current_t)
        counterfactual = prediction_outputs(classifier, regressor, scaled_w, counterfactual_t)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de simulation : {exc}") from exc
    return {
        "pct": pct,
        "current": {**current, "treatment": current_t, "cate": cate},
        "counterfactual": {**counterfactual, "treatment": counterfactual_t, "cate": cate},
    }
