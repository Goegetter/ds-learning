import os
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI
import logging
import joblib
import sklearn
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from contextlib import asynccontextmanager
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bc = load_breast_cancer()
X = bc.data
y = bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
medians = np.median(X_train, axis=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(medians, "models/medians.pkl")
joblib.dump(scaler, "models/scaler.pkl")


mods = {
    'logreg': LogisticRegression(max_iter=10000),
    'rf': RandomForestClassifier(random_state=42, n_estimators=100),
    'gb': GradientBoostingClassifier(random_state=42, n_estimators=100)
}
preds = {}

for k, v in mods.items():
    v.fit(X_train_scaled, y_train)
    prediction = v.predict(X_test_scaled)
    preds[k] = {
        'accuracy': accuracy_score(y_test, prediction),
        'precision': precision_score(y_test, prediction),
        'recall': recall_score(y_test, prediction),
        'f1': f1_score(y_test, prediction),
    }
    joblib.dump(v, f'models/{k}.pkl')
    logger.info(f"{k} -> accuracy: {preds[k]['accuracy']:.4f}, f1: {preds[k]['f1']:.4f}, precision: {preds[k]['precision']:.4f}, recall: {preds[k]['recall']:.4f}")


models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    models["scaler"]  = joblib.load("models/scaler.pkl")
    models["medians"] = joblib.load("models/medians.pkl")
    models["logreg"]  = joblib.load("models/logreg.pkl")
    models["rf"]      = joblib.load("models/rf.pkl")
    models["gb"]      = joblib.load("models/gb.pkl")
    yield
    models.clear()

app = FastAPI(title="Breast Cancer Prediction API", lifespan=lifespan)

class PatientData(BaseModel):
    features: list[float | None] = Field(..., min_length=30, max_length=30)

def preprocess(features: list[float | None]) -> np.ndarray:
    X = np.array(features, dtype=float).reshape(1, -1)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[0, nan_mask[0]] = models["medians"][nan_mask[0]]

    # Normalizacja
    return models["scaler"].transform(X)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=malignant, 1=benign")
    prediction_label: str
    probability_malignant: float = Field(..., ge=0, le=1)
    probability_benign: float = Field(..., ge=0, le=1)


class BatchRequest(BaseModel):
    patients: list[PatientData]

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int

def make_response(model_name: str, model, X: np.ndarray):
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return PredictionResponse(
        prediction=pred,
        prediction_label="benign" if pred == 1 else "malignant",
        probability_malignant=round(float(proba[0]), 4),
        probability_benign=round(float(proba[1]), 4),
    )

@app.get('/')
def root():
    return {'message': 'Breast Cancer Prediction API', 'docs': '/docs'}

@app.get('/health')
def health():
    return {
        'status_LogisticRegression': 'OK' if models['logreg'] is not None else 'Model not found',
        'status_RandomForestClassifier': 'OK' if models['rf'] is not None else 'Model not found',
        'staus_GradientBoosting': 'OK' if models['gb'] is not None else 'Model not found'
        }

@app.post("/predict/v1")
def predict_v1_batch(data: BatchRequest):
    results = [make_response("LogisticRegression", models["logreg"], preprocess(p.features)) for p in data.patients]
    return BatchResponse(predictions=results, count=len(results))

@app.post("/predict/v2")
def predict_v2_batch(data: BatchRequest):
    results = [make_response("RandomForest", models["rf"], preprocess(p.features)) for p in data.patients]
    return BatchResponse(predictions=results, count=len(results))

@app.post("/predict/v3")
def predict_v3_batch(data: BatchRequest):
    results = [make_response("GradientBoosting", models["gb"], preprocess(p.features)) for p in data.patients]
    return BatchResponse(predictions=results, count=len(results))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8886)