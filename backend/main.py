"""FastAPI backend: loads the trained TF-IDF + Logistic Regression model
and serves both the prediction API and the built React frontend."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DIST = BASE.parent / "frontend" / "dist"

model = joblib.load(BASE / "model" / "sentiment_model.pkl")
vectorizer = joblib.load(BASE / "model" / "vectorizer.pkl")

app = FastAPI(title="Social Media Sentiment Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": [str(c) for c in model.classes_]}


@app.post("/api/analyze")
def analyze(body: TextIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text is empty.")
    vec = vectorizer.transform([text])
    label = str(model.predict(vec)[0])
    scores = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        scores = {str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)}
    return {
        "sentiment": label,
        "confidence": scores.get(label),
        "scores": scores,
    }


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")
