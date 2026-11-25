# Social Media Sentiment Analyzer (SMSA)

Classifies social media text as positive, neutral or negative using
**TF-IDF + Logistic Regression**, with a **FastAPI** backend and a **React (Vite)** web interface.

## Project structure
```
SMSA/
├── backend/
│   ├── main.py            # FastAPI app: /api/analyze, /api/health, serves the UI
│   ├── train.py           # trains the model and writes the .pkl files
│   ├── requirements.txt
│   └── model/             # sentiment_model.pkl + vectorizer.pkl (created by train.py)
├── frontend/              # React + Vite source
├── data/                  # Tweets-2.csv and sentimentdataset.csv (not committed)
├── Dockerfile
└── README.md
```

## 1. Train the model
The trained files are already in `backend/model/`. To retrain, put `Tweets-2.csv` and
`sentimentdataset.csv` in `data/` and run:
```bash
pip install -r backend/requirements.txt
python backend/train.py
```
`Tweets-2.csv` already has positive/neutral/negative labels. The emotion labels in
`sentimentdataset.csv` (Joy, Grief, Curiosity, ...) are grouped into the same three classes
inside `train.py`. Both are merged, split 80/20, and the model is evaluated on the held-out 20%
(about 79% overall accuracy).

## 2. Run locally
```bash
# terminal 1 – API
uvicorn backend.main:app --reload --port 8000

# terminal 2 – UI (dev server proxies /api to port 8000)
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## 3. Production build (single server)
```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --port 8000
```
Open http://localhost:8000

## API
`POST /api/analyze` with `{"text": "I love this!"}` returns
`{"sentiment": "...", "confidence": 0.93, "scores": {...}}`.

## Docker
```bash
docker build -t smsa .
docker run -p 7860:7860 smsa
```
