"""Train the sentiment model (TF-IDF + Logistic Regression) on the two project datasets.

Run from the repo root:
    python backend/train.py

Expects in data/:
  - Tweets-2.csv           (columns: text, airline_sentiment)  -> positive/neutral/negative
  - sentimentdataset.csv   (columns: Text, Sentiment)          -> emotion labels grouped into 3 classes
Writes backend/model/sentiment_model.pkl and backend/model/vectorizer.pkl
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "backend" / "model"

POSITIVE = {
    "positive", "joy", "excitement", "contentment", "gratitude", "serenity", "happy", "hopeful",
    "pride", "elation", "euphoria", "enthusiasm", "determination", "playful", "inspiration",
    "happiness", "hope", "empowerment", "inspired", "admiration", "calmness", "compassion",
    "tenderness", "fulfillment", "reverence", "proud", "grateful", "compassionate", "thrill",
    "enchantment", "love", "amusement", "kind", "empathetic", "free-spirited", "confident",
    "satisfaction", "accomplishment", "harmony", "creativity", "wonder", "enjoyment", "affection",
    "adoration", "zest", "radiance", "rejuvenation", "coziness", "resilience", "tranquility",
    "overjoyed", "motivation", "blessed", "appreciation", "confidence", "optimism", "kindness",
    "friendship", "success", "amazement", "romance", "celebration", "ecstasy", "triumph",
    "heartwarming", "relief", "breakthrough", "positivity", "awe", "adventure",
}
NEGATIVE = {
    "negative", "despair", "grief", "loneliness", "sad", "embarrassed", "frustration", "regret",
    "melancholy", "hate", "bad", "disgust", "bitterness", "frustrated", "betrayal", "overwhelmed",
    "desolation", "bitter", "shame", "jealousy", "resentment", "fearful", "jealous", "devastated",
    "envious", "anger", "fear", "sadness", "disappointed", "anxiety", "helplessness", "envy",
    "isolation", "disappointment", "sorrow", "loss", "heartbreak", "suffering", "desperation",
    "heartache", "exhaustion", "darkness", "ruins", "lostlove", "dismissive", "intimidation",
}
NEUTRAL = {
    "neutral", "indifference", "ambivalence", "acceptance", "confusion", "numbness", "boredom",
    "reflection", "contemplation", "pensive", "curiosity", "surprise", "nostalgia",
}


def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = t.replace("#", "")
    t = re.sub(r"[^a-z\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def group(label):
    l = str(label).strip().lower()
    if l in POSITIVE: return "positive"
    if l in NEGATIVE: return "negative"
    if l in NEUTRAL: return "neutral"
    return None


def load():
    tw = read(DATA / "Tweets-2.csv")[["text", "airline_sentiment"]]
    tw.columns = ["text", "label"]
    tw["source"] = "tweets"
    sm = read(DATA / "sentimentdataset.csv")[["Text", "Sentiment"]]
    sm.columns = ["text", "label"]
    sm["label"] = sm["label"].map(group)
    sm["source"] = "social"
    df = pd.concat([tw, sm]).dropna()
    df["clean"] = df["text"].map(clean)
    return df[df["clean"].str.len() > 0].reset_index(drop=True)


def main():
    df = load()
    print(df.groupby(["source", "label"]).size(), "\n")
    strat = df["source"] + "_" + df["label"]
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)

    # the small social-media file would be drowned out by 14k airline tweets, so weight it up
    w = np.where(tr["source"] == "social", 8.0, 1.0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(vec.fit_transform(tr["clean"]), tr["label"], sample_weight=w)

    pred = clf.predict(vec.transform(te["clean"]))
    print("Overall accuracy: %.3f\n" % accuracy_score(te["label"], pred))
    print(classification_report(te["label"], pred, digits=3))
    for s in ["tweets", "social"]:
        m = (te["source"] == s).values
        print(f"Accuracy on {s} test rows: {accuracy_score(te['label'][m], pred[m]):.3f} ({m.sum()} rows)")

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT / "sentiment_model.pkl")
    joblib.dump(vec, OUT / "vectorizer.pkl")
    print("\nSaved model files to", OUT)


if __name__ == "__main__":
    main()
