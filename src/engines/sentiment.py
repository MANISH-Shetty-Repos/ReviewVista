# src/engines/sentiment.py
"""
Enhanced Sentiment Analysis Engine.
Provides granular sentiment classification, emotion detection, and confidence scoring.
"""

import numpy as np
from collections import Counter
from src.core.logger import get_logger

logger = get_logger("sentiment")

# ── Emotion Lexicon (lightweight, no external dependency) ──
EMOTION_LEXICON = {
    "joy": {"love", "happy", "great", "excellent", "perfect", "amazing", "wonderful", "fantastic",
            "delicious", "awesome", "enjoy", "pleased", "satisfied", "impressed", "favorite", "best"},
    "anger": {"terrible", "awful", "horrible", "disgusting", "furious", "angry", "worst", "hate",
              "unacceptable", "outraged", "ridiculous", "infuriating", "pathetic", "rubbish"},
    "sadness": {"disappointed", "sad", "unfortunately", "regret", "unhappy", "depressing",
                "letdown", "sorrow", "miserable", "upset", "dismayed"},
    "surprise": {"shocked", "surprised", "unexpected", "amazed", "astonished", "unbelievable",
                 "incredible", "wow", "stunning", "remarkable", "extraordinary"},
    "fear": {"worried", "concerned", "afraid", "scary", "dangerous", "risky", "alarming",
             "nervous", "anxious", "frightening", "unsafe", "hazardous"},
    "trust": {"reliable", "dependable", "consistent", "trustworthy", "quality", "genuine",
              "authentic", "honest", "solid", "durable", "sturdy", "well-made"},
    "disgust": {"gross", "nasty", "revolting", "repulsive", "vile", "foul", "rancid",
                "putrid", "sickening", "inedible", "moldy", "stale", "rotten"},
}

# ── Sentiment Signal Words ──
POSITIVE_SIGNALS = {
    "excellent", "amazing", "perfect", "love", "great", "wonderful", "fantastic",
    "delicious", "recommend", "best", "superior", "outstanding", "premium", "quality",
    "smooth", "fresh", "flavorful", "satisfied", "impressed", "favorite", "nice", "beautiful"
}

# Add domain-neutral negative signals
NEGATIVE_SIGNALS = {
    "terrible", "awful", "horrible", "worst", "disgusting", "disappointing",
    "bitter", "stale", "expired", "broken", "damaged", "refund", "waste",
    "overpriced", "cheap", "bland", "tasteless", "rotten", "inedible", "gross", "bad", "poor"
}


def analyze_sentiment(reviews: list[dict]) -> dict:
    """
    Comprehensive sentiment analysis for a set of reviews.
    """
    if not reviews:
        return _empty_sentiment()

    ratings = [r.get("rating", 3) for r in reviews]
    texts = [r.get("clean_text", "") for r in reviews]

    # ── Rating-based sentiment ──
    positive = sum(1 for r in ratings if r >= 4)
    negative = sum(1 for r in ratings if r <= 2)
    neutral = len(ratings) - positive - negative

    total = len(ratings)
    sentiment_dist = {
        "positive": round(positive / total * 100, 1),
        "negative": round(negative / total * 100, 1),
        "neutral": round(neutral / total * 100, 1),
    }

    # ── Emotion detection ──
    emotions = _detect_emotions(texts)

    # ── Per-review sentiment scores ──
    review_sentiments = []
    for r in reviews:
        score = _compute_sentiment_score(r)
        rating = r.get("rating", 3)
        
        # 1. Determine sentiment strictly by rating
        if rating >= 4:
            sentiment = "positive"
        elif rating <= 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # 2. Compare with lexicon score to flag disagreement
        lexicon_sentiment = _classify_sentiment(score)
        disagreement = (sentiment != lexicon_sentiment)
        
        # Sanity validation check
        if rating >= 4 and "bad" in r.get("clean_text", "").lower() and len(r.get("clean_text", "")) < 20:
             # Potential rating error, but we honor rating
             logger.warning("Potential rating/sentiment conflict in review: text='%s', rating=%d", r.get("clean_text", ""), rating)

        review_sentiments.append({
            "text_preview": r.get("clean_text", "")[:100],
            "rating": rating,
            "sentiment": sentiment,
            "score": round(score, 3),
            "disagreement": disagreement,
            "review_id": r.get("review_id", "")
        })

    # ── Overall confidence ──
    scores = [rs["score"] for rs in review_sentiments]
    confidence = round(1 - np.std(scores), 3) if len(scores) > 1 else 0.8

    dominant = max(sentiment_dist, key=sentiment_dist.get)

    return {
        "distribution": sentiment_dist,
        "dominant_sentiment": dominant,
        "avg_rating": round(np.mean(ratings), 2),
        "emotions": emotions,
        "confidence": confidence,
        "review_sentiments": review_sentiments,
        "total_analyzed": total,
    }


def _detect_emotions(texts: list[str]) -> dict:
    """Detect emotion distribution across texts using lexicon matching."""
    emotion_counts = Counter()

    for text in texts:
        words = set(text.lower().split())
        for emotion, keywords in EMOTION_LEXICON.items():
            hits = len(words & keywords)
            if hits > 0:
                emotion_counts[emotion] += hits

    total = sum(emotion_counts.values()) or 1
    return {
        emotion: round(count / total * 100, 1)
        for emotion, count in emotion_counts.most_common()
    }


def _compute_sentiment_score(review: dict) -> float:
    """Compute a sentiment score between -1.0 and 1.0 for a single review."""
    rating = review.get("rating", 3)
    text = review.get("clean_text", "").lower()
    words = set(text.split())

    # Base score from rating (normalized to [-1, 1])
    rating_score = (rating - 3) / 2.0

    # Lexicon-based adjustment
    pos_hits = len(words & POSITIVE_SIGNALS)
    neg_hits = len(words & NEGATIVE_SIGNALS)
    
    if pos_hits + neg_hits > 0:
        lexicon_score = (pos_hits - neg_hits) / (pos_hits + neg_hits)
    else:
        lexicon_score = 0.0

    # Weighted combination: rating matters more
    return 0.6 * rating_score + 0.4 * lexicon_score


def _classify_sentiment(score: float) -> str:
    """Classify sentiment score into category."""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


def _empty_sentiment() -> dict:
    return {
        "distribution": {"positive": 0, "negative": 0, "neutral": 0},
        "dominant_sentiment": "neutral",
        "avg_rating": 0,
        "emotions": {},
        "confidence": 0,
        "review_sentiments": [],
        "total_analyzed": 0,
    }


def get_sentiment_trend(df, time_col="timestamp", rating_col="rating", period="M") -> list[dict]:
    """
    Compute sentiment trend over time periods.
    """
    import pandas as pd

    if time_col not in df.columns:
        return []

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if len(df) == 0:
        return []

    grouped = df.groupby(df[time_col].dt.to_period(period))

    trends = []
    for period_val, group in grouped:
        ratings = group[rating_col]
        trends.append({
            "period": str(period_val),
            "avg_rating": round(ratings.mean(), 2),
            "positive_pct": round((ratings >= 4).mean() * 100, 1),
            "negative_pct": round((ratings <= 2).mean() * 100, 1),
            "review_count": len(group),
        })

    return trends
