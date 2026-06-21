# src/engines/comparison.py
"""
Product Comparison Engine.
Compare two products across ratings, sentiment, topics, and strengths/weaknesses.
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.core.data_loader import DataManager
from src.engines.sentiment import POSITIVE_SIGNALS, NEGATIVE_SIGNALS
from src.core.logger import get_logger

logger = get_logger("comparison")


def compare_products(product_a: str, product_b: str) -> dict:
    """
    Compare two products across multiple dimensions.

    Args:
        product_a: Product ID for first product.
        product_b: Product ID for second product.

    Returns:
        Comprehensive comparison dict.
    """
    df = DataManager.get_clusters_df()

    reviews_a = df[df["product_id"] == product_a]
    reviews_b = df[df["product_id"] == product_b]

    if reviews_a.empty or reviews_b.empty:
        return {"error": "One or both products have no reviews."}

    profile_a = _build_product_profile(reviews_a, product_a)
    profile_b = _build_product_profile(reviews_b, product_b)

    # ── Compute comparison metrics ──
    rating_diff = round(profile_a["avg_rating"] - profile_b["avg_rating"], 2)

    # Winner determination
    if abs(rating_diff) < 0.2:
        overall_winner = "tie"
    elif rating_diff > 0:
        overall_winner = product_a
    else:
        overall_winner = product_b

    # AI recommendation
    recommendation = _generate_comparison_recommendation(profile_a, profile_b)

    return {
        "product_a": profile_a,
        "product_b": profile_b,
        "rating_difference": rating_diff,
        "overall_winner": overall_winner,
        "recommendation": recommendation,
        "comparison_dimensions": [
            {
                "dimension": "Average Rating",
                "product_a": profile_a["avg_rating"],
                "product_b": profile_b["avg_rating"],
                "winner": product_a if profile_a["avg_rating"] > profile_b["avg_rating"] else product_b,
            },
            {
                "dimension": "Review Count",
                "product_a": profile_a["review_count"],
                "product_b": profile_b["review_count"],
                "winner": product_a if profile_a["review_count"] > profile_b["review_count"] else product_b,
            },
            {
                "dimension": "Positive %",
                "product_a": profile_a["positive_pct"],
                "product_b": profile_b["positive_pct"],
                "winner": product_a if profile_a["positive_pct"] > profile_b["positive_pct"] else product_b,
            },
            {
                "dimension": "Negative %",
                "product_a": profile_a["negative_pct"],
                "product_b": profile_b["negative_pct"],
                "winner": product_a if profile_a["negative_pct"] < profile_b["negative_pct"] else product_b,
            },
        ],
    }


def _build_product_profile(reviews_df: pd.DataFrame, product_id: str) -> dict:
    """Build a comprehensive profile for a product."""
    ratings = reviews_df["rating"]
    texts = reviews_df["clean_text"].dropna().tolist()

    # Basic metrics
    avg_rating = round(float(ratings.mean()), 2)
    review_count = int(len(reviews_df))
    positive_pct = round(float((ratings >= 4).mean() * 100), 1) if review_count > 0 else 0.0
    negative_pct = round(float((ratings <= 2).mean() * 100), 1) if review_count > 0 else 0.0

    # Rating distribution
    rating_dist = ratings.value_counts().sort_index().to_dict()

    # Top positive and negative keywords
    positive_texts = [t for t, r in zip(texts, ratings) if r >= 4]
    negative_texts = [t for t, r in zip(texts, ratings) if r <= 2]

    positive_keywords = _extract_keywords(positive_texts, POSITIVE_SIGNALS)
    negative_keywords = _extract_keywords(negative_texts, NEGATIVE_SIGNALS)

    # Common keywords overall
    all_words = " ".join(texts).lower().split()
    word_freq = Counter(w for w in all_words if len(w) > 3 and w.isalpha())
    common_keywords = [w for w, _ in word_freq.most_common(10)]

    # Cluster distribution
    cluster_dist = reviews_df["cluster"].value_counts().head(5).to_dict()

    # Representative reviews (up to 2 positive and 2 negative)
    pos_reviews = reviews_df[reviews_df["rating"] >= 4]["review_text"].dropna().head(2).tolist()
    neg_reviews = reviews_df[reviews_df["rating"] <= 2]["review_text"].dropna().head(2).tolist()

    return {
        "product_id": product_id,
        "review_count": review_count,
        "avg_rating": avg_rating,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
        "positive_keywords": positive_keywords,
        "negative_keywords": negative_keywords,
        "common_keywords": common_keywords,
        "top_clusters": {str(k): int(v) for k, v in cluster_dist.items()},
        "representative_positive": pos_reviews,
        "representative_negative": neg_reviews,
    }


def _extract_keywords(texts: list[str], signal_words: set, top_k: int = 8) -> list[str]:
    """Extract most frequent keywords from texts, boosting signal words."""
    if not texts:
        return []

    all_words = " ".join(texts).lower().split()
    word_freq = Counter(w for w in all_words if len(w) > 3 and w.isalpha())

    # Boost signal words
    for word in signal_words:
        if word in word_freq:
            word_freq[word] *= 2

    return [w for w, _ in word_freq.most_common(top_k)]


def _generate_comparison_recommendation(profile_a: dict, profile_b: dict) -> str:
    """Generate a business recommendation based on comparison."""
    a_id = profile_a["product_id"][:12]
    b_id = profile_b["product_id"][:12]
    a_rating = profile_a["avg_rating"]
    b_rating = profile_b["avg_rating"]

    if abs(a_rating - b_rating) < 0.3:
        return (
            f"Both products perform similarly (ratings: {a_rating} vs {b_rating}). "
            f"Differentiation should focus on specific customer pain points identified in negative reviews."
        )
    elif a_rating > b_rating:
        return (
            f"Product {a_id}... outperforms with a rating of {a_rating} vs {b_rating}. "
            f"Product {b_id}... should address its top negative keywords: "
            f"{', '.join(profile_b['negative_keywords'][:3])}."
        )
    else:
        return (
            f"Product {b_id}... outperforms with a rating of {b_rating} vs {a_rating}. "
            f"Product {a_id}... should address its top negative keywords: "
            f"{', '.join(profile_a['negative_keywords'][:3])}."
        )


def get_comparable_products(min_reviews: int = 10) -> list[dict]:
    """Get list of products with enough reviews for meaningful comparison."""
    df = DataManager.get_clusters_df()
    product_counts = df.groupby("product_id").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    # Filter products with enough reviews
    eligible = product_counts[product_counts["review_count"] >= min_reviews]
    eligible = eligible.sort_values("review_count", ascending=False).head(100)

    return [
        {
            "product_id": row["product_id"],
            "review_count": int(row["review_count"]),
            "avg_rating": round(row["avg_rating"], 2),
        }
        for _, row in eligible.iterrows()
    ]
