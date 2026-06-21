# src/engines/topic_intelligence.py
"""
Enhanced Topic Intelligence Engine.
Provides enriched topic cards, trends, domain-agnostic topic mapping, 
and representative reviews aligned with cluster keywords and sentiment.
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.core.data_loader import DataManager
from src.core.logger import get_logger
from src.summary import summarize_reviews

logger = get_logger("topic_intelligence")

# Module level cache for topic cards
_cached_topic_cards = None

# ── Human-readable topic name mapping across multiple domains ──
TOPIC_NAME_MAP = {
    # Food & Beverages
    "coffee": "Coffee Quality",
    "tea": "Tea & Beverages",
    "chocolate": "Chocolate & Sweets",
    "snack": "Snacks",
    "cereal": "Breakfast & Cereal",
    "sauce": "Sauces & Condiments",
    "spice": "Spices & Seasoning",
    "organic": "Organic Products",
    "gluten": "Gluten-Free Products",
    "baby": "Baby Food",
    "protein": "Protein & Health",
    "candy": "Candy & Confectionery",
    "chips": "Chips & Crisps",
    "sugar": "Sugar & Sweeteners",
    "water": "Water & Beverages",
    "juice": "Juice & Drinks",
    "vitamin": "Vitamins & Supplements",
    "oil": "Cooking Oils",
    "flavor": "Flavor & Taste",
    "taste": "Taste Quality",
    "beans": "Coffee Beans",
    "roast": "Roast Profile",
    "fresh": "Freshness",

    # Jewellery & Fashion
    "necklace": "Necklaces & Pendants",
    "ring": "Rings & Bands",
    "earring": "Earrings",
    "bracelet": "Bracelets & Bangles",
    "stone": "Gemstones & Crystals",
    "gold": "Gold Jewellery",
    "silver": "Silver Jewellery",
    "chain": "Chains & Clasps",
    "shine": "Shine & Luster",
    "size": "Size & Fit",
    "wear": "Wearability",
    "fit": "Comfort & Fit",
    "comfort": "Comfort",
    "pretty": "Aesthetics & Style",
    "beautiful": "Appearance & Design",
    "gift": "Gifting & Present",

    # Electronics & PC
    "battery": "Battery & Power",
    "charge": "Charging & Power",
    "screen": "Screen & Display",
    "display": "Display Quality",
    "sound": "Sound & Audio",
    "audio": "Audio Quality",
    "mouse": "Mouse & Input Devices",
    "keyboard": "Keyboard & Input",
    "speed": "Performance & Speed",
    "fast": "Performance",
    "cable": "Cables & Connectivity",
    "port": "Connectivity & Ports",
    "build": "Build Quality",
    "durable": "Durability",
    "broke": "Durability Issues",

    # General / Logistics
    "delivery": "Delivery & Shipping",
    "shipping": "Shipping Service",
    "packaging": "Packaging Quality",
    "package": "Packaging",
    "box": "Packaging & Unboxing",
    "price": "Pricing & Value",
    "cost": "Cost & Value",
    "cheap": "Affordability",
    "value": "Value for Money",
    "service": "Customer Support",
    "support": "Customer Service",
    "return": "Returns & Refunds",
}


def _infer_topic_name(keywords: list[str]) -> str:
    """Infer a human-readable topic name from keywords."""
    for kw in keywords:
        kw_lower = kw.lower()
        for key, name in TOPIC_NAME_MAP.items():
            if key in kw_lower:
                return name
    # Fallback: capitalize first keyword
    return keywords[0].title() if keywords else "General"


def get_all_topics() -> list[dict]:
    """
    Build enriched topic cards from cluster data.
    Caches results and updates when the underlying DataManager is reset.
    """
    global _cached_topic_cards

    # If the DataManager in-memory cache is empty/reset, invalidate our local cache
    if DataManager._clusters_df is None:
        _cached_topic_cards = None

    if _cached_topic_cards is not None:
        return _cached_topic_cards

    df = DataManager.get_clusters_df()
    topic_map = DataManager.get_topic_map()

    if df is None or len(df) == 0:
        return []

    topics = []

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_reviews = df[df["cluster"] == cluster_id]
        kw_info = topic_map.get(cluster_id, {"keywords": [], "domain": "general"})
        keywords = kw_info["keywords"]

        if not keywords:
            continue

        ratings = cluster_reviews["rating"]
        review_count = len(cluster_reviews)

        # Sentiment breakdown
        pos_pct = float(round((ratings >= 4).mean() * 100, 1)) if review_count > 0 else 0.0
        neg_pct = float(round((ratings <= 2).mean() * 100, 1)) if review_count > 0 else 0.0

        # Trend indicator (simple: compare avg rating to global)
        global_avg = float(df["rating"].mean())
        topic_avg = float(ratings.mean()) if review_count > 0 else 3.0
        if topic_avg > global_avg + 0.2:
            trend = "positive"
            trend_icon = "📈"
        elif topic_avg < global_avg - 0.2:
            trend = "negative"
            trend_icon = "📉"
        else:
            trend = "stable"
            trend_icon = "➡️"

        # ── Representative Reviews Selection with Sentiment & Keyword Match ──
        # Filter and rank representative reviews based on keyword hits to ensure relevance
        def keyword_hits(text):
            words = set(str(text).lower().split())
            return len(words & set(keywords))

        cluster_reviews_with_relevance = cluster_reviews.copy()
        cluster_reviews_with_relevance["relevance"] = cluster_reviews_with_relevance["clean_text"].apply(keyword_hits)

        # Positive reviews tab: rating >= 4, sorted by highest keyword relevance
        pos_reviews = (
            cluster_reviews_with_relevance[cluster_reviews_with_relevance["rating"] >= 4]
            .sort_values(by=["relevance", "rating"], ascending=[False, False])
            .head(3)
        )
        
        # Negative reviews tab: rating <= 2, sorted by highest keyword relevance
        neg_reviews = (
            cluster_reviews_with_relevance[cluster_reviews_with_relevance["rating"] <= 2]
            .sort_values(by=["relevance", "rating"], ascending=[False, True])
            .head(3)
        )

        topic_name = _infer_topic_name(keywords)
        
        # Generate dynamic summary from cluster review texts
        from src.summary import summarize_topic_reviews
        topic_summary = summarize_topic_reviews(cluster_reviews)

        topics.append({
            "cluster_id": int(cluster_id),
            "name": topic_name,
            "keywords": keywords[:8],
            "domain": kw_info["domain"],
            "review_count": int(review_count),
            "avg_rating": float(round(topic_avg, 2)),
            "positive_pct": float(pos_pct),
            "negative_pct": float(neg_pct),
            "trend": trend,
            "trend_icon": trend_icon,
            "top_reviews": _format_reviews(pos_reviews),
            "bottom_reviews": _format_reviews(neg_reviews),
            "summary": topic_summary
        })

    # Sort by review count descending
    topics.sort(key=lambda x: x["review_count"], reverse=True)
    _cached_topic_cards = topics
    return topics


def get_topic_by_id(cluster_id: int) -> dict | None:
    """Get detailed topic info for a specific cluster."""
    topics = get_all_topics()
    for t in topics:
        if t["cluster_id"] == cluster_id:
            return t
    return None


def get_topic_summary_stats() -> dict:
    """Aggregate stats across all topics."""
    topics = get_all_topics()
    if not topics:
        return {}

    return {
        "total_topics": len(topics),
        "avg_reviews_per_topic": int(round(np.mean([t["review_count"] for t in topics]))),
        "top_positive_topic": max(topics, key=lambda t: t["positive_pct"])["name"],
        "top_negative_topic": max(topics, key=lambda t: t["negative_pct"])["name"],
        "largest_topic": max(topics, key=lambda t: t["review_count"])["name"],
        "highest_rated_topic": max(topics, key=lambda t: t["avg_rating"])["name"],
        "lowest_rated_topic": min(topics, key=lambda t: t["avg_rating"])["name"],
    }


def _format_reviews(df_slice: pd.DataFrame) -> list[dict]:
    """Format DataFrame rows into review dicts for display."""
    reviews = []
    for _, row in df_slice.iterrows():
        text = str(row.get("review_text", row.get("clean_text", "")))
        reviews.append({
            "text": text[:300],
            "rating": int(row.get("rating", 3)),
            "product_id": row.get("product_id", ""),
            "review_id": row.get("review_id", "")
        })
    return reviews
