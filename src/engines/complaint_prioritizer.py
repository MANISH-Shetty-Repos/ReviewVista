# src/engines/complaint_prioritizer.py
"""
Complaint Prioritization Engine.
Ranks issues by frequency, severity, negative sentiment, and growth trajectory.
"""

import numpy as np
from collections import Counter
from src.core.data_loader import DataManager
from src.core.logger import get_logger

logger = get_logger("complaint_prioritizer")

# ── Issue category keywords ──
ISSUE_CATEGORIES = {
    "Taste & Flavor": {"bitter", "bland", "tasteless", "awful", "terrible", "flavor", "taste", "acidic", "sour", "chemical", "aftertaste", "metallic", "burnt"},
    "Freshness & Expiry": {"stale", "expired", "old", "rancid", "moldy", "spoiled", "rotten", "fresh"},
    "Packaging & Shipping": {"broken", "damaged", "leak", "crushed", "dented", "packaging", "shipped", "arrived", "delivery", "box"},
    "Quality & Value": {"overpriced", "cheap", "waste", "money", "quality", "poor", "inferior", "worth"},
    "Health & Safety": {"sick", "allergy", "allergic", "stomach", "nausea", "unsafe", "harmful", "reaction"},
    "Customer Service": {"refund", "return", "customer service", "response", "complaint", "support", "replacement"},
    "Size & Quantity": {"small", "tiny", "less", "amount", "quantity", "portion", "size", "shrunk"},
    "Consistency": {"inconsistent", "different", "changed", "used to", "formula", "batch", "vary"},
}


def prioritize_complaints(top_n: int = 10) -> list[dict]:
    """
    Analyze the full review dataset and rank complaints by priority.

    Returns:
        Sorted list of complaint dicts with priority level, frequency, severity, etc.
    """
    df = DataManager.get_clusters_df()

    # Focus on negative reviews (rating <= 2)
    negative_reviews = df[df["rating"] <= 2]

    if negative_reviews.empty:
        return []

    complaints = []

    for category, keywords in ISSUE_CATEGORIES.items():
        # Find reviews mentioning this category
        mask = negative_reviews["clean_text"].apply(
            lambda text: bool(set(str(text).lower().split()) & keywords)
        )
        matching = negative_reviews[mask]

        if matching.empty:
            continue

        frequency = len(matching)
        total_negative = len(negative_reviews)
        frequency_pct = round(frequency / total_negative * 100, 1)

        # Severity: average rating of matching reviews (lower = more severe)
        avg_rating = matching["rating"].mean()
        severity_score = round((3 - avg_rating) / 2 * 100, 1)  # Normalize to 0-100

        # Sample negative review excerpts
        sample_texts = matching["clean_text"].head(5).tolist()
        sample_texts = [str(t)[:150] for t in sample_texts]

        # Priority score: weighted combination
        priority_score = (
            0.4 * (frequency_pct / 100) +
            0.3 * (severity_score / 100) +
            0.3 * (frequency / total_negative)
        ) * 100

        # Priority level
        if priority_score >= 30:
            level = "🔴 High"
            level_key = "high"
        elif priority_score >= 15:
            level = "🟡 Medium"
            level_key = "medium"
        else:
            level = "🟢 Low"
            level_key = "low"

        # Explanation
        explanation = _generate_explanation(category, frequency, frequency_pct, avg_rating)

        complaints.append({
            "category": category,
            "priority_level": level,
            "priority_key": level_key,
            "priority_score": round(priority_score, 1),
            "frequency": frequency,
            "frequency_pct": frequency_pct,
            "severity_score": round(severity_score, 1),
            "avg_rating": round(avg_rating, 2),
            "sample_reviews": sample_texts,
            "keywords": list(keywords)[:6],
            "explanation": explanation,
        })

    # Sort by priority score descending
    complaints.sort(key=lambda x: x["priority_score"], reverse=True)
    return complaints[:top_n]


def _generate_explanation(category: str, freq: int, freq_pct: float, avg_rating: float) -> str:
    """Generate human-readable explanation for complaint priority."""
    severity = "critical" if avg_rating <= 1.5 else "significant" if avg_rating <= 2.0 else "moderate"

    return (
        f"{category} issues appear in {freq_pct}% of negative reviews ({freq} reviews) "
        f"with a {severity} severity (avg rating: {avg_rating:.1f}/5). "
        f"{'Immediate action recommended.' if freq_pct > 20 else 'Should be monitored and addressed.'}"
    )


def get_priority_summary() -> dict:
    """Get high-level complaint priority metrics."""
    complaints = prioritize_complaints()

    high = sum(1 for c in complaints if c["priority_key"] == "high")
    medium = sum(1 for c in complaints if c["priority_key"] == "medium")
    low = sum(1 for c in complaints if c["priority_key"] == "low")

    return {
        "total_categories": len(complaints),
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "top_complaint": complaints[0]["category"] if complaints else "None",
        "complaints": complaints,
    }
