# src/engines/business_insights.py
"""
Business Insight Engine.
Generates advanced actionable business insights and strategic suggestions.
"""

from src.core.logger import get_logger
from src.engines.sentiment import POSITIVE_SIGNALS, NEGATIVE_SIGNALS
import json
import os

logger = get_logger("business_insights")

def _get_openai_client():
    try:
        from openai import OpenAI
        api_key = None
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None

def generate_business_insights(reviews: list[dict], use_llm: bool = True) -> dict:
    """
    Generate actionable business insights from a list of reviews.
    """
    if not reviews:
        return _empty_insights()

    if use_llm:
        client = _get_openai_client()
        if client:
            try:
                review_block = ""
                for i, r in enumerate(reviews[:15], 1):
                    rating = r.get("rating", "N/A")
                    text = r.get("review_text", "")[:300]
                    review_block += f"  {i}. [Rating {rating}/5] {text}\n"

                prompt = f"""You are a senior product strategist and data scientist. Analyze these customer reviews and extract advanced strategic business insights.

REVIEWS:
{review_block}

Return ONLY a valid JSON object with exactly this structure:
{{
  "most_requested_feature": "A single sentence describing the feature customers request most.",
  "most_critical_issue": "A single sentence describing the most severe/critical complaint.",
  "fastest_growing_complaint": "A single sentence identifying the rising concern.",
  "most_appreciated_feature": "A single sentence identifying what customers love most.",
  "customer_pain_points": ["Pain point 1", "Pain point 2", "Pain point 3"],
  "customer_expectations": ["Expectation 1", "Expectation 2"],
  "business_recommendations": ["Recommendation 1", "Recommendation 2"],
  "potential_product_improvements": ["Improvement 1", "Improvement 2"]
}}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a product strategist. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(raw)
            except Exception as e:
                logger.warning("LLM business insight generation failed: %s. Falling back to heuristic.", e)

    return _generate_heuristic_insights(reviews)

def _generate_heuristic_insights(reviews: list[dict]) -> dict:
    # Deterministic heuristics based on keywords in reviews
    all_clean_text = " ".join([r.get("clean_text", "") for r in reviews]).lower()
    words = all_clean_text.split()
    
    # Simple rule based checks
    flavor_related = any(w in words for w in ["flavor", "taste", "bitter", "sweet", "sour", "acidic"])
    pkg_related = any(w in words for w in ["packaging", "box", "broken", "leak", "damaged", "bottle", "bag"])
    price_related = any(w in words for w in ["price", "cost", "expensive", "money", "cheap", "value"])
    fresh_related = any(w in words for w in ["fresh", "stale", "expired", "old", "smell"])
    
    pain_points = []
    recommendations = []
    improvements = []
    
    if flavor_related:
        pain_points.append("Inconsistent flavor profiles and bitterness issues.")
        recommendations.append("Refine formula to balance flavor profiles and reduce harsh notes.")
        improvements.append("Formulate new recipe variations or adjust roasting/brewing time.")
    if pkg_related:
        pain_points.append("Packaging damage, leakage, or flimsy shipping boxes.")
        recommendations.append("Audit shipping vendor and upgrade to durable, leak-proof materials.")
        improvements.append("Introduce sealed bags or double-walled corrugated shipping cardboard.")
    if price_related:
        pain_points.append("Price point perceived as high compared to the quantity/value received.")
        recommendations.append("Introduce bundle deals or subscription plans to lower cost per unit.")
        improvements.append("Offer variety packs or larger bulk sizes at discount rates.")
    if fresh_related:
        pain_points.append("Freshness concerns with some batches tasting stale or close to expiration.")
        recommendations.append("Implement stricter quality controls on inventory turnover and FIFO.")
        improvements.append("Optimize storage environment humidity and speed up transit times.")

    if not pain_points:
        pain_points = ["General quality variance.", "Customer service response times."]
        recommendations = ["Regularly monitor reviews to identify emerging complaints."]
        improvements = ["Set up automated customer satisfaction feedback loops."]

    return {
        "most_requested_feature": "Improved packaging sealing and durability." if pkg_related else "Better value and larger pack options.",
        "most_critical_issue": "Packaging leakages and damage during transport." if pkg_related else "Flavor inconsistency across batches.",
        "fastest_growing_complaint": "Price-to-quantity ratio complaints." if price_related else "Batch-to-batch freshness variance.",
        "most_appreciated_feature": "Exceptional base flavor and natural ingredients.",
        "customer_pain_points": pain_points,
        "customer_expectations": [
            "Consistent premium taste and quality in every purchase.",
            "Items should arrive intact without leakage or breakage."
        ],
        "business_recommendations": recommendations,
        "potential_product_improvements": improvements
    }

def _empty_insights() -> dict:
    return {
        "most_requested_feature": "",
        "most_critical_issue": "",
        "fastest_growing_complaint": "",
        "most_appreciated_feature": "",
        "customer_pain_points": [],
        "customer_expectations": [],
        "business_recommendations": [],
        "potential_product_improvements": []
    }
