# src/pipeline.py
"""
Central orchestrator for the runtime pipeline.
Retrieves reviews, maps clusters, runs sentiment, summaries, and business recommendations.
"""

import os
import re
from bs4 import BeautifulSoup
from src.core.data_loader import DataManager
from src.retrieval import retrieve_reviews, get_cluster_distribution
from src.engines.sentiment import analyze_sentiment
from src.engines.business_insights import generate_business_insights
from src.engines.topic_intelligence import get_topic_by_id
from src.summary import summarize_reviews
from src.core.logger import get_logger

logger = get_logger("pipeline")


def clean_text_runtime(text):
    """Clean html tags and special characters from text at runtime."""
    try:
        text = BeautifulSoup(str(text), "html.parser").get_text()
    except Exception:
        pass
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def full_pipeline(query: str, k: int = 10, use_llm: bool = True) -> dict:
    """
    Run the end-to-end feedback intelligence pipeline.
    
    Args:
        query: User's natural language search query.
        k: Number of relevant reviews to retrieve.
        use_llm: Whether to generate insights via OpenAI GPT-4o-mini.
        
    Returns:
        Dict containing top reviews, sentiment, cluster info, summaries, and business insights.
    """
    logger.info("Executing pipeline for query: '%s', k=%d", query, k)

    # 1. Lazy load CSVs and resources via DataManager
    clusters_df = DataManager.get_clusters_df()
    cluster_keywords_df = DataManager.get_cluster_keywords_df()

    # 2. Retrieve Top-K reviews
    results = retrieve_reviews(query, k=k)

    top_reviews = []
    indices = []

    for item in results:
        clean_text = clean_text_runtime(item.get("review_text", ""))
        top_reviews.append({
            "review_text": item.get("review_text", ""),
            "clean_text": clean_text,
            "rating": item.get("rating", 3),
            "product_id": item.get("product_id", "Unknown"),
            "category": item.get("category", "food"),
            "score": item.get("score", 0.0),
            "index": item.get("index")
        })
        indices.append(item["index"])

    if not top_reviews:
        logger.warning("No reviews retrieved for query: '%s'", query)
        return _empty_pipeline_response(query)

    # 3. Compute cluster distribution
    cluster_info = get_cluster_distribution(indices, clusters_df)
    dominant_cluster = cluster_info.get("dominant_cluster", 0)

    # 4. Get topic details
    topic_details = get_topic_by_id(dominant_cluster)
    cluster_keywords = topic_details.get("keywords", []) if topic_details else []

    # 5. Sentiment Analysis
    sentiment = analyze_sentiment(top_reviews)

    # 6. Summarization
    clean_texts = [r["clean_text"] for r in top_reviews]
    summary = summarize_reviews(clean_texts)

    # 7. Actionable Business Insights
    insights = generate_business_insights(top_reviews, use_llm=use_llm)

    # 8. Compile structured output
    return {
        "query": query,
        "top_reviews": top_reviews,
        "cluster_info": cluster_info,
        "topic_details": topic_details,
        "sentiment": sentiment,
        "summary": summary,
        "insight": {
            "dominant_theme": insights.get("customer_pain_points", [])[:3] or cluster_keywords[:3],
            "strengths": [insights.get("most_appreciated_feature", "")] if insights.get("most_appreciated_feature") else [],
            "pain_points": insights.get("customer_pain_points", []),
            "key_observation": insights.get("most_critical_issue", ""),
            "business_recommendation": insights.get("business_recommendations", ["Monitor feedback"])[0],
            "sentiment": sentiment["distribution"],
            "insight_source": "llm" if use_llm else "tfidf"
        },
        "advanced_insights": insights
    }


def _empty_pipeline_response(query: str) -> dict:
    return {
        "query": query,
        "top_reviews": [],
        "cluster_info": {},
        "topic_details": None,
        "sentiment": {},
        "summary": {"short_summary": "", "detailed_summary": ""},
        "insight": {},
        "advanced_insights": {}
    }