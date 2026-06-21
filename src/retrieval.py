# src/retrieval.py
"""
FAISS Retrieval Engine with Metadata-Filtered Search & Validation Checks.
Ensures query-intent matches rating/sentiment constraints before final ranking.
"""

import numpy as np
import faiss
from src.core.data_loader import DataManager
from src.core.logger import get_logger

logger = get_logger("retrieval")


def embed_query(query: str) -> np.ndarray:
    """Embed query string using SentenceTransformer via DataManager."""
    model = DataManager.get_model()
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    return query_emb


def detect_query_filter(query: str) -> str | None:
    """Detect if the query has positive, negative, or neutral sentiment/rating intent."""
    q = query.lower()
    
    # Positive intent words
    pos_keywords = [
        "best", "good", "positive", "great", "excellent", "5-star", "4-star", 
        "5 star", "4 star", "love", "like", "awesome", "perfect", "highly recommend"
    ]
    # Negative intent words
    neg_keywords = [
        "worst", "bad", "negative", "terrible", "awful", "1-star", "2-star", 
        "1 star", "2 star", "poor", "hate", "dislike", "useless", "broken", "waste"
    ]
    # Neutral intent words
    neutral_keywords = [
        "neutral", "average", "3-star", "3 star", "okay", "mediocre"
    ]
    
    if any(kw in q for kw in neg_keywords):
        return "negative"
    if any(kw in q for kw in pos_keywords):
        return "positive"
    if any(kw in q for kw in neutral_keywords):
        return "neutral"
        
    return None


def validate_review_metadata(review_dict: dict, idx: int):
    """Ensure retrieved review metadata is consistent and has required attributes."""
    if not review_dict:
        raise ValueError(f"Validation Error: Retrieved review at index {idx} is empty.")
    
    required_keys = ["review_text", "rating", "product_id"]
    for key in required_keys:
        if key not in review_dict:
            raise ValueError(f"Validation Error: Retrieved review at index {idx} is missing required metadata field '{key}'.")
            
    # Check for obvious rating range inconsistencies
    rating = review_dict["rating"]
    if not (1 <= rating <= 5):
        raise ValueError(f"Validation Error: Review ID {review_dict.get('review_id')} has invalid rating: {rating}")


def retrieve_reviews(query: str, k: int = 10) -> list[dict]:
    """
    Retrieve top-k reviews from FAISS vector store.
    Incorporates query-intent metadata filtering and strict metadata validation.
    """
    logger.info("Retrieving reviews for: '%s', k=%d", query, k)
    
    index = DataManager.get_index()
    mapping = DataManager.get_mapping()

    if not mapping:
        logger.warning("No mappings loaded. Returning empty search results.")
        return []

    # Detect user intent (best, worst, negative, etc.)
    filter_type = detect_query_filter(query)
    
    # If filter is active, fetch a larger pool of results to search from
    search_k = min(2000, len(mapping)) if filter_type else k
    
    query_emb = embed_query(query)
    scores, ids = index.search(query_emb, search_k)

    results = []
    for i, idx in enumerate(ids[0]):
        if idx < 0:
            continue
        idx_str = str(idx)
        if idx_str in mapping:
            review_dict = mapping[idx_str].copy()
            
            # Enforce validation checks before using metadata
            validate_review_metadata(review_dict, int(idx))
            
            review_dict["index"] = int(idx)
            review_dict["score"] = float(scores[0][i])
            
            # Apply rating filter based on query intent
            rating = review_dict.get("rating", 3)
            if filter_type == "negative" and rating >= 4:
                continue
            if filter_type == "positive" and rating <= 2:
                continue
            if filter_type == "neutral" and rating != 3:
                continue
                
            results.append(review_dict)
            if len(results) >= k:
                break
        else:
            logger.warning("FAISS index returned ID %d which was not found in mappings.", idx)

    # Fallback: if filtering was too strict and returned nothing, return unfiltered matches
    if not results and filter_type:
        logger.info("Metadata filter '%s' returned 0 results. Falling back to unfiltered search.", filter_type)
        for i, idx in enumerate(ids[0][:k]):
            if idx < 0:
                continue
            idx_str = str(idx)
            if idx_str in mapping:
                review_dict = mapping[idx_str].copy()
                review_dict["index"] = int(idx)
                review_dict["score"] = float(scores[0][i])
                results.append(review_dict)

    return results


def get_cluster_distribution(indices: list[int], df) -> dict:
    """
    Compute cluster statistics for the retrieved indices.
    """
    if not indices:
        return {
            "dominant_cluster": 0,
            "cluster_counts": {},
            "top_clusters": []
        }

    # Filter out invalid indices just in case
    valid_indices = [idx for idx in indices if idx < len(df)]
    if not valid_indices:
        return {
            "dominant_cluster": 0,
            "cluster_counts": {},
            "top_clusters": []
        }

    clusters = df.iloc[valid_indices]["cluster"].tolist()
    cluster_counts = {}
    for c in clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    if not cluster_counts:
        return {
            "dominant_cluster": 0,
            "cluster_counts": {},
            "top_clusters": []
        }

    dominant_cluster = max(cluster_counts, key=cluster_counts.get)
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "dominant_cluster": int(dominant_cluster),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "top_clusters": [[int(item[0]), int(item[1])] for item in top_clusters]
    }