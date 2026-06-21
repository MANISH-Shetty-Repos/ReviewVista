# src/core/data_loader.py
"""
Centralized data management with lazy loading and caching.
Loads all shared data artifacts once and provides them to all modules.
"""

import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config.settings import (
    CLUSTERS_CSV, CLUSTER_KEYWORDS_CSV,
    FAISS_INDEX_PATH, REVIEW_MAPPING_PATH,
    EMBEDDING_MODEL
)
from .logger import get_logger

logger = get_logger("data_loader")


class DataManager:
    """Singleton-style lazy loader for all data artifacts."""

    _instance = None
    _model = None
    _index = None
    _mapping = None
    _clusters_df = None
    _cluster_keywords_df = None
    _products = None
    _topics = None

    @classmethod
    def reset(cls):
        """Reset all cached data to force reload of newly processed files."""
        cls._index = None
        cls._mapping = None
        cls._clusters_df = None
        cls._cluster_keywords_df = None
        cls._products = None
        cls._topics = None
        
        # Clear module level topic cards cache
        try:
            from src.engines import topic_intelligence
            topic_intelligence._cached_topic_cards = None
        except Exception:
            pass
            
        # Clear Streamlit cache if in streamlit context
        try:
            import streamlit as st
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
            
        logger.info("DataManager state reset - next access will reload files.")

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        return cls._model

    @classmethod
    def get_index(cls):
        if cls._index is None:
            logger.info("Loading FAISS index from: %s", FAISS_INDEX_PATH)
            cls._index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info("FAISS index loaded: %d vectors", cls._index.ntotal)
        return cls._index

    @classmethod
    def get_mapping(cls) -> dict:
        if cls._mapping is None:
            logger.info("Loading review mapping from: %s", REVIEW_MAPPING_PATH)
            with open(REVIEW_MAPPING_PATH) as f:
                raw = json.load(f)
            cls._mapping = {str(k): v for k, v in raw.items()}
            logger.info("Review mapping loaded: %d entries", len(cls._mapping))
        return cls._mapping

    @classmethod
    def get_clusters_df(cls) -> pd.DataFrame:
        if cls._clusters_df is None:
            logger.info("Loading clusters data from: %s", CLUSTERS_CSV)
            cls._clusters_df = pd.read_csv(CLUSTERS_CSV)
            logger.info("Clusters data loaded: %d rows", len(cls._clusters_df))
        return cls._clusters_df

    @classmethod
    def get_cluster_keywords_df(cls) -> pd.DataFrame:
        if cls._cluster_keywords_df is None:
            logger.info("Loading cluster keywords from: %s", CLUSTER_KEYWORDS_CSV)
            cls._cluster_keywords_df = pd.read_csv(CLUSTER_KEYWORDS_CSV)
        return cls._cluster_keywords_df

    @classmethod
    def get_unique_products(cls) -> list:
        """Get list of unique product IDs from the dataset."""
        if cls._products is None:
            df = cls.get_clusters_df()
            cls._products = sorted(df["product_id"].unique().tolist())
        return cls._products

    @classmethod
    def get_topic_map(cls) -> dict:
        """Get mapping of cluster_id -> topic keywords."""
        if cls._topics is None:
            kw_df = cls.get_cluster_keywords_df()
            cls._topics = {}
            for _, row in kw_df.iterrows():
                cls._topics[row["cluster"]] = {
                    "keywords": [k.strip() for k in row["keywords"].split(",") if k.strip()],
                    "domain": row.get("domain", "general")
                }
        return cls._topics

    @classmethod
    def get_reviews_for_product(cls, product_id: str) -> pd.DataFrame:
        """Get all reviews for a specific product."""
        df = cls.get_clusters_df()
        return df[df["product_id"] == product_id]

    @classmethod
    def get_reviews_for_cluster(cls, cluster_id: int, limit: int = 100) -> pd.DataFrame:
        """Get reviews belonging to a specific cluster."""
        df = cls.get_clusters_df()
        cluster_reviews = df[df["cluster"] == cluster_id]
        return cluster_reviews.head(limit)

    @classmethod
    def get_dataset_stats(cls) -> dict:
        """Compute high-level dataset statistics."""
        df = cls.get_clusters_df()
        mapping = cls.get_mapping()

        total_reviews = len(df)
        avg_rating = round(df["rating"].mean(), 2)
        positive_pct = round((df["rating"] >= 4).mean() * 100, 1)
        negative_pct = round((df["rating"] <= 2).mean() * 100, 1)
        neutral_pct = round(100 - positive_pct - negative_pct, 1)
        n_products = df["product_id"].nunique()
        n_clusters = df["cluster"].nunique()

        # Rating distribution
        rating_dist = df["rating"].value_counts().sort_index().to_dict()

        return {
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "neutral_pct": neutral_pct,
            "n_products": n_products,
            "n_clusters": n_clusters,
            "rating_distribution": rating_dist,
            "indexed_vectors": len(mapping),
        }
