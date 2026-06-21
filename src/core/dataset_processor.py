# src/core/dataset_processor.py
"""
ReviewVista Offline Processing Pipeline.
Handles validation, cleaning, embedding generation, FAISS indexing, 
clustering, and keyword extraction for custom datasets.
"""

import os
import json
import re
import numpy as np
import pandas as pd
import faiss
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from src.core.data_loader import DataManager
from src.core.logger import get_logger
from config.settings import (
    CLUSTERS_CSV, CLUSTER_KEYWORDS_CSV,
    FAISS_INDEX_PATH, REVIEW_MAPPING_PATH,
    EMBEDDING_DIM, DATA_DIR
)

logger = get_logger("dataset_processor")


def detect_encoding_and_load(file_bytes) -> pd.DataFrame:
    """Attempts loading CSV bytes using common encodings."""
    encodings = ["utf-8", "latin1", "cp1252", "utf-16", "utf-8-sig"]
    for enc in encodings:
        try:
            # Wrap bytes in StringIO after decoding
            decoded = file_bytes.decode(enc)
            import io
            df = pd.read_csv(io.StringIO(decoded))
            # Clean duplicate column headers if pandas renamed them with suffix
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            continue
    raise ValueError("Failed to parse CSV with standard encodings. Please verify the file format.")


import html

def clean_html(text: str) -> str:
    """Decode HTML entities and remove HTML tags."""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_review_dataset(
    df: pd.DataFrame,
    text_col: str,
    rating_col: str = None,
    product_col: str = None,
    category_col: str = None,
    date_col: str = None,
    user_col: str = None,
    min_word_count: int = 5
) -> tuple[pd.DataFrame, dict]:
    """
    Cleans the input dataframe according to strict data quality standards.
    """
    stats = {}
    original_count = len(df)
    stats["original_count"] = original_count

    if original_count == 0:
        raise ValueError("The uploaded dataset is empty.")

    if text_col not in df.columns:
        raise ValueError(f"Selected Review Text column '{text_col}' not found in the dataset.")

    # 1. Clean the review text column and strip HTML
    df[text_col] = df[text_col].astype(str).fillna("").apply(clean_html).str.strip()
    
    # 2. Count empty reviews
    empty_mask = df[text_col].apply(lambda x: len(x) == 0 or x.isspace())
    empty_count = sum(empty_mask)
    stats["removed_empty_count"] = int(empty_count)
    df = df[~empty_mask].copy()

    # 3. Clean spaces
    df[text_col] = df[text_col].apply(lambda x: re.sub(r"\s+", " ", x).strip())

    # 4. Remove rows with only special characters or numbers
    special_only_mask = df[text_col].apply(lambda x: re.match(r"^[0-9\W_]+$", x) is not None)
    special_only_count = sum(special_only_mask)
    stats["removed_special_count"] = int(special_only_count)
    df = df[~special_only_mask].copy()

    # 5. Remove extremely short reviews based on word count
    word_count_mask = df[text_col].apply(lambda x: len(x.split()) < min_word_count)
    short_count = sum(word_count_mask)
    stats["removed_short_count"] = int(short_count)
    df = df[~word_count_mask].copy()

    # 6. Deduplicate based on review text
    duplicate_mask = df.duplicated(subset=[text_col], keep="first")
    duplicate_count = sum(duplicate_mask)
    stats["removed_duplicate_count"] = int(duplicate_count)
    df = df[~duplicate_mask].copy()

    stats["cleaned_count"] = len(df)

    if len(df) == 0:
        raise ValueError("All records were filtered out during the data cleaning process. Try adjusting the thresholds.")

    # 7. Standardize columns
    standard_df = pd.DataFrame()
    standard_df["review_text"] = df[text_col]
    
    # Clean text representation (lower, letters only)
    standard_df["clean_text"] = df[text_col].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9\s]", " ", str(x)).lower().strip()
    )

    # Handle ratings (supporting both 1-5 star ratings and 0/1 binary ratings)
    if rating_col and rating_col in df.columns:
        # Cast to numeric, fallback to 3 for invalid
        raw_ratings = pd.to_numeric(df[rating_col], errors="coerce").fillna(3).astype(int)
        unique_vals = set(raw_ratings.unique())
        if unique_vals.issubset({0, 1}):
            # Map binary 1 -> 5 (Positive) and 0 -> 1 (Negative)
            standard_df["rating"] = raw_ratings.map({1: 5, 0: 1}).fillna(3).astype(int)
        else:
            standard_df["rating"] = raw_ratings.clip(1, 5)
    else:
        # Try to infer rating from sentiment column if present
        sentiment_cols = [c for c in df.columns if "sentiment" in c.lower()]
        if sentiment_cols:
            sent_col = sentiment_cols[0]
            standard_df["rating"] = df[sent_col].astype(str).str.lower().map({
                "positive": 5, "pos": 5, "1": 5, "negative": 1, "neg": 1, "0": 1
            }).fillna(3).astype(int)
        else:
            standard_df["rating"] = 5

    # Handle Product ID
    if product_col and product_col in df.columns:
        standard_df["product_id"] = df[product_col].fillna("unknown_product").astype(str)
    else:
        standard_df["product_id"] = "unknown_product"

    # Handle Category
    if category_col and category_col in df.columns:
        standard_df["category"] = df[category_col].fillna("general").astype(str).str.lower()
    else:
        standard_df["category"] = "general"

    # Handle Date
    if date_col and date_col in df.columns:
        # Standardize format or keep as string
        standard_df["timestamp"] = df[date_col].fillna("").astype(str)
    else:
        standard_df["timestamp"] = datetime.now().strftime("%Y-%m-%d")

    # Handle User
    if user_col and user_col in df.columns:
        standard_df["user_id"] = df[user_col].fillna("unknown_user").astype(str)
    else:
        standard_df["user_id"] = "unknown_user"

    # 8. Reset indices to guarantee perfect alignment with FAISS sequential indexing
    standard_df = standard_df.reset_index(drop=True)
    
    # 9. Add unique review ID for verification checks
    standard_df["review_id"] = [f"rev_{i}" for i in range(len(standard_df))]

    return standard_df, stats


def process_dataset_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 10,
    progress_callback=None
) -> bool:
    """
    Executes the entire offline ML pipeline on the cleaned review dataframe.
    """
    try:
        # Delete previous generated assets to ensure clean slate
        for path in [
            CLUSTERS_CSV,
            CLUSTER_KEYWORDS_CSV,
            FAISS_INDEX_PATH,
            REVIEW_MAPPING_PATH,
            DATA_DIR / "processed" / "reviews_processed.csv",
            DATA_DIR / "processed" / "cluster_keywords.csv"
        ]:
            path_str = str(path)
            if os.path.exists(path_str):
                try:
                    os.remove(path_str)
                except Exception as ex:
                    logger.warning(f"Could not remove stale file {path_str}: {ex}")

        # Step 1: Generating Embeddings
        if progress_callback:
            progress_callback("Generating Embeddings...", 0.2)
        
        texts = df["clean_text"].tolist()
        model = DataManager.get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype("float32")

        # Step 2: Building FAISS Search Index
        if progress_callback:
            progress_callback("Building Search Index...", 0.4)
            
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)

        # Step 3: Clustering Reviews (K-Means)
        if progress_callback:
            progress_callback("Clustering Reviews...", 0.6)
            
        # Handle cases where clusters exceed count of items
        clusters_count = min(n_clusters, len(df))
        kmeans = KMeans(n_clusters=clusters_count, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        df["cluster"] = cluster_labels

        # Step 4: Extracting Topics (TF-IDF per cluster)
        if progress_callback:
            progress_callback("Extracting Topics...", 0.8)
            
        keywords_df = _extract_cluster_keywords(df)

        # Step 5: Saving Results to live directories
        if progress_callback:
            progress_callback("Saving Results...", 0.9)
            
        # Ensure directories exist
        os.makedirs(os.path.dirname(CLUSTERS_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(CLUSTER_KEYWORDS_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(REVIEW_MAPPING_PATH), exist_ok=True)

        # Save CSV files
        df.to_csv(CLUSTERS_CSV, index=False)
        keywords_df.to_csv(CLUSTER_KEYWORDS_CSV, index=False)

        # Save FAISS Index
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        # Save Review Mapping dictionary
        mapping = {}
        for idx, row in df.iterrows():
            mapping[str(idx)] = {
                "review_id": row["review_id"] if "review_id" in row else f"rev_{idx}",
                "review_text": row["review_text"],
                "clean_text": row["clean_text"],
                "rating": int(row["rating"]),
                "product_id": row["product_id"],
                "category": row["category"],
                "timestamp": row["timestamp"]
            }
            
        with open(REVIEW_MAPPING_PATH, "w") as f:
            json.dump(mapping, f, indent=2)

        # Save backup copies under structured data directory
        data_uploads_dir = str(DATA_DIR / "uploads")
        data_processed_dir = str(DATA_DIR / "processed")
        os.makedirs(data_uploads_dir, exist_ok=True)
        os.makedirs(data_processed_dir, exist_ok=True)
        
        # Save historical backups
        df.to_csv(os.path.join(data_processed_dir, "reviews_processed.csv"), index=False)
        keywords_df.to_csv(os.path.join(data_processed_dir, "cluster_keywords.csv"), index=False)

        # Reset global cache state to force immediate reload
        DataManager.reset()

        if progress_callback:
            progress_callback("Processing Complete", 1.0)
            
        return True

    except Exception as e:
        logger.exception("Error executing dataset processing pipeline: %s", str(e))
        raise RuntimeError(f"Processing Pipeline Failed: {str(e)}")


def _extract_cluster_keywords(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Helper method to run TF-IDF per cluster."""
    results = []
    custom_stopwords = {
        "good", "great", "like", "love", "product", "buy",
        "really", "nice", "best", "better", "well", "also",
        "review", "customer", "feedback", "amazon", "item", "purchase",
        "bought", "got", "ordered", "one", "get", "use", "would", "will",
        "br", "href", "http", "https"
    }
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

    for cluster_id in sorted(df["cluster"].unique()):
        texts = df[df["cluster"] == cluster_id]["clean_text"].dropna()
        
        if len(texts) == 0:
            keywords = []
        else:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=stop_words,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.mean(axis=0).A1
                top_indices = scores.argsort()[::-1][:top_k]
                keywords = [feature_names[i] for i in top_indices]
            except Exception:
                keywords = ["feedback", "customer", "review"]

        results.append({
            "cluster": cluster_id,
            "keywords": ", ".join(keywords),
            "domain": "general"
        })

    return pd.DataFrame(results)
