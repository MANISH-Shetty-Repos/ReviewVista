# config/settings.py
"""
Centralized configuration for ReviewVista platform.
All paths, model parameters, and feature flags in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_DIR = BASE_DIR / "faiss_index"
REPORTS_DIR = BASE_DIR / "reports"
EXPORTS_DIR = BASE_DIR / "exports"
CACHE_DIR = BASE_DIR / ".cache"

# Ensure directories exist
for d in [EXPORTS_DIR, CACHE_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Files ──────────────────────────────────────────────
CLUSTERS_CSV = ANALYSIS_DIR / "reviews_with_clusters.csv"
CLUSTER_KEYWORDS_CSV = ANALYSIS_DIR / "cluster_keywords.csv"
FAISS_INDEX_PATH = FAISS_DIR / "amazon_reviews_index.faiss"
REVIEW_MAPPING_PATH = EMBEDDINGS_DIR / "review_id_mapping.json"

# ── Embedding Model ────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ── LLM Configuration ──────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# ── Retrieval ───────────────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

# ── Clustering ──────────────────────────────────────────────
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", "50"))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "50"))

# ── Platform Metadata ──────────────────────────────────────
PLATFORM_NAME = "ReviewVista"
PLATFORM_VERSION = "2.0.0"
PLATFORM_TAGLINE = "AI-Powered Customer Review Intelligence Platform"
PLATFORM_DESCRIPTION = "Transform millions of unstructured customer reviews into actionable business intelligence"

# ── Feature Flags ───────────────────────────────────────────
ENABLE_LLM = bool(OPENAI_API_KEY)
ENABLE_CHAT = bool(OPENAI_API_KEY)
ENABLE_COMPARISON = True
ENABLE_REPORTS = True

# ── Cache TTL (seconds) ────────────────────────────────────
CACHE_TTL_SEARCH = int(os.getenv("CACHE_TTL_SEARCH", "300"))
CACHE_TTL_ANALYTICS = int(os.getenv("CACHE_TTL_ANALYTICS", "600"))
