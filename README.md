# ReviewVista

## Overview

ReviewVista is a customer feedback intelligence platform designed to transform unstructured product reviews into actionable business insights. The platform combines semantic vector search, unsupervised topic discovery, sentiment analysis, and Retrieval-Augmented Generation (RAG) to help businesses understand customer opinions, identify recurring issues, compare products, and generate evidence-based recommendations.

The project demonstrates an end-to-end AI pipeline covering data ingestion, preprocessing, embedding generation, vector indexing, semantic retrieval, topic clustering, analytics, report generation, and an interactive RAG-powered assistant.

---

## Features

- Semantic search using dense vector embeddings
- FAISS-based high-performance vector retrieval
- Retrieval-Augmented Generation (RAG) assistant
- Automatic topic discovery using K-Means clustering
- Customer sentiment analysis
- Interactive analytics dashboard
- Product comparison and benchmarking
- Real-time KPI dashboard
- Custom dataset upload and preprocessing
- Executive, technical, and customer insight report generation
- Metadata-based filtering
- Modular and scalable architecture

---

## System Architecture

```text
                      Customer Review Dataset
                               │
                               ▼
                    Data Ingestion & Validation
                               │
                               ▼
                  Data Cleaning & Preprocessing
                               │
                               ▼
          SentenceTransformer Embedding Generation
                               │
                               ▼
                     FAISS Vector Index Creation
                               │
                               ▼
                 Topic Clustering (K-Means Model)
                               │
                               ▼
                 Keyword Extraction (TF-IDF Model)
                               │
──────────────────────────────────────────────────────────
                               │
                               ▼
                           User Query
                               │
                               ▼
                      Query Embedding Generation
                               │
                               ▼
                  Semantic Search (FAISS Top-K)
                               │
                               ▼
                 Metadata & Intent-Based Filtering
                               │
                               ▼
                 Retrieved Customer Review Context
                               │
                               ▼
             Retrieval-Augmented Generation (RAG)
                               │
                               ▼
                     OpenAI GPT Response Generation
                               │
                               ▼
          Business Insights & Strategic Recommendations
```

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.10+ |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Embedding Model | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (IndexFlatIP) |
| Large Language Model | OpenAI GPT Models |
| Text Processing | BeautifulSoup4, Regular Expressions |
| Environment Management | python-dotenv |
| Version Control | Git & GitHub |

---

## Processing Pipeline

### Offline Processing Pipeline

1. Load customer review datasets.
2. Detect dataset encoding.
3. Validate dataset schema.
4. Clean and normalize review text.
5. Remove duplicate and invalid records.
6. Generate dense embeddings using SentenceTransformers.
7. Normalize embedding vectors.
8. Build the FAISS vector index.
9. Discover topics using K-Means clustering.
10. Extract representative keywords using TF-IDF.
11. Store processed artifacts for runtime retrieval.

### Runtime Retrieval Pipeline

1. Accept a natural language query.
2. Generate semantic embeddings for the query.
3. Retrieve the Top-K most relevant reviews using FAISS.
4. Apply metadata and intent-based filtering.
5. Construct the Retrieval-Augmented Generation (RAG) prompt.
6. Generate grounded insights using OpenAI GPT.
7. Execute heuristic summarization when LLM services are unavailable.

---

## Dataset Format

ReviewVista supports importing custom CSV datasets.

| Field | Required | Description |
|------|----------|-------------|
| Review Text | Yes | Customer review content |
| Rating | No | Rating from 1–5 |
| Product | No | Product identifier or name |
| Category | No | Product category |
| Date | No | Review timestamp |
| User | No | Reviewer identifier |

Only the **Review Text** column is mandatory.

---

## Current Limitations

- Supports CSV datasets only.
- FAISS index is stored locally.
- K-Means cluster count is fixed during index generation.
- In-memory caching may limit scalability for very large datasets.
- RAG response quality depends on the relevance of retrieved reviews.