# pages/7_Settings.py
"""
Settings & Configurations Page.
Allows customization of the platform settings: model selection, cache control, and parameters.
"""

import streamlit as st
from src.core.styles import apply_custom_css
from config.settings import (
    PLATFORM_NAME, PLATFORM_VERSION, PLATFORM_TAGLINE,
    EMBEDDING_MODEL, LLM_MODEL
)

st.set_page_config(
    page_title="Settings - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Platform Settings")
st.markdown("View and configure ReviewVista machine learning, indexing, and LLM orchestration settings.")

st.divider()

# --- Model Configurations ---
st.subheader("Machine Learning & LLM Config")
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.text_input("Active Embedding Model", value=EMBEDDING_MODEL, disabled=True)
    st.caption("SentenceTransformers model utilized to encode queries and documents into 384-dimensional space.")
with col_m2:
    llm_select = st.selectbox("LLM Model Selection", options=[LLM_MODEL, "gpt-4o", "gpt-3.5-turbo"], index=0)
    st.caption("Active LLM orchestration module for generating summaries and business strategic recommendations.")

st.divider()

# --- Retrieval Parameters ---
st.subheader("Retrieval Configuration")
col_r1, col_r2 = st.columns(2)
with col_r1:
    top_k_val = st.number_input("Default Top-K Retrieval Size", min_value=1, max_value=100, value=st.session_state.get("top_k", 10))
    if top_k_val != st.session_state.get("top_k", 10):
        st.session_state.top_k = top_k_val
    st.caption("Number of relevant reviews FAISS semantic search engine retrieves at query runtime.")
with col_r2:
    st.selectbox("Similarity Metric", options=["Inner Product (Cosine Similarity)", "L2 Distance"], index=0, disabled=True)
    st.caption("Metric used by FAISS index to compare embedding vectors.")

st.divider()

# --- Cache and Optimization ---
st.subheader("Memory & Cache Optimization")
col_c1, col_c2 = st.columns(2)
with col_c1:
    st.write("**Pre-warmed memory cache details:**")
    st.write("- FAISS Vector Store Index: *Active (Cached)*")
    st.write("- Review ID Mapping dictionary: *Active (Cached)*")
    st.write("- Clusters pandas DataFrame: *Active (Cached)*")
with col_c2:
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    if st.button("Clear Application RAM Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Successfully cleared Streamlit cached memory buffers!")

st.divider()

# --- Platform Info ---
st.subheader("System Information")
st.write(f"**Platform Name:** {PLATFORM_NAME}")
st.write(f"**Description:** {PLATFORM_TAGLINE}")
st.write(f"**Active Version:** {PLATFORM_VERSION}")
st.write(f"**Environment Profile:** `Local Development`")
st.write(f"**Python Runtime:** `3.10+` (Headless Server)")
