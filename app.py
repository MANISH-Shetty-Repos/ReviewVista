# app.py
"""
ReviewVista Landing & Dashboard Home.
Displays high-level KPI cards, recent activities, top complaints, strengths, and AI executive summaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.engines.complaint_prioritizer import get_priority_summary
from src.engines.topic_intelligence import get_all_topics

# --- Dynamic Insight Generators ---
def generate_dynamic_executive_summary(stats: dict, priority_stats: dict, topics: list) -> dict:
    """Generate a fully dynamic executive summary based on the active dataset."""
    # 1. Dominant positive topics (highest positive_pct or avg_rating >= 4.0)
    pos_topics = [t for t in topics if t["avg_rating"] >= 4.0]
    pos_topics.sort(key=lambda x: x["review_count"], reverse=True)
    dominant_pos = [t["name"] for t in pos_topics[:2]]
    
    # 2. Dominant negative topics (lowest avg_rating or highest negative_pct)
    neg_topics = [t for t in topics if t["avg_rating"] < 4.0]
    # Fallback to sorting all topics if no topic is < 4.0
    if not neg_topics:
        neg_topics = topics.copy()
    neg_topics.sort(key=lambda x: (x["avg_rating"], -x["review_count"]))
    dominant_neg = [t["name"] for t in neg_topics[:2]]
    
    # 3. Most discussed products
    df = DataManager.get_clusters_df()
    most_discussed_products = []
    if df is not None and not df.empty and "product_id" in df.columns:
        # Exclude unknown_product if possible
        prods = df[df["product_id"] != "unknown_product"]["product_id"]
        if not prods.empty:
            most_discussed_products = prods.value_counts().head(2).index.tolist()
        else:
            most_discussed_products = df["product_id"].value_counts().head(2).index.tolist()
        
    # 4. Most common customer complaints
    complaints = priority_stats.get("complaints", [])
    common_complaints = [c["category"] for c in complaints[:2]]
    
    # Let's format human-readable text!
    # Primary Strengths
    if dominant_pos:
        strengths_text = f"Customers express high satisfaction with **{', '.join(dominant_pos)}**. "
        # Grab a praise excerpt if available
        praise_reviews = []
        for t in pos_topics[:1]:
            if t["top_reviews"]:
                praise_reviews.append(t["top_reviews"][0]["text"])
        if praise_reviews:
            strengths_text += f"For example, positive feedback highlights: *\"{praise_reviews[0][:150]}...\"*"
    else:
        strengths_text = "Overall feedback sentiment is stable, with no specific dominant strength categories identified."
        
    # Primary Pain Points
    if dominant_neg:
        pain_text = f"The main concerns are centered around **{', '.join(dominant_neg)}**. "
        complaint_reviews = []
        for t in neg_topics[:1]:
            if t["bottom_reviews"]:
                complaint_reviews.append(t["bottom_reviews"][0]["text"])
        if complaint_reviews:
            pain_text += f"Customers frequently mention issues such as: *\"{complaint_reviews[0][:150]}...\"*"
    else:
        pain_text = "No severe pain points or critical product defects were detected as dominant themes."
        
    # Strategic Action Items
    action_items = []
    if dominant_neg:
        action_items.append(f"Audit and review the quality control and customer experience for the '{dominant_neg[0]}' theme.")
    if common_complaints:
        action_items.append(f"Investigate the root causes of negative feedback related to '{common_complaints[0]}'.")
    if most_discussed_products:
        action_items.append(f"Monitor customer sentiment updates closely for the highly active product: {most_discussed_products[0]}.")
        
    if not action_items:
        action_items = [
            "Maintain standard quality checks across all product lines.",
            "Gather more customer feedback to increase analysis confidence."
        ]
        
    total_revs = stats.get('total_reviews', 0)
    n_prods = stats.get('n_products', 0)
    n_clusts = stats.get('n_clusters', 0)
    avg_rat = stats.get('avg_rating', 0.0)
    pos_p = stats.get('positive_pct', 0.0)

    intelligence_summary = (
        f"The active dataset contains {total_revs:,} reviews for {n_prods:,} unique products, "
        f"organized into {n_clusts} discovered themes. The overall rating is {avg_rat} / 5.0, "
        f"with positive reviews comprising {pos_p}% of the dataset."
    )
    
    return {
        "intelligence_summary": intelligence_summary,
        "strengths": strengths_text,
        "pain_points": pain_text,
        "action_items": action_items
    }


def generate_dynamic_strengths(stats: dict, topics: list) -> list:
    """Generate operational strengths dynamically from discovered topics."""
    dynamic_strengths = []
    # Get topics with rating >= 3.8, sorted by review count descending
    pos_topics = [t for t in topics if t["avg_rating"] >= 3.8]
    pos_topics.sort(key=lambda x: x["review_count"], reverse=True)
    
    total_reviews = stats.get("total_reviews", 1)
    
    for t in pos_topics[:4]:
        # Get description
        top_rev = t["top_reviews"][0]["text"] if t["top_reviews"] else "No specific positive feedback text recorded."
        if len(top_rev) > 120:
            top_rev = top_rev[:117] + "..."
        desc = f"Customers praised this aspect: \"{top_rev}\""
        
        pct_of_total = round((t["review_count"] / total_reviews) * 100, 1)
        
        dynamic_strengths.append({
            "name": t["name"],
            "rating": f"{t['avg_rating']:.1f}",
            "desc": desc,
            "coverage": f"{pct_of_total}% of all reviews"
        })
        
    # If not enough positive topics, add generic one
    if not dynamic_strengths:
        dynamic_strengths = [
            {
                "name": "General Satisfaction",
                "rating": f"{stats.get('avg_rating', 4.5)}",
                "desc": "Overall high rating across general feedback categories.",
                "coverage": "100.0% of feedback"
            }
        ]
    return dynamic_strengths

# 1. Page Configuration
st.set_page_config(
    page_title="ReviewVista - Customer Review Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark theme CSS
apply_custom_css()

# Initialize session state for settings
if "use_llm" not in st.session_state:
    st.session_state.use_llm = True
if "top_k" not in st.session_state:
    st.session_state.top_k = 10

# 2. Sidebar Navigation & Global Config
st.sidebar.title("ReviewVista")
st.sidebar.markdown("*AI-Powered Customer Review Intelligence Platform*")
st.sidebar.divider()

st.sidebar.subheader("Global Settings")
st.session_state.use_llm = st.sidebar.toggle("Enable LLM Insights", value=st.session_state.use_llm)
st.session_state.top_k = st.sidebar.slider("Retrieval Size (K)", min_value=5, max_value=30, value=st.session_state.top_k)

st.sidebar.info("Use the sidebar pages to navigate through the intelligence modules.")

# 3. Main Dashboard Home
st.title("Dashboard Home")
st.markdown("Real-time telemetry and predictive feedback intelligence overview.")

# Fetch statistics
stats = DataManager.get_dataset_stats()
priority_stats = get_priority_summary()
topics = get_all_topics()

# 4. KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-container">
            <p style='margin:0; font-size:13px; color:#CBD5E1;'>Total Reviews</p>
            <h2 style='margin:5px 0 0 0; font-size:26px; color:#F8FAFC;'>{stats['total_reviews']:,}</h2>
            <p style='margin:5px 0 0 0; font-size:11px; color:#10B981;'>100% Index Coverage</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-container">
            <p style='margin:0; font-size:13px; color:#CBD5E1;'>Average Rating</p>
            <h2 style='margin:5px 0 0 0; font-size:26px; color:#F8FAFC;'>{stats['avg_rating']} / 5.0</h2>
            <p style='margin:5px 0 0 0; font-size:11px; color:#CBD5E1;'>Based on 200k Sample</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-container">
            <p style='margin:0; font-size:13px; color:#CBD5E1;'>Positive Sentiment</p>
            <h2 style='margin:5px 0 0 0; font-size:26px; color:#10B981;'>{stats['positive_pct']}%</h2>
            <p style='margin:5px 0 0 0; font-size:11px; color:#CBD5E1;'>4 & 5 Star Reviews</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-container">
            <p style='margin:0; font-size:13px; color:#CBD5E1;'>Critical Complaints</p>
            <h2 style='margin:5px 0 0 0; font-size:26px; color:#EF4444;'>{priority_stats['high_priority']} High</h2>
            <p style='margin:5px 0 0 0; font-size:11px; color:#CBD5E1;'>Requires attention</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-container">
            <p style='margin:0; font-size:13px; color:#CBD5E1;'>Discovered Topics</p>
            <h2 style='margin:5px 0 0 0; font-size:26px; color:#3B82F6;'>{stats['n_clusters']}</h2>
            <p style='margin:5px 0 0 0; font-size:11px; color:#CBD5E1;'>K-Means Cluster Map</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# 5. Row 2: AI Executive Summary & Quick Actions
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("AI Executive Feedback Summary")
    
    exec_summary = generate_dynamic_executive_summary(stats, priority_stats, topics)
    action_items_html = "".join([f"<li>{item}</li>" for item in exec_summary["action_items"]])
    
    st.markdown(
        f"""
        <div class="glass-card">
            <h4 style="margin-top:0;">Intelligence Summary</h4>
            <p style="font-size: 14px; line-height: 1.5; color: #CBD5E1;">{exec_summary['intelligence_summary']}</p>
            <p><strong>Primary Strengths:</strong> {exec_summary['strengths']}</p>
            <p><strong>Primary Pain Points:</strong> {exec_summary['pain_points']}</p>
            <p><strong>Strategic Action Items:</strong>
            <ol style="margin-top: 5px; padding-left: 20px;">
                {action_items_html}
            </ol>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_right:
    st.subheader("Quick Actions")
    st.markdown(
        """
        <div class="glass-card">
            <h5 style="margin-top:0;">Available Workflows</h5>
            <ul style="padding-left:20px; margin:0;">
                <li><strong>Advanced Search:</strong> Find specific review sentiments.</li>
                <li><strong>Topic Explorer:</strong> Browse thematic keyword cards.</li>
                <li><strong>Product Compare:</strong> Run side-by-side benchmarking.</li>
                <li><strong>Export Report:</strong> Compile CSV and JSON outputs.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# 6. Row 3: Top Complaints & Top Strengths
st.divider()
col_comp, col_strength = st.columns(2)

with col_comp:
    st.subheader("Prioritized Complaints")
    complaints = priority_stats.get("complaints", [])
    
    for c in complaints[:4]:
        st.markdown(
            f"""
            <div style="background-color:#1F2937; border-left:4px solid #EF4444; border:1px solid #334155; padding:15px; margin-bottom:10px; border-radius:6px;">
                <div style="display:flex; justify-content:space-between;">
                    <strong style="color:#F8FAFC;">{c['category']}</strong>
                    <span class="badge badge-high">{c['priority_level'].replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', '')}</span>
                </div>
                <p style="margin:8px 0; font-size:13px; color:#CBD5E1;">{c['explanation']}</p>
                <div style="font-size:11px; color:#CBD5E1; opacity:0.8;">Keywords: {", ".join(c['keywords'])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_strength:
    st.subheader("Operational Strengths")
    
    strengths = generate_dynamic_strengths(stats, topics)
    
    for s in strengths:
        st.markdown(
            f"""
            <div style="background-color:#1F2937; border-left:4px solid #10B981; border:1px solid #334155; padding:15px; margin-bottom:10px; border-radius:6px;">
                <div style="display:flex; justify-content:space-between;">
                    <strong style="color:#F8FAFC;">{s['name']}</strong>
                    <span style="color:#10B981; font-weight:600; font-size:14px;">Rating: {s['rating']} / 5.0</span>
                </div>
                <p style="margin:8px 0; font-size:13px; color:#CBD5E1;">{s['desc']}</p>
                <div style="font-size:11px; color:#CBD5E1; opacity:0.8;">Topic Impact: {s['coverage']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.divider()
st.caption("ReviewVista 2.0.0 — Powered by FAISS semantic search and GPT-4o-mini.")