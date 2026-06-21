# pages/2_Topic_Explorer.py
"""
Topic Explorer Page.
Allows users to browse and drill down into automatically discovered discussion themes.
"""

import streamlit as st
import pandas as pd
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.engines.topic_intelligence import get_all_topics, get_topic_summary_stats

st.set_page_config(
    page_title="Topic Explorer - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Topic Explorer")
st.markdown("Discover thematic discussions automatically grouped using K-Means and PCA dimensionality reduction.")

# Fetch all topics
topics = get_all_topics()
stats = get_topic_summary_stats()

# Display summary stats
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Total Discovered Themes", stats.get("total_topics", 0))
with col_s2:
    st.metric("Largest Discussion Theme", stats.get("largest_topic", "N/A"))
with col_s3:
    st.metric("Highest Rated Theme", stats.get("highest_rated_topic", "N/A"))
with col_s4:
    st.metric("Lowest Rated Theme", stats.get("lowest_rated_topic", "N/A"))

st.divider()

# Controls & Filters
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    search_topic = st.text_input("Filter Topics by Name or Keyword", placeholder="e.g. taste, delivery")
with col_c2:
    sort_by = st.selectbox("Sort Topics By", options=["Review Count (Highest)", "Average Rating (Highest)", "Average Rating (Lowest)"])
with col_c3:
    available_domains = sorted(list(set(t["domain"] for t in topics))) if topics else ["general"]
    domain_filter = st.multiselect("Filter by Domain", options=available_domains, default=available_domains)

# Apply filters
filtered_topics = []
for t in topics:
    # Keyword search
    if search_topic:
        term = search_topic.lower()
        if term not in t["name"].lower() and not any(term in kw.lower() for kw in t["keywords"]):
            continue
            
    # Domain match
    if t["domain"] not in domain_filter:
        continue
        
    filtered_topics.append(t)

# Sorting
if sort_by == "Review Count (Highest)":
    filtered_topics.sort(key=lambda x: x["review_count"], reverse=True)
elif sort_by == "Average Rating (Highest)":
    filtered_topics.sort(key=lambda x: x["avg_rating"], reverse=True)
elif sort_by == "Average Rating (Lowest)":
    filtered_topics.sort(key=lambda x: x["avg_rating"])

# Display Topic Grid/Cards
if not filtered_topics:
    st.info("No topics match your current filter parameters.")
else:
    for idx, t in enumerate(filtered_topics):
        rating_color = "#10B981" if t["avg_rating"] >= 4.0 else "#EF4444" if t["avg_rating"] <= 3.0 else "#F59E0B"
        
        with st.expander(f"Theme: {t['name']} ({t['review_count']} reviews) | Rating: {t['avg_rating']} / 5.0", expanded=(idx==0)):
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("##### Topic Details")
                st.markdown(f"**Keywords:** {', '.join(t['keywords'])}")
                st.markdown(f"**Cluster ID:** {t['cluster_id']}")
                st.markdown(f"**Category / Domain:** `{t['domain']}`")
                
                # Progress bars for sentiment
                st.markdown("**Sentiment Distribution:**")
                st.progress(t["positive_pct"] / 100, text=f"Positive ({t['positive_pct']}%)")
                st.progress((100 - t["positive_pct"] - t["negative_pct"]) / 100, text=f"Neutral ({round(100 - t['positive_pct'] - t['negative_pct'], 1)}%)")
                st.progress(t["negative_pct"] / 100, text=f"Negative ({t['negative_pct']}%)")

            with col_right:
                st.markdown("##### AI Context Summary")
                if t.get("summary"):
                    st.write(t["summary"])
                else:
                    st.write(
                        f"This theme covers discussion surrounding {t['name'].lower()}. "
                        f"The overall sentiment is tending towards {t['trend']}. Customers mentioning this topic "
                        f"typically report {'highly positive satisfaction' if t['avg_rating'] >= 4.0 else 'moderate sentiment with complaints'}. "
                        f"Key keywords focus heavily on: {', '.join(t['keywords'][:4])}."
                    )
                
            # Representative reviews tabs
            st.markdown("##### Customer Voice (Representative Reviews)")
            tab_pos, tab_neg = st.tabs(["Positive Reviews", "Negative Reviews"])
            
            with tab_pos:
                for rev in t["top_reviews"][:3]:
                    st.markdown(f"* Rating: {rev['rating']}/5 | \"{rev['text']}...\"")
                    
            with tab_neg:
                if not t["bottom_reviews"]:
                    st.write("No negative feedback recorded for this theme.")
                for rev in t["bottom_reviews"][:3]:
                    st.markdown(f"* Rating: {rev['rating']}/5 | \"{rev['text']}...\"")
