# pages/3_Analytics.py
"""
Analytics Dashboard.
Displays interactive charts, ratings distributions, sentiment trends over time, and heatmap breakdowns.
"""

import streamlit as st
import pandas as pd
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.engines.sentiment import get_sentiment_trend

st.set_page_config(
    page_title="Advanced Analytics - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Advanced Analytics Dashboard")
st.markdown("Interact with historical trends, distributions, and review intelligence analytics.")

# Fetch datasets
df = DataManager.get_clusters_df()

# --- Interactive Sidebar Filters ---
st.sidebar.subheader("Analytics Filter Panel")
selected_ratings = st.sidebar.slider("Ratings Range", min_value=1, max_value=5, value=(1, 5))
selected_clusters = st.sidebar.multiselect("Select Specific Clusters", options=sorted(df["cluster"].unique().tolist()), default=[])

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["rating"] >= selected_ratings[0]) & (filtered_df["rating"] <= selected_ratings[1])]
if selected_clusters:
    filtered_df = filtered_df[filtered_df["cluster"].isin(selected_clusters)]

# --- Layout Grid ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Star Ratings Distribution")
    rating_counts = filtered_df["rating"].value_counts().sort_index()
    
    # Render interactive bar chart
    chart_data = pd.DataFrame({
        "Reviews count": rating_counts.values
    }, index=rating_counts.index)
    
    st.bar_chart(chart_data)
    st.caption("Distribution of star ratings based on selected filters.")

with col_right:
    st.subheader("Sentiment Composition")
    pos = sum(filtered_df["rating"] >= 4)
    neg = sum(filtered_df["rating"] <= 2)
    neu = len(filtered_df) - pos - neg
    
    sentiment_data = pd.DataFrame({
        "Count": [pos, neu, neg]
    }, index=["Positive", "Neutral", "Negative"])
    
    st.bar_chart(sentiment_data)
    st.caption("Breakdown of positive (4-5), neutral (3), and negative (1-2) sentiments.")

st.divider()

# --- Trends Section ---
st.subheader("Customer Satisfaction Trend (Monthly)")

# Calculate trend
trend_data = get_sentiment_trend(filtered_df, period="M")

if not trend_data:
    st.info("Timestamp metadata missing or invalid in current sample.")
else:
    trend_df = pd.DataFrame(trend_data)
    trend_df = trend_df.set_index("period")
    
    # Render line chart for avg rating
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.line_chart(trend_df[["avg_rating"]])
        st.caption("Monthly average rating trajectory.")
    with col_t2:
        st.area_chart(trend_df[["positive_pct", "negative_pct"]])
        st.caption("Sentiment ratio shift over time (Positive vs Negative %).")

st.divider()

# --- Heatmap / Grid Breakdown ---
st.subheader("Review Intensity Heatmap (Clusters vs Ratings)")

pivot_table = filtered_df.pivot_table(
    index="cluster", 
    columns="rating", 
    aggfunc="size", 
    fill_value=0
)

# Render as styled dataframe table
st.dataframe(
    pivot_table.style.background_gradient(cmap="Blues", axis=None),
    use_container_width=True
)
st.caption("Heatmap matrix showing count of reviews across different clusters (rows) and star ratings (columns).")
