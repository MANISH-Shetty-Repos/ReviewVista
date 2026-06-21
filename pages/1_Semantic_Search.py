# pages/1_Semantic_Search.py
"""
Semantic Search Page.
Allows natural language search across reviews with advanced metadata filters, highlighting, and exports.
"""

import streamlit as st
import pandas as pd
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.pipeline import full_pipeline
from src.engines.report_generator import generate_report

st.set_page_config(
    page_title="Semantic Search - ReviewVista",
    layout="wide"
)

apply_custom_css()

# Initialize session state for searches
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []
if "saved_searches" not in st.session_state:
    st.session_state.saved_searches = []
if "use_llm" not in st.session_state:
    st.session_state.use_llm = True
if "top_k" not in st.session_state:
    st.session_state.top_k = 10

st.title("Semantic Search Engine")
st.markdown("Search customer opinions using natural language, powered by SentenceTransformers and FAISS.")

# Fetch filter lists
products = DataManager.get_unique_products()

# --- Search Interface ---
col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input(
        "Enter search query:",
        placeholder="e.g. why are customers complaining about bitterness?",
        value="bitter coffee flavor issues",
        label_visibility="collapsed"
    )
with col_btn:
    run_search = st.button("Search Database", use_container_width=True)

# Add search query suggestions
st.markdown("Suggestions: **bad coffee taste** | **fresh dog treats** | **damaged packaging issues** | **stale snack products**")

# --- Advanced Filters ---
with st.expander("Advanced Metadata Filters", expanded=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        rating_filter = st.multiselect("Filter by Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with col_f2:
        sentiment_filter = st.multiselect("Filter by Sentiment", options=["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])
    with col_f3:
        product_filter = st.selectbox("Filter by Product ID", options=["All Products"] + products[:50])

# Execute pipeline
if run_search or query:
    if query not in st.session_state.recent_searches:
        st.session_state.recent_searches = [query] + st.session_state.recent_searches[:4]

    with st.spinner("Running semantic retrieval..."):
        # Retrieve slightly more than requested to allow metadata filtering
        output = full_pipeline(query, k=st.session_state.top_k + 15, use_llm=st.session_state.use_llm)

    if not output["top_reviews"]:
        st.info("No reviews matched your search criteria.")
    else:
        # Filter results based on metadata filter inputs
        filtered_reviews = []
        for r in output["top_reviews"]:
            # Rating match
            if r["rating"] not in rating_filter:
                continue
            
            # Sentiment match
            score = (r["rating"] - 3) / 2.0
            sentiment = "neutral"
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            if sentiment not in sentiment_filter:
                continue
                
            # Product match
            if product_filter != "All Products" and r["product_id"] != product_filter:
                continue
                
            filtered_reviews.append(r)
            
        # Limit to top-k
        filtered_reviews = filtered_reviews[:st.session_state.top_k]

        st.divider()

        # Display Summary
        st.subheader("Semantic Insight Summary")
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown(f"**Dominant Themes:** {', '.join(output['insight'].get('dominant_theme', []))}")
            st.markdown(f"**Key Observation:** {output['insight'].get('key_observation', 'Mixed reviews')}")
        with col_sum2:
            st.markdown(f"**Recommendation:** {output['insight'].get('business_recommendation', 'Maintain standard operations')}")

        st.divider()

        # Results Layout
        col_results, col_side = st.columns([3, 1])

        with col_results:
            st.subheader(f"Results ({len(filtered_reviews)} reviews matched)")
            
            for idx, r in enumerate(filtered_reviews, 1):
                rating_stars = "⭐" * r["rating"]
                score_pct = round(r.get("score", 0.0) * 100, 1)
                
                # Simple highlight word match helper
                raw_text = r["review_text"]
                clean_display = raw_text
                
                # Highlight matching terms from query
                query_terms = [t for t in query.lower().split() if len(t) > 3]
                for term in query_terms:
                    import re
                    clean_display = re.sub(
                        f"({term})", 
                        r"<span class='highlight'>\1</span>", 
                        clean_display, 
                        flags=re.IGNORECASE
                    )

                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; font-size:14px; color:#CBD5E1;">#{idx} | Product: {r['product_id']}</span>
                            <span style="color:#F59E0B; font-weight:600;">Rating: {r['rating']} / 5.0</span>
                        </div>
                        <p style="margin:10px 0; line-height:1.5; font-size:14px; color:#F8FAFC;">{clean_display}</p>
                        <div style="display:flex; justify-content:space-between; font-size:12px; color:#CBD5E1; opacity:0.8;">
                            <span>Category: {r['category'].upper()}</span>
                            <span>Semantic Similarity: {score_pct}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_side:
            # Action Panel & Analytics
            st.subheader("Actions")
            
            # Export Reports
            export_payload = {
                "top_reviews": filtered_reviews,
                "insight": output["insight"],
                "summary": output["summary"]
            }
            
            # Download JSON
            json_report = generate_report("search_results", export_payload, "json")
            st.download_button(
                label="Export as JSON",
                data=json_report,
                file_name=f"search_{query.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Download CSV
            csv_report = generate_report("search_results", export_payload, "csv")
            st.download_button(
                label="Export as CSV",
                data=csv_report,
                file_name=f"search_{query.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.divider()
            
            st.subheader("Saved Searches")
            if st.button("Save Current Query", use_container_width=True):
                if query not in st.session_state.saved_searches:
                    st.session_state.saved_searches.append(query)
                    st.success("Query saved!")
                    
            for sq in st.session_state.saved_searches:
                st.markdown(f"Pin: *{sq}*")

            st.divider()

            st.subheader("Recent Searches")
            for rq in st.session_state.recent_searches:
                st.markdown(f"Query: *{rq}*")
