# pages/4_Product_Comparison.py
"""
Product Comparison Page.
Performs side-by-side benchmarking of customer feedback between two product ASINs.
"""

import streamlit as st
import pandas as pd
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.engines.comparison import compare_products, get_comparable_products

st.set_page_config(
    page_title="Product Comparison - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Product Benchmarking & Comparison")
st.markdown("Select two products to compare ratings, common complaints, key strengths, and overall customer sentiment.")

# Fetch comparison products list
comparable_products = get_comparable_products(min_reviews=5)
if len(comparable_products) < 2:
    comparable_products = get_comparable_products(min_reviews=2)
if len(comparable_products) < 2:
    comparable_products = get_comparable_products(min_reviews=1)

prod_options = [p["product_id"] for p in comparable_products]

df = DataManager.get_clusters_df()
unique_prods_count = df["product_id"].nunique() if df is not None and not df.empty else 0

if len(prod_options) < 2:
    if unique_prods_count == 1:
        st.info(f"Only one unique product detected in the active dataset ('{df['product_id'].iloc[0]}'). At least two unique products are required for comparison benchmarking.")
    elif unique_prods_count == 0:
        st.info("No reviews or products found in the active dataset to compare.")
    else:
        st.info("Not enough product IDs found with review counts to enable comparison benchmarking.")
else:
    # Selection
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        prod_a = st.selectbox("Select Product A", options=prod_options, index=0)
    with col_sel2:
        prod_b = st.selectbox("Select Product B", options=prod_options, index=min(1, len(prod_options)-1))

    if prod_a == prod_b:
        st.warning("Please select two different products to run comparison.")
    else:
        with st.spinner("Benchmarking products..."):
            results = compare_products(prod_a, prod_b)

        if "error" in results:
            st.error(results["error"])
        else:
            profile_a = results["product_a"]
            profile_b = results["product_b"]

            st.divider()

            # --- Winner Alert Banner ---
            winner = results["overall_winner"]
            if winner == "tie":
                st.info("Benchmark Result: It's a tie! Both products showcase similar user rating scores.")
            else:
                st.success(f"Benchmark Winner: Product {winner} performs better overall.")

            st.markdown(f"**AI Recommendation:** {results['recommendation']}")

            st.divider()

            # --- Side-by-Side Comparison ---
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #4F46E5;">
                        <h3 style="margin-top:0;">Product A Details</h3>
                        <p><strong>ASIN:</strong> <code>{profile_a['product_id']}</code></p>
                        <h2>Rating: {profile_a['avg_rating']} / 5.0</h2>
                        <p>Total Reviews Analyzed: {profile_a['review_count']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Sentiment metrics
                st.markdown("**Sentiment Breakdown:**")
                st.progress(profile_a["positive_pct"] / 100, text=f"Positive ({profile_a['positive_pct']}%)")
                st.progress(profile_a["negative_pct"] / 100, text=f"Negative ({profile_a['negative_pct']}%)")
                
                st.subheader("Top Appreciated Features")
                for kw in profile_a["positive_keywords"][:5]:
                    st.write(f"✓ {kw.title()}")
                    
                st.subheader("Top Complaints")
                for kw in profile_a["negative_keywords"][:5]:
                    st.write(f"✗ {kw.title()}")

                st.subheader("Representative Reviews")
                if profile_a.get("representative_positive"):
                    st.markdown("**Positive Highlights:**")
                    for r_text in profile_a["representative_positive"]:
                        st.markdown(f"*\"{r_text[:150]}\"*")
                if profile_a.get("representative_negative"):
                    st.markdown("**Negative Highlights:**")
                    for r_text in profile_a["representative_negative"]:
                        st.markdown(f"*\"{r_text[:150]}\"*")

            with col_b:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top:4px solid #3B82F6;">
                        <h3 style="margin-top:0;">Product B Details</h3>
                        <p><strong>ASIN:</strong> <code>{profile_b['product_id']}</code></p>
                        <h2>Rating: {profile_b['avg_rating']} / 5.0</h2>
                        <p>Total Reviews Analyzed: {profile_b['review_count']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Sentiment metrics
                st.markdown("**Sentiment Breakdown:**")
                st.progress(profile_b["positive_pct"] / 100, text=f"Positive ({profile_b['positive_pct']}%)")
                st.progress(profile_b["negative_pct"] / 100, text=f"Negative ({profile_b['negative_pct']}%)")
                
                st.subheader("Top Appreciated Features")
                for kw in profile_b["positive_keywords"][:5]:
                    st.write(f"✓ {kw.title()}")
                    
                st.subheader("Top Complaints")
                for kw in profile_b["negative_keywords"][:5]:
                    st.write(f"✗ {kw.title()}")

                st.subheader("Representative Reviews")
                if profile_b.get("representative_positive"):
                    st.markdown("**Positive Highlights:**")
                    for r_text in profile_b["representative_positive"]:
                        st.markdown(f"*\"{r_text[:150]}\"*")
                if profile_b.get("representative_negative"):
                    st.markdown("**Negative Highlights:**")
                    for r_text in profile_b["representative_negative"]:
                        st.markdown(f"*\"{r_text[:150]}\"*")

            st.divider()

            # --- Comparison Table ---
            st.subheader("Head-to-Head Telemetry Metrics")
            comp_df = pd.DataFrame(results["comparison_dimensions"])
            st.table(comp_df.set_index("dimension"))
