# pages/8_Upload_Dataset.py
"""
Upload Dataset Page.
Allows business stakeholders to upload custom feedback datasets from different domains
and run the offline ML pipeline to enable immediate, interactive exploration.
"""

import os
import streamlit as st
import pandas as pd
from src.core.styles import apply_custom_css
from src.core.dataset_processor import detect_encoding_and_load, clean_review_dataset, process_dataset_pipeline

st.set_page_config(
    page_title="Upload Dataset - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Upload Dataset")
st.markdown(
    "Import custom customer review datasets (Amazon, Flipkart, Google, Hotel, Restaurant, IMDb, App, Twitter/X) "
    "and process them to generate new semantic databases, clusters, and interactive dashboards."
)

st.divider()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload review dataset (CSV format):", type=["csv"])

if uploaded_file is not None:
    # 1. Read file bytes and detect encoding
    file_bytes = uploaded_file.read()
    
    try:
        df = detect_encoding_and_load(file_bytes)
        row_count = len(df)
        
        if row_count == 0:
            st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
        else:
            st.success(f"File parsed successfully. Detected {row_count:,} records and {len(df.columns)} columns.")
            
            # Preview first 10 rows
            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.divider()
            
            # --- Column Selection & Configuration ---
            st.subheader("Configure Dataset Schema")
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                text_col = st.selectbox(
                    "Review Text Column (Required)",
                    options=df.columns,
                    help="Select the column containing the main review text to analyze."
                )
                
                rating_col = st.selectbox(
                    "Rating Column (Optional)",
                    options=["None"] + list(df.columns),
                    help="Select the column containing the numerical rating (1 to 5 stars)."
                )
                
                product_col = st.selectbox(
                    "Product Column (Optional)",
                    options=["None"] + list(df.columns),
                    help="Select the column containing the product ID, model, or name."
                )

            with col_c2:
                category_col = st.selectbox(
                    "Category Column (Optional)",
                    options=["None"] + list(df.columns),
                    help="Select the column containing the product category or domain."
                )
                
                date_col = st.selectbox(
                    "Date Column (Optional)",
                    options=["None"] + list(df.columns),
                    help="Select the column containing the review submission date/timestamp."
                )
                
                user_col = st.selectbox(
                    "User Column (Optional)",
                    options=["None"] + list(df.columns),
                    help="Select the column containing the reviewer username or ID."
                )

            # --- Advanced Processing Parameters ---
            st.divider()
            st.subheader("Processing Parameters")
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                min_words = st.slider(
                    "Minimum Word Count Threshold",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Reviews with fewer words than this threshold will be filtered out."
                )
            with col_p2:
                n_clusters = st.slider(
                    "Number of Topic Clusters",
                    min_value=3,
                    max_value=30,
                    value=10,
                    help="Select the number of thematic clusters to build using K-Means."
                )

            # Validate input columns
            valid_config = True
            if not text_col:
                valid_config = False
                st.error("Please specify the Review Text column.")

            st.divider()
            
            # --- Processing Button & Execution ---
            process_btn = st.button("Process Dataset", disabled=not valid_config, use_container_width=True)
            
            if process_btn:
                # Resolve optional columns
                sel_rating = rating_col if rating_col != "None" else None
                sel_product = product_col if product_col != "None" else None
                sel_category = category_col if category_col != "None" else None
                sel_date = date_col if date_col != "None" else None
                sel_user = user_col if user_col != "None" else None
                
                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()
                
                # Step 1: Cleaning and validation
                status_placeholder.markdown("### Status: Cleaning Reviews...")
                
                cleaned_df, clean_stats = clean_review_dataset(
                    df=df,
                    text_col=text_col,
                    rating_col=sel_rating,
                    product_col=sel_product,
                    category_col=sel_category,
                    date_col=sel_date,
                    user_col=sel_user,
                    min_word_count=min_words
                )
                
                # Show Clean Statistics
                st.markdown("### Preprocessing Results")
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("Original Record Count", clean_stats["original_count"])
                with col_r2:
                    st.metric("Cleaned Record Count", clean_stats["cleaned_count"])
                with col_r3:
                    st.metric("Removed Duplicates", clean_stats["removed_duplicate_count"])
                with col_r4:
                    st.metric("Removed Empty/Short", clean_stats["removed_empty_count"] + clean_stats["removed_short_count"])
                
                # Callback to handle pipeline stages
                def pipeline_callback(stage_text, progress_val):
                    status_placeholder.markdown(f"### Status: {stage_text}")
                    progress_bar.progress(progress_val)
                
                # Execute full ML processing pipeline
                try:
                    success = process_dataset_pipeline(
                        df=cleaned_df,
                        n_clusters=n_clusters,
                        progress_callback=pipeline_callback
                    )
                    
                    if success:
                        st.success(
                            "Processing Complete! The new dataset has been integrated successfully. "
                            "You can now navigate back to the Dashboard or Semantic Search to explore the fresh analytics instantly."
                        )
                        st.balloons()
                        
                except Exception as ex:
                    st.error(f"Error executing processing pipeline: {str(ex)}")

    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin configuring the feedback pipeline.")
