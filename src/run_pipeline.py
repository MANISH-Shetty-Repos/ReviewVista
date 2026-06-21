# src/run_pipeline.py
"""
Script to execute the preprocessing and ML processing pipeline offline
using the correct jewelry dataset.
"""

import sys
import pandas as pd
from src.core.dataset_processor import clean_review_dataset, process_dataset_pipeline
from src.core.data_loader import DataManager

def run_offline_import():
    csv_path = "/home/manishshetty/Downloads/archive/amazon_jwellery_Data.csv"
    print(f"Loading raw dataset from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
        
    print(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")
    
    # Run cleaning step
    print("Cleaning review dataset...")
    cleaned_df, stats = clean_review_dataset(
        df=df,
        text_col="review_body",
        rating_col="star_rating",
        product_col="product_id",
        category_col="product_category",
        date_col="review_date",
        user_col="customer_id",
        min_word_count=5
    )
    
    print(f"Cleaning stats: {stats}")
    print(f"Cleaned dataset: {len(cleaned_df)} rows")
    
    # Run ML pipeline
    print("Executing offline processing pipeline (embeddings, indexing, clustering, topics)...")
    def progress_cb(stage_text, val):
        print(f"[{val*100:.0f}%] {stage_text}")
        
    success = process_dataset_pipeline(
        df=cleaned_df,
        n_clusters=10,
        progress_callback=progress_cb
    )
    
    if success:
        print("Pipeline processed and saved successfully!")
    else:
        print("Pipeline execution failed.")

if __name__ == "__main__":
    run_offline_import()
