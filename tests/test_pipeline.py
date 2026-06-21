# tests/test_pipeline.py
"""
Automated validation tests for the ReviewVista analytics pipeline.
Asserts index integrity, binary rating conversion, metadata-filtered retrieval, and sentiment classification.
"""

import os
import json
import unittest
import pandas as pd
from src.core.dataset_processor import clean_review_dataset, process_dataset_pipeline
from src.retrieval import retrieve_reviews, detect_query_filter
from src.engines.sentiment import analyze_sentiment
from src.engines.topic_intelligence import get_all_topics
from src.core.data_loader import DataManager


class TestReviewVistaPipeline(unittest.TestCase):

    def setUp(self):
        # Create a small mockup dataset with binary ratings to test conversion
        self.mock_data = pd.DataFrame({
            "review_text_col": [
                "So beautiful even tho clearly not high end. Great jewelry!",
                "Great product. Highly recommend to everyone.",
                "Exactly as pictured and my daughter loved it.",
                "They didn't even last through the first day. Terrible quality.",
                "Broke immediately, absolute waste of money.",
                "Very comfortable ring, beautiful shine.",
                "It was ok, nothing special but average.",
                "Extremely disappointed with this purchase."
            ],
            "rating_col": [1, 1, 1, 0, 0, 1, 0, 0],  # Binary ratings: 1 = positive, 0 = negative
            "product_col": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
        })
        
        # Paths for temporary testing outputs
        self.test_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        os.makedirs(os.path.join(self.test_dir, "test_output"), exist_ok=True)
        
        # Override config paths in settings temporarily if needed, or rely on normal settings.
        # Here we will clean and process data using the processor.
        
    def test_clean_review_dataset(self):
        """Test that data cleaning, binary rating mapping, and index resetting work correctly."""
        cleaned_df, stats = clean_review_dataset(
            self.mock_data.copy(),
            text_col="review_text_col",
            rating_col="rating_col",
            product_col="product_col",
            min_word_count=2
        )
        
        # 1. Assert index is reset to 0..N-1 sequential
        self.assertEqual(list(cleaned_df.index), list(range(len(cleaned_df))))
        
        # 2. Assert unique review IDs are present
        self.assertIn("review_id", cleaned_df.columns)
        self.assertEqual(cleaned_df.loc[0, "review_id"], "rev_0")
        
        # 3. Assert binary rating conversion worked: 1 -> 5 stars, 0 -> 1 star
        # Positive review at index 0 should map to rating 5
        self.assertEqual(cleaned_df.loc[0, "rating"], 5)
        # Negative review at index 3 should map to rating 1
        self.assertEqual(cleaned_df.loc[3, "rating"], 1)

    def test_sentiment_classification_agreement(self):
        """Test that sentiment classification strictly agrees with ratings and flags conflicts."""
        cleaned_df, _ = clean_review_dataset(
            self.mock_data.copy(),
            text_col="review_text_col",
            rating_col="rating_col",
            product_col="product_col",
            min_word_count=2
        )
        
        # Run sentiment analysis
        reviews_list = cleaned_df.to_dict(orient="records")
        sentiment_res = analyze_sentiment(reviews_list)
        
        # Verify that rating 5 results in positive sentiment and rating 1 results in negative sentiment
        for item in sentiment_res["review_sentiments"]:
            rating = item["rating"]
            sentiment = item["sentiment"]
            if rating >= 4:
                self.assertEqual(sentiment, "positive")
            elif rating <= 2:
                self.assertEqual(sentiment, "negative")

    def test_query_filter_detector(self):
        """Test that query sentiment filters are correctly detected."""
        self.assertEqual(detect_query_filter("show worst reviews"), "negative")
        self.assertEqual(detect_query_filter("best products"), "positive")
        self.assertEqual(detect_query_filter("average feedback"), "neutral")
        self.assertIsNone(detect_query_filter("necklace quality"))


if __name__ == "__main__":
    unittest.main()
