# src/core/styles.py
"""
UI/UX Styling Helper for ReviewVista.
Provides modern, premium CSS injections to give the Streamlit dashboard a high-end SaaS feel.
"""

import streamlit as st

def apply_custom_css():
    """Injects custom CSS to style Streamlit components with a modern SaaS aesthetic."""
    st.markdown(
        """
        <style>
        /* Modern Font and Background */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }

        /* Metric Cards */
        div.metric-container {
            background-color: #1F2937 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            padding: 18px !important;
            transition: border-color 0.2s, transform 0.2s;
        }
        
        div.metric-container:hover {
            border-color: #4F46E5 !important;
            transform: translateY(-1px);
        }

        /* Status Badge Styling */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-high {
            background-color: rgba(239, 68, 68, 0.1);
            color: #EF4444;
            border: 1px solid #EF4444;
        }
        .badge-medium {
            background-color: rgba(245, 158, 11, 0.1);
            color: #F59E0B;
            border: 1px solid #F59E0B;
        }
        .badge-low {
            background-color: rgba(16, 185, 129, 0.1);
            color: #10B981;
            border: 1px solid #10B981;
        }

        /* Standardized Enterprise Cards */
        .glass-card {
            background-color: #1F2937 !important;
            border-radius: 8px !important;
            border: 1px solid #334155 !important;
            padding: 24px !important;
            margin-bottom: 20px !important;
            color: #CBD5E1 !important;
        }

        /* Sidebar Navigation Cleanup */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #334155 !important;
        }

        /* Input Fields, textareas and Select boxes styling override */
        input, select, textarea {
            background-color: #1F2937 !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }

        /* Buttons styling */
        div.stButton > button:first-child {
            background-color: #4F46E5 !important;
            color: #F8FAFC !important;
            border-radius: 6px !important;
            border: 1px solid #334155 !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            transition: background-color 0.2s, border-color 0.2s;
        }
        div.stButton > button:first-child:hover {
            background-color: #3B82F6 !important;
            border-color: #3B82F6 !important;
        }
        
        /* Highlight term wrapper */
        .highlight {
            background-color: rgba(79, 70, 229, 0.25) !important;
            border-bottom: 2px solid #4F46E5 !important;
            color: #F8FAFC !important;
            padding: 0 4px !important;
            border-radius: 2px !important;
        }

        /* Titles and Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        /* Text customization */
        p, span, li, label {
            color: #CBD5E1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
