# pages/6_Reports.py
"""
Reports Compilation Page.
Generates customizable executive, technical, and customer feedback reports for download in CSV, JSON, or Markdown formats.
"""

import streamlit as st
from src.core.data_loader import DataManager
from src.core.styles import apply_custom_css
from src.engines.report_generator import generate_report
from src.engines.complaint_prioritizer import get_priority_summary

st.set_page_config(
    page_title="Executive Reports - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("Feedback Reports & Exports")
st.markdown("Compile, customize, and download structured reports on customer opinions, complaints, and trends.")

st.divider()

col_config, col_preview = st.columns([1, 2])

with col_config:
    st.subheader("Report Settings")
    
    report_type = st.selectbox(
        "Select Report Template",
        options=["Executive Summary Report", "Technical Analysis Report", "Customer Insight Report"]
    )
    
    format_type = st.selectbox(
        "Select File Format",
        options=["JSON Schema", "CSV Data Sheet", "Markdown Document"]
    )
    
    include_reviews = st.checkbox("Include Representative Reviews", value=True)
    include_complaints = st.checkbox("Include Prioritized Complaint Metrics", value=True)
    
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    compile_btn = st.button("Compile Document", use_container_width=True)

with col_preview:
    st.subheader("Document Preview")
    
    if compile_btn:
        with st.spinner("Compiling document payload..."):
            # Gather relevant data
            data_payload = {}
            df = DataManager.get_clusters_df()
            
            if include_complaints:
                priority_summary = get_priority_summary()
                data_payload["complaints"] = priority_summary.get("complaints", [])
                
            if include_reviews:
                # Use sample of reviews
                sample_reviews = []
                for _, row in df.head(15).iterrows():
                    sample_reviews.append({
                        "review_text": row.get("review_text", ""),
                        "clean_text": row.get("clean_text", ""),
                        "rating": int(row.get("rating", 3)),
                        "product_id": row.get("product_id", "")
                    })
                data_payload["top_reviews"] = sample_reviews
                
            # Maps selection to format keys
            fmt_map = {
                "JSON Schema": "json",
                "CSV Data Sheet": "csv",
                "Markdown Document": "markdown"
            }
            
            template_map = {
                "Executive Summary Report": "executive",
                "Technical Analysis Report": "technical",
                "Customer Insight Report": "customer_insight"
            }
            
            file_exts = {
                "json": "json",
                "csv": "csv",
                "markdown": "md"
            }
            
            selected_fmt = fmt_map[format_type]
            selected_template = template_map[report_type]
            
            report_bytes = generate_report(selected_template, data_payload, selected_fmt)
            
            # Show preview
            st.success("Document successfully compiled!")
            
            # Preview format rendering
            if selected_fmt == "json":
                st.code(report_bytes.decode("utf-8")[:1000] + "\n\n... (payload truncated for preview)", language="json")
            elif selected_fmt == "csv":
                st.code(report_bytes.decode("utf-8")[:500] + "\n\n... (data sheet truncated for preview)", language="text")
            elif selected_fmt == "markdown":
                st.markdown(
                    f"""
                    <div style="background-color:#1F2937; border:1px solid #334155; padding:15px; max-height:400px; overflow-y:auto; border-radius:8px; color:#CBD5E1;">
                        {report_bytes.decode("utf-8")[:2000]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            
            # Download Button
            st.download_button(
                label="Download Compiled Report File",
                data=report_bytes,
                file_name=f"{selected_template}_report.{file_exts[selected_fmt]}",
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.info("Configure report settings on the left panel and click 'Compile Document' to preview and download.")
