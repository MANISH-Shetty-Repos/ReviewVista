# src/engines/report_generator.py
"""
Report Generation Engine.
Exports analysis results, search queries, and summaries into CSV, JSON, and formatted reports.
"""

import json
import csv
import io
import pandas as pd
from datetime import datetime
from src.core.logger import get_logger

logger = get_logger("report_generator")

def generate_report(report_type: str, data: dict, format_type: str = "json") -> bytes:
    """
    Generate and export a report in specified format.

    Args:
        report_type: "executive", "technical", or "customer_insight"
        data: Combined data payload to include in the report
        format_type: "json", "csv", or "markdown"

    Returns:
        Bytes representing the file contents.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Structure the report metadata
    payload = {
        "report_type": report_type,
        "generated_at": timestamp,
        "platform": "ReviewVista Feedback Platform",
        "version": "2.0.0",
        "data": data
    }

    if format_type.lower() == "json":
        return json.dumps(payload, indent=2).encode("utf-8")
        
    elif format_type.lower() == "csv":
        return _generate_csv(data)
        
    elif format_type.lower() == "markdown" or format_type.lower() == "txt":
        return _generate_markdown(report_type, data, timestamp).encode("utf-8")

    else:
        raise ValueError(f"Unsupported format: {format_type}")

def _generate_csv(data: dict) -> bytes:
    """Convert flat lists or dictionaries in the report data to a downloadable CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Check if there is a 'top_reviews' key
    if "top_reviews" in data:
        writer.writerow(["Index", "Rating", "Clean Text", "Original Review Text"])
        for idx, r in enumerate(data["top_reviews"], 1):
            writer.writerow([
                idx,
                r.get("rating", 3),
                r.get("clean_text", ""),
                r.get("review_text", "")
            ])
    elif "complaints" in data:
        writer.writerow(["Category", "Priority Level", "Priority Score", "Frequency", "Frequency Pct", "Avg Rating"])
        for c in data["complaints"]:
            writer.writerow([
                c.get("category", ""),
                c.get("priority_level", ""),
                c.get("priority_score", 0),
                c.get("frequency", 0),
                c.get("frequency_pct", 0.0),
                c.get("avg_rating", 0.0)
            ])
    elif "topics" in data:
        writer.writerow(["Topic ID", "Name", "Keywords", "Review Count", "Avg Rating", "Positive %", "Negative %"])
        for t in data["topics"]:
            writer.writerow([
                t.get("cluster_id", 0),
                t.get("name", ""),
                ", ".join(t.get("keywords", [])),
                t.get("review_count", 0),
                t.get("avg_rating", 0.0),
                t.get("positive_pct", 0.0),
                t.get("negative_pct", 0.0)
            ])
    else:
        # Generic key-value CSV fallback
        writer.writerow(["Key", "Value"])
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)):
                writer.writerow([k, v])
            else:
                writer.writerow([k, str(v)])
                
    return output.getvalue().encode("utf-8")

def _generate_markdown(report_type: str, data: dict, timestamp: str) -> str:
    """Build a professional formatted text/markdown report."""
    title = f"ReviewVista {report_type.replace('_', ' ').title()} Report"
    
    md = f"""# {title}
Generated At: {timestamp}
Platform: ReviewVista Customer Feedback Intelligence Platform

---

## Executive Summary
"""

    if "insight" in data:
        ins = data["insight"]
        md += f"""- **Dominant Theme:** {", ".join(ins.get("dominant_theme", []))}
- **Key Observation:** {ins.get("key_observation", "")}
- **Actionable Recommendation:** {ins.get("business_recommendation", "")}

### Sentiment Overview
- **Average Rating:** {ins.get("sentiment", {}).get("avg_rating", "N/A")} / 5.0
- **Positive Ratio:** {ins.get("sentiment", {}).get("positive_ratio", 0) * 100}%
- **Negative Ratio:** {ins.get("sentiment", {}).get("negative_ratio", 0) * 100}%
"""
    elif "summary" in data:
        md += f"""{data["summary"].get("short_summary", "No summary available.")}\n"""

    if "top_reviews" in data:
        md += "\n## Representative Customer Feedback\n"
        for idx, r in enumerate(data["top_reviews"][:5], 1):
            md += f"\n**{idx}. [Rating: {r.get('rating')}/5]**\n> {r.get('review_text')}\n"

    if "complaints" in data:
        md += "\n## Prioritized Customer Complaints\n"
        for c in data["complaints"][:5]:
            md += f"\n### {c.get('category')} ({c.get('priority_level')})\n"
            md += f"- **Priority Score:** {c.get('priority_score')}/100\n"
            md += f"- **Severity Score:** {c.get('severity_score')}/100\n"
            md += f"- **Explanation:** {c.get('explanation')}\n"

    md += "\n---\n*End of Report.*"
    return md
