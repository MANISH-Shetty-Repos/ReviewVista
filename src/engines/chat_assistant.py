# src/engines/chat_assistant.py
"""
RAG-Powered AI Chat Assistant.
Answers customer questions strictly using retrieved reviews as context to avoid hallucination.
"""

from src.core.logger import get_logger
from src.retrieval import retrieve_reviews
import os
import json

logger = get_logger("chat_assistant")

def _get_openai_client():
    try:
        from openai import OpenAI
        api_key = None
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None

def answer_query(query: str, history: list[dict] = None, use_llm: bool = True) -> dict:
    """
    RAG-powered conversational assistant response.
    Retrieves relevant reviews and formats them as context to answer the user query.
    """
    # 1. Retrieve top reviews relevant to the query
    retrieved = retrieve_reviews(query, k=15)
    
    context_str = ""
    for i, r in enumerate(retrieved, 1):
        rating = r.get("rating", "N/A")
        text = r.get("review_text", r.get("clean_text", ""))[:400]
        prod_id = r.get("product_id", "Unknown")
        context_str += f"Review {i} (Product: {prod_id}, Rating: {rating}/5): {text}\n\n"

    # 2. Get OpenAI response
    client = _get_openai_client()
    
    if use_llm and client:
        try:
            # Build conversation context
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Customer Feedback Intelligence Assistant. "
                        "Your job is to answer user queries based STRICTLY on the retrieved customer reviews provided below. "
                        "Never make up facts or hallucinate. If the context doesn't contain enough information to answer "
                        "the query, state clearly that you cannot find the answer in the retrieved reviews.\n\n"
                        f"REVIEWS CONTEXT:\n{context_str}"
                    )
                }
            ]
            
            # Add history if present
            if history:
                for msg in history[-5:]:  # Last 5 messages to avoid overflow
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
            messages.append({"role": "user", "content": query})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=600,
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "sources": retrieved,
                "is_llm": True
            }
            
        except Exception as e:
            logger.warning("LLM chat assistant failed: %s. Falling back to heuristic answer.", e)

    # Heuristic fallback answer based on the reviews retrieved
    fallback_answer = _build_fallback_answer(query, retrieved)
    return {
        "answer": fallback_answer,
        "sources": retrieved,
        "is_llm": False
    }

def _build_fallback_answer(query: str, reviews: list[dict]) -> str:
    """Deterministic fallback compiler when LLM is unavailable."""
    if not reviews:
        return "No reviews found in the system for this query."
        
    avg_rating = round(sum(r.get("rating", 3) for r in reviews) / len(reviews), 2)
    positives = sum(1 for r in reviews if r.get("rating", 3) >= 4)
    negatives = sum(1 for r in reviews if r.get("rating", 3) <= 2)
    
    response = (
        f"I've retrieved {len(reviews)} reviews matching your query.\n\n"
        f"**Summary of Retrieved Reviews:**\n"
        f"- **Average Rating:** {avg_rating} / 5.0\n"
        f"- **Positive Reviews:** {positives}\n"
        f"- **Negative Reviews:** {negatives}\n\n"
        f"Here are the top representative reviews matching your inquiry:\n"
    )
    
    for i, r in enumerate(reviews[:3], 1):
        rating = r.get("rating", "N/A")
        text = r.get("review_text", "")[:200]
        response += f"\n* {i}. **[Rating: {rating}/5]** \"{text}...\""
        
    response += "\n\n*(Note: OpenAI API key not configured or failed; showing rule-based summary fallback)*"
    return response
