# src/summary.py
"""
Domain-agnostic Review Summarization Engine.
Uses TF-IDF sentence ranking and deduplication to generate representative
summaries of reviews for any domain (Retail, Hotels, Apps, IMDb, etc.).
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def fix_text(text: str) -> str:
    """Fix common tokenization splits in preprocessed text."""
    fixes = {
        "wasn t": "wasn't",
        "can t": "can't",
        "it s": "it's",
        "i ve": "i've",
        "doesn t": "doesn't",
        "don t": "don't",
        "didn t": "didn't",
        "won t": "won't",
        "isn t": "isn't",
        "aren t": "aren't",
        "ve ": "have ",
        " re ": " are ",
        "ll ": "will ",
        "  ": " "
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text.strip()


def is_relevant_sentence(sentence: str) -> bool:
    """Generic check to filter out non-informative or transactional sentences."""
    s = sentence.lower().strip()
    
    # Ignore purely transactional or web template patterns
    banned_patterns = [
        "add to cart", "click here", "read more", "i bought", 
        "i purchased", "arrived yesterday", "shipping was", 
        "highly recommend", "great product", "five stars", 
        "one star", "two stars", "three stars", "four stars"
    ]
    for pattern in banned_patterns:
        if pattern in s:
            return False
            
    words = s.split()
    if len(words) < 5 or len(words) > 30:
        return False
        
    # Ignore sentences that are just numbers/symbols
    if re.match(r"^[0-9\W_]+$", s):
        return False
        
    return True


def split_into_sentences(texts: list[str]) -> list[str]:
    """Split raw reviews into constituent sentences for summary extraction."""
    sentences = []
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    
    for text in texts:
        if not isinstance(text, str):
            continue
        # Split by punctuation (period, question mark, exclamation)
        raw_sentences = sentence_end.split(text)
        for s in raw_sentences:
            s_clean = fix_text(s)
            # Further split on conjunctions if the sentence is very long
            if len(s_clean.split()) > 20:
                sub_chunks = re.split(r'\b(but|because|although)\b', s_clean, flags=re.IGNORECASE)
                for chunk in sub_chunks:
                    if chunk.strip():
                        sentences.append(chunk.strip())
            else:
                if s_clean:
                    sentences.append(s_clean)
                    
    # Filter using relevance check
    return [s for s in sentences if is_relevant_sentence(s)]


def rank_sentences(sentences: list[str]) -> list[tuple[str, float]]:
    """Rank sentences using TF-IDF word relevance score."""
    if not sentences:
        return []
    
    # Set of stopwords
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    
    try:
        X = vectorizer.fit_transform(sentences)
        # Sum TF-IDF scores for words in each sentence to measure sentence importance
        scores = np.asarray(X.sum(axis=1)).flatten()
        return sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    except Exception:
        # Fallback ranking by sentence length if TF-IDF fails (e.g. too few words)
        return [(s, float(len(s))) for s in sentences]


def deduplicate_sentences(ranked_sentences: list[tuple[str, float]], threshold: float = 0.5) -> list[str]:
    """Select the most representative, non-redundant sentences."""
    selected = []
    
    for sentence, _ in ranked_sentences:
        is_duplicate = False
        words_set = set(sentence.lower().split())
        
        for sel in selected:
            sel_words_set = set(sel.lower().split())
            if not words_set or not sel_words_set:
                continue
            # Calculate token overlap (Jaccard similarity)
            overlap = len(words_set & sel_words_set) / max(len(words_set), len(sel_words_set))
            if overlap > threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            # Capitalize first letter and format ending
            formatted = sentence.strip()
            if not formatted.endswith(('.', '!', '?')):
                formatted += '.'
            formatted = formatted[0].upper() + formatted[1:]
            selected.append(formatted)
            
        if len(selected) >= 5:
            break
            
    return selected


def summarize_reviews(texts: list[str]) -> dict:
    """
    Summarize a collection of review texts.
    Returns both a short summary and a detailed summary.
    """
    if not texts:
        return {
            "short_summary": "No feedback data available.",
            "detailed_summary": "No reviews have been uploaded or processed yet."
        }
        
    sentences = split_into_sentences(texts)
    
    # Fallback if text cleaning was too aggressive or input has short unstructured comments
    if not sentences:
        # Fallback to simple representative snippets from original texts
        clean_snippets = []
        for t in texts:
            if isinstance(t, str) and len(t.strip()) > 15:
                snippet = fix_text(t)[:150].strip()
                if not snippet.endswith(('.', '!', '?')):
                    snippet += '.'
                snippet = snippet[0].upper() + snippet[1:]
                clean_snippets.append(snippet)
                if len(clean_snippets) >= 4:
                    break
        if clean_snippets:
            return {
                "short_summary": clean_snippets[0],
                "detailed_summary": " ".join(clean_snippets)
            }
        else:
            return {
                "short_summary": "Customer opinions are mixed across general feedback.",
                "detailed_summary": "Reviews express basic customer reactions with various general comments about the experience."
            }

    ranked = rank_sentences(sentences)
    selected = deduplicate_sentences(ranked)
    
    if len(selected) >= 3:
        short_summary = " ".join(selected[:2])
        detailed_summary = " ".join(selected[:4])
    elif selected:
        short_summary = selected[0]
        detailed_summary = " ".join(selected)
    else:
        short_summary = "Customer opinions are mixed across general feedback."
        detailed_summary = "Reviews express basic customer reactions with various general comments about the experience."
        
    return {
        "short_summary": short_summary,
        "detailed_summary": detailed_summary
    }


def summarize_topic_reviews(df_cluster: "pd.DataFrame") -> str:
    """
    Generate a coherent, structured summary based on reviews in a cluster/topic.
    Includes:
    - Overview
    - Positive Findings
    - Negative Findings
    - Overall Customer Opinion
    """
    if df_cluster is None or df_cluster.empty:
        return "No feedback data available for this topic."

    # 1. Overview
    total_count = len(df_cluster)
    avg_rating = df_cluster["rating"].mean()
    pos_pct = round((df_cluster["rating"] >= 4).mean() * 100, 1) if total_count > 0 else 0.0
    neg_pct = round((df_cluster["rating"] <= 2).mean() * 100, 1) if total_count > 0 else 0.0
    
    overview = (
        f"**Overview:** This theme contains {total_count} customer reviews with an average rating of {avg_rating:.1f}/5.0. "
        f"Approximately {pos_pct}% of the feedback is positive, while {neg_pct}% contains criticisms or complaints."
    )
    
    # 2. Positive Findings
    pos_reviews = df_cluster[df_cluster["rating"] >= 4]["review_text"].dropna().tolist()
    if pos_reviews:
        pos_sum_info = summarize_reviews(pos_reviews)
        pos_findings = f"**Positive Findings:** Customers highlight key compliments: {pos_sum_info['detailed_summary']}"
    else:
        pos_findings = "**Positive Findings:** No prominent positive reviews were recorded for this theme."
        
    # 3. Negative Findings
    neg_reviews = df_cluster[df_cluster["rating"] <= 2]["review_text"].dropna().tolist()
    if neg_reviews:
        neg_sum_info = summarize_reviews(neg_reviews)
        neg_findings = f"**Negative Findings:** Key issues raised include: {neg_sum_info['detailed_summary']}"
    else:
        neg_findings = "**Negative Findings:** No critical issues or recurring complaints were reported for this theme."
        
    # 4. Overall Customer Opinion
    overall_opinion = (
        f"**Overall Customer Opinion:** The general sentiment is tending towards "
        f"{'highly positive satisfaction' if avg_rating >= 4.0 else 'moderate sentiment with complaints' if avg_rating >= 3.0 else 'negative sentiment requiring action'}. "
        f"Customers recommend monitoring this area for quality and consistency."
    )
    
    return f"{overview}\n\n{pos_findings}\n\n{neg_findings}\n\n{overall_opinion}"