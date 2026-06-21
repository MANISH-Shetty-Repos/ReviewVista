# pages/5_AI_Assistant.py
"""
AI Chat Assistant Page.
Allows conversational exploration of feedback, grounded strictly in retrieved review context to prevent hallucination.
"""

import streamlit as st
from src.core.styles import apply_custom_css
from src.engines.chat_assistant import answer_query

st.set_page_config(
    page_title="AI Chat Assistant - ReviewVista",
    layout="wide"
)

apply_custom_css()

st.title("AI Feedback Assistant")
st.markdown("Ask natural language questions about customer complaints, product strengths, or overall sentiment.")

# Initialize conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}
if "use_llm" not in st.session_state:
    st.session_state.use_llm = True

# Suggested Quick Prompts
st.markdown("##### Suggested Questions:")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    q1 = st.button("Why are customers unhappy with taste?", use_container_width=True)
with col_p2:
    q2 = st.button("Summarize delivery & packaging issues", use_container_width=True)
with col_p3:
    q3 = st.button("What are positive comments saying about value?", use_container_width=True)

# Select question if suggestions clicked
preset_query = ""
if q1:
    preset_query = "Why are customers unhappy with taste?"
elif q2:
    preset_query = "Summarize delivery & packaging issues"
elif q3:
    preset_query = "What are positive comments saying about value?"

st.divider()

# --- Chat Interface Container ---
chat_container = st.container()

with chat_container:
    # Display historical chat messages
    for idx, msg in enumerate(st.session_state.chat_history):
        role_label = "You" if msg["role"] == "user" else "AI Assistant"
        
        st.markdown(f"**{role_label}**")
        st.write(msg["content"])
        
        # Display source reviews if it was an assistant response
        if msg["role"] == "assistant" and idx in st.session_state.sources_history:
            sources = st.session_state.sources_history[idx]
            with st.expander("Show Source Reviews Grounding this Response"):
                for s_idx, src in enumerate(sources[:5], 1):
                    st.markdown(
                        f"""
                        <div style="background-color:#1F2937; border:1px solid #334155; padding:12px; margin-bottom:8px; border-radius:6px; font-size:13px; color:#CBD5E1;">
                            <strong>Source {s_idx} (Product: {src.get('product_id')}, Rating: {src.get('rating')}/5)</strong><br>
                            "{src.get('review_text', src.get('clean_text'))[:200]}..."
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.divider()

# Input box
user_input = st.chat_input("Ask a feedback question...")
query_to_send = preset_query or user_input

if query_to_send:
    # 1. Append user message to history
    st.session_state.chat_history.append({"role": "user", "content": query_to_send})
    
    # Rerender immediately for feedback feel
    st.rerun()

# Execute model response generation if last message is from user
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    last_query = st.session_state.chat_history[-1]["content"]
    
    with st.spinner("Retrieving semantic reviews and generating answer..."):
        # Format chat history in format expected by chat assistant
        history_formatted = []
        for msg in st.session_state.chat_history[:-1]:
            history_formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        result = answer_query(last_query, history_formatted, use_llm=st.session_state.use_llm)
        
    # Append assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
    
    # Store source reviews metadata mapped to index
    new_assistant_idx = len(st.session_state.chat_history) - 1
    st.session_state.sources_history[new_assistant_idx] = result["sources"]
    
    st.rerun()
