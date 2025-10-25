import os
import streamlit as st
import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from gtts import gTTS
import tempfile
import pickle
import random
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import string
import base64
from datetime import datetime

# ==========================
# NLTK DATA DOWNLOADS
# ==========================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================
# CONFIG - SECURE API KEYS
# ==========================
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "")
except (KeyError, FileNotFoundError) as e:
    st.error(f"""
    ⚠️ API Keys not configured properly. Error: {e}
    
    Please check your Streamlit Cloud secrets:
    1. Go to App Settings → Secrets
    2. Make sure OPENROUTER_API_KEY is set exactly
    3. Format: OPENROUTER_API_KEY = "your-key-here"
    """)
    st.stop()

BASE_URL = "https://openrouter.ai/api/v1"
INDEX_FILE = "faiss_advanced_index.bin"
METADATA_FILE = "metadata.pkl"

# ==========================
# CACHED RESOURCE LOADING
# ==========================
@st.cache_resource
def load_models_and_index():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        return embedding_model, reranker_model, index, metadata
    except FileNotFoundError:
        st.error("Knowledge base files not found. Please run build_index_advanced.py first.")
        return embedding_model, reranker_model, None, None

# ==========================
# IMPROVED API CALL FUNCTION
# ==========================
def call_openrouter(model, prompt):
    """Make API call with comprehensive error handling"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.streamlit.app",  # Required by OpenRouter
        "X-Title": "Legal Research Assistant"  # Required by OpenRouter
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions", 
            headers=headers, 
            json=data, 
            timeout=30
        )
        
        # Detailed error handling
        if response.status_code == 401:
            error_msg = "❌ Authentication failed (401). Please check:"
            error_msg += "\n• Your OpenRouter API key in Streamlit secrets"
            error_msg += "\n• That the key is active and has credits"
            error_msg += "\n• There are no extra spaces in the key"
            st.error(error_msg)
            return "Authentication error: Please check your API key configuration.", 0
            
        elif response.status_code == 402:
            st.error("💳 Payment required. Please add credits to your OpenRouter account.")
            return "Payment required: Please add credits to your OpenRouter account.", 0
            
        elif response.status_code == 429:
            st.error("⏰ Rate limit exceeded. Please wait a moment and try again.")
            return "Rate limit exceeded. Please try again in a moment.", 0
            
        elif response.status_code >= 400:
            st.error(f"🔌 API Error {response.status_code}: {response.text[:200]}...")
            return f"API Error {response.status_code}: Please try again.", 0
        
        # Success case
        response.raise_for_status()
        result = response.json()
        
        if "choices" not in result or not result["choices"]:
            st.error("❌ Invalid response format from API")
            return "Invalid response from API. Please try again.", 0
            
        answer = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return answer, tokens_used
        
    except requests.exceptions.Timeout:
        st.error("⏰ Request timeout. The API is taking too long to respond.")
        return "Request timeout. Please try again.", 0
        
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Please check your internet connection.")
        return "Connection error. Please check your internet.", 0
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}", 0

# ==========================
# SIMPLIFIED PROMPT HANDLING
# ==========================
def dynamic_query_generator(question):
    """Generate 3 prompt variations"""
    base_prompt = question.strip()
    variations = [base_prompt]
    
    # Add legal-specific variations
    variations.append(f"Legal analysis of: {base_prompt}")
    variations.append(f"Provide detailed legal guidance on: {base_prompt}")
    
    return variations[:3]

def handle_prompt_selection(selected_prompt):
    """Process the selected prompt immediately"""
    with st.spinner("⚖️ Researching legal information..."):
        try:
            # Get context from knowledge base
            rag_context, sources = retrieve_and_rerank(
                selected_prompt, folder_index, folder_metadata,
                embed_model, reranker_model, top_k=3
            )
            
            # Construct final prompt
            final_prompt = f"""
            As a legal expert, provide comprehensive guidance on:
            
            {selected_prompt}
            
            Relevant legal context:
            {rag_context if rag_context else "No specific legal context found."}
            
            Please provide:
            1. Legal analysis
            2. Relevant statutes/case law if available  
            3. Practical steps
            4. Format suggestions if drafting is requested
            """
            
            # Call API
            answer, tokens = call_openrouter(st.session_state.selected_model, final_prompt)
            st.session_state.tokens_used += tokens
            
            # Add to chat
            assistant_message = {
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            }
            st.session_state.messages.append(assistant_message)
            
            # Update session
            if st.session_state.active_chat:
                st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages.copy()
            
            # Clear pending prompts
            st.session_state.pending_prompts = None
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# ==========================
# EXISTING FUNCTIONS (keep these as-is from your working code)
# ==========================
def create_extractive_summary(text, max_sentences=5):
    if not text:
        return ""
    try:
        sentences = nltk.sent_tokenize(text)
        summary = " ".join(sentences[:max_sentences])
        return summary
    except LookupError:
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        summary = " ".join(sentences[:max_sentences])
        return summary

def truncate_text(text, max_words=200):
    words = text.split()
    return " ".join(words[:max_words])

def preprocess_text(text):
    if not text:
        return []
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    return [w for w in tokens if w not in stop_words]

def bm25_search(query, metadata, top_k=10):
    tokenized_corpus = [preprocess_text(m['text']) for m in metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = preprocess_text(query)
    if not tokenized_query:
        return []
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        results.append({
            'source': metadata[idx]['source'],
            'text': metadata[idx]['text'],
            'score': scores[idx]
        })
    return results

def reciprocal_rank_fusion(results_list, k=60):
    fused_ranks = {}
    for results in results_list:
        if not results:
            continue
        for rank, item in enumerate(results):
            doc_id = item['source'] + '_' + item['text']
            fused_ranks[doc_id] = fused_ranks.get(doc_id, 0) + 1 / (k + rank)
    sorted_items = sorted(fused_ranks.items(), key=lambda x: x[1], reverse=True)
    unique_items = {item['source'] + '_' + item['text']: item for results in results_list if results for item in
                    results}
    fused_results = [unique_items[doc_id] for doc_id, _ in sorted_items if doc_id in unique_items]
    return fused_results

def retrieve_and_rerank(query, index, metadata, embed_model, reranker_model, top_k=2):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k * 4)
    semantic_results = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    bm25_results = bm25_search(query, metadata, top_k=top_k * 4)
    fused_candidates = reciprocal_rank_fusion([semantic_results, bm25_results])
    pairs = [[query, c["text"]] for c in fused_candidates]
    if not pairs:
        return "", []
    scores = reranker_model.predict(pairs)
    reranked = sorted(zip(fused_candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    top_chunks = [truncate_text(item[0]["text"], max_words=200) for item in reranked]
    sources = [{"text": item[0]["text"], "source": item[0]["source"]} for item in reranked]
    return "\n\n".join(top_chunks), sources

# ==========================
# STREAMLIT APP SETUP
# ==========================
st.set_page_config(page_title="Legal Research Assistant", layout="wide")

# Load models with error handling
embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()
if folder_index is None:
    st.stop()

# Initialize session state
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "pending_prompts" not in st.session_state: 
    st.session_state.pending_prompts = None
if "uploaded_pdfs_data" not in st.session_state: 
    st.session_state.uploaded_pdfs_data = None
if "tokens_used" not in st.session_state: 
    st.session_state.tokens_used = 0
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai/gpt-3.5-turbo"  # Use working model

# ==========================
# SIMPLIFIED UI
# ==========================
st.title("⚖️ Legal Research Assistant")
st.markdown("Professional Legal Analysis & Case Research")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model selection with working models
    st.session_state.selected_model = st.selectbox("Select Model", [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "anthropic/claude-3-sonnet",
        "google/gemini-2.0-flash-exp"
    ], index=0)
    
    st.info(f"Knowledge Base: {folder_index.ntotal} legal documents")
    
    # API Status
    if OPENROUTER_API_KEY:
        st.success("✅ API Key Loaded")
    else:
        st.error("❌ API Key Missing")

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt selection interface
if st.session_state.pending_prompts:
    st.write("### Choose a prompt variation:")
    for i, prompt in enumerate(st.session_state.pending_prompts):
        if st.button(f"Option {i+1}: {prompt}", key=f"prompt_{i}", use_container_width=True):
            handle_prompt_selection(prompt)
            st.rerun()

# Chat input
if prompt := st.chat_input("Enter your legal question..."):
    # Create new chat if needed
    if st.session_state.active_chat is None:
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[chat_id] = {
            "name": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "messages": [],
            "created_at": datetime.now()
        }
        st.session_state.active_chat = chat_id
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate prompt variations
    st.session_state.pending_prompts = dynamic_query_generator(prompt)
    st.rerun()
