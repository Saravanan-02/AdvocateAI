import os
import streamlit as st
import fitz  # PyMuPDF
import gdown
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
# CONFIG
# ==========================
OPENROUTER_API_KEY = "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"
GOOGLE_VISION_API_KEY = "AIzaSyBFh_YqGdkvUjQPT6ihyur2mlvETJcOF_k"
BASE_URL = "https://openrouter.ai/api/v1"

INDEX_FILE = "faiss_advanced_index.bin"
METADATA_FILE = "metadata.pkl"

# Google Drive File IDs
FAISS_FILE_ID = "1Ctr4N_eDIGL5Nmeb5Mx0M8BpesxHGxwg"
METADATA_FILE_ID = "1zx1Xm-B1SsLLt-7y50amAHPwikGUaG90"

# ==========================
# RELIABLE GOOGLE DRIVE DOWNLOAD FUNCTION (gdown)
# ==========================
def download_from_drive(file_id, destination):
    """Download file from Google Drive using gdown safely."""
    try:
        if os.path.exists(destination):
            return True

        st.info(f"📥 Downloading {os.path.basename(destination)} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        if not os.path.exists(destination) or os.path.getsize(destination) == 0:
            raise Exception("Downloaded file is empty.")
        return True
    except Exception as e:
        st.error(f"Error downloading {destination}: {e}")
        st.warning(f"You can manually check the file here:\nhttps://drive.google.com/uc?id={file_id}")
        return False

# ==========================
# FAISS VALIDATION HELPER
# ==========================
def is_valid_faiss_index(file_path):
    """Quickly check if file is a valid FAISS binary or HTML (corrupted)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(10)
            if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
                return False
        _ = faiss.read_index(file_path)
        return True
    except Exception:
        return False

# ==========================
# LOAD MODELS AND KNOWLEDGE BASE
# ==========================
@st.cache_resource
def load_models_and_index():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # Download FAISS and metadata
    faiss_ok = download_from_drive(FAISS_FILE_ID, INDEX_FILE)
    metadata_ok = download_from_drive(METADATA_FILE_ID, METADATA_FILE)

    # Validate FAISS index
    if not os.path.exists(INDEX_FILE) or not is_valid_faiss_index(INDEX_FILE):
        st.error("❌ FAISS file is invalid or corrupted (HTML file detected). Using fallback mini index.")
        fallback_metadata = [
            {'text': 'Article 226 of the Indian Constitution provides writ jurisdiction to High Courts', 'source': 'Constitution'},
            {'text': 'Suspension without notice violates natural justice principles', 'source': 'Service Law'},
            {'text': 'Madras High Court has jurisdiction over Tamil Nadu and Puducherry', 'source': 'Jurisdiction'}
        ]
        texts = [m['text'] for m in fallback_metadata]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        fallback_index = faiss.IndexFlatIP(dimension)
        fallback_index.add(embeddings)
        return embedding_model, reranker_model, fallback_index, fallback_metadata

    # Load FAISS index and metadata
    try:
        with st.spinner("🔧 Loading knowledge base..."):
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
        st.success(f"✅ Knowledge base loaded successfully with {index.ntotal} chunks")
        return embedding_model, reranker_model, index, metadata
    except Exception as e:
        st.error(f"❌ Error loading FAISS or metadata: {e}")
        st.warning("Switching to fallback minimal knowledge base...")
        fallback_metadata = [
            {'text': 'Article 226 of Indian Constitution provides writ jurisdiction to High Courts', 'source': 'Constitution'},
            {'text': 'Suspension without notice violates natural justice principles', 'source': 'Service Law'},
            {'text': 'Madras High Court has jurisdiction over Tamil Nadu and Puducherry', 'source': 'Jurisdiction'}
        ]
        texts = [m['text'] for m in fallback_metadata]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        fallback_index = faiss.IndexFlatIP(dimension)
        fallback_index.add(embeddings)
        return embedding_model, reranker_model, fallback_index, fallback_metadata

# ==========================
# STREAMLIT APP CONFIG
# ==========================
st.set_page_config(page_title="Advocate AI Pro", layout="wide")

with st.spinner("🚀 Loading AI models and knowledge base..."):
    embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()

st.success(f"Knowledge base ready with {folder_index.ntotal} chunks.", icon="✅")

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_most_relevant_text(query, top_k=3):
    """Retrieve top K similar documents from FAISS index."""
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    scores, indices = folder_index.search(query_embedding, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(folder_metadata):
            results.append((folder_metadata[idx]["text"], score, folder_metadata[idx]["source"]))
    return results

def generate_llm_response(prompt):
    """Call OpenRouter AI API to generate legal answers."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are AdvocateAI, an Indian legal assistant providing accurate legal guidance."},
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ API Error: {e}"

def text_to_speech(text):
    """Convert response text to audio."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text)
        tts.save(tmpfile.name)
        return tmpfile.name

# ==========================
# STREAMLIT UI
# ==========================
st.title("⚖️ Advocate AI Pro")
st.markdown("Your AI-powered Indian Legal Assistant")

query = st.text_input("Enter your legal query:", placeholder="e.g., Can a government employee be suspended without notice?")

if query:
    st.write("🔍 Searching knowledge base...")
    results = get_most_relevant_text(query)
    context_texts = "\n\n".join([r[0] for r in results])

    combined_prompt = f"Context:\n{context_texts}\n\nUser Query:\n{query}\n\nProvide a clear, factual, and concise legal explanation with relevant laws."
    response = generate_llm_response(combined_prompt)

    st.markdown("### 🧾 Answer:")
    st.write(response)

    audio_file = text_to_speech(response)
    st.audio(audio_file)

    with st.expander("📚 Retrieved Legal References"):
        for i, (text, score, source) in enumerate(results):
            st.markdown(f"**{i+1}. {source}** — {text} (score: {round(score, 3)})")
