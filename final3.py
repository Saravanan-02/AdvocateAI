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
import io

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

# ==========================
# FALLBACK KNOWLEDGE BASE SETUP
# ==========================
def setup_fallback_knowledge_base():
    """Create a minimal fallback knowledge base if downloads fail"""
    st.warning("📁 Setting up minimal fallback knowledge base...")
    
    # Create minimal metadata
    fallback_metadata = [
        {
            'text': 'The Indian Constitution is the supreme law of India. It establishes the framework defining fundamental political principles.',
            'source': 'Constitution of India'
        },
        {
            'text': 'The Supreme Court of India is the highest judicial court and the final court of appeal under the Constitution of India.',
            'source': 'Indian Judiciary System'
        },
        {
            'text': 'Fundamental Rights are enshrined in Part III of the Indian Constitution from Articles 12 to 35.',
            'source': 'Fundamental Rights'
        },
        {
            'text': 'The Code of Civil Procedure, 1908 governs the procedure to be followed in civil courts in India.',
            'source': 'Civil Procedure Code'
        },
        {
            'text': 'The Indian Penal Code, 1860 is the official criminal code of India which covers all substantive aspects of criminal law.',
            'source': 'Indian Penal Code'
        }
    ]
    
    # Create minimal embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    texts = [item['text'] for item in fallback_metadata]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save files locally for future use
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(fallback_metadata, f)
        st.success("✅ Fallback knowledge base created successfully!")
    except Exception as e:
        st.warning(f"Could not save fallback files: {e}")
    
    return index, fallback_metadata

# ==========================
# LOAD MODELS (CACHED)
# ==========================
@st.cache_resource
def load_models():
    """Load only the AI models (cached)"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return embedding_model, reranker_model

# ==========================
# LOAD KNOWLEDGE BASE (NOT CACHED)
# ==========================
def load_knowledge_base():
    """Load knowledge base files"""
    # Try loading existing local files first
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
            if index.ntotal > 0 and len(metadata) > 0:
                st.success("✅ Loaded existing knowledge base from local files")
                return index, metadata
        except Exception as e:
            st.warning(f"Could not load local files: {e}")
    
    # Create fallback knowledge base
    st.warning("🚨 No knowledge base found. Creating minimal fallback...")
    index, metadata = setup_fallback_knowledge_base()
    
    return index, metadata

# ==========================
# HANDLE FILE UPLOADS
# ==========================
def handle_file_uploads():
    """Handle file uploads for knowledge base"""
    st.info("📤 Upload your knowledge base files (optional)")
    
    col1, col2 = st.columns(2)
    
    uploaded_faiss = None
    uploaded_meta = None
    
    with col1:
        uploaded_faiss = st.file_uploader("Upload FAISS Index (.bin)", type=['bin'], key="faiss_upload")
    
    with col2:
        uploaded_meta = st.file_uploader("Upload Metadata (.pkl)", type=['pkl'], key="meta_upload")
    
    if uploaded_faiss is not None and uploaded_meta is not None:
        if st.button("💾 Save Uploaded Files", use_container_width=True):
            try:
                # Save uploaded files
                with open(INDEX_FILE, "wb") as f:
                    f.write(uploaded_faiss.getvalue())
                
                with open(METADATA_FILE, "wb") as f:
                    f.write(uploaded_meta.getvalue())
                
                st.success("✅ Knowledge base files saved successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving uploaded files: {e}")
    
    return uploaded_faiss, uploaded_meta

# ==========================
# RULE-BASED PROMPT GENERATION
# ==========================
def dynamic_query_generator(question):
    base_prompt = question.strip()
    variations = [base_prompt]

    if base_prompt.lower().startswith("how to"):
        variations.append(f"Steps for {base_prompt[7:]}")
        variations.append(f"Guide to {base_prompt[7:]}")
    elif base_prompt.lower().startswith("what is") or base_prompt.lower().startswith("define"):
        topic = re.sub(r'^(what is|define)\s+', '', base_prompt, flags=re.IGNORECASE)
        variations.append(f"Definition of {topic}")
        variations.append(f"Explanation of {topic}")
    variations.append(f"Explain {base_prompt}")
    variations.append(f"Provide key information on {base_prompt}")
    return list(set(variations))

# ==========================
# LIGHTWEIGHT SUMMARIZATION
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

# ==========================
# LLM-BASED SOURCE SUMMARY
# ==========================
def summarize_source_with_llm(source_text, model="openai/gpt-5-mini"):
    """
    Summarize a given source text in exactly 2 lines using the selected LLM model.
    """
    prompt = f"Summarize the following legal text in exactly 2 lines:\n\n{source_text}"
    try:
        summary, _ = call_openrouter(model, prompt)
        summary = " ".join(summary.strip().splitlines())
        return summary
    except Exception as e:
        print(f"LLM summary error: {e}")
        return create_extractive_summary(source_text, max_sentences=2)

# ==========================
# HYBRID SEARCH
# ==========================
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

def highlight_keywords(text):
    text = re.sub(r'\b([A-Z][a-z]+)\b', r'**\1**', text)  # Names
    text = re.sub(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'**\1**', text)  # Dates
    text = re.sub(r'\b\d+\b', r'**\g<0>**', text)  # Numbers
    return text

# ==================================
# GOOGLE VISION OCR FUNCTION
# ==================================
def ocr_page_with_google_vision(page):
    """
    Performs OCR on a single PDF page using Google Cloud Vision API.
    """
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    b64_image = base64.b64encode(img_bytes).decode('utf-8')

    payload = {
        "requests": [{
            "image": {"content": b64_image},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
        }]
    }

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    headers = {"Content-Type": "application/json; charset=utf-8"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if 'error' in result['responses'][0]:
            st.error(f"Google Vision API Error: {result['responses'][0]['error']['message']}")
            return ""

        if 'fullTextAnnotation' in result['responses'][0]:
            return result['responses'][0]['fullTextAnnotation']['text']
        else:
            return ""  # No text found
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Google Vision API: {e}")
        return ""

# ======================================
# PDF LOADER WITH OCR
# ======================================
def process_uploaded_pdfs(uploaded_files, use_ocr=False):
    all_text = []
    if not uploaded_files:
        return ""
        
    progress_bar = st.progress(0, text="Processing PDFs...")
    total_pages = 0
    docs = []
    for pdf in uploaded_files:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        docs.append({'doc': doc, 'name': pdf.name})
        total_pages += len(doc)
    
    pages_processed = 0
    for item in docs:
        doc = item['doc']
        pdf_name = item['name']
        for i, page in enumerate(doc):
            text = page.get_text()
            # If OCR is enabled and the page has very little text, assume it's an image
            if use_ocr and len(text.strip()) < 50:
                with st.spinner(f"Running OCR on page {i+1} of '{pdf_name}'..."):
                    ocr_text = ocr_page_with_google_vision(page)
                all_text.append(ocr_text)
            else:
                all_text.append(text)
            
            pages_processed += 1
            progress_bar.progress(pages_processed / total_pages, text=f"Processing page {pages_processed}/{total_pages}...")
            
    progress_bar.empty()
    return "\n".join(all_text)

# ==========================
# PROMPT CONSTRUCTION (LIMIT KNOWLEDGE TO 3 LINES)
# ==========================
def construct_final_prompt(question, rag_summary, uploaded_summary):
    rag_lines = rag_summary.strip().splitlines()
    rag_summary = " ".join(rag_lines[:3])  # keep only first 3 lines
    uploaded_summary = truncate_text(uploaded_summary, max_words=200)
    return f"Advocate AI, answer clearly and cite references.\nQ: {question}\nKnowledge: {rag_summary}\nUploaded Docs: {uploaded_summary}"

# ==========================
# OPENROUTER API CALL WITH ERROR HANDLING
# ==========================
def call_openrouter(model, prompt):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used
    except requests.exceptions.HTTPError as e:
        st.error(f"OpenRouter API Error: {e}")
        st.error(f"Response: {response.text}")
        return f"Error: Unable to get response from AI model. Please check your API key and try again.", 0
    except Exception as e:
        st.error(f"Error calling OpenRouter: {e}")
        return f"Error: Unable to connect to AI service. Please try again later.", 0

# ==========================
# TEXT TO SPEECH
# ==========================
def text_to_audio(text):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return open(fp.name, "rb").read()
    except Exception as e:
        print(f"gTTS error: {e}")
        return None

# ==========================
# ENHANCED SEARCH FUNCTION
# ==========================
def search_chat_history(search_query, chat_sessions):
    """
    Enhanced search through all chat sessions with context highlighting.
    """
    results = []
    if not search_query.strip():
        return results
        
    search_terms = search_query.lower().split()
    
    for chat_id, session in chat_sessions.items():
        matches = []
        
        # Search in chat name
        chat_name_lower = session["name"].lower()
        if any(term in chat_name_lower for term in search_terms):
            matches.append({
                "type": "chat_name",
                "content": session["name"],
                "highlighted": highlight_search_terms(session["name"], search_terms)
            })
        
        # Search in messages with context
        for i, message in enumerate(session["messages"]):
            content_lower = message["content"].lower()
            if any(term in content_lower for term in search_terms):
                # Extract context around the match
                highlighted_content = highlight_search_terms(message["content"], search_terms)
                matches.append({
                    "type": "message",
                    "role": message["role"],
                    "message_index": i,
                    "content": message["content"],
                    "highlighted": highlighted_content
                })
        
        if matches:
            results.append({
                "chat_id": chat_id,
                "chat_name": session["name"],
                "matches": matches,
                "total_matches": len(matches)
            })
    
    return results

def highlight_search_terms(text, search_terms):
    """
    Highlight search terms in text with markdown bold.
    """
    highlighted_text = text
    for term in search_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
    return highlighted_text

# ==========================
# STREAMLIT APP
# ==========================
st.set_page_config(page_title="Advocate AI Optimized", layout="wide")

# Add a startup message
st.info("🚀 Loading AI models and knowledge base...")

# Load AI models (cached)
embed_model, reranker_model = load_models()

# Initialize session state
if "knowledge_base_loaded" not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if "folder_index" not in st.session_state:
    st.session_state.folder_index = None
if "folder_metadata" not in st.session_state:
    st.session_state.folder_metadata = None
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

# Load knowledge base (not cached)
if not st.session_state.knowledge_base_loaded:
    with st.spinner("Loading knowledge base..."):
        st.session_state.folder_index, st.session_state.folder_metadata = load_knowledge_base()
        st.session_state.knowledge_base_loaded = True

folder_index = st.session_state.folder_index
folder_metadata = st.session_state.folder_metadata

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    # ==========================
    # KNOWLEDGE BASE STATUS
    # ==========================
    if st.session_state.knowledge_base_loaded:
        if folder_index.ntotal > 5:  # Assuming fallback has 5 items
            st.success(f"✅ Knowledge Base: {folder_index.ntotal} chunks")
        else:
            st.warning(f"⚠️ Fallback KB: {folder_index.ntotal} basic legal concepts")
    else:
        st.error("❌ Knowledge Base: Not loaded")
    
    # ==========================
    # KNOWLEDGE BASE UPLOAD SECTION
    # ==========================
    with st.expander("📁 Upload Knowledge Base", expanded=False):
        handle_file_uploads()
    
    st.markdown("---")
    
    # ==========================
    # CHAT HISTORY
    # ==========================
    st.header("💬 Chat History")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        st.session_state.chat_sessions[chat_id] = {
            "name": "New Chat",
            "messages": [],
            "created_at": datetime.now()
        }
        st.session_state.active_chat = chat_id
        st.session_state.messages = []
        st.rerun()

    # Enhanced Search Bar
    search_query = st.text_input("🔍 Search chat history...", placeholder="Search by keyword, name, date, number...")
    
    # Search Results Section
    if search_query:
        search_results = search_chat_history(search_query, st.session_state.chat_sessions)
        if search_results:
            st.write(f"**Found {len(search_results)} chat(s):**")
            
            for result in search_results:
                with st.container():
                    st.markdown(f"**{result['chat_name']}** ({result['total_matches']} matches)")
                    
                    for match in result['matches'][:3]:
                        if match['type'] == 'chat_name':
                            st.caption("📁 Chat name:")
                        else:
                            role_icon = "👤" if match['role'] == 'user' else "🤖"
                            st.caption(f"{role_icon} Message:")
                        
                        st.markdown(f"`{match['highlighted']}`")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button("Open Chat", key=f"open_{result['chat_id']}", use_container_width=True):
                            st.session_state.active_chat = result['chat_id']
                            st.session_state.messages = st.session_state.chat_sessions[result['chat_id']]["messages"]
                            st.rerun()
                    with col2:
                        with st.popover("⋮"):
                            new_name = st.text_input("Rename chat", 
                                                   value=result['chat_name'], 
                                                   key=f"rename_{result['chat_id']}")
                            if st.button("Save", key=f"save_{result['chat_id']}"):
                                st.session_state.chat_sessions[result['chat_id']]["name"] = new_name
                                st.rerun()
                            
                            if st.button("Delete", key=f"delete_{result['chat_id']}"):
                                del st.session_state.chat_sessions[result['chat_id']]
                                if st.session_state.active_chat == result['chat_id']:
                                    st.session_state.active_chat = None
                                    st.session_state.messages = []
                                st.rerun()
                    
                    st.divider()
        else:
            st.info("No matching chats found. Try different keywords.")
    
    # Regular Chat List
    else:
        st.write("**Your Chats:**")
        
        if not st.session_state.chat_sessions:
            st.info("No chats yet. Start a new conversation!")
        else:
            sorted_chats = sorted(
                st.session_state.chat_sessions.items(),
                key=lambda x: x[1].get('created_at', datetime.min),
                reverse=True
            )
            
            for chat_id, session in sorted_chats:
                is_active = st.session_state.active_chat == chat_id
                
                chat_col1, chat_col2 = st.columns([4, 1])
                
                with chat_col1:
                    button_type = "primary" if is_active else "secondary"
                    if st.button(
                        session["name"], 
                        key=f"chat_btn_{chat_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.active_chat = chat_id
                        st.session_state.messages = session["messages"]
                        st.rerun()
                
                with chat_col2:
                    with st.popover("⋮"):
                        new_name = st.text_input("Rename chat", 
                                               value=session["name"], 
                                               key=f"rename_{chat_id}")
                        if st.button("Save", key=f"save_{chat_id}"):
                            st.session_state.chat_sessions[chat_id]["name"] = new_name
                            st.rerun()
                        
                        if st.button("Share", key=f"share_{chat_id}"):
                            st.info("Share functionality coming soon!")
                        
                        if st.button("Delete", key=f"delete_{chat_id}"):
                            del st.session_state.chat_sessions[chat_id]
                            if st.session_state.active_chat == chat_id:
                                st.session_state.active_chat = None
                                st.session_state.messages = []
                            st.rerun()

    st.markdown("---")
    
    # ==========================
    # MODEL SELECTION
    # ==========================
    
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="
            display: flex; 
            align-items: center; 
            justify-content: center; 
            height: 50px; 
            width: 50px; 
            border-radius: 8px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white; 
            font-size: 24px; 
            margin-right: 12px;
            border: 2px solid #d4af37;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">⚖️</div>
        <div>
            <h3 style="margin: 0; color: #1e3c72; font-weight: bold;">Advocate AI Pro</h3>
            <p style="margin: 0; font-size: 12px; color: #666;">Legal Research Assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model = st.selectbox("Select Model", [
        "openai/gpt-5-mini",  # Put cheaper model first
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "x-ai/grok-code-fast-1",
    ], index=0)
    
    # ==========================
    # FILE UPLOAD & OCR
    # ==========================
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs: 
        st.session_state.uploaded_pdfs_data = uploaded_pdfs
    
    use_ocr_toggle = st.toggle("Enable OCR for scanned PDFs", help="Uses Google Vision to extract text from image-based PDFs. Slower and requires your API key.")

    # ==========================
    # COURTS IN INDIA
    # ==========================
    st.markdown("---")
    with st.expander("⚖️ Courts in India"):
        st.markdown("[Supreme Court of India](https://www.sci.gov.in/)")
        st.markdown("[eCourts Services](https://ecourts.gov.in/)")
        st.markdown("[High Courts (All States)](https://ecommitteesci.gov.in/high-courts/)")
        st.markdown("[District Courts](https://districts.ecourts.gov.in/)")
        st.markdown("[Judgment Search Portal](https://judgments.ecourts.gov.in/)")
        st.markdown("[National Green Tribunal (NGT)](https://greentribunal.gov.in/)")
        st.markdown("[Consumer Disputes (NCDRC)](https://ncdrc.nic.in/)")
        st.markdown("[Central Administrative Tribunal (CAT)](https://cgat.gov.in/)")
        st.markdown("[Debt Recovery Tribunal (DRT)](https://drt.gov.in/)")
        st.markdown("[Income Tax Appellate Tribunal (ITAT)](https://itat.gov.in/)")
        st.markdown("[Armed Forces Tribunal (AFT)](https://aftdelhi.nic.in/)")
        st.markdown("[Election Commission / Tribunals](https://eci.gov.in/)")

# ==========================
# MAIN CHAT INTERFACE
# ==========================
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; border-left: 5px solid #1e3c72;">
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        height: 70px; 
        width: 70px; 
        border-radius: 10px; 
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; 
        font-size: 32px; 
        margin-right: 20px;
        border: 3px solid #d4af37;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    ">⚖️</div>
    <div>
        <h1 style="margin: 0; color: #1e3c72; font-weight: 700; font-size: 2.5rem;">Legal Research Assistant</h1>
        <p style="margin: 5px 0 0 0; color: #495057; font-size: 1.1rem; font-weight: 400;">
            Professional Legal Analysis & Case Research
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Knowledge Base Status Banner
if folder_index and folder_index.ntotal <= 5:
    st.warning("""
    ⚠️ **Using Fallback Knowledge Base** 
    - The app is running with basic legal concepts
    - For full functionality, upload your knowledge base files in the sidebar
    - You can still upload PDFs and ask legal questions
    """)

# Display current chat context
if st.session_state.active_chat and st.session_state.active_chat in st.session_state.chat_sessions:
    current_chat_name = st.session_state.chat_sessions[st.session_state.active_chat]["name"]
    st.subheader(f"💬 {current_chat_name}")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            highlighted_content = highlight_keywords(message["content"])
            st.markdown(highlighted_content)
            
            if message.get("sources"):
                with st.expander("Show Sources"):
                    for source in message["sources"]:
                        llm_summary = summarize_source_with_llm(source["text"], model=model)
                        st.markdown(f"**Source:** `{source['source']}`")
                        st.markdown(f"**Summary:** {llm_summary}")
            
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
    
    if st.session_state.tokens_used > 0:
        st.markdown(f"**Total Tokens Used:** {st.session_state.tokens_used}")

# Chat Input with Auto Chat Creation and Auto Name Update
question = st.chat_input("Enter your legal question...")
if question:
    # Auto-create new chat if none exists
    if st.session_state.active_chat is None:
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        st.session_state.chat_sessions[chat_id] = {
            "name": question[:50] + "..." if len(question) > 50 else question,
            "messages": [],
            "created_at": datetime.now()
        }
        st.session_state.active_chat = chat_id
        st.session_state.messages = []
    else:
        # Auto-update chat name for new chats when first question is asked
        current_chat = st.session_state.chat_sessions[st.session_state.active_chat]
        if current_chat["name"] == "New Chat" and len(current_chat["messages"]) == 0:
            current_chat["name"] = question[:50] + "..." if len(question) > 50 else question
    
    # Add user message to current chat
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Update the chat session with current messages
    if st.session_state.active_chat:
        st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages.copy()
    
    # Generate prompt variations
    refined_prompts = dynamic_query_generator(question)
    st.session_state.pending_prompts = refined_prompts
    st.rerun()

# Prompt Selection Interface
if st.session_state.pending_prompts:
    st.write("### Choose the best prompt:")
    choice = st.radio("Select prompt:", st.session_state.pending_prompts, key="prompt_selector")
    
    if st.button("Confirm and Generate Response"):
        with st.spinner("Generating response..."):
            # Retrieve context and generate response
            rag_context_full, sources = retrieve_and_rerank(choice, folder_index, folder_metadata, embed_model,
                                                            reranker_model, top_k=2)
            uploaded_context_full = ""
            if st.session_state.uploaded_pdfs_data:
                uploaded_context_full = process_uploaded_pdfs(st.session_state.uploaded_pdfs_data, use_ocr=use_ocr_toggle)

            final_prompt = construct_final_prompt(choice, rag_context_full, uploaded_context_full)
            answer, tokens = call_openrouter(model, final_prompt)
            
            # Update tokens and add assistant response
            st.session_state.tokens_used += tokens
            audio_bytes = text_to_audio(answer)
            
            assistant_message = {"role": "assistant", "content": answer, "sources": sources}
            if audio_bytes:
                assistant_message["audio"] = audio_bytes
            
            st.session_state.messages.append(assistant_message)
            
            # Update chat session
            if st.session_state.active_chat:
                st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages.copy()
            
            st.session_state.pending_prompts = None
            st.rerun()
