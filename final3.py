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
except (KeyError, FileNotFoundError):
    st.error("""
    ⚠️ API Keys not found. Please configure your secrets in Streamlit Cloud:
    
    1. Go to your app settings → Secrets
    2. Add: OPENROUTER_API_KEY = "your-openrouter-key"
    3. Optional: Add GOOGLE_VISION_API_KEY for OCR functionality
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
        return embedding_model, reranker_model, None, None

# ==========================
# FIXED PROMPT GENERATION
# ==========================
def dynamic_query_generator(question):
    """Generate prompt variations for legal research"""
    base_prompt = question.strip()
    variations = [base_prompt]

    # Legal-specific variations
    if any(term in base_prompt.lower() for term in ['writ petition', 'affidavit', 'high court', 'suspended']):
        if 'suspended' in base_prompt.lower() and 'teacher' in base_prompt.lower():
            variations.extend([
                f"Legal grounds for challenging suspension without notice for government teacher {base_prompt}",
                f"Madras High Court writ petition format for teacher suspension case {base_prompt}",
                f"Constitutional provisions and case laws for suspension without notice {base_prompt}"
            ])
    
    # General legal variations
    if len(variations) < 3:
        variations.extend([
            f"Legal analysis of {base_prompt}",
            f"Case law and precedents related to {base_prompt}"
        ])
    
    return variations[:3]  # Return exactly 3 variations

# ==========================
# IMPROVED PROMPT HANDLING FUNCTIONS
# ==========================
def generate_prompt_variations(user_question):
    """Generate variations and store in pending state"""
    st.session_state.pending_prompts = dynamic_query_generator(user_question)
    st.session_state.current_question = user_question

def handle_prompt_selection(selected_prompt):
    """Handle when user selects a prompt variation"""
    st.session_state.selected_prompt = selected_prompt
    st.session_state.pending_prompts = None  # Clear pending state
    
    # Generate response immediately
    generate_response(selected_prompt)

def generate_response(prompt):
    """Generate the actual response using RAG"""
    with st.spinner("⚖️ Conducting legal research..."):
        try:
            # RAG from Database
            rag_context, sources = retrieve_and_rerank(
                prompt, folder_index, folder_metadata, 
                embed_model, reranker_model, top_k=3
            )
            
            # RAG from Uploaded PDFs
            uploaded_context = ""
            if st.session_state.uploaded_pdfs_data:
                uploaded_context = process_uploaded_pdfs(
                    st.session_state.uploaded_pdfs_data, 
                    use_ocr=st.session_state.get('use_ocr', False)
                )

            # Construct final prompt
            final_prompt = construct_final_prompt(prompt, rag_context, uploaded_context)
            
            # LLM Call
            answer, tokens = call_openrouter(st.session_state.selected_model, final_prompt)
            st.session_state.tokens_used += tokens

            # Generate follow-up suggestions
            try:
                suggested_prompts = generate_suggested_prompts(prompt, answer)
                st.session_state.suggested_prompts = suggested_prompts
            except Exception as e:
                st.session_state.suggested_prompts = []

            # Add assistant message
            assistant_message = {
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            }
            st.session_state.messages.append(assistant_message)
            
            # Update chat session
            if st.session_state.active_chat:
                st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages.copy()
                
        except Exception as e:
            st.error(f"Error generating response: {e}")

# ==========================
# IMPROVED PROMPT CONSTRUCTION
# ==========================
def construct_final_prompt(question, rag_summary, uploaded_summary):
    """Construct final prompt with limited context"""
    rag_lines = rag_summary.strip().splitlines()
    if not rag_lines:
        knowledge_block = "No relevant knowledge found in the database."
    else:
        knowledge_block = " ".join(rag_lines[:3])  # Keep only first 3 lines
        
    if not uploaded_summary:
        uploaded_block = "No context provided from uploaded documents."
    else:
        uploaded_block = truncate_text(uploaded_summary, max_words=200)

    return f"""
    Advocate AI: Provide a professional legal response to the following question.
    
    **Question:** {question}
    
    **Database Knowledge:**
    {knowledge_block}
    
    **Uploaded Documents Context:**
    {uploaded_block}
    
    **Instructions:**
    - Provide a structured legal analysis
    - Cite relevant laws and precedents if available
    - Use clear headings and bullet points
    - Focus on practical legal guidance
    """

# ==========================
# IMPROVED API CALL WITH ERROR HANDLING
# ==========================
def call_openrouter(model, prompt):
    """Make API call with proper error handling"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
        "Content-Type": "application/json"
    }
    data = {
        "model": model, 
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
        
        if response.status_code == 401:
            st.error("❌ Authentication failed. Please check your OpenRouter API key in Streamlit secrets.")
            return "Authentication error: Please check your API configuration.", 0
        elif response.status_code == 429:
            st.error("⚠️ Rate limit exceeded. Please try again later.")
            return "Rate limit exceeded. Please try again in a moment.", 0
        elif response.status_code >= 400:
            st.error(f"API Error {response.status_code}: {response.text}")
            return f"API Error: {response.status_code}", 0
            
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used
        
    except requests.exceptions.Timeout:
        st.error("⏰ Request timeout. Please try again.")
        return "Request timeout. Please try again.", 0
    except Exception as e:
        st.error(f"🔌 Connection error: {e}")
        return f"Connection error: {str(e)}", 0

# ==========================
# SUGGESTED PROMPTS GENERATION
# ==========================
def generate_suggested_prompts(question, answer):
    """Generate follow-up prompts based on conversation"""
    prompt = f"""
    Based on this legal Q&A, suggest 3 brief follow-up questions:
    
    Question: {question}
    Answer: {answer[:500]}...
    
    Return ONLY 3 questions, one per line, no numbering.
    """
    
    try:
        suggestions, _ = call_openrouter("openai/gpt-5-mini", prompt)
        prompts = [p.strip() for p in suggestions.split('\n') if p.strip() and p.strip().endswith('?')]
        return prompts[:3]
    except:
        return []

# ==========================
# EXISTING HELPER FUNCTIONS (keep as-is)
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

def summarize_source_with_llm(source_text, model="openai/gpt-5-mini"):
    prompt = f"Summarize the following legal text in exactly 2 lines:\n\n{source_text}"
    try:
        summary, _ = call_openrouter(model, prompt)
        summary = " ".join(summary.strip().splitlines())
        return summary
    except Exception as e:
        return create_extractive_summary(source_text, max_sentences=2)

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

def ocr_page_with_google_vision(page):
    if not GOOGLE_VISION_API_KEY:
        st.warning("Google Vision API key not configured")
        return ""
        
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
            return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Google Vision API: {e}")
        return ""

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
# STREAMLIT APP SETUP
# ==========================
st.set_page_config(page_title="Advocate AI Optimized", layout="wide")
embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()

if folder_index is None:
    st.error("Knowledge base not found. Run build_index_advanced.py")
    st.stop()

# Initialize session state with proper structure
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "pending_prompts" not in st.session_state: 
    st.session_state.pending_prompts = None
if "suggested_prompts" not in st.session_state:
    st.session_state.suggested_prompts = []
if "uploaded_pdfs_data" not in st.session_state: 
    st.session_state.uploaded_pdfs_data = None
if "tokens_used" not in st.session_state: 
    st.session_state.tokens_used = 0
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai/gpt-5"
if "use_ocr" not in st.session_state:
    st.session_state.use_ocr = False

# ==========================
# SIDEBAR 
# ==========================
with st.sidebar:
    st.header("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        st.session_state.chat_sessions[chat_id] = {
            "name": "New Chat",
            "messages": [],
            "created_at": datetime.now()
        }
        st.session_state.active_chat = chat_id
        st.session_state.messages = []
        st.session_state.pending_prompts = None
        st.session_state.suggested_prompts = []
        st.rerun()

    # Chat list
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
                    st.session_state.pending_prompts = None
                    st.session_state.suggested_prompts = []
                    st.rerun()
            
            with chat_col2:
                with st.popover("⋮"):
                    new_name = st.text_input("Rename chat", 
                                           value=session["name"], 
                                           key=f"rename_{chat_id}")
                    if st.button("Save", key=f"save_{chat_id}"):
                        st.session_state.chat_sessions[chat_id]["name"] = new_name
                        st.rerun()
                    
                    if st.button("Delete", key=f"delete_{chat_id}"):
                        del st.session_state.chat_sessions[chat_id]
                        if st.session_state.active_chat == chat_id:
                            st.session_state.active_chat = None
                            st.session_state.messages = []
                        st.rerun()

    st.markdown("---")
    
    # Model selection
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
    
    st.session_state.selected_model = st.selectbox("Select Model", [
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "x-ai/grok-code-fast-1",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "openai/gpt-5-mini"
    ], index=0)
    
    st.success(f"Knowledge Base loaded with {folder_index.ntotal} chunks", icon="✅")
    
    # File upload
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs: 
        st.session_state.uploaded_pdfs_data = uploaded_pdfs
    
    # OCR toggle
    st.session_state.use_ocr = st.toggle("Enable OCR for scanned PDFs", 
                                        disabled=not GOOGLE_VISION_API_KEY,
                                        help="Uses Google Vision to extract text from image-based PDFs." if GOOGLE_VISION_API_KEY else "OCR disabled - Google Vision API Key not configured")

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

# Display current chat context
if st.session_state.active_chat and st.session_state.active_chat in st.session_state.chat_sessions:
    current_chat_name = st.session_state.chat_sessions[st.session_state.active_chat]["name"]
    st.subheader(f"💬 {current_chat_name}")

# Chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            highlighted_content = highlight_keywords(message["content"])
            st.markdown(highlighted_content)
            
            if message.get("sources"):
                with st.expander("Show Sources"):
                    for source in message["sources"]:
                        llm_summary = summarize_source_with_llm(source["text"], model=st.session_state.selected_model)
                        st.markdown(f"**Source:** `{source['source']}`")
                        st.markdown(f"**Summary:** {llm_summary}")
            
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
    
    if st.session_state.tokens_used > 0:
        st.markdown(f"**Total Tokens Used:** {st.session_state.tokens_used}")

# ==========================
# FIXED PROMPT SELECTION UI
# ==========================
if st.session_state.pending_prompts:
    st.write("### Choose a prompt variation:")
    
    # Display prompt variations as buttons
    for i, prompt in enumerate(st.session_state.pending_prompts):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**Option {i+1}:** {prompt}")
        with col2:
            if st.button("Select", key=f"select_{i}"):
                handle_prompt_selection(prompt)
                st.rerun()

# ==========================
# SUGGESTED PROMPTS UI
# ==========================
if st.session_state.suggested_prompts and not st.session_state.pending_prompts:
    st.write("### Suggested follow-up questions:")
    
    for i, prompt in enumerate(st.session_state.suggested_prompts):
        if st.button(prompt, key=f"suggested_{i}", use_container_width=True):
            generate_prompt_variations(prompt)
            st.rerun()

# ==========================
# CHAT INPUT
# ==========================
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
        # Auto-update chat name for new chats
        current_chat = st.session_state.chat_sessions[st.session_state.active_chat]
        if current_chat["name"] == "New Chat" and len(current_chat["messages"]) == 0:
            current_chat["name"] = question[:50] + "..." if len(question) > 50 else question
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Update chat session
    if st.session_state.active_chat:
        st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages.copy()
    
    # Generate prompt variations
    generate_prompt_variations(question)
    st.rerun()
