import os
import streamlit as st
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
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
from io import BytesIO

# --- PDF Dependencies ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer 
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch

# --- WORD (.docx) Dependencies ---
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_PARAGRAPH_ALIGN
except ImportError:
    st.error("The 'python-docx' library is not installed. Please run 'pip install python-docx' to enable Word export.")
    if "Document" not in globals():
        st.stop()

# --- PDF Text Extraction Dependencies ---
try:
    import PyPDF2
    from PyPDF2.errors import PdfReadError
except ImportError:
    st.error("The 'PyPDF2' library is not installed. Please run 'pip install PyPDF2' to enable PDF processing.")
    if "PyPDF2" not in globals():
        st.stop()


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
# CONFIG & API KEY
# ==========================

# --- IMPROVED API KEY HANDLING ---
OPENROUTER_API_KEY = None
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
except:
    pass

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

BASE_URL = "https://openrouter.ai/api/v1"
INDEX_FILE = "faiss_advanced_index.bin"
METADATA_FILE = "metadata.pkl"

# Stop the app if the key is not available
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it in your Streamlit secrets or environment variables.")
    st.info("To set it in Streamlit Community Cloud, go to 'Settings' > 'Secrets' and add `OPENROUTER_API_KEY = 'your_key_here'`")
    st.stop()

# ==========================
# JAVASCRIPT FOR NATIVE TTS
# ==========================
def inject_tts_javascript():
    js_code = """
    function speakText(text) {
        if ('speechSynthesis' in window) {
            let cleanText = text.replace(/<[^>]*>/g, '');
            cleanText = cleanText.replace(/\\*\\*(.*?)\\*\\*/g, '$1');
            cleanText = cleanText.replace(/\\n/g, ' ');
            if (window.speechSynthesis.speaking) {
                window.speechSynthesis.cancel();
            }
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.rate = 1.0; 
            utterance.pitch = 1.0; 
            utterance.volume = 1.0;
            utterance.lang = 'en-US'; 
            window.speechSynthesis.speak(utterance);
        } else {
            alert("Speech Synthesis not supported in this browser.");
        }
    }
    """
    return f"<script>{js_code}</script>"

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
# HELPER FUNCTIONS 
# ==========================
def create_extractive_summary(text, max_sentences=3):
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

def truncate_text(text, max_words=500):
    words = text.split()
    return " ".join(words[:max_words])

def summarize_source_with_llm(source_text, model="openai/gpt-5-mini"):
    return create_extractive_summary(source_text, max_sentences=3) 

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
    if not tokenized_corpus:
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = preprocess_text(query)
    if not tokenized_query:
        return []
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        results.append({
            'source': metadata[idx].get('source', 'BM25 Result'),
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

def dynamic_query_generator(question):
    """Generates three distinct prompt variations for legal research."""
    base_prompt = question.strip()
    q_lower = base_prompt.lower()
    variations = [base_prompt]
    topic = re.sub(r'^(tell me about|what is|define|explain|how to)\s+', '', q_lower, flags=re.IGNORECASE).strip()
    if q_lower.startswith("how to"):
        refined_prompt_2 = f"What are the steps, legal guidelines, and mandatory procedure required for {topic}?"
    elif topic and len(topic) < len(q_lower):
        refined_prompt_2 = f"Provide a complete definition, key elements, and relevant section references for the legal concept: {topic}."
    else:
        refined_prompt_2 = f"Conduct a formal structural analysis of the legal principle or section: {base_prompt}."
    if re.search(r'ipc|act|section|law', q_lower):
        refined_prompt_3 = f"Cite recent Supreme Court case law, mandatory punishment/penalty, and any recent amendments related to: {base_prompt}."
    else:
        refined_prompt_3 = f"Compare {base_prompt} with related legal concepts and explain its application in a modern context, referencing relevant precedents."
    if refined_prompt_2 not in variations: variations.append(refined_prompt_2)
    if refined_prompt_3 not in variations and len(variations) < 3: variations.append(refined_prompt_3)
    if len(variations) < 3:
        variations.append(f"What are the common challenges or defenses related to {base_prompt}?")
    
    return variations[:3]

@st.cache_data(ttl=600)
def extract_text_from_pdfs(uploaded_files):
    if not uploaded_files:
        return ""
    all_text = []
    with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
        for file in uploaded_files:
            try:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
            except PdfReadError:
                st.warning(f"Could not read '{file.name}'. The file might be corrupt or encrypted.")
            except Exception as e:
                st.error(f"An error occurred while processing '{file.name}': {e}")
            file.seek(0)
    return "\n\n".join(all_text)

# ==========================
# PROMPT CONSTRUCTION
# ==========================
def construct_final_prompt(question, rag_summary, uploaded_context_summary):
    rag_lines = rag_summary.strip().splitlines()
    if not rag_lines:
        knowledge_block = "No relevant knowledge found in the database."
    else:
        knowledge_block = "\n".join(rag_lines[:3]) 
        
    if not uploaded_context_summary:
        uploaded_block = "No context provided from uploaded documents."
    else:
        uploaded_block = truncate_text(uploaded_context_summary, max_words=1500)

    final_prompt = f"""
    Advocate AI: Answer the following question clearly, concisely, and formally.
    
    **Instructions:**
    1. Provide a professional, structured legal response based on the **Selected Question** below.
    2. **Crucially, maintain clear line breaks and use markdown (like bullet points or list numbering) for structure.**
    3. Do NOT include any initial greetings, intros, or headings. Start directly with the answer content.
    4. You have two sources of information. Prioritize context from 'Uploaded Docs' if it is relevant.
    5. Cite references clearly, e.g., (Source: SourceFileName.pdf) or (Source: Uploaded Document).
    
    **Selected Question:** {question}
    
    ---
    **Knowledge (From Database):**
    {knowledge_block}
    
    ---
    **Knowledge (From Uploaded Docs):**
    {uploaded_block}
    """
    return final_prompt.strip()

# ==========================
# LLM API CALL
# ==========================
def get_cheating_draft_placeholder(sources):
    source_citations = ""
    if sources:
        source_citations = f"The intent to deceive (Source: {sources[0]['source']}) and the subsequent delivery of property (Source: {sources[0]['source']}) must be clearly established."
    return f"""
**CRITICAL AUTHENTICATION ERROR (401)**. Please ensure your `OPENROUTER_API_KEY` is set correctly.
---
### Placeholder Draft: Cheating (IPC Section 420)
This is a structural draft for a Criminal Complaint (First Information Report/Private Complaint) for the offense of Cheating and Dishonest Inducement to Deliver Property, usually covered under **Section 420, Indian Penal Code (IPC)**.
... (rest of placeholder) ...
{source_citations} Always ensure all oral representations, documents, and money transfer records are attached as evidence (Annexures).
"""

def call_openrouter(model, prompt, sources=None):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(f"https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("Authentication Failed (401). Please check the OPENROUTER_API_KEY.")
            if "cheating case" in prompt.lower() and sources is not None:
                return get_cheating_draft_placeholder(sources), 100
            return f"**LLM Error (401 Unauthorized):** Cannot connect. Please fix your API Key. {e}", 0
        st.error(f"HTTP Error connecting to LLM: {e}")
        return f"Error connecting to LLM: {e}", 0
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return "Error: Request timed out.", 0
    except Exception as e:
        st.error(f"Error connecting to LLM: {e}")
        return f"Error connecting to LLM: {e}", 0

def generate_suggested_prompts(last_question, last_answer):
    """Generates 3 follow-up prompts."""
    suggestion_prompt = f"""
    Based on the last legal question and its answer, generate 3 brief, distinct, and relevant follow-up questions a legal professional might ask next.
    
    **Rules:**
    1. Return ONLY the 3 questions.
    2. Separate each question with a newline.
    3. Do NOT number them or use bullet points.
    4. The questions should be specific and insightful.
    
    **Last Question:**
    {last_question}
    
    **Last Answer:**
    {truncate_text(last_answer, 300)}
    """
    answer, tokens = call_openrouter("openai/gpt-5-mini", suggestion_prompt)
    if "Error" in answer:
        return []
    prompts = [
        p.strip() for p in answer.split('\n') 
        if p.strip() and p.strip().endswith('?')
    ]
    return prompts[:3]


# =========================================
# KEYWORD HIGHLIGHTING
# =========================================
def highlight_with_style(text, search_terms_raw):
    """
    Highlights terms and returns text, plus a boolean if anything was found.
    """
    found_keyword = False
    if not search_terms_raw:
        return text, found_keyword
    search_terms = [re.escape(term.strip()) for term in search_terms_raw.split(',') if term.strip()]
    if not search_terms:
        return text, found_keyword
        
    pattern = re.compile(r'(' + '|'.join(search_terms) + r')', re.IGNORECASE)
    
    highlight_style = 'background-color: #ADD8E6; color: black; border-radius: 3px; padding: 2px 4px; display: inline; font-weight: bold;'
    
    def replacer(match):
        nonlocal found_keyword
        found_keyword = True
        return f'<span style="{highlight_style}">{match.group(0)}</span>'
        
    highlighted_text = pattern.sub(replacer, text)
    return highlighted_text, found_keyword

def check_keywords_in_chat(messages, keyword_terms):
    """
    Loops through chat history to check if keywords exist.
    This runs *before* the sidebar is rendered to prevent the "Not Found" bug.
    """
    if not keyword_terms:
        return True
    
    for message in messages:
        content = message["content"]
        for term in keyword_terms.split(','):
            if term.strip() and term.strip().lower() in content.lower():
                return True
    return False


# ===================================================================
# DOWNLOAD CHAT CONTENT (PDF & WORD)
# ===================================================================

def clean_markdown_and_linebreaks(text):
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text, flags=re.DOTALL)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)
    text = re.sub(r'^[\s]*[\*-]\s+', '&bull; ', text, flags=re.MULTILINE)
    text = re.sub(r'\n\n', '<br/><br/>', text)
    text = re.sub(r'\n', '<br/>', text)
    return text.strip()

def clean_text_for_docx(text):
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.*?)\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)
    text = re.sub(r'^[\s]*[\*-]\s+', '\u2022 ', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

def generate_qna_content_pdf(messages, chat_name):
    buffer = BytesIO()
    margin = 0.75 * inch 
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    story = []
    line_spacing = 1.5
    font_size_main = 15
    font_size_source = 12
    styles.add(ParagraphStyle(name='Question', fontName='Times-Bold', fontSize=font_size_main, leading=font_size_main * line_spacing, spaceAfter=0, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='AnswerHTML', fontName='Times-Roman', fontSize=font_size_main, leading=font_size_main * line_spacing, spaceAfter=8, leftIndent=10))
    styles.add(ParagraphStyle(name='SourceHeader', fontName='Times-Bold', fontSize=font_size_source, leading=font_size_source * line_spacing, spaceBefore=5, spaceAfter=2, leftIndent=10))
    styles.add(ParagraphStyle(name='SourceItem', fontName='Times-Roman', fontSize=font_size_source, leading=font_size_source * line_spacing, spaceAfter=2, leftIndent=20))
    styles.add(ParagraphStyle(name='ChatTitle', fontName='Times-Bold', fontSize=18, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='SubTitle', fontName='Times-Bold', fontSize=15, spaceAfter=5, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Date', fontName='Times-Roman', fontSize=12, spaceAfter=20, alignment=TA_LEFT))

    story.append(Paragraph("ADVOCATE AI CHAT SESSION", styles['ChatTitle']))
    story.append(Paragraph(f"Topic: {chat_name}", styles['SubTitle']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Date']))
    story.append(Spacer(1, 0.25 * inch))

    q_count = 1
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                answer_message = messages[i+1]
                question_text = f"Q{q_count}: {clean_markdown_and_linebreaks(message['content'])}"
                story.append(Paragraph(question_text, styles['Question']))
                story.append(Spacer(1, 15))
                html_answer = clean_markdown_and_linebreaks(answer_message['content'])
                story.append(Paragraph(html_answer, styles['AnswerHTML']))
                if answer_message.get("sources"):
                    story.append(Paragraph("--- Sources Cited (from Database) ---", styles['SourceHeader']))
                    for j, source in enumerate(answer_message["sources"]):
                        source_summary = summarize_source_with_llm(source['text']) 
                        source_line = (f"&bull; <b>Source {j+1}:</b> {source['source']}<br/>"
                                       f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Snippet:</i> {source_summary}")
                        story.append(Paragraph(source_line, styles['SourceItem']))
                story.append(Spacer(1, 0.2 * inch))
                q_count += 1
    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return b""

def generate_qna_content_word(messages, chat_name):
    document = Document()
    style = document.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(15)
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing = 1.5
    title = document.add_heading("ADVOCATE AI CHAT SESSION", level=1)
    title.runs[0].font.name = 'Times New Roman'
    topic = document.add_paragraph()
    topic.add_run(f"Topic: {chat_name}").bold = True
    document.add_paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").runs[0].font.size = Pt(12)
    document.add_paragraph()
    q_count = 1
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                answer_message = messages[i+1]
                q_para = document.add_paragraph()
                q_run = q_para.add_run(f"Q{q_count}: {clean_text_for_docx(message['content'])}")
                q_run.bold = True
                document.add_paragraph()
                a_para = document.add_paragraph(clean_text_for_docx(answer_message['content']))
                a_para.paragraph_format.left_indent = Inches(0.25)
                if answer_message.get("sources"):
                    s_header_para = document.add_paragraph()
                    s_header_para.paragraph_format.left_indent = Inches(0.25)
                    s_run = s_header_para.add_run("--- Sources Cited (from Database) ---")
                    s_run.font.size = Pt(12)
                    s_run.italic = True
                    for j, source in enumerate(answer_message["sources"]):
                        source_summary = summarize_source_with_llm(source['text'])
                        source_text = f"\u2022 Source {j+1}: {source['source']}\n\tSnippet: {source_summary}"
                        src_para = document.add_paragraph(source_text)
                        src_para.paragraph_format.left_indent = Inches(0.5)
                        for run in src_para.runs:
                            run.font.size = Pt(12)
                document.add_paragraph()
                q_count += 1
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()

# ==========================
# STREAMLIT APP SETUP
# ==========================
st.set_page_config(page_title="Advocate AI Optimized", layout="wide")
st.markdown(inject_tts_javascript(), unsafe_allow_html=True) 
embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()

if folder_index is None:
    st.error("Knowledge base not found. Run build_index_advanced.py.")
    st.stop()

# Initialize session state 
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_pdfs_data" not in st.session_state: st.session_state.uploaded_pdfs_data = None
if "tokens_used" not in st.session_state: st.session_state.tokens_used = 0
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {}
if "active_chat" not in st.session_state: st.session_state.active_chat = None
if "selected_model" not in st.session_state: st.session_state.selected_model = "openai/gpt-5"
if "keyword_found" not in st.session_state: st.session_state.keyword_found = True
if "suggested_prompts" not in st.session_state: st.session_state.suggested_prompts = []
if "pending_prompts" not in st.session_state: st.session_state.pending_prompts = []


# Inject Custom CSS
custom_css = f"""
<style>
/* Base UI Cleanup */
.main {{ padding-top: 20px; }}
/* Sidebar Chat Button Styling */
.stButton>button {{
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
    margin-top: 5px;
    height: 40px;
    border: 1px solid #ccc;
}}
.stButton>button:hover {{ 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
    border: 1px solid #1e3c72;
}}

/* Primary button for Active Chat Session */
.stButton[data-testid*="primary"]>button {{
    background-color: #1e3c72;
    color: white;
    border: 1px solid #1e3c72;
}}
/* Secondary button for inactive chats and suggestions */
.stButton[data-testid*="secondary"]>button {{
    background-color: white;
    color: #333;
    border: 1px solid #ccc;
}}
/* Chat Message Bubbles (User/Assistant differentiation) */
div[data-testid="chat-message-container"] {{
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}}
/* User Message Bubble */
div[data-testid="chat-message-container"]:has(div.st-emotion-cache-1c7v0l):nth-child(even) {{
    background-color: #e6f0ff !important;
    border-left: 5px solid #1e3c72;
}}
/* Assistant Message Bubble */
div[data-testid="chat-message-container"]:has(div.st-emotion-cache-1c7v0l):nth-child(odd) {{
    background-color: #f0f2f6 !important;
    border-right: 5px solid #d4af37;
}}

/* TTS Button Styling */
.stButton button[kind="secondary"] {{
    background-color: #f9f9f9;
    color: #1e3c72;
    border: 1px solid #ccc;
    font-size: 10px;
    padding: 2px 5px;
    height: 25px;
    margin-bottom: 10px;
}}

/* Suggestion/Pending Button Styling */
div[data-testid="stVerticalBlock"] .stButton>button {{
    height: 80px;
    white-space: normal;
    text-align: left;
    background-color: #f0f2f6;
    border: 1px solid #ccc;
    margin-bottom: 5px;
}}
div[data-testid="stVerticalBlock"] .stButton>button:hover {{
    background-color: #e6f0ff;
    border-color: #1e3c72;
}}
/* Primary suggestion button */
div[data-testid="stVerticalBlock"] .stButton>button[data-testid*="primary"] {{
    background-color: #e6f0ff;
    border-color: #1e3c72;
    color: #1e3c72;
    font-weight: normal;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- KEYWORD BUG FIX: Run check *before* sidebar is rendered ---
if "keyword_highlight_input" in st.session_state:
    st.session_state.keyword_found = check_keywords_in_chat(
        st.session_state.messages, 
        st.session_state.keyword_highlight_input
    )


# ==========================
# SIDEBAR 
# ==========================
with st.sidebar:
    
    st.header("🔑 Keyword Highlight")
    keyword_search_input = st.text_input("Enter keywords (comma-separated)", key="keyword_highlight_input", placeholder="e.g., defense, doctrine, contract")
    
    st.markdown("---") 

    if not st.session_state.keyword_found and st.session_state.keyword_highlight_input:
        st.warning("⚠️ Keywords not found.", icon="🔍")
    elif st.session_state.keyword_highlight_input and st.session_state.messages:
        st.success("Keywords found.", icon="✨")

    st.markdown("---") 

    st.header("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        st.session_state.chat_sessions[chat_id] = {"name": "New Chat", "messages": [], "created_at": datetime.now()}
        st.session_state.active_chat = chat_id
        st.session_state.messages = []
        st.session_state.keyword_found = True
        st.session_state.suggested_prompts = []
        st.session_state.pending_prompts = []
        st.rerun()

    st.write("**Your Chats:**")
    if not st.session_state.chat_sessions:
        st.info("No chats yet.")
    else:
        sorted_chats = sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1].get('created_at', datetime.min), reverse=True)
        for chat_id, session in sorted_chats:
            is_active = st.session_state.active_chat == chat_id
            chat_col1, chat_col2 = st.columns([4, 1])
            with chat_col1:
                button_type = "primary" if is_active else "secondary"
                if st.button(session["name"], key=f"chat_btn_{chat_id}", use_container_width=True, type=button_type):
                    st.session_state.active_chat = chat_id
                    st.session_state.messages = session["messages"]
                    st.session_state.keyword_found = True
                    st.session_state.suggested_prompts = []
                    st.session_state.pending_prompts = []
                    st.rerun()
            with chat_col2:
                with st.popover("⋮"):
                    new_name = st.text_input("Rename chat", value=session["name"], key=f"rename_{chat_id}")
                    if st.button("Save", key=f"save_{chat_id}"):
                        st.session_state.chat_sessions[chat_id]["name"] = new_name
                        st.rerun()
                    if st.button("Delete", key=f"delete_{chat_id}"):
                        del st.session_state.chat_sessions[chat_id]
                        if st.session_state.active_chat == chat_id:
                            st.session_state.active_chat = None
                            st.session_state.messages = []
                            st.session_state.keyword_found = True
                            st.session_state.suggested_prompts = []
                            st.session_state.pending_prompts = []
                        st.rerun()
    st.markdown("---")

    # --- DOWNLOAD & CLEAR HISTORY ---
    pdf_content_bytes = b''
    word_content_bytes = b''
    pdf_file_name = "AdvocateAI_Session.pdf"
    word_file_name = "AdvocateAI_Session.docx"
    current_messages = []
    current_chat_name = "Current_Unsaved_Chat"

    if st.session_state.active_chat and st.session_state.active_chat in st.session_state.chat_sessions:
        current_chat = st.session_state.chat_sessions[st.session_state.active_chat]
        current_chat_name = current_chat["name"]
        current_messages = current_chat["messages"]
    elif st.session_state.messages:
        current_messages = st.session_state.messages

    if current_messages:
        base_file_name = f"AdvocateAI_{current_chat_name.replace(' ', '_')[:20]}"
        pdf_file_name = f"{base_file_name}.pdf"
        word_file_name = f"{base_file_name}.docx"
        pdf_content_bytes = generate_qna_content_pdf(current_messages, current_chat_name)
        word_content_bytes = generate_qna_content_word(current_messages, current_chat_name)

    st.download_button(label="⬇️ Download Q&A as PDF", data=pdf_content_bytes, file_name=pdf_file_name, mime="application/pdf", use_container_width=True, disabled=(not pdf_content_bytes))
    st.download_button(label="⬇️ Download Q&Details as Word", data=word_content_bytes, file_name=word_file_name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, disabled=(not word_content_bytes))
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_sessions = {}
        st.session_state.active_chat = None
        st.session_state.messages = []
        st.session_state.keyword_found = True
        st.session_state.suggested_prompts = []
        st.session_state.pending_prompts = []
        st.rerun()
    st.markdown("---")
    
    # --- MODEL & UPLOAD SETTINGS ---
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
        "openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro",
        "x-ai/grok-code-fast-1", "google/gemini-2.5-flash", 
        "google/gemini-2.5-flash-lite", "openai/gpt-5-mini", "tngtech/deepseek-r1t2-chimera:free"
    ], index=0)
    
    st.success(f"Knowledge Base loaded with {folder_index.ntotal} chunks", icon="✅")
    
    uploaded_pdfs = st.file_uploader("Upload PDFs to use as context", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        st.session_state.uploaded_pdfs_data = uploaded_pdfs
        st.info(f"{len(uploaded_pdfs)} PDF(s) loaded. They will be used as context.")
    elif st.session_state.uploaded_pdfs_data:
        st.session_state.uploaded_pdfs_data = None
        
    st.markdown("---")

# ==========================
# MAIN CHAT INTERFACE
# ==========================
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; border-left: 5px solid #d4af37;">
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        height: 70px; 
        width: 70px; 
        border-radius: 10px; 
        background: white; 
        color: #1e3c72; 
        font-size: 32px; 
        margin-right: 20px;
        border: 3px solid #d4af37;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    ">⚖️</div>
    <div>
        <h1 style="margin: 0; color: white; font-weight: 700; font-size: 2.5rem;">Legal Research Assistant</h1>
        <p style="margin: 5px 0 0 0; color: #f0f2f6; font-size: 1.1rem; font-weight: 400;">
            Professional Legal Analysis & Case Research
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.active_chat and st.session_state.active_chat in st.session_state.chat_sessions:
    current_chat_name = st.session_state.chat_sessions[st.session_state.active_chat]["name"]
    st.subheader(f"💬 {current_chat_name}")

keyword_terms = st.session_state.keyword_highlight_input

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            
            split_point = "Writ Petition under Article 226 of the Constitution of India"
            if content.count(split_point) > 1:
                content = content.split(split_point, 1)[0] + split_point + content.split(split_point, 2)[-1]
            
            if message["role"] == "assistant":
                js_safe_content = content.replace("'", "\\'").replace('\n', ' ')
                st.markdown(
                    f'<button onclick="speakText(\'{js_safe_content}\')" class="stButton secondary">🔊 Read Aloud</button>',
                    unsafe_allow_html=True
                )

            # --- RENDER HIGHLIGHTS ---
            if keyword_terms:
                highlighted_content, _ = highlight_with_style(content, keyword_terms)
                st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.markdown(content)
            
            if message.get("sources"):
                with st.expander("Show Sources (from Database)"):
                    for source in message["sources"]:
                        llm_summary = summarize_source_with_llm(source["text"], model=st.session_state.selected_model) 
                        st.markdown(f"**Source:** `{source['source']}`")
                        st.markdown(f"**Snippet (3 Lines):** {llm_summary}")
    
    if st.session_state.tokens_used > 0:
        st.markdown(f"---")
        st.caption(f"Total tokens used in this session: **{st.session_state.tokens_used}**")

# ==================================
# NEW PROMPT WORKFLOW FUNCTIONS
# ==================================

def generate_prompt_variations(prompt):
    """
    Called by chat_input. Generates prompts and stores them in 'pending_prompts'.
    """
    st.session_state.suggested_prompts = []
    
    if not st.session_state.active_chat:
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        new_chat_name = prompt[:30].strip() + "..." if len(prompt) > 30 else prompt.strip()
        st.session_state.chat_sessions[chat_id] = {"name": new_chat_name, "messages": [], "created_at": datetime.now()}
        st.session_state.active_chat = chat_id
        st.session_state.messages = st.session_state.chat_sessions[chat_id]["messages"]
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Generating query variations..."):
        variations = dynamic_query_generator(prompt)
        st.session_state.pending_prompts = variations
        
    if st.session_state.active_chat:
        st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages


def handle_selected_prompt(selected_prompt):
    """
    Called by button click. Clears pending prompts and runs the full RAG pipeline.
    """
    st.session_state.pending_prompts = []

    with st.spinner("⚖️ Conducting research..."):
        
        # RAG from Database
        all_sources = []
        try:
            chunks, sources = retrieve_and_rerank(selected_prompt, folder_index, folder_metadata, embed_model, reranker_model, top_k=3)
            all_sources.extend(sources)
        except Exception as e:
            st.warning(f"Database retrieval error: {e}")

        unique_sources = {}
        for source in all_sources:
            key = source['source'] + source['text'][:100]
            if key not in unique_sources:
                unique_sources[key] = source
        final_rag_context = "\n\n".join([truncate_text(s['text'], max_words=200) for s in unique_sources.values()])
        
        # RAG from Uploaded PDFs
        uploaded_context_summary = ""
        if st.session_state.uploaded_pdfs_data:
            full_pdf_text = extract_text_from_pdfs(st.session_state.uploaded_pdfs_data)
            if full_pdf_text:
                pdf_chunks = full_pdf_text.split('\n\n')
                pdf_metadata = [{'text': chunk, 'source': 'Uploaded Document'} for chunk in pdf_chunks if chunk.strip()]
                if pdf_metadata:
                    pdf_search_results = bm25_search(selected_prompt, pdf_metadata, top_k=5)
                    uploaded_context_summary = "\n\n".join([res['text'] for res in pdf_search_results])

        # Construct Final Prompt
        final_prompt = construct_final_prompt(selected_prompt, final_rag_context, uploaded_context_summary)
        
        # LLM Call for Answer
        llm_answer, tokens_used = call_openrouter(st.session_state.selected_model, final_prompt, list(unique_sources.values()))
        st.session_state.tokens_used += tokens_used

        # LLM Call for Follow-up Suggestions
        try:
            suggested_prompts = generate_suggested_prompts(selected_prompt, llm_answer)
            st.session_state.suggested_prompts = suggested_prompts
        except Exception as e:
            st.warning(f"Could not generate suggestions: {e}")

        # Add assistant message to state
        assistant_message = {
            "role": "assistant",
            "content": llm_answer,
            "sources": list(unique_sources.values())
        }
        st.session_state.messages.append(assistant_message)
        
        if st.session_state.active_chat:
            st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages
    
    # CRITICAL FIX: Added st.rerun()
    st.rerun()

# ==================================
# MAIN CHAT INPUT & SUGGESTION UI
# ==================================

# --- NEW: Conditional UI for prompts ---
if st.session_state.pending_prompts:
    st.caption("Select a prompt to continue:")
    
    labels = ["Original Question", "Refined Prompt", "Refined Prompt"]
    
    for i, prompt_text in enumerate(st.session_state.pending_prompts):
        button_label = f"**{labels[i]}**\n\n{prompt_text}"
        st.button(
            button_label, 
            key=f"pending_{i}",
            on_click=handle_selected_prompt,
            args=(prompt_text,),
            use_container_width=True,
            type="primary"
        )
elif st.session_state.suggested_prompts:
    st.caption("Suggested Follow-ups:")
    
    for i, prompt_text in enumerate(st.session_state.suggested_prompts):
        st.button(
            prompt_text, 
            key=f"suggestion_{i}",
            on_click=generate_prompt_variations,
            args=(prompt_text,),
            use_container_width=True
        )

# --- Streamlit Chat Input (calls generate_prompt_variations) ---
chat_input_disabled = bool(st.session_state.pending_prompts)
if prompt := st.chat_input("Ask a legal question (e.g., 'What is the doctrine of Res Gestae?')", disabled=chat_input_disabled):
    generate_prompt_variations(prompt)

