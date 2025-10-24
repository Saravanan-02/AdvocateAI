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
from io import BytesIO

# --- PDF Dependencies (FROM NEW CODE) ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch

# --- WORD (.docx) Dependencies (FROM NEW CODE) ---
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_PARAGRAPH_ALIGN
except ImportError:
    st.error("The 'python-docx' library is not installed. Please run 'pip install python-docx' to enable Word export.")
    if "Document" not in globals():
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
# CONFIG
# ==========================
OPENROUTER_API_KEY = "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"
GOOGLE_VISION_API_KEY = "AIzaSyBFh_YqGdkvUjQPT6ihyur2mlvETJcOF_k"
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
# LIGHTWEIGHT SUMMARIZATION (REPLACED WITH V2's 3-LINE VERSION)
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

def truncate_text(text, max_words=200):
    words = text.split()
    return " ".join(words[:max_words])

# ==========================
# LLM-BASED SOURCE SUMMARY (REPLACED WITH V2's FAST EXTRACTIVE VERSION)
# ==========================
def summarize_source_with_llm(source_text, model=None): # Model arg kept for compatibility but unused
    """
    Summarize a given source text using a fast 3-line extractive summary.
    """
    return create_extractive_summary(source_text, max_sentences=3)

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

# ==================================
# KEYWORD HIGHLIGHTING (REPLACED WITH V2's FUNCTION)
# ==================================
def highlight_with_style(text, search_terms_raw):
    """
    Highlights search terms regardless of case in all paragraphs.
    Returns the highlighted text and a boolean indicating if any keyword was found.
    """
    found_keyword = False
    if not search_terms_raw:
        return text, found_keyword

    search_terms = [re.escape(term.strip()) for term in search_terms_raw.split(',') if term.strip()]
    if not search_terms:
        return text, found_keyword

    # Create a single pattern to match all terms, case-insensitively, non-word boundary
    pattern = re.compile(r'(' + '|'.join(search_terms) + r')', re.IGNORECASE)
    highlight_style = 'background-color: #ffff00; color: black; border-radius: 3px; padding: 2px 4px; display: inline; font-weight: bold;'

    def replacer(match):
        nonlocal found_keyword
        found_keyword = True
        # match.group(0) is the actual text found (preserving its original casing)
        return f'<span style="{highlight_style}">{match.group(0)}</span>'

    highlighted_text = pattern.sub(replacer, text)
    return highlighted_text, found_keyword

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
# OPENROUTER API CALL (REPLACED WITH V2's ENHANCED VERSION)
# ==========================
def get_cheating_draft_placeholder(sources):
    """
    Provides a detailed, structured placeholder response for the 'cheating case' request.
    This is used when the API call fails due to authentication or other issues.
    """
    source_citations = ""
    if sources:
        source_citations = f"The intent to deceive (Source: {sources[0]['source']}) and the subsequent delivery of property (Source: {sources[0]['source']}) must be clearly established."

    return f"""
**CRITICAL AUTHENTICATION ERROR:** The LLM failed to connect due to an **Authorization Error (401)**. Please ensure your `OPENROUTER_API_KEY` is set correctly in the `CONFIG` section of the code.

---

### Placeholder Draft: Cheating (IPC Section 420)

This is a structural draft for a Criminal Complaint (First Information Report/Private Complaint) for the offense of Cheating and Dishonest Inducement to Deliver Property, usually covered under **Section 420, Indian Penal Code (IPC)**.

**1. Essential Elements of Cheating (Sec. 415 IPC):**
* **Deception:** The accused deceived the victim.
* **Fraudulent/Dishonest Inducement:** The deception was intended to fraudulently or dishonestly induce the victim.
* **Harm:** The victim was induced to:
    * Deliver property to any person.
    * Consent that any person should retain property.
    * Do or omit to do anything which caused or was likely to cause damage or harm in body, mind, reputation, or property.

**2. Aggravated Cheating (Sec. 420 IPC):**
This section applies when the cheating is specifically related to the **dishonest delivery of property**.

**Draft of the Petition (Complaint):**

| Section | Content | Example |
| :--- | :--- | :--- |
| **Title** | The full name of the Court (e.g., Court of the Chief Judicial Magistrate, Pune) | **IN THE COURT OF THE CJM, BANGALORE** |
| **Parties** | Complainant/Victim (A) vs. Accused (B) | **Complainant:** Mr. Vijay Sharma, S/o [Father's Name] |
| **Heading** | Subject matter and primary sections | **Complaint under Section 200 CrPC read with Section 420, 406 IPC** |
| **Facts (Body)** | <ul><li>**Date 1 (Deception):** When the accused made the false representation (e.g., promised to sell genuine gold).</li><li>**Date 2 (InducEMENT):** When the complainant believed the representation.</li><li>**Date 3 (Delivery):** When the complainant delivered the money/property (the loss).</li><li>**Date 4 (Discovery):** When the complainant realized they were cheated.</li></ul>| The Accused falsely represented that he possessed clear title to the property... |
| **Prayer** | The relief sought (register FIR, investigate, prosecute) | The Accused be summoned, tried, and punished according to law. |

{source_citations} Always ensure all oral representations, documents, and money transfer records are attached as evidence (Annexures).
"""

def call_openrouter(model, prompt, sources):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(f"https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        return answer, tokens_used

    except requests.exceptions.HTTPError as e:
        # --- 401 Error Handling ---
        if e.response.status_code == 401:
            st.error("Authentication Failed (401). Please check the OPENROUTER_API_KEY in the CONFIG section.")
            # Fallback for the specific request if authentication fails
            if "cheating case" in prompt.lower():
                return get_cheating_draft_placeholder(sources), 100
            return f"**LLM Error (401 Unauthorized):** Cannot connect to LLM. Please fix your API Key. {e}", 0

        # --- Other HTTP Errors ---
        st.error(f"HTTP Error connecting to LLM: {e}")
        return f"Error connecting to LLM: {e}", 0
    except Exception as e:
        # --- Other Errors (e.g., URL formatting) ---
        st.error(f"Error connecting to LLM: {e}")
        return f"Error connecting to LLM: {e}", 0

# ==========================
# TEXT TO SPEECH (v1's gTTS)
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
# ENHANCED SEARCH FUNCTION (v1's advanced search)
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
                "chat_name": session["name"],
                "matches": matches,
                "total_matches": len(matches)
            })

    return results

def highlight_search_terms(text, search_terms):
    """
    Highlight search terms in text with markdown bold. (For search results)
    """
    highlighted_text = text
    for term in search_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
    return highlighted_text


# ===================================================================
# DOWNLOAD CHAT CONTENT (PDF & WORD) - ADDED FROM V2
# ===================================================================

# --- Text Cleaning Helper for PDF (HTML compatible) ---
def clean_markdown_and_linebreaks(text):
    """
    Converts markdown (bold, italics, list items) to HTML for ReportLab's Paragraph element,
    while preserving line breaks.
    """
    text = re.sub(r'#+\s*', '', text) # Remove headings (#)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL) # Replace **bold** with <b>html bold</b>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text, flags=re.DOTALL) # Replace *italics* with <i>html italics</i>
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text) # Remove markdown links but keep text

    # Replace markdown list items (* or - followed by a space) with bullet HTML entity
    text = re.sub(r'^[\s]*[\*-]\s+', '&bull; ', text, flags=re.MULTILINE)

    # Convert double line breaks to HTML paragraph break, single breaks to simple line break
    text = re.sub(r'\n\n', '<br/><br/>', text)
    text = re.sub(r'\n', '<br/>', text)

    text = text.strip()
    return text

# --- Text Cleaning Helper for DOCX (Plain text) ---
def clean_text_for_docx(text):
    """
    Cleans markdown and HTML for pure plain text insertion into DOCX.
    """
    text = re.sub(r'#+\s*', '', text) # Remove headings
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL) # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text, flags=re.DOTALL) # Remove italics
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text) # Links
    text = re.sub(r'^[\s]*[\*-]\s+', '\u2022 ', text, flags=re.MULTILINE) # Bullets
    text = re.sub(r'<[^>]*>', '', text) # Strip HTML
    text = text.strip()
    return text

# --- PDF Generation ---
def generate_qna_content_pdf(messages, chat_name):
    """
    Generates a PDF with Times New Roman 15pt font and 1.5 line spacing,
    using PLATYPUS SimpleDocTemplate to handle multi-page content correctly.
    """
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
    
    styles.add(ParagraphStyle(name='Question',
                              fontName='Times-Bold',
                              fontSize=font_size_main,
                              leading=font_size_main * line_spacing,
                              spaceAfter=0,
                              alignment=TA_LEFT))
                              
    styles.add(ParagraphStyle(name='AnswerHTML',
                              fontName='Times-Roman',
                              fontSize=font_size_main,
                              leading=font_size_main * line_spacing,
                              spaceAfter=8,
                              leftIndent=10))
    
    styles.add(ParagraphStyle(name='SourceHeader',
                              fontName='Times-Bold',
                              fontSize=font_size_source,
                              leading=font_size_source * line_spacing,
                              spaceBefore=5,
                              spaceAfter=2,
                              leftIndent=10))
                              
    styles.add(ParagraphStyle(name='SourceItem',
                              fontName='Times-Roman',
                              fontSize=font_size_source,
                              leading=font_size_source * line_spacing,
                              spaceAfter=2,
                              leftIndent=20))
    
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
                    story.append(Paragraph("--- Sources Cited ---", styles['SourceHeader']))
                    
                    for j, source in enumerate(answer_message["sources"]):
                        # This now uses the FAST extractive summary
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
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(72, 720, f"Error: Could not generate PDF. Details: {e}")
        p.save()
        return buffer.getvalue()

# --- WORD Generation ---
def generate_qna_content_word(messages, chat_name):
    """
    Generates a .docx file with Times New Roman 15pt font and 1.5 line spacing.
    """
    document = Document()
    
    style = document.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(15)
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing = 1.5

    title = document.add_heading("ADVOCATE AI CHAT SESSION", level=1)
    title_run = title.runs[0]
    title_run.font.name = 'Times New Roman'
    
    topic = document.add_paragraph()
    topic_run = topic.add_run(f"Topic: {chat_name}")
    topic_run.font.name = 'Times New Roman'
    topic_run.bold = True
    topic_run.font.size = Pt(15)

    date = document.add_paragraph()
    date_run = date.add_run(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    date_run.font.name = 'Times New Roman'
    date_run.font.size = Pt(12)
    
    document.add_paragraph() # Spacer

    q_count = 1
    
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                answer_message = messages[i+1]
                
                q_para = document.add_paragraph()
                q_para.paragraph_format.line_spacing = 1.5
                q_run = q_para.add_run(f"Q{q_count}: {clean_text_for_docx(message['content'])}")
                q_run.bold = True
                q_run.font.name = 'Times New Roman'
                q_run.font.size = Pt(15)
                
                document.add_paragraph()

                a_para = document.add_paragraph(clean_text_for_docx(answer_message['content']))
                a_para.paragraph_format.line_spacing = 1.5
                a_para.paragraph_format.left_indent = Inches(0.25)
                for run in a_para.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(15)
                
                if answer_message.get("sources"):
                    s_header_para = document.add_paragraph()
                    s_header_para.paragraph_format.line_spacing = 1.5
                    s_header_para.paragraph_format.left_indent = Inches(0.25)
                    s_run = s_header_para.add_run("--- Sources Cited ---")
                    s_run.font.name = 'Times New Roman'
                    s_run.font.size = Pt(12)
                    s_run.italic = True

                    for j, source in enumerate(answer_message["sources"]):
                        # This now uses the FAST extractive summary
                        source_summary = summarize_source_with_llm(source['text'])
                        source_text = f"\u2022 Source {j+1}: {source['source']}\n\tSnippet: {source_summary}"
                        
                        src_para = document.add_paragraph(source_text)
                        src_para.paragraph_format.line_spacing = 1.5
                        src_para.paragraph_format.left_indent = Inches(0.5)
                        for run in src_para.runs:
                            run.font.name = 'Times New Roman'
                            run.font.size = Pt(12)
                
                document.add_paragraph() # Spacer
                q_count += 1
                
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ==========================
# STREAMLIT APP
# ==========================
st.set_page_config(page_title="Advocate AI Optimized", layout="wide")
embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()

if folder_index is None:
    st.error("Knowledge base not found. Run build_index_advanced.py")
    st.stop()

# --- ADDED: Custom CSS from V2 ---
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
}}
.stButton>button:hover {{ box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }}

/* Primary button for Active Chat Session */
.stButton[data-testid*="primary"]>button {{
    background-color: #1e3c72;
    color: white;
    border: 1px solid #1e3c72;
}}
/* Secondary button for inactive chats */
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
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


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
# --- ADDED: Keyword state from V2 ---
if "keyword_found" not in st.session_state:
    st.session_state.keyword_found = True # Start as True to prevent initial message
if "keyword_search_terms" not in st.session_state:
    st.session_state.keyword_search_terms = ""

# ==========================
# SIDEBAR - MERGED LAYOUT
# ==========================
with st.sidebar:
    
    # --- ADDED: Keyword Highlight Section from V2 (at the top) ---
    st.header("🔑 Keyword Highlight")
    keyword_search_input = st.text_input("Enter keywords (comma-separated)", 
                                         key="keyword_highlight_input", 
                                         placeholder="e.g., defense, doctrine, contract")
    st.session_state.keyword_search_terms = keyword_search_input
    
    # --- Display Keyword Search Status ---
    if not st.session_state.keyword_found and st.session_state.keyword_search_terms:
        st.warning("⚠️ Keywords not found in the current conversation.", icon="🔍")
    elif st.session_state.keyword_search_terms:
        st.success("Keywords found and highlighted in the chat.", icon="✨")
    st.markdown("---")

    # ==========================
    # FIRST: CHAT HISTORY (v1's advanced history)
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
        st.session_state.keyword_found = True # Reset status
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
                    
                    for match in result['matches'][:3]:  # Show max 3 matches per chat
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

    # Regular Chat List (when not searching)
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
                        st.session_state.keyword_found = True # Reset status
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
    
    # --- ADDED: Export Section from V2 ---
    st.header("📥 Export Chat")
    
    # Prepare file contents and names (logic from v2)
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
    elif st.session_state.messages: # Handle case where chat isn't saved yet
        current_messages = st.session_state.messages

    if current_messages:
        base_file_name = f"AdvocateAI_{current_chat_name.replace(' ', '_')[:20]}"
        pdf_file_name = f"{base_file_name}.pdf"
        word_file_name = f"{base_file_name}.docx"
        
        pdf_content_bytes = generate_qna_content_pdf(current_messages, current_chat_name)
        word_content_bytes = generate_qna_content_word(current_messages, current_chat_name)

    st.download_button(
        label="⬇️ Download Q&A as PDF",
        data=pdf_content_bytes,
        file_name=pdf_file_name,
        mime="application/pdf",
        use_container_width=True,
        disabled=(not pdf_content_bytes)
    )
    
    st.download_button(
        label="⬇️ Download Q&A as Word",
        data=word_content_bytes,
        file_name=word_file_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
        disabled=(not word_content_bytes)
    )
    
    if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
        st.session_state.chat_sessions = {}
        st.session_state.active_chat = None
        st.session_state.messages = []
        st.session_state.keyword_found = True
        st.rerun()

    st.markdown("---")

    # ==========================
    # SECOND: SELECT MODEL
    # ==========================
    
    # Professional Advocate Logo with Legal Symbol
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
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "x-ai/grok-code-fast-1",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "openai/gpt-5-mini"
    ], index=0)
    
    st.success(f"Knowledge Base loaded with {folder_index.ntotal} chunks", icon="✅")
    
    # ==========================
    # THIRD: FILE UPLOAD & OCR ENABLE
    # ==========================
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        st.session_state.uploaded_pdfs_data = uploaded_pdfs
    
    use_ocr_toggle = st.toggle("Enable OCR for scanned PDFs", help="Uses Google Vision to extract text from image-based PDFs. Slower and requires your API key.")

    # ==========================
    # FOURTH: COURTS IN INDIA
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

# Display current chat context
if st.session_state.active_chat and st.session_state.active_chat in st.session_state.chat_sessions:
    current_chat_name = st.session_state.chat_sessions[st.session_state.active_chat]["name"]
    st.subheader(f"💬 {current_chat_name}")


# --- MERGED CHAT DISPLAY LOOP (v2 Highlighting + v1 Sources/Audio) ---
global_keyword_found_in_session = False
keyword_terms = st.session_state.get("keyword_search_terms", "")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # --- v2: Keyword Highlighting Logic ---
            if keyword_terms:
                highlighted_content, found_in_message = highlight_with_style(content, keyword_terms)
                if found_in_message:
                    global_keyword_found_in_session = True
                st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.markdown(content) # Original display
            
            # --- v1: Source Expander Logic ---
            if message.get("sources"):
                with st.expander("Show Sources"):
                    for source in message["sources"]:
                        # This now uses the FAST (v2) summary function
                        llm_summary = summarize_source_with_llm(source["text"])
                        st.markdown(f"**Source:** `{source['source']}`")
                        st.markdown(f"**Summary (3 lines):** {llm_summary}")
            
            # --- v1: gTTS Audio Logic ---
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
    
    # --- v2: Keyword Status Update ---
    st.session_state.keyword_found = global_keyword_found_in_session

    if st.session_state.tokens_used > 0:
        st.markdown(f"---")
        st.caption(f"Total tokens used in this session: **{st.session_state.tokens_used}**")
# --- END: Merged Chat Display Loop ---


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
        with st.spinner("⚖️ Conducting multi-query legal research and analysis..."):
            # Retrieve context and generate response
            rag_context_full, sources = retrieve_and_rerank(choice, folder_index, folder_metadata, embed_model,
                                                            reranker_model, top_k=2)
            uploaded_context_full = ""
            if st.session_state.uploaded_pdfs_data:
                uploaded_context_full = process_uploaded_pdfs(st.session_state.uploaded_pdfs_data, use_ocr=use_ocr_toggle)

            final_prompt = construct_final_prompt(choice, rag_context_full, uploaded_context_full)
            
            # --- UPDATED: Call to v2's OpenRouter function ---
            answer, tokens = call_openrouter(model, final_prompt, sources)
            
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
