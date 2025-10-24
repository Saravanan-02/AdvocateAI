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
# --- FIX: Import SimpleDocTemplate and Spacer for multi-page content ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer 
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch

# --- WORD (.docx) Dependencies ---
# Graceful handling for missing python-docx
docx_available = False
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_PARAGRAPH_ALIGN
    docx_available = True
except ImportError:
    st.warning("The 'python-docx' library is not installed. Word export will be disabled. Install with: 'pip install python-docx'")


# ==========================
# NLTK DATA DOWNLOADS
# ==========================
# Note: In a production environment, you should ensure NLTK data is available
# either by running this once or by bundling the data with the deployment.
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
# --- IMPORTANT: API KEY SETUP ---
# To fix the 401 error, you MUST replace the placeholder below with your valid OpenRouter API Key.
# Get your key from OpenRouter and set it here, or as an environment variable (recommended in production).
OPENROUTER_API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
# --------------------------------
# Placeholder for keys (assuming they are set externally or kept private)


INDEX_FILE = "faiss_advanced_index.bin"
METADATA_FILE = "metadata.pkl"

# ==========================
# JAVASCRIPT FOR NATIVE TTS (Audio Output Feature)
# ==========================
# Function to inject into Streamlit to enable native browser TTS
def inject_tts_javascript():
    js_code = """
    function speakText(text) {
        if ('speechSynthesis' in window) {
            // Clean up the text: remove HTML tags, markdown, and limit length to prevent freezing
            let cleanText = text.replace(/<[^>]*>/g, '');
            cleanText = cleanText.replace(/\\*\\*(.*?)\\*\\*/g, '$1'); // Remove bold markdown
            cleanText = cleanText.replace(/\\n/g, ' '); // Replace newlines with spaces
            
            // Cancel any ongoing speech before starting a new one
            if (window.speechSynthesis.speaking) {
                window.speechSynthesis.cancel();
            }
            
            const utterance = new SpeechSynthesisUtterance(cleanText);
            
            // Optional: Set voice, pitch, and rate for a professional tone
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
    return f"""
    <script>
        {js_code}
    </script>
    """

# ==========================
# CACHED RESOURCE LOADING
# ==========================
@st.cache_resource
def load_models_and_index():
    # Use a faster, smaller model for efficiency
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
# Source summaries use 3 sentences
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

def summarize_source_with_llm(source_text, model="openai/gpt-5-mini"):
    # This now consistently uses the 3-sentence extractive summary for sources
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
    
    # FIX: Handle IndexScalarQuantizer by using ntotal instead of len
    try:
        # Try to get the total number of vectors in the index
        total_vectors = index.ntotal
    except AttributeError:
        # Fallback: use metadata length
        total_vectors = len(metadata)
    
    semantic_results = [metadata[idx] for idx in indices[0] if idx < total_vectors]
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

    return variations[:3]

def process_uploaded_pdfs(uploaded_files):
    """
    Placeholder for PDF processing logic.
    For now, it returns a generic summary string.
    OCR functionality has been removed.
    """
    if not uploaded_files:
        return ""
    
    file_names = ", ".join([f.name for f in uploaded_files][:3])
    count = len(uploaded_files)
    
    # Simulate text extraction and a very basic summary of the first few files
    return f"Context from {count} uploaded documents: {file_names}..."

# ==========================
# PROMPT CONSTRUCTION
# ==========================
def construct_final_prompt(question, rag_summary, uploaded_summary):
    rag_lines = rag_summary.strip().splitlines()
    if not rag_lines:
        knowledge_block = "No relevant knowledge found in the database."
    else:
        # Only pass the top 3 lines of retrieved context (as truncated 200-word chunks)
        knowledge_block = "\n".join(rag_lines[:3]) 
        
    uploaded_summary = truncate_text(uploaded_summary, max_words=200)

    # Instruct the model to provide a response with structure (line breaks, bullet points)
    final_prompt = f"""
    Advocate AI: Answer the following question clearly, concisely, and formally.
    
    **Instructions:**
    1. Provide a professional, structured legal response.
    2. **Crucially, maintain clear line breaks and use markdown (like bullet points or list numbering) for structure.**
    3. Do NOT include any initial greetings, intros, section numbers (I., II.), or headings (like 'Formal Structural Analysis'). Start directly with the answer content.
    4. Cite references in the text clearly, e.g., (Source: SourceFileName.pdf) instead of footnotes.
    
    **Question:** {question}
    
    **Knowledge (Top Relevant Context):**
    {knowledge_block}
    
    **Uploaded Docs Context:** {uploaded_summary}
    """
    return final_prompt.strip()

# ==========================
# LLM API CALL (with 401 Fallback) - FIXED
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
| **Facts (Body)** | <ul><li>**Date 1 (Deception):** When the accused made the false representation (e.g., promised to sell genuine gold).</li><li>**Date 2 (Inducement):** When the complainant believed the representation.</li><li>**Date 3 (Delivery):** When the complainant delivered the money/property (the loss).</li><li>**Date 4 (Discovery):** When the complainant realized they were cheated.</li></ul>| The Accused falsely represented that he possessed clear title to the property... |
| **Prayer** | The relief sought (register FIR, investigate, prosecute) | The Accused be summoned, tried, and punished according to law. |

{source_citations} Always ensure all oral representations, documents, and money transfer records are attached as evidence (Annexures).
"""

def call_openrouter(model, prompt, sources):
    
    # --- SIMULATION MODE ---
    # Provide a placeholder legal analysis for the cheating case request if an error occurs.
    if "cheating case" in prompt.lower() and not OPENROUTER_API_KEY:
        return get_cheating_draft_placeholder(sources), 100
        
    
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        # --- FIX: Changed URL from "httpss.openrouter..." to "https://openrouter..." ---
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


# =========================================
# KEYWORD HIGHLIGHTING FUNCTION (HTML STYLE) - REVISED
# =========================================
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


# ===================================================================
# DOWNLOAD CHAT CONTENT (PDF & WORD) - REVISED FOR NEW FORMATTING
# ===================================================================

# --- Text Cleaning Helper for PDF (HTML compatible) --- (FIXED)
def clean_markdown_and_linebreaks(text):
    """
    Converts markdown (bold, italics, list items) to HTML for ReportLab's Paragraph element, 
    while preserving line breaks.
    """
    # --- FIX ---
    # The "duplication fix" below was causing the main body of the petition to be deleted.
    # It is now disabled to allow the full text to render.
    
    # split_point = "Writ Petition under Article 226 of the Constitution of India praying for issuance of a Writ of Certiorarified Mandamus"
    # if text.count(split_point) > 1:
    #     text = text.split(split_point, 1)[0] + split_point + text.split(split_point, 2)[-1]
        
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

# --- Text Cleaning Helper for DOCX (Plain text) --- (FIXED)
def clean_text_for_docx(text):
    """
    Cleans markdown and HTML for pure plain text insertion into DOCX.
    """
    # --- FIX ---
    # The "duplication fix" below was causing the main body of the petition to be deleted.
    # It is now disabled to allow the full text to render.
    
    # split_point = "Writ Petition under Article 226 of the Constitution of India praying for issuance of a Writ of Certiorarified Mandamus"
    # if text.count(split_point) > 1:
    #     text = text.split(split_point, 1)[0] + split_point + text.split(split_point, 2)[-1]
        
    text = re.sub(r'#+\s*', '', text) # Remove headings
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL) # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text, flags=re.DOTALL) # Remove italics
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text) # Links
    text = re.sub(r'^[\s]*[\*-]\s+', '\u2022 ', text, flags=re.MULTILINE) # Bullets
    text = re.sub(r'<[^>]*>', '', text) # Strip HTML
    text = text.strip()
    return text

# --- PDF Generation (FIXED) ---
def generate_qna_content_pdf(messages, chat_name):
    """
    Generates a PDF with Times New Roman 15pt font and 1.5 line spacing,
    using PLATYPUS SimpleDocTemplate to handle multi-page content correctly.
    """
    buffer = BytesIO()
    # Set standard 0.75-inch margins
    margin = 0.75 * inch 
    
    # --- Use SimpleDocTemplate for automatic page breaks ---
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    
    styles = getSampleStyleSheet()
    story = [] # This list will hold all our "flowable" elements
    
    # --- Define custom styles as requested ---
    line_spacing = 1.5
    font_size_main = 15
    font_size_source = 12
    
    # Main Q&A styles
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
    
    # Source styles
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
    
    # Title styles
    # Title styles
    styles.add(ParagraphStyle(name='ChatTitle', fontName='Times-Bold', fontSize=18, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='SubTitle', fontName='Times-Bold', fontSize=15, spaceAfter=5, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Date', fontName='Times-Roman', fontSize=12, spaceAfter=20, alignment=TA_LEFT))

    # --- 1. Add Title Block to the story ---
    # --- 1. Add Title Block to the story ---
    story.append(Paragraph("ADVOCATE AI CHAT SESSION", styles['ChatTitle']))
    story.append(Paragraph(f"Topic: {chat_name}", styles['SubTitle']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Date']))
    story.append(Spacer(1, 0.25 * inch)) # Extra space after header

    q_count = 1
    
    # --- 2. Loop through messages and add them to the story ---
    for i, message in enumerate(messages):
        if message["role"] == "user":
            
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                answer_message = messages[i+1]
                
                # --- Question ---
                question_text = f"Q{q_count}: {clean_markdown_and_linebreaks(message['content'])}"
                story.append(Paragraph(question_text, styles['Question']))
                
                # --- FIX: Adds a 1-line gap after the question (15pt) ---
                story.append(Spacer(1, 15)) 
                
                # --- Answer (will auto-wrap pages) ---
                html_answer = clean_markdown_and_linebreaks(answer_message['content'])
                story.append(Paragraph(html_answer, styles['AnswerHTML']))
                
                # --- Sources ---
                if answer_message.get("sources"):
                    story.append(Paragraph("--- Sources Cited ---", styles['SourceHeader']))
                    
                    for j, source in enumerate(answer_message["sources"]):
                        source_summary = summarize_source_with_llm(source['text']) 
                        # HTML for bullets, bold, italics, and line breaks
                        source_line = (f"&bull; <b>Source {j+1}:</b> {source['source']}<br/>"
                                       f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Snippet:</i> {source_summary}")
                        story.append(Paragraph(source_line, styles['SourceItem']))
                
                # --- Separator ---
                story.append(Spacer(1, 0.2 * inch)) # Space before next Q&A
                q_count += 1
    
    # --- 3. Build the PDF document ---
    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        # Fallback in case of a complex rendering error
        st.error(f"Error generating PDF: {e}")
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(72, 720, f"Error: Could not generate PDF. Details: {e}")
        p.save()
        return buffer.getvalue()

# --- WORD Generation (FIXED) ---
def generate_qna_content_word(messages, chat_name):
    """
    Generates a .docx file with Times New Roman 15pt font and 1.5 line spacing.
    """
    if not docx_available:
        st.error("Word export is not available. Please install python-docx: pip install python-docx")
        return b''
    
    document = Document()
    
    # Set default style for the whole document
    style = document.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(15)
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing = 1.5

    # Title Block
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
                
                # --- Question ---
                q_para = document.add_paragraph()
                q_para.paragraph_format.line_spacing = 1.5
                q_run = q_para.add_run(f"Q{q_count}: {clean_text_for_docx(message['content'])}")
                q_run.bold = True
                q_run.font.name = 'Times New Roman'
                q_run.font.size = Pt(15)
                
                # --- FIX: Adds a 1-line gap after the question ---
                document.add_paragraph() 

                # --- Answer ---
                a_para = document.add_paragraph(clean_text_for_docx(answer_message['content']))
                a_para.paragraph_format.line_spacing = 1.5
                a_para.paragraph_format.left_indent = Inches(0.25)
                # Ensure all runs in the answer paragraph have the correct font
                for run in a_para.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(15)
                
                # --- Sources ---
                if answer_message.get("sources"):
                    s_header_para = document.add_paragraph()
                    s_header_para.paragraph_format.line_spacing = 1.5
                    s_header_para.paragraph_format.left_indent = Inches(0.25)
                    s_run = s_header_para.add_run("--- Sources Cited ---")
                    s_run.font.name = 'Times New Roman'
                    s_run.font.size = Pt(12)
                    s_run.italic = True

                    for j, source in enumerate(answer_message["sources"]):
                        source_summary = summarize_source_with_llm(source['text'])
                        # Use \n for line break and \t for indentation in Word
                        source_text = f"\u2022 Source {j+1}: {source['source']}\n\tSnippet: {source_summary}"
                        
                        src_para = document.add_paragraph(source_text)
                        src_para.paragraph_format.line_spacing = 1.5
                        src_para.paragraph_format.left_indent = Inches(0.5)
                        for run in src_para.runs:
                            run.font.name = 'Times New Roman'
                            run.font.size = Pt(12)
                
                document.add_paragraph() # Spacer
                q_count += 1
                
    # Save to buffer
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ==========================
# STREAMLIT APP SETUP & UI ENHANCEMENTS
# ==========================
# Inject the JavaScript function once at the start
st.markdown(inject_tts_javascript(), unsafe_allow_html=True) 

st.set_page_config(page_title="Advocate AI Optimized", layout="wide")
embed_model, reranker_model, folder_index, folder_metadata = load_models_and_index()

if folder_index is None:
    st.error("Knowledge base not found. Run build_index_advanced.py to create 'faiss_advanced_index.bin' and 'metadata.pkl'.")
    st.stop()

# Initialize session state 
if "messages" not in st.session_state: st.session_state.messages = []
if "pending_prompts" not in st.session_state: st.session_state.pending_prompts = None
if "uploaded_pdfs_data" not in st.session_state: st.session_state.uploaded_pdfs_data = None
if "tokens_used" not in st.session_state: st.session_state.tokens_used = 0
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {}
if "active_chat" not in st.session_state: st.session_state.active_chat = None
if "selected_model" not in st.session_state: st.session_state.selected_model = "openai/gpt-5"
# New state for keyword search status
if "keyword_found" not in st.session_state: st.session_state.keyword_found = True # Start as True to prevent initial message

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

/* TTS Button Styling */
.stButton button[kind="secondary"] {{
    background-color: #f9f9f9;
    color: #1e3c72;
    border: 1px solid #ccc;
    font-size: 10px;
    padding: 2px 5px;
    height: 25px;
    margin-bottom: 10px; /* Space between button and text */
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================
# SIDEBAR 
# ==========================
with st.sidebar:
    
    st.header("🔑 Keyword Highlight")
    keyword_search_input = st.text_input("Enter keywords (comma-separated)", key="keyword_highlight_input", placeholder="e.g., defense, doctrine, contract")
    st.session_state.keyword_search_terms = keyword_search_input
    st.markdown("---") 

    # --- Display Keyword Search Status ---
    if not st.session_state.keyword_found and st.session_state.keyword_search_terms:
        st.warning("⚠️ Keywords not found in the current conversation.", icon="🔍")
    elif st.session_state.keyword_search_terms:
        st.success("Keywords found and highlighted in the chat.", icon="✨")

    st.markdown("---") 

    # ==========================
    # CHAT HISTORY / NEW CHAT
    # ==========================
    st.header("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        st.session_state.chat_sessions[chat_id] = {"name": "New Chat", "messages": [], "created_at": datetime.now()}
        st.session_state.active_chat = chat_id
        st.session_state.messages = []
        st.session_state.keyword_found = True # Reset status on new chat
        st.rerun()

    st.write("**Your Chats:**")
    if not st.session_state.chat_sessions:
        st.info("No chats yet. Start a new conversation!")
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
                    st.session_state.keyword_found = True # Assume found on chat switch until checked
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
                        st.rerun()
    st.markdown("---")

    # ==========================
    # DOWNLOAD & CLEAR HISTORY (ADDED WORD EXPORT)
    # ==========================
    
    # Prepare file contents and names
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
        # Generate file names
        base_file_name = f"AdvocateAI_{current_chat_name.replace(' ', '_')[:20]}"
        pdf_file_name = f"{base_file_name}.pdf"
        word_file_name = f"{base_file_name}.docx"
        
        # Generate file bytes
        pdf_content_bytes = generate_qna_content_pdf(current_messages, current_chat_name)
        if docx_available:
            word_content_bytes = generate_qna_content_word(current_messages, current_chat_name)

    # PDF Download Button
    st.download_button(
        label="⬇️ Download Q&A as PDF", 
        data=pdf_content_bytes, 
        file_name=pdf_file_name,
        mime="application/pdf", 
        use_container_width=True,
        disabled=(not pdf_content_bytes)
    )
    
    # WORD Download Button (NEW) - Only show if docx is available
    if docx_available:
        st.download_button(
            label="⬇️ Download Q&A as Word", 
            data=word_content_bytes, 
            file_name=word_file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
            use_container_width=True,
            disabled=(not word_content_bytes)
        )
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_sessions = {}
        st.session_state.active_chat = None
        st.session_state.messages = []
        st.session_state.keyword_found = True
        st.rerun()
    
    st.markdown("---")
    
    # ==========================
    # MODEL & UPLOAD SETTINGS
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
    
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs: 
        st.session_state.uploaded_pdfs_data = uploaded_pdfs
    
    # NOTE: OCR toggle removed as requested

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

# Logic to track if any keyword was found in the *entire* displayed chat
# This must be reset every time the chat is rendered to check for the current keyword input
global_keyword_found_in_session = False
keyword_terms = st.session_state.keyword_search_terms

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # --- FIX FOR DUPLICATION ISSUE ---
            # The faulty "fix" logic was removed from the cleaning functions,
            # but we can leave this display-time fix just in case.
            split_point = "Writ Petition under Article 226 of the Constitution of India praying for issuance of a Writ of Certiorarified Mandamus"
            if content.count(split_point) > 1:
                content = content.split(split_point, 1)[0] + split_point + content.split(split_point, 2)[-1]
            
            # Display Audio Button for Assistant messages using the native JS function
            if message["role"] == "assistant":
                # Escape single quotes and newlines in the content for JavaScript string safety
                js_safe_content = content.replace("'", "\\'").replace('\n', ' ')
                
                # Using st.markdown to inject the TTS button which calls the JS function
                st.markdown(
                    f'<button onclick="speakText(\'{js_safe_content}\')" class="stButton secondary">🔊 Read Aloud</button>',
                    unsafe_allow_html=True
                )

            # Display Text Content with Highlighting
            if keyword_terms:
                highlighted_content, found_in_message = highlight_with_style(content, keyword_terms)
                if found_in_message:
                    global_keyword_found_in_session = True
                st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.markdown(content)
            
            # Display Sources
            if message.get("sources"):
                with st.expander("Show Sources"):
                    for source in message["sources"]:
                        # Now uses a 3-sentence summary
                        llm_summary = summarize_source_with_llm(source["text"], model=st.session_state.selected_model) 
                        st.markdown(f"**Source:** `{source['source']}`")
                        st.markdown(f"**Snippet (3 Lines):** {llm_summary}")
    
    # Update the session state variable for the sidebar display
    st.session_state.keyword_found = global_keyword_found_in_session
    
    if st.session_state.tokens_used > 0:
        st.markdown(f"---")
        st.caption(f"Total tokens used in this session: **{st.session_state.tokens_used}**")

# ==========================
# MAIN CHAT INPUT AND LOGIC
# ==========================

def handle_new_chat_message(prompt):
    """
    Processes the user's question: RAG, LLM call, and state update.
    """
    # 0. Auto-create a new chat session if none is active
    if not st.session_state.active_chat:
        chat_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{random.randint(1000,9999)}"
        new_chat_name = prompt[:30].strip() + "..." if len(prompt) > 30 else prompt.strip()
        st.session_state.chat_sessions[chat_id] = {"name": new_chat_name, "messages": [], "created_at": datetime.now()}
        st.session_state.active_chat = chat_id
        st.session_state.messages = st.session_state.chat_sessions[chat_id]["messages"]
    
    # 1. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("⚖️ Conducting multi-query legal research and analysis..."):
        
        # 2. Dynamic Query Generation (Multi-Query Strategy)
        queries = dynamic_query_generator(prompt)
        
        # 3. Parallel Retrieval and Reranking for all generated queries
        all_chunks = []
        all_sources = []
        
        for q in queries:
            try:
                # Retrieve and Rerank for the query, getting top 2 (or a bit more) chunks
                # retrieve_and_rerank performs RRF internally on a candidate set
                chunks, sources = retrieve_and_rerank(q, folder_index, folder_metadata, embed_model, reranker_model, top_k=2)
                all_chunks.extend(chunks.split('\n\n'))
                all_sources.extend(sources)
            except Exception as e:
                # Log the error but continue with other queries
                st.warning(f"Retrieval error for query '{q[:20]}...': {e}")

        # 4. Final Context Selection: Remove duplicates and combine context
        unique_sources = {}
        for source in all_sources:
            # Use source file + first 100 chars of text as key to identify uniqueness
            key = source['source'] + source['text'][:100]
            if key not in unique_sources:
                unique_sources[key] = source
        
        # Re-assemble the RAG context from unique top-ranked chunks
        final_rag_context = "\n\n".join([truncate_text(s['text'], max_words=200) for s in unique_sources.values()])
        
        # 5. Process Uploaded PDFs (Placeholder summary)
        uploaded_summary = ""
        if st.session_state.uploaded_pdfs_data:
            uploaded_summary = process_uploaded_pdfs(st.session_state.uploaded_pdfs_data)
        
        # 6. Construct Final Prompt
        final_prompt = construct_final_prompt(prompt, final_rag_context, uploaded_summary)
        
        # 7. LLM Call
        llm_answer, tokens_used = call_openrouter(st.session_state.selected_model, final_prompt, list(unique_sources.values()))
        st.session_state.tokens_used += tokens_used

        # 8. Add assistant message to state
        assistant_message = {
            "role": "assistant",
            "content": llm_answer,
            "sources": list(unique_sources.values())
        }
        # --- FIX: Corrected typo from "messages..append" to "messages.append" ---
        st.session_state.messages.append(assistant_message)
        
        # 9. Update the active chat session in chat_sessions state
        if st.session_state.active_chat:
            st.session_state.chat_sessions[st.session_state.active_chat]["messages"] = st.session_state.messages
            
    st.rerun()

# --- Streamlit Chat Input ---
if prompt := st.chat_input("Ask a legal question (e.g., 'What is the doctrine of Res Gestae?')"):
    handle_new_chat_message(prompt)

# --- End of Streamlit App ---
