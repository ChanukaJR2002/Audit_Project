import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# 1.Configuration
st.set_page_config(page_title="Think Before You Sign - AI Contract Auditor", page_icon="⚖️", layout="wide")

## 2.Assign Gemini API key
try:
    # Look inside Streamlit secrets (.streamlit/secrets.toml)
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=API_KEY)
    else:
        st.error("🔑 Critical Error: GEMINI_API_KEY not found in your Streamlit secrets! Please add it to .streamlit/secrets.toml.")
        client = None
except Exception as e:
    st.error(f"Gemini API Client failed to initialize. Error details: {e}")
    client = None

# 3. Initialize the Embedding Model 
@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# 4.Processing PDFs
def is_valid_pdf(pdf_file):
    """Checks the actual file signature/structure, not just the extension, to confirm it's a real PDF."""
    try:
        # Check the magic bytes PDF files must start with
        pdf_file.seek(0)
        header = pdf_file.read(5)
        pdf_file.seek(0)
        if header != b"%PDF-":
            return False
        # Try to actually open it with PyPDF2 to catch corrupted/fake PDFs
        PdfReader(pdf_file)
        pdf_file.seek(0)
        return True
    except Exception:
        pdf_file.seek(0)
        return False

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    """Splits text into overlapping chunks so context isn't lost at boundaries."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def run_audit(audit_query, analysis_type, faiss_index, text_chunks):
    """Runs the RAG + Gemini step and returns (display_text, risk_percentage), or (None, None) on failure."""
    query_vector = embedding_model.encode([audit_query])
    distances, indices = faiss_index.search(np.array(query_vector).astype('float32'), k=6)

    retrieved_context = ""
    for idx in indices[0]:
        if idx < len(text_chunks):
            retrieved_context += f"\n--- Contract Segment ---\n{text_chunks[idx]}\n"

    system_prompt = (
    "You are an expert contract auditor and legal analyst. Read the provided contract segments "
    "and answer the user's request. Always start your response with exactly one line indicating "
    "the risk percentage from 0 to 100 based on the severity of the clauses found, in this exact format:\n"
    "RISK_PERCENTAGE: X\n"
    "Use this calibration: 0-30 = standard/fair clauses common in this contract type with no unusual "
    "one-sidedness; 31-60 = some clauses favor one party more than typical, but nothing predatory or "
    "trapping; 61-100 = clauses that trap, penalize disproportionately, or are deceptive/non-standard "
    "for this type of agreement (e.g., compounding penalties, mandatory non-waivable fees, hard-to-exit "
    "auto-renewals). Judge severity relative to what's typical for this specific contract type, not in "
    "the abstract. "
    "Followed by your plain-language breakdown. Highlight risks in bold text and give practical advice."
)

    user_prompt = f"""
    User Audit Goal: {analysis_type} - {audit_query}

    Relevant Contract Context:
    {retrieved_context}

    Please provide a comprehensive analysis based strictly on the text provided above.
    Important: First identify what TYPE of contract this is (e.g., employment, lease, consulting/services, 
    SaaS/subscription, NDA) from the context given, and calibrate your risk severity judgment to what is 
    standard/typical for that specific contract type, per the calibration guide in your instructions.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={"system_instruction": system_prompt}
        )
        raw_output = response.text
        display_text = raw_output
        risk_num = 50  # fallback if parsing fails

        if "RISK_PERCENTAGE:" in raw_output:
            try:
                line = [l for l in raw_output.split('\n') if "RISK_PERCENTAGE:" in l][0]
                risk_num = int(line.split(":")[1].strip())
                display_text = display_text.replace(line, "").strip()
            except Exception:
                risk_num = 50

        return display_text, risk_num
    except Exception as e:
        st.error(f"Error communicating with Gemini: {e}")
        return None, None

# 5. Setup the Session State  
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # bump this to force-clear the file uploader
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = None

# 6. Streamlit UI 
st.title("⚖️ Think Before You Sign ⚖️")
st.subheader("Upload any contract or lease to scan for sneaky clauses and hidden risks.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a contract PDF",
        type="pdf",
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        st.markdown("---")
        if st.button("🔄 Reset", use_container_width=True, help="Removes the document and clears all analysis/chat history."):
            st.session_state.analysis_results = None
            st.session_state.analysis_type = None
            st.session_state.chat_history = []
            st.session_state.uploader_key += 1
            st.rerun()

# Upload the document
if uploaded_file is not None:

    # Validate that this is actually a real PDF (not a renamed .txt/.docx/etc.)
    if not is_valid_pdf(uploaded_file):
        st.error(
            f"❌ '{uploaded_file.name}' is not a valid PDF file. "
            "It may be a different file type (renamed) or a corrupted/unsupported PDF. "
            "Please upload a genuine PDF contract."
        )
        st.stop()

    st.success(f"Loaded: {uploaded_file.name}")

    # Show the uploaded document in the interface
    with st.expander("📄 View Uploaded Document", expanded=False):
        pdf_bytes = uploaded_file.getvalue()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )


    with st.spinner("Analyzing document structure..."):
        raw_text = extract_text_from_pdf(uploaded_file)

        # Catch PDFs that parsed but contain no extractable text (e.g. scanned image-only PDFs)
        if not raw_text or not raw_text.strip():
            st.error(
                "⚠️ No readable text could be extracted from this PDF. "
                "It may be a scanned/image-only document with no real text layer. "
                "Please upload a text-based PDF contract."
            )
            st.stop()

        text_chunks = chunk_text(raw_text)

        # Create Vector Embeddings using Sentence Transformers
        embeddings = embedding_model.encode(text_chunks)

        # Index chunks with FAISS for fast similarity matching
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings).astype('float32'))

    # Pre-defined Auditing Actions
    st.markdown("### Choose an Audit Action:")
    col1, col2, col3 = st.columns(3)

    audit_query = ""
    analysis_type = ""

    with col1:
        if st.button("🚩 Scan for Red Flags & Hidden Fees", use_container_width=True):
            audit_query = "hidden fees, penalties, automatic renewals, termination charges, liabilities"
            analysis_type = "Red Flags"
    with col2:
        if st.button("🔄 Check Termination & Renewal Clauses", use_container_width=True):
            audit_query = "how to terminate contract, cancellation notice period, automatic renewal terms"
            analysis_type = "Termination Terms"
    with col3:
        if st.button("💡 Generate Counter-Offer & Negotiation Points", use_container_width=True):
            audit_query = "payment terms, responsibilities, liabilities, governing law"
            analysis_type = "Negotiation Prep"

    # Run the audit if one of the buttons was just clicked
    if audit_query:
        with st.spinner(f"Retrieving context and auditing for {analysis_type}..."):
            result_text, risk_num = run_audit(audit_query, analysis_type, faiss_index, text_chunks)
            if result_text is not None:
                st.session_state.analysis_results = result_text
                st.session_state.analysis_type = analysis_type
                st.session_state.risk_pct = risk_num
                st.session_state.chat_history = []  # fresh audit, clear old Q&A thread

    # Display persisted audit results
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.markdown(f"## 📊 Audit Results: {st.session_state.analysis_type}")

        risk_pct = st.session_state.risk_pct
        if risk_pct is not None:
            if risk_pct >= 71:
                risk_status, risk_msg = "HIGH RISK 🚩", "Action Required: Dangerous or heavily one-sided clauses found."
            elif risk_pct >= 31:
                risk_status, risk_msg = "MEDIUM RISK ⚠️", "Caution: Terms heavily favor the other party."
            else:
                risk_status, risk_msg = "LOW RISK ✅", "Safe: Terms look reasonable and balanced."

            metric_col1, metric_col2 = st.columns([1, 2])
            with metric_col1:
                st.metric(label="Calculated Risk Level", value=f"{risk_pct}%", delta=risk_status,
                          delta_color="inverse" if risk_pct > 30 else "normal")
            with metric_col2:
                st.caption("Threat Assessment")
                st.progress(risk_pct / 100)
                st.write(risk_msg)

        st.markdown(st.session_state.analysis_results)

        # Ask any question according to analyzed document
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Follow-up Questions")
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

    # Display chat bar in the bottom after uploading a doument
    custom_query = st.chat_input("Ask a specific question about this contract...")
    if custom_query:
        with st.spinner("Retrieving context and answering your question..."):
            answer_text, _ = run_audit(custom_query, "Custom Query", faiss_index, text_chunks)
            if answer_text is not None:
                # Keep the existing analysis on screen; just append this Q&A to the thread
                st.session_state.chat_history.append((custom_query, answer_text))
                st.rerun()

else:
    st.info("Please upload a contract PDF in the sidebar to begin.")
