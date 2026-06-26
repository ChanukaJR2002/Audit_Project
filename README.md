# Think Before You Sign - AI Contract Auditor

An AI-powered Streamlit application that scans contracts and lease documents (PDF) for risky clauses, hidden fees, and one-sided terms. It uses Retrieval-Augmented Generation (RAG) with Google Gemini to analyze contract text and produce a plain-language risk assessment.

## Features

- Upload any contract or lease as a PDF and validate that it is a genuine, readable PDF
- Automatic text extraction and chunking of contract content
- Semantic search over contract chunks using Sentence Transformers and FAISS
- AI-generated audit covering:
  - Red flags and hidden fees
  - Termination and renewal clauses
  - Negotiation and counter-offer points
- Automatic risk scoring (0-100) with Low / Medium / High risk classification
- Follow-up chat to ask custom questions about the uploaded contract
- In-app PDF preview

## Tech Stack

- Streamlit - web app interface
- Google Gemini API (`gemini-2.5-flash`) - contract analysis and risk scoring
- Sentence Transformers (`all-MiniLM-L6-v2`) - text embeddings
- FAISS - similarity search over contract chunks
- PyPDF2 - PDF parsing and text extraction

## Project Structure

```
Audit_Project/
├── app1.py                  # Main Streamlit application
├── .streamlit/
│   └── secrets.toml          # API keys (not committed to git)
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or later
- A Google Gemini API key

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ChanukaJR2002/Audit_Project.git
   cd Audit_Project
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv env
   # Windows
   env\Scripts\activate
   # macOS/Linux
   source env/bin/activate
   ```

3. Install dependencies:
   
   pip install -r requirements.txt
   ```

4. Add your Gemini API key. Create a file at `.streamlit/secrets.toml` with the following content:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

   This file is listed in `.gitignore` and should never be committed to version control.

## Running the App

```
streamlit run app.py
```

The app will open in your browser, usually at `http://localhost:8501`.

## Usage

1. Upload a contract PDF using the sidebar file uploader.
2. Wait for the document to be processed (text extraction and embedding).
3. Choose one of the audit actions:
   - Scan for Red Flags and Hidden Fees
   - Check Termination and Renewal Clauses
   - Generate Counter-Offer and Negotiation Points
4. Review the generated risk score and plain-language breakdown.
5. Use the chat box at the bottom to ask follow-up questions about the contract.
6. Use the Reset button in the sidebar to clear the current document and start over.

## Notes

- Only text-based PDFs are supported. Scanned or image-only PDFs without a text layer will be rejected.
- The risk score (0-100) is calibrated by contract type (e.g. employment, lease, NDA, SaaS) rather than judged in the abstract.
- This tool provides an automated, AI-generated analysis and is not a substitute for advice from a qualified legal professional.

## Limitations

- Only supports text-based PDFs — scanned/image-only PDFs are rejected since there's no OCR step
- No persistent storage — uploaded documents and chat history are lost when the session ends or the page refreshes
- Depends on an external API (Gemini), so it requires internet access and a valid API key to function; no offline mode

## Future Implementations

- Add OCR support (e.g. Tesseract) to handle scanned/image PDFs
- Support additional file formats (like DOCX, plain text)
- Add downloadable audit reports (PDF/DOCX export of results)
- Add user authentication and persistent history across sessions
- Add other languages
- Add a feedback mechanism to flag inaccurate AI analysis(e.g. Thumbs up or down)

## Author Details
---------------
## Name - Chanuka Rajapaksa
## GitHub - https://github.com/ChanukaJR2002
## Linkedin - https://www.linkedin.com/in/chanuka-rajapaksa-14b9533a1/