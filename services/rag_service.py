import os
import re
import json
import uuid
import logging
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sqlalchemy.orm import Session

from models.document_model import DocumentModel

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# STEP 1: PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF, prioritizing table structure detection
    (critical for lab reports / structured documents where naive text
    extraction scrambles column order).
    """
    text_parts = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            if tables:
                for table in tables:
                    for row in table:
                        if not any(cell for cell in row):
                            continue
                        row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                        text_parts.append(row_text)

            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# STEP 2: TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """
    Removes repeated page-header/footer boilerplate that appears on every
    page of many structured PDFs (patient info blocks, barcodes, page
    numbers). This boilerplate otherwise pollutes retrieval by producing
    many near-duplicate, low-information chunks.
    """
    text = re.sub(
        r"Report Status.*?Phone:\s*\d+",
        "",
        text,
        flags=re.DOTALL
    )
    text = re.sub(r"\*\d{6,}\*", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# STEP 3: CHUNKING
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """
    Splits text into smaller, tightly-bounded chunks. Smaller chunks work
    better for dense tabular data (lab reports, invoices, etc.) because
    they keep individual rows/facts isolated rather than buried inside
    a large block dominated by unrelated data.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.strip()) > 10]


# ---------------------------------------------------------------------------
# STEP 4: DOCUMENT PROCESSING PIPELINE
# ---------------------------------------------------------------------------

def process_document(file_path: str, document_id: int, db: Session):
    """
    Full pipeline: extract -> clean -> chunk -> embed -> store in a
    dedicated Chroma collection. Updates the document's status when done,
    and marks it 'failed' with the error logged if anything breaks.
    """
    try:
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_extracted_text(raw_text)
        chunks = chunk_text(cleaned_text)

        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        collection_name = f"doc_{document_id}_{uuid.uuid4().hex[:8]}"

        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR
        )

        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        document.collection_name = collection_name
        document.status = "ready"
        db.commit()

        logger.info(f"Document {document_id} processed successfully: {len(chunks)} chunks created.")

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if document:
            document.status = "failed"
            db.commit()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# STEP 5: QUERY DECOMPOSITION
# ---------------------------------------------------------------------------

def extract_search_terms(question: str) -> list[str]:
    """
    Uses a cheap LLM call to break a multi-part question into individual
    search terms, so each field the user asked about gets its own
    dedicated retrieval pass instead of competing in one shared search.
    Falls back to treating the whole question as one term if extraction fails.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the distinct data fields or topics the user is asking about "
                        "as a JSON array of short search strings. Example: "
                        "'What is RBC, MCH and platelet count?' -> "
                        '["RBC Count", "MCH", "Platelet Count"]. '
                        "Return ONLY the JSON array, nothing else."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        terms = json.loads(response.choices[0].message.content)
        if isinstance(terms, list) and terms:
            return terms
    except Exception as e:
        logger.warning(f"Search term extraction failed, using full question: {e}")

    return [question]


# ---------------------------------------------------------------------------
# STEP 6: QUESTION ANSWERING (MULTI-QUERY RETRIEVAL + GENERATION)
# ---------------------------------------------------------------------------

def answer_question(question: str, collection_name: str) -> dict:
    """
    Decomposes the question into individual search terms, retrieves
    relevant chunks separately for EACH term (so no single field gets
    crowded out by others in a multi-part question), merges and
    deduplicates the results, then asks the LLM to answer strictly
    grounded in that combined context.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    search_terms = extract_search_terms(question)
    logger.info(f"Decomposed question into search terms: {search_terms}")

    seen_content = set()
    merged_docs = []

    for term in search_terms:
        # k=5 per term: gives each requested field more chances to surface
        results = vectorstore.similarity_search(term, k=5)
        for doc in results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                merged_docs.append(doc)

    if not merged_docs:
        return {
            "reply": "I couldn't find relevant information in this document.",
            "sources": []
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in merged_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document assistant. Answer ONLY using the "
                "context provided below. For every value you state, it must "
                "appear verbatim in the context — never estimate, infer, average, "
                "or reuse a number from a different field.\n\n"
                "If the user asks for multiple values and only some are present "
                "in the context, answer the ones you can find and explicitly say "
                "'Not found in the provided context' for each missing one. "
                "Never guess to fill a gap."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0
    )

    answer = response.choices[0].message.content
    sources = [doc.page_content[:200] + "..." for doc in merged_docs]

    return {"reply": answer, "sources": sources}


# ---------------------------------------------------------------------------
# DEBUG UTILITY (temporary — remove before final production deploy)
# ---------------------------------------------------------------------------

def debug_raw_search(query: str, collection_name: str, k: int = 10) -> list[str]:
    """
    Bypasses the LLM entirely and returns raw retrieved chunks for a query.
    Useful for diagnosing whether a missing answer is a retrieval problem
    or a generation problem.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]