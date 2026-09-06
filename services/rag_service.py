import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from models.document_model import DocumentModel
from models.document_chunk_model import DocumentChunkModel

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# STEP 1: PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF, prioritizing table structure detection
    (critical for lab reports, invoices, forms — anything with tabular
    data where naive text extraction scrambles column order).
    """
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
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
# STEP 2: CHUNKING
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.strip()) > 10]


# ---------------------------------------------------------------------------
# STEP 3: EMBEDDING
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# STEP 4: DOCUMENT PROCESSING PIPELINE
# ---------------------------------------------------------------------------

def process_document(file_path: str, document_id: int, db: Session):
    try:
        raw_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(raw_text)

        if not chunks:
            raise ValueError("No extractable text found in this PDF.")

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            db_chunk = DocumentChunkModel(
                document_id=document_id,
                chunk_text=chunk,
                chunk_index=index,
                embedding=embedding
            )
            db.add(db_chunk)

        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        document.status = "ready"
        db.commit()

        logger.info(f"Document {document_id} processed: {len(chunks)} chunks stored in PostgreSQL.")

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
# STEP 5: QUESTION CLASSIFICATION + DECOMPOSITION
# ---------------------------------------------------------------------------

def analyze_question(question: str) -> dict:
    """
    Uses a cheap LLM call to figure out:
    1. Is this a BROAD question (summary, overview, "what is this about")
       or a SPECIFIC question (asking for particular fields/values)?
    2. If specific, break it into individual search terms so multi-part
       questions ("what is X, Y, and Z") don't lose any part to retrieval
       competition.
    Falls back to treating it as one specific term if this call fails.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the user's question about a document. Respond ONLY with "
                        "a JSON object with two fields:\n"
                        '"is_broad": true if the question asks for a summary, overview, '
                        "general description, or 'what is this document about' — false if "
                        "it asks about specific facts, fields, or values.\n"
                        '"search_terms": a JSON array of distinct topics/fields being asked '
                        "about. For broad questions, return an empty array. For specific "
                        "questions, extract each distinct field, e.g. "
                        "'What is RBC, MCH and platelet count?' -> "
                        '["RBC Count", "MCH", "Platelet Count"]'
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "is_broad": result.get("is_broad", False),
            "search_terms": result.get("search_terms", [question])
        }
    except Exception as e:
        logger.warning(f"Question analysis failed, using fallback: {e}")
        return {"is_broad": False, "search_terms": [question]}


# ---------------------------------------------------------------------------
# STEP 6: RETRIEVAL
# ---------------------------------------------------------------------------

def retrieve_chunks_for_term(document_id: int, term: str, db: Session, k: int = 5):
    """Retrieves the top-k most similar chunks for a single search term."""
    embedding = get_embedding(term)
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.embedding.cosine_distance(embedding))
        .limit(k)
        .all()
    )


def retrieve_all_chunks(document_id: int, db: Session):
    """
    Retrieves EVERY chunk for a document, ordered by original position.
    Used for broad/summary questions where the whole document matters,
    not just the top-matching fragments.
    """
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index)
        .all()
    )


# ---------------------------------------------------------------------------
# STEP 7: QUESTION ANSWERING (RETRIEVAL + GENERATION)
# ---------------------------------------------------------------------------

def answer_question(question: str, document_id: int, db: Session) -> dict:
    """
    Handles ANY kind of question about the document:
    - Broad questions (summary, overview) -> pull the whole document
      (up to a safe token limit) so nothing is missed
    - Specific/multi-part questions -> decompose into individual search
      terms, retrieve each separately, merge and deduplicate
    """
    analysis = analyze_question(question)
    logger.info(f"Question analysis: {analysis}")

    seen_ids = set()
    merged_chunks = []

    if analysis["is_broad"]:
        # Broad question: pull the entire document, capped to avoid
        # excessive token usage on very long PDFs.
        all_chunks = retrieve_all_chunks(document_id, db)
        MAX_CHUNKS_FOR_BROAD = 60  # ~500 chars each, keeps context reasonable
        merged_chunks = all_chunks[:MAX_CHUNKS_FOR_BROAD]
    else:
        search_terms = analysis["search_terms"] or [question]
        for term in search_terms:
            results = retrieve_chunks_for_term(document_id, term, db, k=5)
            for chunk in results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    merged_chunks.append(chunk)

    if not merged_chunks:
        return {
            "reply": "I couldn't find relevant information in this document.",
            "sources": []
        }

    context = "\n\n---\n\n".join(chunk.chunk_text for chunk in merged_chunks)

    system_prompt = (
        "You are a precise document assistant. Answer using ONLY the context "
        "provided below.\n\n"
        "For specific facts/values, they must appear verbatim in the context — "
        "never estimate, infer, or guess. If the user asks for multiple values "
        "and only some are present, answer what you can find and clearly say "
        "'Not found in the provided context' for each missing one.\n\n"
        "For summary or overview questions, synthesize a clear, well-organized "
        "answer covering the key sections and information present in the context."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0
    )

    answer = response.choices[0].message.content
    sources = [chunk.chunk_text[:200] + "..." for chunk in merged_chunks[:10]]  # cap displayed sources

    return {"reply": answer, "sources": sources}