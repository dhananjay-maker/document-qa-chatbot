import os
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
    (critical for lab reports / structured documents where naive text
    extraction scrambles column order).
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
    """
    Splits text into smaller, tightly-bounded chunks. Smaller chunks work
    better for dense tabular data (lab reports, invoices, etc.) because
    they keep individual rows/facts isolated.
    """
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
    """
    Full pipeline: extract -> chunk -> embed -> store DIRECTLY in
    PostgreSQL via pgvector. This persists properly across container
    restarts, unlike the earlier file-based Chroma approach.
    """
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
# STEP 5: QUESTION ANSWERING (RETRIEVAL + GENERATION)
# ---------------------------------------------------------------------------

def answer_question(question: str, document_id: int, db: Session) -> dict:
    """
    Embeds the question, finds the most similar chunks for THIS document
    using pgvector's cosine distance search directly in SQL, then asks
    the LLM to answer grounded in that retrieved context.
    """
    question_embedding = get_embedding(question)

    relevant_chunks = (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.embedding.cosine_distance(question_embedding))
        .limit(6)
        .all()
    )

    if not relevant_chunks:
        return {"reply": "I couldn't find relevant information in this document.", "sources": []}

    context = "\n\n---\n\n".join(chunk.chunk_text for chunk in relevant_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document assistant. Answer ONLY using the "
                "context provided below. For every value you state, it must "
                "appear verbatim in the context — never estimate, infer, or guess.\n\n"
                "If the answer isn't in the context, say so clearly."
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
    sources = [chunk.chunk_text[:200] + "..." for chunk in relevant_chunks]

    return {"reply": answer, "sources": sources}