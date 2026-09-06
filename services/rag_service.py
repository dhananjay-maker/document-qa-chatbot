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
# STEP 5: PRODUCTION GUARDRAIL — REQUEST INTENT CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_request_intent(question: str) -> dict:
    """
    Production guardrail: classifies the user's request BEFORE running
    the full RAG pipeline, to catch off-topic, malformed, code-generation,
    or raw-data-dump requests early with a clean, safe response instead
    of letting them fall through to confusing or unsafe outputs.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's message about a document into exactly one "
                        "category. Respond ONLY with JSON: {\"category\": \"...\"}\n\n"
                        "Categories:\n"
                        "- 'document_question': A genuine question about the document's content, "
                        "including short/single-word topic mentions (e.g. 'RAG', 'skills', "
                        "'other use case') — these should be treated as document questions, "
                        "NOT unclear, since users often type short queries.\n"
                        "- 'code_generation_request': Explicitly asking to WRITE or GENERATE new "
                        "code, not asking what the document says about a technical topic.\n"
                        "- 'raw_data_request': Explicitly asking for raw chunks, internal data "
                        "dumps, or debug-style output rather than a normal answer.\n"
                        "- 'off_topic': Clearly unrelated to any document (weather, general "
                        "chitchat, requests with no connection to documents at all)."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return {"category": "document_question"}  # fail open


# ---------------------------------------------------------------------------
# STEP 6: QUESTION CLASSIFICATION + DECOMPOSITION
# ---------------------------------------------------------------------------

def analyze_question(question: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the user's message about a document, even if it's a single "
                        "word or short fragment — infer their likely intent as a genuine "
                        "question about that topic. Respond ONLY with a JSON object with "
                        "two fields:\n"
                        '"is_broad": true if this is a general/overview question about a '
                        "topic — including single-word topic mentions (treat 'RAG' as 'tell "
                        "me about RAG'), 'tell me about X', summaries, or 'what is this "
                        "document about'. false only for narrow questions asking about one "
                        "specific isolated fact or value.\n"
                        '"search_terms": a JSON array of distinct topics/fields being asked '
                        "about, with synonyms included. Empty array for broad questions."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "is_broad": result.get("is_broad", True),
            "search_terms": result.get("search_terms", [question])
        }
    except Exception as e:
        logger.warning(f"Question analysis failed, using fallback: {e}")
        return {"is_broad": True, "search_terms": [question]}


# ---------------------------------------------------------------------------
# STEP 7: RETRIEVAL
# ---------------------------------------------------------------------------

def retrieve_chunks_for_term(document_id: int, term: str, db: Session, k: int = 8):
    embedding = get_embedding(term)
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.embedding.cosine_distance(embedding))
        .limit(k)
        .all()
    )


def retrieve_all_chunks(document_id: int, db: Session):
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index)
        .all()
    )


# ---------------------------------------------------------------------------
# STEP 8: ANSWER GENERATION (shared by primary attempt and fallback)
# ---------------------------------------------------------------------------

def _generate_answer(question: str, chunks: list) -> dict:
    context = "\n\n---\n\n".join(chunk.chunk_text for chunk in chunks)

    wants_detail = any(
        word in question.lower()
        for word in ["detail", "in-depth", "explain fully", "elaborate", "comprehensive", "thorough", "full"]
    )

    length_instruction = (
        "Give a detailed, well-organized answer covering the key points."
        if wants_detail else
        "Keep your answer SHORT and DIRECT — 2-4 sentences for most questions. "
        "Do not use headers, numbered sections, or bold formatting unless the "
        "question specifically asks for a list. Get straight to the point."
    )

    system_prompt = (
        "You are a precise document assistant. Answer using ONLY the context "
        "provided below — never use outside knowledge to fill gaps.\n\n"
        f"{length_instruction}\n\n"
        "CRITICAL: Read the ENTIRE context carefully. If a term is mentioned or "
        "briefly explained (even just an acronym expansion) but not explained in "
        "full depth, say what IS present rather than claiming nothing is found. "
        "Only say information is missing if you have genuinely checked the entire "
        "context and it truly does not appear.\n\n"
        "NEVER include raw formatting artifacts, page numbers, barcode text, or "
        "OCR noise from the source document in your answer — synthesize a clean, "
        "readable response in your own words.\n\n"
        "SAFETY RULE FOR MEDICAL/HEALTH DOCUMENTS: Never recommend contacting "
        "specific named individuals (lab technicians, pathologists, report "
        "signatories) as substitutes for consulting an actual treating doctor. "
        "If asked for medical advice or next steps, clearly state you cannot "
        "provide medical guidance and recommend consulting a qualified "
        "healthcare professional — do not name lab staff as 'doctors to contact.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=150 if not wants_detail else 600
    )

    answer = response.choices[0].message.content
    sources = [chunk.chunk_text[:200] + "..." for chunk in chunks[:10]]

    return {"reply": answer, "sources": sources}


# ---------------------------------------------------------------------------
# STEP 9: MAIN QUESTION ANSWERING (guardrails + fallback escalation)
# ---------------------------------------------------------------------------

def answer_question(question: str, document_id: int, db: Session) -> dict:
    intent = classify_request_intent(question)
    category = intent.get("category", "document_question")
    logger.info(f"Request intent: {category}")

    if category == "off_topic":
        return {
            "reply": "I can only answer questions about this document. Try asking something specific about its content!",
            "sources": []
        }

    if category == "code_generation_request":
        return {
            "reply": "I can explain concepts from the document, but I don't generate new code that isn't in the document itself. Try asking me to explain the concept instead.",
            "sources": []
        }

    if category == "raw_data_request":
        return {
            "reply": "I can summarize or answer specific questions about the document, but I can't dump raw internal data. What would you like to know?",
            "sources": []
        }

    # category == "document_question" — proceed with normal RAG pipeline
    analysis = analyze_question(question)
    logger.info(f"Question analysis: {analysis}")

    seen_ids = set()
    merged_chunks = []

    if analysis["is_broad"]:
        all_chunks = retrieve_all_chunks(document_id, db)
        merged_chunks = all_chunks[:60]
    else:
        search_terms = analysis["search_terms"] or [question]
        for term in search_terms:
            results = retrieve_chunks_for_term(document_id, term, db, k=8)
            for chunk in results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    merged_chunks.append(chunk)

    if not merged_chunks:
        return {
            "reply": "I couldn't find relevant information in this document.",
            "sources": []
        }

    result = _generate_answer(question, merged_chunks)

    not_found_phrases = ["does not provide", "not found", "couldn't find", "no information", "not mentioned"]
    if not analysis["is_broad"] and any(phrase in result["reply"].lower() for phrase in not_found_phrases):
        logger.info("Narrow retrieval failed to answer — escalating to full document.")
        all_chunks = retrieve_all_chunks(document_id, db)
        fallback_result = _generate_answer(question, all_chunks[:60])
        if not any(phrase in fallback_result["reply"].lower() for phrase in not_found_phrases):
            return fallback_result

    return result