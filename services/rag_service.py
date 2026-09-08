import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
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
        chunk_size=800,
        chunk_overlap=150,
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
# STEP 4: DOCUMENT PROCESSING PIPELINE (unchanged — pgvector, persistent)
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
# STEP 5: AGENT-BASED RETRIEVAL (persistent pgvector backend)
# ---------------------------------------------------------------------------

def answer_question(question: str, document_id: int, db: Session) -> dict:
    """
    Uses a ReAct agent (LangGraph) that decides when and how to search the
    document — including retrying with a different query if the first
    search doesn't find what's needed. Retrieval hits PostgreSQL/pgvector
    (persistent), not an in-memory store, so results survive restarts.
    """
    retrieved_sources = []

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve document passages relevant to the query. Always call this
        before answering any question about the document. If the first
        result doesn't seem sufficient, call again with a rephrased query."""
        try:
            embedding = get_embedding(query)
            results = (
                db.query(DocumentChunkModel)
                .filter(DocumentChunkModel.document_id == document_id)
                .order_by(DocumentChunkModel.embedding.cosine_distance(embedding))
                .limit(6)
                .all()
            )
            if not results:
                return "No relevant context found."

            for r in results:
                snippet = r.chunk_text[:200] + "..."
                if snippet not in retrieved_sources:
                    retrieved_sources.append(snippet)

            return "\n\n".join(r.chunk_text for r in results)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return f"Retrieval error: {e}"

    system_prompt = (
        "You are a helpful assistant that answers questions based on the uploaded "
        "document.\n\n"
        "RULES:\n"
        "- ALWAYS call retrieve_context before answering ANY question about the document.\n"
        "- If the first retrieval does not find what you need, call retrieve_context "
        "again with a different/rephrased query before giving up.\n"
        "- Answer only based on retrieved content — never use outside knowledge to "
        "fill gaps or make up information.\n"
        "- If the answer genuinely isn't in the document after retrying, say so clearly.\n"
        "- Be concise (2-4 sentences for most answers) unless the user asks for detail.\n"
        "- NEVER dump raw chunks, debug data, or write new code that isn't in the "
        "document — explain concepts in your own words instead.\n"
        "- For medical/health documents: never name lab staff or report signatories as "
        "'doctors to contact' — always recommend consulting an actual healthcare "
        "professional for medical advice.\n"
        "- If the question is clearly unrelated to any document (e.g. weather, general "
        "chitchat), politely say you can only answer questions about the uploaded document."
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    agent = create_react_agent(model=llm, tools=[retrieve_context], prompt=system_prompt)

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = response["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        answer = "Sorry, something went wrong while processing your question. Please try again."

    return {"reply": answer, "sources": retrieved_sources[:10]}